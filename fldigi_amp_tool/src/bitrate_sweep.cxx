// SPDX-License-Identifier: GPL-3.0-or-later
// bitrate_sweep - standalone diagnostic CLI (not part of the shipped
// fldigi_amp_tool): sweeps MP3 bitrates and reports block-level
// recovery quality, to help pick an optimal bitrate (smallest file
// with comfortable margin), not just a pass/fail cliff.
//
// Uses a simple, ASCII/Baudot-safe block protocol (uppercase hex +
// per-block CRC16, one block per line, single pass - no repeat, so the
// numbers reflect raw per-bitrate channel quality, not the 3x-repeat
// mechanism already characterized separately) over two payload modems:
//   - MFSK128L: fldigi's real payload modem (Viterbi FEC + long interleave)
//   - RTTY: fldigi bakes Viterbi FEC into every MFSK submode
//     unconditionally (checked mfsk.cxx - MFSK128 vs MFSK128L differ
//     only in interleave depth), so there's no FEC-off MFSK sibling.
//     RTTY has no FEC at all and is already linked, so it stands in as
//     the "similar modem without FEC" comparator.
//
// Usage: bitrate_sweep <input-file> <workdir> [bitrate_kbps...]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <fcntl.h>
#include <sndfile.h>

#include "config.h"
#include "threads.h"
#include "trx.h"
#include "modem.h"
#include "mfsk.h"
#include "rtty.h"
#include "sound.h"
#include "configuration.h"

#include "harness_io.h"

// CRC16 - same algorithm as amp_proto.cxx (verbatim port of FLAMP's
// Ccrc16), duplicated here so this diagnostic has no dependency on the
// AMP-2 wire format (its block protocol is deliberately simpler/safe
// for both RTTY and MFSK - see header comment).
static uint16_t crc16(const std::string& s)
{
	uint32_t crc = 0xFFFF;
	for (unsigned char c : s) {
		crc ^= c;
		for (int i = 0; i < 8; i++)
			crc = (crc & 1) ? (0xA001 ^ (crc >> 1)) : (crc >> 1);
	}
	return (uint16_t)(crc & 0xFFFF);
}

static std::string to_hex(const std::vector<uint8_t>& raw)
{
	static const char* hexd = "0123456789ABCDEF";
	std::string out;
	out.reserve(raw.size() * 2);
	for (uint8_t b : raw) { out += hexd[b >> 4]; out += hexd[b & 0xF]; }
	return out;
}

static std::vector<uint8_t> from_hex(const std::string& hex)
{
	std::vector<uint8_t> out;
	auto val = [](char c) -> int {
		if (c >= '0' && c <= '9') return c - '0';
		if (c >= 'A' && c <= 'F') return c - 'A' + 10;
		return -1;
	};
	for (size_t i = 0; i + 1 < hex.size(); i += 2) {
		int hi = val(hex[i]), lo = val(hex[i + 1]);
		if (hi < 0 || lo < 0) break;
		out.push_back((uint8_t)((hi << 4) | lo));
	}
	return out;
}

// "BLK <n> <crc4hex> <hexdata>\n" - letters/digits/space/newline only,
// safe for RTTY's 5-bit Baudot/ITA2 alphabet (unlike AMP-2's own
// '<' '>' '{' '}' framing, which RTTY silently eats - see main.cxx).
static std::string build_blocks(const std::vector<uint8_t>& raw, int blocksize_hexchars)
{
	std::string hex = to_hex(raw);
	int numblocks = (int)(hex.size() / blocksize_hexchars + (hex.size() % blocksize_hexchars ? 1 : 0));
	std::string out;
	for (int b = 0; b < numblocks; b++) {
		std::string chunk = hex.substr((size_t)b * blocksize_hexchars, blocksize_hexchars);
		char hdr[32];
		snprintf(hdr, sizeof(hdr), "BLK %d %04X ", b + 1, crc16(chunk));
		out += hdr;
		out += chunk;
		out += "\n";
	}
	return out;
}

// MFSK128L's deinterleaver is a cascade of `depth` (800) diagonal
// shift-registers chained on every symbol (interleave.cxx:57-76) - a real
// fill-latency of ~depth*symbits symbols (~25s of audio) before the
// decode pipeline is warmed up, regardless of channel quality. Sending
// numbered blocks starting at t=0 (as this tool did) charges that
// unavoidable cold-start cost against the very thing this sweep is
// supposed to isolate: per-bitrate compression loss. Prepend disposable
// junk "BLK 0" lines (never counted - parse_blocks/total_blocks only
// look at 1..N) so the interleaver is already primed before the first
// real, measured block starts.
static std::string build_warmup(int blocksize_hexchars, int lines)
{
	std::string out;
	for (int i = 0; i < lines; i++) {
		std::string chunk(blocksize_hexchars, '0' + (i % 10));
		char hdr[32];
		snprintf(hdr, sizeof(hdr), "BLK 0 %04X ", crc16(chunk));
		out += hdr;
		out += chunk;
		out += "\n";
	}
	return out;
}

struct BlockRxResult {
	int numblocks_sent = 0;
	std::map<int, std::string> blocks; // 1-based -> hex chunk (CRC-verified)
};

static void parse_blocks(const std::string& text, BlockRxResult& r)
{
	size_t pos = 0;
	while (true) {
		size_t p = text.find("BLK ", pos);
		if (p == std::string::npos) break;
		int blknum; unsigned crc;
		int consumed = 0;
		if (sscanf(text.c_str() + p, "BLK %d %4X %n", &blknum, &crc, &consumed) != 2) {
			pos = p + 4;
			continue;
		}
		size_t data_start = p + consumed;
		size_t nl = text.find('\n', data_start);
		if (nl == std::string::npos) nl = text.size();
		std::string chunk = text.substr(data_start, nl - data_start);
		// trim trailing whitespace/CR that RTTY/MFSK varicode framing may add
		while (!chunk.empty() && (chunk.back() == '\r' || chunk.back() == ' ')) chunk.pop_back();
		if (crc16(chunk) == crc && blknum > 0)
			r.blocks[blknum] = chunk;
		pos = nl;
	}
}

static void tx_phase(modem* m, const std::string& text)
{
	active_modem = m;
	modem::XMLRPC_CPS_TEST = true;
	// modem::stopflag is only cleared in modem::init()/mfsk's own
	// constructor - rtty's constructor leaves it uninitialized (its
	// own init() sets it, but init() is an RX/TX-shared setup step this
	// headless harness never calls, only tx_init()). Clear it
	// explicitly so a stray nonzero garbage value doesn't make
	// tx_process() think a stop was already requested on the first call.
	m->set_stopflag(false);
	harness_tx_reset((const uint8_t*)text.data(), text.size());
	m->tx_init();
	int guard = 0;
	while (m->tx_process() >= 0) {
		if (++guard > 200000000) break;
	}
}

static void tx_to_wav(modem* m, const std::string& text, const std::string& wavpath)
{
	active_modem = m;
	TXscard = new SoundNull();
	TXscard->Open(O_WRONLY, m->get_samplerate());
	TXscard->startGenerate(wavpath, SF_FORMAT_WAV | SF_FORMAT_PCM_16);
	tx_phase(m, text);
	active_modem = nullptr;
	TXscard->stopGenerate();
	TXscard->Close();
	delete TXscard; TXscard = nullptr;
}

static size_t wav_frame_count(const std::string& path)
{
	SF_INFO info; memset(&info, 0, sizeof(info));
	SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
	if (!f) return 0;
	size_t frames = (size_t)info.frames;
	sf_close(f);
	return frames;
}

// modem::get_metric() (modem.h:154) exposes the live Viterbi decode-confidence
// value (0-100, mfsk.cxx:583-587) - the same "signal quality" number fldigi's
// GUI shows, and the FEC decoder's own real-time sense of how marginal each
// decode is. A single min/avg over the whole multi-hundred-second RX pass
// is too coarse to say anything about one specific block out of 345 - it
// gets smoothed away by the other 344. Record the full per-chunk time
// series instead, so a caller can look at the metric *locally*, in the
// window a specific block was actually being decoded.
struct MetricStats {
	double sum = 0.0;
	double min_ss = 1e9; // steady-state min (post warm-up)
	size_t count = 0;
	std::vector<double> series; // one sample per rx_process() call, SCBLOCKSIZE/samplerate apart
};

static std::string rx_from_wav(modem* m, const std::string& wavpath, size_t tail_flush_blocks,
	MetricStats* mstats = nullptr)
{
	active_modem = m;
	m->set_stopflag(false);
	bHighSpeed = true;
	RXscard = new SoundNull();
	RXscard->Open(O_RDONLY, m->get_samplerate());
	RXscard->startPlayback(wavpath, 0);
	harness_rx_reset();
	m->rx_init();

	size_t frames = wav_frame_count(wavpath);
	size_t blocks = frames / SCBLOCKSIZE + 1;
	size_t total_iters = blocks + tail_flush_blocks;
	size_t warmup_skip = total_iters / 10;
	std::vector<float> fbuf(SCBLOCKSIZE);
	std::vector<double> dbuf(SCBLOCKSIZE);
	for (size_t b = 0; b < total_iters; b++) {
		bHighSpeed = true;
		size_t n = RXscard->Read(fbuf.data(), SCBLOCKSIZE);
		if (n == 0) n = SCBLOCKSIZE;
		for (size_t i = 0; i < n; i++) dbuf[i] = fbuf[i];
		m->rx_process(dbuf.data(), (int)n);
		if (mstats) {
			double met = m->get_metric();
			mstats->series.push_back(met);
			if (b >= warmup_skip) {
				mstats->sum += met;
				mstats->count++;
				if (met < mstats->min_ss) mstats->min_ss = met;
			}
		}
	}
	RXscard->Close();
	delete RXscard; RXscard = nullptr;
	active_modem = nullptr;
	auto& bytes = harness_rx_bytes();
	return std::string(bytes.begin(), bytes.end());
}

// Look at the metric *locally* around a specific line (warmup or real block)
// in the TX sequence, instead of over the whole run. line_index is 0-based
// position in the full transmitted sequence (warmup lines then real blocks,
// in order); total_lines is the full sequence length. Returns local min
// over the samples whose estimated timestamp falls in that line's window.
static double local_min_metric(const MetricStats& mstats, int line_index, int total_lines, double samples_per_sec)
{
	if (mstats.series.empty() || total_lines <= 0) return -1.0;
	double sample_period_s = (double)SCBLOCKSIZE / samples_per_sec;
	double total_time_s = mstats.series.size() * sample_period_s;
	double line_dur_s = total_time_s / total_lines;
	double win_start = line_index * line_dur_s;
	double win_end = win_start + line_dur_s;
	size_t i0 = (size_t)std::max(0.0, win_start / sample_period_s);
	size_t i1 = (size_t)std::min((double)mstats.series.size(), win_end / sample_period_s + 1);
	double mn = 1e9;
	for (size_t i = i0; i < i1 && i < mstats.series.size(); i++)
		if (mstats.series[i] < mn) mn = mstats.series[i];
	return mn < 1e9 ? mn : -1.0;
}

static bool mp3_roundtrip(const std::string& in_wav, const std::string& out_wav, int kbps)
{
	std::string mp3 = in_wav + "." + std::to_string(kbps) + ".mp3";
	char cmd[2048];
	snprintf(cmd, sizeof(cmd),
		"ffmpeg -y -loglevel error -i '%s' -ar 8000 -b:a %dk -c:a libmp3lame '%s' && "
		"ffmpeg -y -loglevel error -i '%s' -ar 8000 '%s'",
		in_wav.c_str(), kbps, mp3.c_str(), mp3.c_str(), out_wav.c_str());
	return system(cmd) == 0;
}

// Runs the post-MP3 WAV through harness/airgap_sim.py (speaker/mic acoustic
// path: bandpass, soft-clip, echo, noise, dropouts/cutouts/repeats - see
// REQUIREMENTS.md's airgap step and SUMMARY.md task #5). Order matches the
// real pipeline: MP3 compression happens at upload time, the acoustic path
// happens later at physical playback, so airgap runs on the MP3-roundtrip
// output, not the clean pre-MP3 WAV. Toggled by SWEEP_AIRGAP=1 (off by
// default, so plain MP3-only sweeps are unaffected). SWEEP_AIRGAP_SCRIPT
// overrides the script path (default: airgap_sim.py, relative to cwd, same
// as this tool is normally run from harness/). SWEEP_AIRGAP_SEED (default 0)
// keeps the same glitch pattern across bitrates in one sweep, so bitrate is
// the only thing varying. SWEEP_AIRGAP_ARGS passes extra raw args straight
// through to airgap_sim.py (e.g. "--snr-db 12 --no-glitches").
static bool airgap_sim(const std::string& in_wav, const std::string& out_wav)
{
	std::string script = getenv("SWEEP_AIRGAP_SCRIPT") ? getenv("SWEEP_AIRGAP_SCRIPT") : "airgap_sim.py";
	int seed = getenv("SWEEP_AIRGAP_SEED") ? atoi(getenv("SWEEP_AIRGAP_SEED")) : 0;
	std::string extra_args = getenv("SWEEP_AIRGAP_ARGS") ? getenv("SWEEP_AIRGAP_ARGS") : "";
	char cmd[2048];
	snprintf(cmd, sizeof(cmd), "python3 '%s' '%s' '%s' --seed %d %s >&2",
		script.c_str(), in_wav.c_str(), out_wav.c_str(), seed, extra_args.c_str());
	return system(cmd) == 0;
}

struct ModemSpec {
	const char* name;
	trx_mode mode;
	bool is_mfsk; // vs rtty
};

int main(int argc, char** argv)
{
	SET_THREAD_ID(FLMAIN_TID);
	if (argc < 3) {
		fprintf(stderr, "usage: %s <input-file> <workdir> [bitrate_kbps...]\n", argv[0]);
		fprintf(stderr, "  default bitrates: 8 16 24 32 40 48 56 64 80 96 112\n");
		return 2;
	}
	std::string in_path = argv[1];
	std::string workdir = argv[2];

	std::vector<int> bitrates;
	if (argc > 3) {
		for (int i = 3; i < argc; i++) bitrates.push_back(atoi(argv[i]));
	} else {
		bitrates = {8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112};
	}

	progdefaults.wavSampleRate = 0;
	progdefaults.sample_converter = 0;
	progdefaults.record_both_channels = false;
	progdefaults.rsid = false;
	progdefaults.rx_lowercase = false;
	// RTTY TX auto-wraps at progdefaults.rtty_autocount (default 72) chars by
	// splicing in a CR+LF - fine for human-readable prose, but our BLK lines
	// run ~75-77 chars, so it was silently corrupting every block's payload
	// before transmission even started (nothing to do with the channel).
	progdefaults.rtty_autocrlf = false;

	FILE* f = fopen(in_path.c_str(), "rb");
	if (!f) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
	std::vector<uint8_t> raw;
	uint8_t buf[65536]; size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) raw.insert(raw.end(), buf, buf + n);
	fclose(f);

	// Same blocksize (in hex chars) for both modems, single pass, no
	// repeat - deliberately isolates raw per-bitrate channel quality.
	// RTTY runs at ~45 baud (a few chars/sec) - a multi-bitrate sweep
	// over the full input would take hours, so RTTY gets a documented
	// truncated prefix of the same input instead of the whole thing;
	// it's a qualitative no-FEC comparator, not a second product.
	const int blocksize_hexchars = 64;
	const size_t rtty_bytes = raw.size() < 400 ? raw.size() : 400;
	std::vector<uint8_t> raw_rtty(raw.begin(), raw.begin() + rtty_bytes);

	std::vector<ModemSpec> modems = {
		{"MFSK128L(FEC)", MODE_MFSK128L, true},
		{"RTTY(no-FEC)", MODE_RTTY, false},
	};
	std::map<std::string, int> total_blocks_by_modem;
	std::map<std::string, std::string> clean_wav;
	int mfsk_warmup_lines = getenv("SWEEP_WARMUP_LINES") ? atoi(getenv("SWEEP_WARMUP_LINES")) : 20;
	// Mirror of the warm-up fix, at the other end: the deep interleaver's
	// last real symbols run out of *future* context to spread against (the
	// same reason the first symbols run out of *past* context), so they get
	// less effective protection. Invisible on a clean channel (mfsk.cxx's
	// own flushtx(preamble) already drains it enough there - see the earlier
	// 100%-down-to-32kbps result), but under any real airgap degradation
	// this shows up as the same few trailing blocks failing at every
	// bitrate, regardless of compression. Append disposable trailing
	// "BLK 0" lines so the real last block isn't literally the last thing
	// sent through the interleaver.
	int mfsk_cooldown_lines = getenv("SWEEP_COOLDOWN_LINES") ? atoi(getenv("SWEEP_COOLDOWN_LINES")) : 20;
	for (auto& spec : modems) {
		const std::vector<uint8_t>& payload = spec.is_mfsk ? raw : raw_rtty;
		std::string blocktext = spec.is_mfsk ? build_warmup(blocksize_hexchars, mfsk_warmup_lines) : std::string();
		blocktext += build_blocks(payload, blocksize_hexchars);
		if (spec.is_mfsk)
			blocktext += build_warmup(blocksize_hexchars, mfsk_cooldown_lines);
		int total_blocks = (int)(to_hex(payload).size() / blocksize_hexchars +
			(to_hex(payload).size() % blocksize_hexchars ? 1 : 0));
		total_blocks_by_modem[spec.name] = total_blocks;
		fprintf(stderr, "[sweep] %s input: %zu bytes -> %d blocks%s\n", spec.name, payload.size(),
			total_blocks, spec.is_mfsk ? "" : " (truncated prefix - RTTY is too slow for the full file)");

		std::string wavpath = workdir + "/sweep_" + spec.name + ".wav";
		modem* m = spec.is_mfsk ? (modem*)new mfsk(spec.mode) : (modem*)new rtty(spec.mode);
		tx_to_wav(m, blocktext, wavpath);
		size_t frames = wav_frame_count(wavpath);
		fprintf(stderr, "[sweep] %s clean TX: %.1fs @ 8kHz\n", spec.name, frames / 8000.0);
		delete m;
		clean_wav[spec.name] = wavpath;
	}

	bool do_airgap = getenv("SWEEP_AIRGAP") && atoi(getenv("SWEEP_AIRGAP")) != 0;
	if (do_airgap)
		fprintf(stderr, "[sweep] airgap simulation ENABLED (SWEEP_AIRGAP=1) - "
			"applying harness/airgap_sim.py to each MP3-roundtrip WAV before decode\n");

	fprintf(stderr, "\n[sweep] %-6s  %-28s  %-28s\n", "kbps", "MFSK128L(FEC)", "RTTY(no-FEC)");
	for (int kbps : bitrates) {
		std::string row = "", metric_row = "";
		for (auto& spec : modems) {
			std::string rt_wav = workdir + "/sweep_rt_" + spec.name + ".wav";
			bool conv_ok = mp3_roundtrip(clean_wav[spec.name], rt_wav, kbps);
			std::string decode_wav = rt_wav;
			if (conv_ok && do_airgap) {
				std::string ag_wav = workdir + "/sweep_ag_" + spec.name + ".wav";
				conv_ok = airgap_sim(rt_wav, ag_wav);
				decode_wav = ag_wav;
			}
			char cell[64], metcell[64] = "";
			if (!conv_ok) {
				snprintf(cell, sizeof(cell), do_airgap ? "mp3/airgap-failed" : "mp3-encode-failed");
			} else {
				modem* m = spec.is_mfsk ? (modem*)new mfsk(spec.mode) : (modem*)new rtty(spec.mode);
				MetricStats mstats;
				std::string text = rx_from_wav(m, decode_wav, spec.is_mfsk ? 400 : 100, &mstats);
				delete m;
				BlockRxResult r;
				parse_blocks(text, r);
				int recovered = (int)r.blocks.size();
				int total_blocks = total_blocks_by_modem[spec.name];
				double pct = total_blocks ? 100.0 * recovered / total_blocks : 0.0;
				snprintf(cell, sizeof(cell), "%3d/%3d blocks (%5.1f%%)", recovered, total_blocks, pct);
				double avg_metric = mstats.count ? mstats.sum / mstats.count : 0.0;
				double min_metric = mstats.count ? mstats.min_ss : 0.0;
				snprintf(metcell, sizeof(metcell), "metric avg=%5.1f min=%5.1f", avg_metric, min_metric);
				// Whole-run min/avg is too coarse to say anything about one
				// specific failing block out of hundreds - look at the metric
				// LOCALLY, in the window each block was actually decoded in,
				// and compare lost blocks' local min against surviving ones'.
				if (spec.is_mfsk) {
					int total_lines = mfsk_warmup_lines + total_blocks + mfsk_cooldown_lines;
					std::vector<double> recovered_local_mins, lost_local_mins;
					std::vector<int> lost_block_nums;
					for (int b = 1; b <= total_blocks; b++) {
						int line_index = mfsk_warmup_lines + (b - 1);
						double lm = local_min_metric(mstats, line_index, total_lines, 8000.0);
						if (lm < 0) continue;
						if (r.blocks.count(b)) recovered_local_mins.push_back(lm);
						else { lost_local_mins.push_back(lm); lost_block_nums.push_back(b); }
					}
					double rec_avg = 0, rec_min = 1e9;
					for (double v : recovered_local_mins) { rec_avg += v; if (v < rec_min) rec_min = v; }
					if (!recovered_local_mins.empty()) rec_avg /= recovered_local_mins.size();
					fprintf(stderr, "[sweep]   MFSK128L @ %dkbps: per-block LOCAL min metric - "
						"surviving blocks avg=%.1f worst=%.1f (n=%zu)%s\n",
						kbps, rec_avg, recovered_local_mins.empty() ? -1.0 : rec_min,
						recovered_local_mins.size(), lost_block_nums.empty() ? ", no blocks lost" : "");
					for (size_t i = 0; i < lost_block_nums.size(); i++)
						fprintf(stderr, "[sweep]     lost block %d: local min metric = %.1f\n",
							lost_block_nums[i], lost_local_mins[i]);
				}
				if (getenv("SWEEP_DEBUG")) {
					fprintf(stderr, "[debug] %s @ %dkbps: raw decoded text = %zu bytes\n",
						spec.name, kbps, text.size());
					std::string missing;
					for (int b = 1; b <= total_blocks; b++)
						if (!r.blocks.count(b)) { missing += " "; missing += std::to_string(b); }
					fprintf(stderr, "[debug] %s @ %dkbps: missing blocks (%d):%s\n",
						spec.name, kbps, total_blocks - recovered, missing.c_str());
					fprintf(stderr, "[debug] %s @ %dkbps: raw text head(300)=%s\n",
						spec.name, kbps, text.substr(0, 300).c_str());
					fprintf(stderr, "[debug] %s @ %dkbps: raw text tail(300)=%s\n",
						spec.name, kbps, text.size() > 300 ? text.substr(text.size()-300).c_str() : text.c_str());
				}
			}
			row += "  ";
			row += cell;
			// pad
			while (row.size() % 30 != 0) row += ' ';
			metric_row += "  ";
			metric_row += metcell;
			while (metric_row.size() % 30 != 0) metric_row += ' ';
		}
		fprintf(stderr, "[sweep] %-6d%s\n", kbps, row.c_str());
		fprintf(stderr, "[sweep] %-6s%s\n", "", metric_row.c_str());
	}
	fprintf(stderr, "\n[sweep] note: single pass, no repeat - real fldigi_amp_tool applies 3x repeat\n"
		"[sweep] + per-block CRC on top of this, so its effective reliability at a given\n"
		"[sweep] bitrate is higher than the MFSK128L column alone suggests (see task #2\n"
		"[sweep] in SUMMARY.md for how much repeats buy you against transient vs static loss).\n");
	return 0;
}
