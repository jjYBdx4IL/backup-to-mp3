// SPDX-License-Identifier: GPL-3.0-or-later
// mfsk_tool - headless encode/decode using fldigi's real MFSK32
// (AMP-2 header/announce modem) and MFSK128L (AMP-2 payload modem, with
// Viterbi FEC + long interleave) modem objects, framed with the real
// AMP-2 protocol FLAMP uses - see amp_proto.h/.cxx. No fldigi process,
// no GUI, no soundcard: TX writes a WAV file at full CPU speed via
// fldigi's own SoundNull + startGenerate() fast path; RX reads it back
// the same way via startPlayback() + bHighSpeed.
//
// Header modem is MFSK32, not RTTY: RTTY's 5-bit Baudot/ITA2 alphabet
// has no representation for '<' '>' '{' '}' or lowercase letters, all
// of which appear in AMP-2's literal header line framing, so RTTY
// silently eats them in transit (verified empirically). MFSK's varicode
// is full-ASCII, matching what the MFSK128L payload channel already
// relies on for the very same framing.
//
// Usage:
//   mfsk_tool encode [--bitrate KBPS] [--samplerate HZ] [--no-verify] <input-file> <output.wav|output.mp3>
//   mfsk_tool decode <input.wav|input.mp3> <output-file>
//
// MP3 in/out (by output/input extension) goes straight through libmp3lame
// (encode) / libmpg123 (decode, mp3_io.cxx) - no ffmpeg process, no
// shellout; this is what used to be amp_pipeline.sh. The modem itself only
// ever runs at its fixed native rate (8kHz), so .mp3 output is resampled up
// to --samplerate (default 32kHz - some MP3 playback targets don't reliably
// accept 8kHz MP3s) via libsamplerate (resample_wav(), mp3_io.cxx) before
// compressing; .wav output stays at the modem's native 8kHz. Decode needs no
// counterpart resample: fldigi's own playback path already resamples an
// arbitrary source rate back down to the modem's native rate on the fly.
// encode to .mp3 verifies by default (decode the result back, byte-compare
// against the input, nonzero exit on mismatch) - skip with --no-verify.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <cctype>
#include <csignal>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sndfile.h>

#include "config.h"
#include "threads.h"
#include "trx.h"
#include "modem.h"
#include "mfsk.h"
#include "sound.h"
#include "configuration.h"

#include "flinput2.h"
#include "harness_io.h"
#include "amp_proto.h"
#include "mp3_io.h"

extern Fl_Input2 *inpFreq; // stubs.cxx: real (never-shown) widget; tag_file() reads its ->value()

// Center (carrier) frequency both modems transmit/receive around. fldigi's
// own modem::modem() constructor defaults this to 1000 Hz when there's no
// waterfall (always true here, headless), so encode/decode must both set
// it explicitly and identically or RX's bandpass mixing won't line up
// with what TX actually sent.
static const double CENTER_FREQ = 1500;

// TX volume: our headless harness never runs fldigi's own status-restore
// path, so progStatus.txlevel (fldigi's normal TXATTEN dB knob, applied
// inside modem::ModulateXmtr) sits at stubs.cxx's zero-initialized 0dB,
// meaning generated samples ride right up against modem.cxx's SIGLIMIT
// (0.95) clamp - too loud, and leaves no headroom for a real fldigi
// receiver's AFC (which tracks the tone off a signal that's already
// clipped/at full-scale less reliably than one with margin below it).
// Simpler to scale the finished WAV directly than to fight that clamp
// from inside the TX path.
static const double GAIN_DB = -10.0;

// encode_to_wav() always writes at this rate (progdefaults.wavSampleRate = 0
// there selects sndfile_samplerate[0] == 8000) since it's the modem's own
// fixed native rate - see DEFAULT_MP3_SAMPLERATE_HZ below.
static const int NATIVE_MODEM_SAMPLERATE_HZ = 8000;

// Default CBR MP3 bitrate when encode's output path ends in .mp3, matching
// amp_pipeline.sh's former hardcoded 112kbit/s - overridable with --bitrate.
static const int DEFAULT_MP3_KBPS = 112;

// The modem itself only ever runs at its fixed native rate (MFSKSampleRate
// == 8000 in fldigi's mfsk.h - not configurable per-mode), so encode_to_wav()
// always produces an 8kHz WAV regardless of this setting. Default MP3 output
// sample rate is 32kHz instead of that native rate - overridable with
// --samplerate - since some MP3 playback targets (e.g. the Huawei D2 watch
// this tool was built for) don't reliably accept 8kHz MP3s. Getting there is
// a resample_wav() pass (mp3_io.cxx, via libsamplerate) between encode_to_wav
// and mp3_encode_wav; decode needs no counterpart change since fldigi's own
// playback path already resamples an arbitrary source rate back down to the
// modem's native rate on the fly (see soundcard/sound.cxx read_file()).
static const int DEFAULT_MP3_SAMPLERATE_HZ = 32000;

// fwd decl, used by encode()'s verify pass. strict_redundancy requires every
// redundant copy of every header line and data block to have been recovered
// (not just one copy of each, the minimum amp_rx_complete() needs to
// reconstruct the file) - see decode_from_wav()'s definition below.
static void decode_from_wav(const std::string& in_wav, const std::string& out_path, bool strict_redundancy = false);

// --- temp-file cleanup on Ctrl-C / SIGTERM ---------------------------------
// encode()/decode() normally unlink() their own scratch files on every exit
// path (success rename or failure), but none of that runs if the process
// dies from a signal mid-operation - and rx_full_scan()/mp3_encode_wav() are
// long enough that a Ctrl-C during them is routine, not an edge case. So every
// live scratch path is tracked here and a SIGINT/SIGTERM/SIGHUP handler
// unlink()s all of them before the process actually dies. unlink() is
// POSIX-async-signal-safe, but growing/scanning an STL container isn't
// (it can allocate), so the registry is a fixed-size array of plain char
// buffers instead, mutated only with those signals blocked so the handler
// never observes it mid-update.
static const int MAX_TMP_FILES = 16;
static char g_tmp_paths[MAX_TMP_FILES][4096];
static volatile sig_atomic_t g_tmp_count = 0;

static void block_cleanup_signals(sigset_t* old_set)
{
	sigset_t set;
	sigemptyset(&set);
	sigaddset(&set, SIGINT);
	sigaddset(&set, SIGTERM);
	sigaddset(&set, SIGHUP);
	sigprocmask(SIG_BLOCK, &set, old_set);
}

static void register_tmp_file(const std::string& path)
{
	sigset_t old_set;
	block_cleanup_signals(&old_set);
	if (g_tmp_count < MAX_TMP_FILES) {
		strncpy(g_tmp_paths[g_tmp_count], path.c_str(), sizeof(g_tmp_paths[0]) - 1);
		g_tmp_paths[g_tmp_count][sizeof(g_tmp_paths[0]) - 1] = '\0';
		g_tmp_count++;
	}
	sigprocmask(SIG_SETMASK, &old_set, nullptr);
}

static void unregister_tmp_file(const std::string& path)
{
	sigset_t old_set;
	block_cleanup_signals(&old_set);
	for (sig_atomic_t i = 0; i < g_tmp_count; i++) {
		if (path == g_tmp_paths[i]) {
			for (sig_atomic_t j = i; j < g_tmp_count - 1; j++)
				memcpy(g_tmp_paths[j], g_tmp_paths[j + 1], sizeof(g_tmp_paths[0]));
			g_tmp_count--;
			break;
		}
	}
	sigprocmask(SIG_SETMASK, &old_set, nullptr);
}

// unlink() + drop-from-registry in one step - every unlink() of a tracked
// scratch path should go through this (instead of calling unlink()
// directly) so the registry never retains a stale entry for a file that's
// already gone by the time a signal might arrive.
static void cleanup_tmp_file(const std::string& path)
{
	unlink(path.c_str());
	unregister_tmp_file(path);
}

static void tmp_cleanup_signal_handler(int sig)
{
	for (sig_atomic_t i = 0; i < g_tmp_count; i++)
		unlink(g_tmp_paths[i]);
	// Restore the default disposition and re-raise instead of exit()ing
	// here, so the process still dies the normal way for this signal
	// (correct exit status, no stdio flushed from within a handler).
	signal(sig, SIG_DFL);
	raise(sig);
}

static void install_tmp_cleanup_handlers()
{
	struct sigaction sa;
	memset(&sa, 0, sizeof(sa));
	sa.sa_handler = tmp_cleanup_signal_handler;
	sigemptyset(&sa.sa_mask);
	sa.sa_flags = 0;
	sigaction(SIGINT, &sa, nullptr);
	sigaction(SIGTERM, &sa, nullptr);
	sigaction(SIGHUP, &sa, nullptr);
}

// Reserves a unique, already-created, hidden ("." prefixed) scratch file
// next to out_path (same directory, so the eventual rename() onto out_path
// is same-filesystem and atomic) via mkstemp() - collision-proof even
// against a second concurrent invocation targeting the same out_path,
// unlike hand-rolling a fixed "<out_path>.tmp" name. Callers reopen the
// returned path by name (libsndfile/LAME/mpg123 all want a path, not an
// fd), so the fd mkstemp() hands back is just closed here.
static std::string make_tmp_path(const std::string& out_path)
{
	size_t slash = out_path.find_last_of('/');
	std::string dir = (slash == std::string::npos) ? "." : out_path.substr(0, slash);
	std::string base = (slash == std::string::npos) ? out_path : out_path.substr(slash + 1);

	std::vector<char> tmpl(dir.begin(), dir.end());
	std::string rest = "/." + base + ".XXXXXX";
	tmpl.insert(tmpl.end(), rest.begin(), rest.end());
	tmpl.push_back('\0');

	int fd = mkstemp(tmpl.data());
	if (fd < 0) {
		fprintf(stderr, "cannot create temp file near %s: %s\n", out_path.c_str(), strerror(errno));
		exit(1);
	}
	// mkstemp() always creates at mode 0600, unlike a plain fopen()/sf_open()
	// write (0666 & ~umask) - since out_tmp gets rename()d onto out_path
	// as-is, fix the mode up front or the final output would silently end
	// up owner-only instead of matching what every other file this tool
	// writes gets.
	mode_t mask = umask(0); umask(mask);
	fchmod(fd, 0666 & ~mask);
	close(fd);
	std::string path(tmpl.data());
	register_tmp_file(path);
	return path;
}

static bool has_ext(const std::string& path, const std::string& ext)
{
	if (path.size() < ext.size()) return false;
	return std::equal(ext.rbegin(), ext.rend(), path.rbegin(), [](char a, char b) {
		return std::tolower((unsigned char)a) == std::tolower((unsigned char)b);
	});
}

static bool files_equal(const std::string& a_path, const std::string& b_path)
{
	FILE* a = fopen(a_path.c_str(), "rb");
	FILE* b = fopen(b_path.c_str(), "rb");
	if (!a || !b) { if (a) fclose(a); if (b) fclose(b); return false; }

	bool equal = true;
	char bufa[65536], bufb[65536];
	while (true) {
		size_t na = fread(bufa, 1, sizeof(bufa), a);
		size_t nb = fread(bufb, 1, sizeof(bufb), b);
		if (na != nb || memcmp(bufa, bufb, na) != 0) { equal = false; break; }
		if (na == 0) break; // both at EOF
	}
	fclose(a); fclose(b);
	return equal;
}

static void apply_gain_db(const std::string& path, double db)
{
	SF_INFO info; memset(&info, 0, sizeof(info));
	SNDFILE* in = sf_open(path.c_str(), SFM_READ, &info);
	if (!in) { fprintf(stderr, "cannot open %s for gain scaling\n", path.c_str()); exit(1); }

	double mult = pow(10.0, db / 20.0);
	std::vector<short> samples((size_t)info.frames * info.channels);
	sf_count_t got = sf_readf_short(in, samples.data(), info.frames);
	sf_close(in);

	for (auto& s : samples) {
		long v = std::lround(s * mult);
		if (v > 32767) v = 32767;
		if (v < -32768) v = -32768;
		s = (short)v;
	}

	SNDFILE* out = sf_open(path.c_str(), SFM_WRITE, &info);
	if (!out) { fprintf(stderr, "cannot rewrite %s with gain applied\n", path.c_str()); exit(1); }
	sf_writef_short(out, samples.data(), got);
	sf_close(out);
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

// Send an RSID/TxID burst announcing whatever modem is currently
// active_modem, exactly like fldigi's own real TX loop does once per TX
// stint (trx.cxx trx_trx_transmit_loop(), right after tx_init() and
// before the tx_process() loop begins) - so a genuine, unmodified fldigi
// listening in real time can auto-detect and switch to the modem we're
// about to use, instead of requiring the operator to have pre-selected it.
static void send_txid()
{
	fprintf(stderr, "[encode] TxID/RSID: announcing %s\n", active_modem->get_mode_name());
	ReedSolomon->send(/*preRSID=*/true);
}

// Runs one modem segment start-to-finish, with its own RSID/TxID burst up
// front. Each call is a distinct logical transmission (header or data) - see
// AmpTransmission - so one send_txid() per call is exactly the real fldigi
// behavior (one RSID per continuous TX stint).
static void tx_phase(modem* m, const std::string& text)
{
	active_modem = m;
	modem::XMLRPC_CPS_TEST = true; // skip realtime throttle in SoundNull::Write
	harness_tx_reset((const uint8_t*)text.data(), text.size());
	m->tx_init();
	send_txid();

	int guard = 0;
	while (true) {
		if (m->tx_process() < 0) break;
		if (++guard > 200000000) { fprintf(stderr, "tx_process guard tripped\n"); break; }
	}
}

static void encode_to_wav(const std::string& in_path, const std::string& out_wav)
{
	FILE* f = fopen(in_path.c_str(), "rb");
	if (!f) { fprintf(stderr, "cannot open %s\n", in_path.c_str()); exit(1); }
	std::vector<uint8_t> raw;
	uint8_t buf[65536]; size_t n;
	while ((n = fread(buf, 1, sizeof(buf), f)) > 0) raw.insert(raw.end(), buf, buf + n);
	fclose(f);

	AmpFileParams params;
	size_t slash = in_path.find_last_of("/\\");
	params.filename = (slash == std::string::npos) ? in_path : in_path.substr(slash + 1);

	AmpMessage msg = amp_build_message(raw, params);

	fprintf(stderr, "[encode] %zu bytes, hash=%s, %zu transmission(s)\n",
		raw.size(), msg.hash.c_str(), msg.transmissions.size());
	for (size_t t = 0; t < msg.transmissions.size(); t++) {
		fprintf(stderr, "[encode] transmission %zu/%zu header (MFSK32) text:\n%s\n",
			t + 1, msg.transmissions.size(), msg.transmissions[t].header_text.c_str());
		fprintf(stderr, "[encode] transmission %zu/%zu data (MFSK128L) text: %zu bytes\n",
			t + 1, msg.transmissions.size(), msg.transmissions[t].data_text.size());
	}

	progdefaults.wavSampleRate = 0; // sndfile_samplerate[0] == 8000 == RTTY/MFSK samplerate
	progdefaults.sample_converter = 0;
	progdefaults.record_both_channels = false;
	progdefaults.rsid = false;

	// SND_SUPPORT::tag_file() (soundcard/sound.cxx) unconditionally embeds
	// progdefaults.myCall as the WAV's SF_STR_ARTIST tag and inpFreq->value()
	// into its SF_STR_COMMENT tag ("<mode> freq=<freq>") - this is the WAV's
	// own identification metadata, separate from the in-band RSID/TxID audio
	// burst and the AMP-2 <ID> header line. Our headless harness never runs
	// configuration.cxx's readDefaultsXML() (so progdefaults.myCall defaults
	// to "") and never shows the real frequency-display widget (so inpFreq's
	// text defaults to ""), which left every produced WAV with an empty
	// artist tag and a comment of literally "MFSK32 freq=" with no value -
	// not a valid identification. Populate both from what we actually use.
	progdefaults.myCall = params.my_call;
	inpFreq->value(std::to_string((long)CENTER_FREQ).c_str());

	// TX-side RSID (TxID): announce the upcoming modem before each segment,
	// same as real fldigi's default (all modes enabled except CW/PSK31/RTTY -
	// see configuration.cxx readDefaultsXML(), which our headless harness
	// never runs). ReedSolomon itself is normally constructed by fldigi's
	// own trx_init(), which we also bypass, so build it here instead.
	if (!ReedSolomon) ReedSolomon = new cRsId;
	progdefaults.rsid_tx_modes.set();

	// Open the WAV once, up front - tag_file() (called from startGenerate) needs
	// a live active_modem to dereference, so a throwaway probe modem covers just
	// that call. MFSK32 and MFSK128L share the same 8000Hz samplerate, so which
	// one opens the file doesn't matter.
	mfsk* probe = new mfsk(MODE_MFSK32);
	active_modem = probe;
	TXscard = new SoundNull();
	TXscard->Open(O_WRONLY, probe->get_samplerate());
	TXscard->startGenerate(out_wav, SF_FORMAT_WAV | SF_FORMAT_PCM_16);
	active_modem = nullptr;
	delete probe;

	for (size_t t = 0; t < msg.transmissions.size(); t++) {
		AmpTransmission& tx = msg.transmissions[t];

		fprintf(stderr, "[encode] transmission %zu/%zu: header (MFSK32, x%d)...\n",
			t + 1, msg.transmissions.size(), params.repeat_header);
		mfsk* hdr = new mfsk(MODE_MFSK32);
		hdr->set_freq(CENTER_FREQ);
		tx_phase(hdr, tx.header_text);
		active_modem = nullptr;
		delete hdr;

		fprintf(stderr, "[encode] transmission %zu/%zu: payload (MFSK128L)...\n",
			t + 1, msg.transmissions.size());
		mfsk* pay = new mfsk(MODE_MFSK128L);
		pay->set_freq(CENTER_FREQ);
		tx_phase(pay, tx.data_text);
		active_modem = nullptr;
		delete pay;
	}

	TXscard->stopGenerate();
	TXscard->Close();
	delete TXscard; TXscard = nullptr;

	fprintf(stderr, "[encode] applying %.1fdB gain to %s (headroom for fldigi's AFC)...\n", GAIN_DB, out_wav.c_str());
	apply_gain_db(out_wav, GAIN_DB);

	size_t frames = wav_frame_count(out_wav);
	fprintf(stderr, "[encode] wrote %s (%zu frames, %.1fs @ 8kHz)\n", out_wav.c_str(), frames, frames / 8000.0);
}

// Top-level encode: WAV output goes straight through encode_to_wav(), same
// as always (this is what the test suite drives). MP3 output additionally
// compresses it directly via libmp3lame (mp3_io.cxx) - no more shelling out
// to ffmpeg. The intermediate WAV is written to a mkstemp()-reserved hidden
// scratch file right next to the output, not off in /tmp - keeps it on the
// same filesystem and easy to find if a step fails.
//
// Neither branch ever writes to out_path directly: everything lands in a
// make_tmp_path(out_path) scratch file first (same dir, so the final
// rename() is atomic - no window where a reader/watcher can see a
// truncated or half-written file at out_path) and gets renamed into place
// only once it's known good. Any failure along the way (encode_to_wav's own
// exit(1) calls, a bad bitrate/channel combo, a verify mismatch) unlinks
// the tmp file instead of leaving it - or a partial out_path - behind.
//
// amp_pipeline.sh used to gate this step on `checksilence` (any non-edge
// silence gap in the pre-MP3 WAV aborts). That check can't tell a harmless
// pause between two independently-decoded modem TX phases (header vs.
// payload, or between repeated transmissions - routinely 0.4-0.9s here)
// apart from a real dropout (airgap_sim.py's simulated glitches are much
// shorter, 15-120ms) - it fires on every multi-transmission message
// regardless of whether anything is actually wrong, as proved by
// roundtrip_test.sh/roundtrip_generated_archive_test.sh decoding these
// exact gaps byte-exact every time. Dropped in favor of the verify pass
// below, which checks the thing that actually matters (does the finished
// MP3 decode back to the original) instead of a heuristic proxy for it.
//
// When verify is true (the default - see main()'s --no-verify), the
// finished MP3 is decoded back through the full AMP-2 decode path (straight
// off the pre-rename tmp file - has_ext()'s ".mp3" check wouldn't match a
// mkstemp() name, so this calls mp3_decode_to_wav/decode_from_wav directly
// instead of going through decode()'s extension dispatch) and diffed
// byte-for-byte against in_path, so a bad bitrate/channel combo is caught
// right here instead of silently producing an undecodable file that only
// fails later on the watch.
static void encode(const std::string& in_path, const std::string& out_path, int mp3_bitrate_kbps, int mp3_samplerate_hz, bool verify)
{
	std::string out_tmp = make_tmp_path(out_path);

	if (!has_ext(out_path, ".mp3")) {
		encode_to_wav(in_path, out_tmp);
		if (rename(out_tmp.c_str(), out_path.c_str()) != 0) {
			fprintf(stderr, "[encode] FAIL: could not rename %s to %s\n", out_tmp.c_str(), out_path.c_str());
			exit(1);
		}
		unregister_tmp_file(out_tmp); // renamed away, not a live scratch file anymore
		fprintf(stderr, "[encode] wrote %s\n", out_path.c_str());
		return;
	}

	std::string tmpwav = make_tmp_path(out_path);
	encode_to_wav(in_path, tmpwav);

	std::string mp3_src_wav = tmpwav;
	if (mp3_samplerate_hz != NATIVE_MODEM_SAMPLERATE_HZ) {
		fprintf(stderr, "[encode] resampling %dHz -> %dHz for MP3 output...\n",
			NATIVE_MODEM_SAMPLERATE_HZ, mp3_samplerate_hz);
		mp3_src_wav = make_tmp_path(out_path);
		if (!resample_wav(tmpwav, mp3_src_wav, mp3_samplerate_hz)) {
			cleanup_tmp_file(tmpwav);
			cleanup_tmp_file(mp3_src_wav);
			fprintf(stderr, "[encode] FAIL: resampling to %dHz failed\n", mp3_samplerate_hz);
			exit(1);
		}
		cleanup_tmp_file(tmpwav);
	}

	fprintf(stderr, "[encode] compressing to MP3 at %dkbit/s CBR (quality=0, matching ffmpeg's "
		"-compression_level 0), %dHz...\n",
		mp3_bitrate_kbps, mp3_samplerate_hz);
	bool ok = mp3_encode_wav(mp3_src_wav, out_tmp, mp3_bitrate_kbps);
	cleanup_tmp_file(mp3_src_wav);
	if (!ok) {
		cleanup_tmp_file(out_tmp);
		fprintf(stderr, "[encode] FAIL: MP3 encoding failed\n");
		exit(1);
	}

	if (verify) {
		fprintf(stderr, "[encode] verify: decoding %s back and comparing against %s...\n",
			out_tmp.c_str(), in_path.c_str());
		std::string verify_wav = make_tmp_path(out_path);
		if (!mp3_decode_to_wav(out_tmp, verify_wav)) {
			cleanup_tmp_file(verify_wav);
			cleanup_tmp_file(out_tmp);
			fprintf(stderr, "[encode] FAIL: could not decode freshly-encoded MP3 for verify\n");
			exit(1);
		}
		std::string verify_out = make_tmp_path(out_path);
		// strict_redundancy=true: this is a clean digital round-trip (no
		// acoustic channel), so verify should require every redundant header
		// and data block copy to come back, not just enough to reconstruct
		// the file - exits(1) itself on failure (missing/incomplete blocks).
		decode_from_wav(verify_wav, verify_out, /*strict_redundancy=*/true);
		cleanup_tmp_file(verify_wav);
		bool match = files_equal(in_path, verify_out);
		cleanup_tmp_file(verify_out);
		if (!match) {
			cleanup_tmp_file(out_tmp);
			fprintf(stderr, "[encode] FAIL: verify mismatch - decoding %s does not reproduce %s "
				"byte-for-byte (skip this check with --no-verify)\n", out_path.c_str(), in_path.c_str());
			exit(1);
		}
		fprintf(stderr, "[encode] verify PASS: MP3 round-trip matches original exactly\n");
	}

	if (rename(out_tmp.c_str(), out_path.c_str()) != 0) {
		fprintf(stderr, "[encode] FAIL: could not rename %s to %s\n", out_tmp.c_str(), out_path.c_str());
		exit(1);
	}
	unregister_tmp_file(out_tmp); // renamed away, not a live scratch file anymore
	fprintf(stderr, "[encode] wrote %s\n", out_path.c_str());
}

// RX: run one modem continuously across the *entire* file and return
// everything put_rx_char() ever emitted. No attempt to find segment
// boundaries in the sample domain: AMP-2 lines are self-delimiting
// ("<TYPE len crc>content"), so we just scan the decoded text for them
// (see amp_proto.cxx) - robust to preamble/postamble noise from either
// modem and to whatever timing drift the airgap path introduces.
static std::string rx_full_scan(modem* m, const std::string& wavpath, size_t tail_flush_blocks)
{
	active_modem = m;
	bHighSpeed = true;

	RXscard = new SoundNull();
	RXscard->Open(O_RDONLY, m->get_samplerate());
	RXscard->startPlayback(wavpath, 0);

	harness_rx_reset();
	m->rx_init();

	size_t frames = wav_frame_count(wavpath);
	size_t blocks = frames / SCBLOCKSIZE + 1;

	std::vector<float> fbuf(SCBLOCKSIZE);
	std::vector<double> dbuf(SCBLOCKSIZE);

	for (size_t b = 0; b < blocks + tail_flush_blocks; b++) {
		bHighSpeed = true; // read_file() resets this to false on EOF; force it back every block
		size_t n = RXscard->Read(fbuf.data(), SCBLOCKSIZE);
		if (n == 0) n = SCBLOCKSIZE;
		for (size_t i = 0; i < n; i++) dbuf[i] = fbuf[i];
		m->rx_process(dbuf.data(), (int)n);
	}

	RXscard->Close();
	delete RXscard; RXscard = nullptr;
	active_modem = nullptr;

	auto& bytes = harness_rx_bytes();
	return std::string(bytes.begin(), bytes.end());
}

static void decode_from_wav(const std::string& in_wav, const std::string& out_path, bool strict_redundancy)
{
	// strict_redundancy is only ever set by encode()'s own verify pass (see
	// its call site below), never by the top-level `decode` command - reuse
	// it to pick a distinct log tag too, so verify's internal decode (which
	// writes its reconstructed-original-file output to a throwaway scratch
	// path, NOT to the final audio output) can't be mistaken in the log for
	// a real `decode` run, or for something being written into the MP3.
	const char* tag = strict_redundancy ? "[verify-decode]" : "[decode]";

	progdefaults.wavSampleRate = 0;
	progdefaults.sample_converter = 0;
	progdefaults.rsid = false;
	progdefaults.rx_lowercase = false;

	fprintf(stderr, "%s scanning MFSK32 header track...\n", tag);
	mfsk* hdr = new mfsk(MODE_MFSK32);
	hdr->set_freq(CENTER_FREQ);
	std::string hdr_text = rx_full_scan(hdr, in_wav, /*tail_flush_blocks=*/200);
	delete hdr;
	fprintf(stderr, "%s MFSK32 header track: %zu bytes decoded\n", tag, hdr_text.size());

	fprintf(stderr, "%s scanning MFSK128L payload track...\n", tag);
	mfsk* pay = new mfsk(MODE_MFSK128L);
	pay->set_freq(CENTER_FREQ);
	std::string mfsk_text = rx_full_scan(pay, in_wav, /*tail_flush_blocks=*/4000);
	delete pay;
	fprintf(stderr, "%s MFSK128L payload track: %zu bytes decoded\n", tag, mfsk_text.size());

	if (const char* dumpdir = getenv("AMP_DUMP_DIR")) {
		FILE* f1 = fopen((std::string(dumpdir) + "/hdr_text.txt").c_str(), "wb");
		if (f1) { fwrite(hdr_text.data(), 1, hdr_text.size(), f1); fclose(f1); }
		FILE* f2 = fopen((std::string(dumpdir) + "/mfsk_text.txt").c_str(), "wb");
		if (f2) { fwrite(mfsk_text.data(), 1, mfsk_text.size(), f2); fclose(f2); }
	}

	AmpReceiveState st;
	amp_parse_lines(hdr_text, st);
	amp_parse_lines(mfsk_text, st);

	fprintf(stderr, "%s hash=%s file=%s size(fs=%d nb=%d bs=%d) blocks=%zu/%d\n", tag,
		st.hash.c_str(), st.filename.c_str(), st.filesize, st.numblocks, st.blocksize,
		st.blocks.size(), st.numblocks);

	// Redundancy stats: amp_build_message() sends repeat_header copies of
	// every header line per transmission, and one copy of every data block
	// per transmission (num_transmissions total) - see amp_proto.cxx.
	// Recovering just one copy of everything is enough to reconstruct the
	// file (amp_rx_complete(), checked below), but a clean digital
	// round-trip - encode()'s own verify pass, no acoustic channel in
	// between - should recover *every* copy; losing any without a real
	// channel loss to blame points at a modem/timing problem worth failing
	// on, even though the reconstructed file itself may still turn out
	// byte-exact off whichever single copy survived.
	AmpFileParams default_params;
	int expected_tx = default_params.num_transmissions;
	int expected_hdr_copies = default_params.repeat_header * expected_tx;

	struct HdrStat { const char* name; int hits; };
	HdrStat hdr_stats[] = {
		{ "FILE", st.file_hits }, { "ID", st.id_hits },
		{ "SIZE", st.size_hits }, { "PROG", st.prog_hits },
	};
	int hdr_incomplete = 0;
	fprintf(stderr, "%s header redundancy (expect %d copies/line = repeat_header %d x %d transmissions):\n",
		tag, expected_hdr_copies, default_params.repeat_header, expected_tx);
	for (auto& h : hdr_stats) {
		bool ok = h.hits >= expected_hdr_copies;
		if (!ok) hdr_incomplete++;
		fprintf(stderr, "%s   %-4s: %d/%d%s\n", tag, h.name, h.hits, expected_hdr_copies, ok ? "" : "  <-- INCOMPLETE");
	}

	int block_copies_expected = st.numblocks * expected_tx;
	int block_copies_recovered = 0;
	int blocks_incomplete = 0;
	for (int b = 1; b <= st.numblocks; b++) {
		int hits = 0;
		auto it = st.block_hits.find(b);
		if (it != st.block_hits.end()) hits = it->second;
		block_copies_recovered += hits;
		if (hits < expected_tx) blocks_incomplete++;
	}
	fprintf(stderr, "%s data block redundancy: %d/%d copies recovered across %d block(s) x %d transmissions"
		" (%d block(s) missing at least one copy)\n",
		tag, block_copies_recovered, block_copies_expected, st.numblocks, expected_tx, blocks_incomplete);

	bool fully_redundant = hdr_incomplete == 0 && blocks_incomplete == 0;

	if (!amp_rx_complete(st)) {
		fprintf(stderr, "%s FAIL: incomplete (missing %d of %d blocks): ",
			tag, st.numblocks - (int)st.blocks.size(), st.numblocks);
		for (int b = 1; b <= st.numblocks; b++)
			if (st.blocks.find(b) == st.blocks.end()) fprintf(stderr, "%d ", b);
		fprintf(stderr, "\n");
		exit(1);
	}

	if (strict_redundancy && !fully_redundant) {
		fprintf(stderr, "%s FAIL: verify requires every redundant copy of every header line and "
			"data block to be recoverable; %d header line type(s) and %d data block(s) above were "
			"missing at least one copy\n", tag, hdr_incomplete, blocks_incomplete);
		exit(1);
	}

	std::vector<uint8_t> raw = amp_rx_extract_file(st);
	FILE* out = fopen(out_path.c_str(), "wb");
	if (!out) { fprintf(stderr, "cannot open %s for write\n", out_path.c_str()); exit(1); }
	fwrite(raw.data(), 1, raw.size(), out);
	fclose(out);
	if (strict_redundancy)
		// verify's own decode: out_path here is a throwaway scratch file
		// (about to be byte-compared against the original input, then
		// deleted), NOT the MP3 being produced - say so explicitly instead
		// of printing "written to <path that looks like it's inside the
		// .mp3>", which reads as if audio content were being overwritten.
		fprintf(stderr, "%s PASS: %zu bytes of the original file reconstructed for comparison "
			"(scratch copy, not the MP3 - discarded once the byte-compare below finishes)\n",
			tag, raw.size());
	else
		fprintf(stderr, "%s PASS: %zu bytes written to %s\n", tag, raw.size(), out_path.c_str());
}

// Top-level decode: WAV input goes straight through decode_from_wav(), same
// as always. MP3 input is decompressed to "<input>.tmp" via libmpg123
// (mp3_io.cxx) first - no more shelling out to ffmpeg.
static void decode(const std::string& in_path, const std::string& out_path)
{
	if (!has_ext(in_path, ".mp3")) {
		decode_from_wav(in_path, out_path);
		return;
	}

	std::string tmpwav = in_path + ".tmp";
	register_tmp_file(tmpwav); // not from make_tmp_path(), so track it explicitly
	fprintf(stderr, "[decode] decompressing %s...\n", in_path.c_str());
	bool ok = mp3_decode_to_wav(in_path, tmpwav);
	if (!ok) {
		cleanup_tmp_file(tmpwav);
		fprintf(stderr, "[decode] FAIL: MP3 decoding failed\n");
		exit(1);
	}

	decode_from_wav(tmpwav, out_path);
	cleanup_tmp_file(tmpwav);
}

// Robustness check (REQUIREMENTS.md): detect silence gaps/dropouts
// anywhere in a WAV, via sliding-window RMS against a fraction of the
// file's peak RMS. A gap mid-transmission (as opposed to natural
// lead-in/lead-out silence) would desync a modem's symbol clock, so
// this is worth being able to check on any WAV in the pipeline -
// straight off mfsk_tool encode, or after an MP3 round trip, or
// after the airgap acoustic simulation.
static int check_silence(const std::string& wavpath, double window_ms, double rel_thresh)
{
	SF_INFO info; memset(&info, 0, sizeof(info));
	SNDFILE* f = sf_open(wavpath.c_str(), SFM_READ, &info);
	if (!f) { fprintf(stderr, "cannot open %s\n", wavpath.c_str()); return 2; }

	size_t win = (size_t)(info.samplerate * window_ms / 1000.0);
	if (win < 1) win = 1;
	std::vector<short> buf(win * info.channels);

	std::vector<double> rms_per_window;
	sf_count_t n;
	while ((n = sf_readf_short(f, buf.data(), win)) > 0) {
		double sumsq = 0;
		for (sf_count_t i = 0; i < n * info.channels; i++) sumsq += (double)buf[i] * buf[i];
		rms_per_window.push_back(sqrt(sumsq / (n * info.channels)));
	}
	sf_close(f);

	double peak = 0;
	for (double r : rms_per_window) if (r > peak) peak = r;
	double thresh = peak * rel_thresh;

	fprintf(stderr, "[checksilence] %s: %zu windows of %.0fms, peak rms=%.1f, threshold=%.1f\n",
		wavpath.c_str(), rms_per_window.size(), window_ms, peak, thresh);

	int gaps = 0;
	size_t i = 0;
	while (i < rms_per_window.size()) {
		if (rms_per_window[i] < thresh) {
			size_t j = i;
			while (j < rms_per_window.size() && rms_per_window[j] < thresh) j++;
			double start_s = i * window_ms / 1000.0;
			double dur_s = (j - i) * window_ms / 1000.0;
			bool at_edge = (i == 0 || j == rms_per_window.size());
			fprintf(stderr, "[checksilence] %s gap: %.3fs starting at %.3fs%s\n",
				at_edge ? "edge (lead-in/out)" : "MID-STREAM", dur_s, start_s,
				at_edge ? "" : " -- would desync a modem in flight");
			if (!at_edge) gaps++;
			i = j;
		} else {
			i++;
		}
	}
	fprintf(stderr, "[checksilence] %d mid-stream gap(s) found\n", gaps);
	return gaps > 0 ? 1 : 0;
}

static void print_usage_and_exit(const char* prog)
{
	fprintf(stderr,
		"usage:\n"
		"  %s encode [--bitrate KBPS] [--samplerate HZ] [--no-verify] <input-file> <output.wav|output.mp3>\n"
		"  %s decode <input.wav|input.mp3> <output-file>\n"
		"  %s checksilence <input.wav> [window_ms=50] [rel_thresh=0.01]\n"
		"\n"
		"  encode's output format is chosen by <output>'s extension. For .mp3,\n"
		"  --bitrate sets the CBR bitrate in kbit/s (default %d) and --samplerate\n"
		"  resamples the audio to HZ before compressing (default %d); both are\n"
		"  ignored for .wav, which is always written at the modem's native %dHz.\n"
		"  --bitrate/--samplerate/--no-verify only apply to .mp3 output.\n"
		"\n"
		"  For .mp3 output, encode verifies by default: it decodes the finished MP3\n"
		"  back and requires a byte-exact match against <input-file>, exiting nonzero\n"
		"  if it doesn't match. --no-verify skips this check.\n",
		prog, prog, prog, DEFAULT_MP3_KBPS, DEFAULT_MP3_SAMPLERATE_HZ, NATIVE_MODEM_SAMPLERATE_HZ);
	exit(2);
}

int main(int argc, char** argv)
{
	install_tmp_cleanup_handlers();
	SET_THREAD_ID(FLMAIN_TID);

	if (argc < 2) print_usage_and_exit(argv[0]);
	std::string verb = argv[1];

	if (verb == "encode") {
		int bitrate_kbps = DEFAULT_MP3_KBPS;
		int samplerate_hz = DEFAULT_MP3_SAMPLERATE_HZ;
		bool verify = true;
		std::vector<std::string> pos;
		for (int i = 2; i < argc; i++) {
			std::string a = argv[i];
			if (a == "--bitrate") {
				if (i + 1 >= argc) print_usage_and_exit(argv[0]);
				bitrate_kbps = atoi(argv[++i]);
				if (bitrate_kbps <= 0) print_usage_and_exit(argv[0]);
			} else if (a == "--samplerate") {
				if (i + 1 >= argc) print_usage_and_exit(argv[0]);
				samplerate_hz = atoi(argv[++i]);
				if (samplerate_hz <= 0) print_usage_and_exit(argv[0]);
			} else if (a == "--no-verify") {
				verify = false;
			} else {
				pos.push_back(a);
			}
		}
		if (pos.size() != 2) print_usage_and_exit(argv[0]);
		encode(pos[0], pos[1], bitrate_kbps, samplerate_hz, verify);
	} else if (verb == "decode") {
		if (argc != 4) print_usage_and_exit(argv[0]);
		decode(argv[2], argv[3]);
	} else if (verb == "checksilence") {
		if (argc < 3) print_usage_and_exit(argv[0]);
		double window_ms = argc > 3 ? atof(argv[3]) : 50.0;
		double rel_thresh = argc > 4 ? atof(argv[4]) : 0.01;
		return check_silence(argv[2], window_ms, rel_thresh);
	} else {
		print_usage_and_exit(argv[0]);
	}
	return 0;
}
