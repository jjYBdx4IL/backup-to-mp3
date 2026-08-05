// SPDX-License-Identifier: GPL-3.0-or-later
// fldigi_amp_tool - headless encode/decode using fldigi's real MFSK32
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
//   fldigi_amp_tool encode <input-file> <output.wav>
//   fldigi_amp_tool decode <input.wav> <output-file>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <fcntl.h>
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

static void encode(const std::string& in_path, const std::string& out_wav)
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

static void decode(const std::string& in_wav, const std::string& out_path)
{
	progdefaults.wavSampleRate = 0;
	progdefaults.sample_converter = 0;
	progdefaults.rsid = false;
	progdefaults.rx_lowercase = false;

	fprintf(stderr, "[decode] scanning MFSK32 header track...\n");
	mfsk* hdr = new mfsk(MODE_MFSK32);
	hdr->set_freq(CENTER_FREQ);
	std::string hdr_text = rx_full_scan(hdr, in_wav, /*tail_flush_blocks=*/200);
	delete hdr;
	fprintf(stderr, "[decode] MFSK32 header track: %zu bytes decoded\n", hdr_text.size());

	fprintf(stderr, "[decode] scanning MFSK128L payload track...\n");
	mfsk* pay = new mfsk(MODE_MFSK128L);
	pay->set_freq(CENTER_FREQ);
	std::string mfsk_text = rx_full_scan(pay, in_wav, /*tail_flush_blocks=*/4000);
	delete pay;
	fprintf(stderr, "[decode] MFSK128L payload track: %zu bytes decoded\n", mfsk_text.size());

	if (const char* dumpdir = getenv("AMP_DUMP_DIR")) {
		FILE* f1 = fopen((std::string(dumpdir) + "/hdr_text.txt").c_str(), "wb");
		if (f1) { fwrite(hdr_text.data(), 1, hdr_text.size(), f1); fclose(f1); }
		FILE* f2 = fopen((std::string(dumpdir) + "/mfsk_text.txt").c_str(), "wb");
		if (f2) { fwrite(mfsk_text.data(), 1, mfsk_text.size(), f2); fclose(f2); }
	}

	AmpReceiveState st;
	amp_parse_lines(hdr_text, st);
	amp_parse_lines(mfsk_text, st);

	fprintf(stderr, "[decode] hash=%s file=%s size(fs=%d nb=%d bs=%d) blocks=%zu/%d\n",
		st.hash.c_str(), st.filename.c_str(), st.filesize, st.numblocks, st.blocksize,
		st.blocks.size(), st.numblocks);

	if (!amp_rx_complete(st)) {
		fprintf(stderr, "[decode] FAIL: incomplete (missing %d of %d blocks): ",
			st.numblocks - (int)st.blocks.size(), st.numblocks);
		for (int b = 1; b <= st.numblocks; b++)
			if (st.blocks.find(b) == st.blocks.end()) fprintf(stderr, "%d ", b);
		fprintf(stderr, "\n");
		exit(1);
	}

	std::vector<uint8_t> raw = amp_rx_extract_file(st);
	FILE* out = fopen(out_path.c_str(), "wb");
	if (!out) { fprintf(stderr, "cannot open %s for write\n", out_path.c_str()); exit(1); }
	fwrite(raw.data(), 1, raw.size(), out);
	fclose(out);
	fprintf(stderr, "[decode] PASS: %zu bytes written to %s\n", raw.size(), out_path.c_str());
}

// Robustness check (REQUIREMENTS.md): detect silence gaps/dropouts
// anywhere in a WAV, via sliding-window RMS against a fraction of the
// file's peak RMS. A gap mid-transmission (as opposed to natural
// lead-in/lead-out silence) would desync a modem's symbol clock, so
// this is worth being able to check on any WAV in the pipeline -
// straight off fldigi_amp_tool encode, or after an MP3 round trip, or
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

int main(int argc, char** argv)
{
	SET_THREAD_ID(FLMAIN_TID);

	bool is_encode = argc >= 4 && strcmp(argv[1], "encode") == 0;
	bool is_decode = argc >= 4 && strcmp(argv[1], "decode") == 0;
	bool is_check = argc >= 3 && strcmp(argv[1], "checksilence") == 0;

	if (!is_encode && !is_decode && !is_check) {
		fprintf(stderr,
			"usage:\n  %s encode <input-file> <output.wav>\n  %s decode <input.wav> <output-file>\n"
			"  %s checksilence <input.wav> [window_ms=50] [rel_thresh=0.01]\n",
			argv[0], argv[0], argv[0]);
		return 2;
	}

	if (is_encode) {
		encode(argv[2], argv[3]);
	} else if (is_decode) {
		decode(argv[2], argv[3]);
	} else {
		double window_ms = argc > 3 ? atof(argv[3]) : 50.0;
		double rel_thresh = argc > 4 ? atof(argv[4]) : 0.01;
		return check_silence(argv[2], window_ms, rel_thresh);
	}
	return 0;
}
