// SPDX-License-Identifier: GPL-3.0-or-later
// Diagnostic-only, not part of the shipped tool: runs fldigi's real RSID
// *receiver* (cRsId::receive, same object code cRsId::send() in main.cxx
// drives on TX) over a WAV file, exactly like trx.cxx's high-speed RX loop
// does (progdefaults.rsid gate + ReedSolomon->receive() call), to check
// whether the TxID bursts mfsk_tool writes actually get recognized
// by fldigi's own RSID decoder.
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <fcntl.h>
#include <sndfile.h>

#include "config.h"
#include "threads.h"
#include "trx.h"
#include "modem.h"
#include "mfsk.h"
#include "sound.h"
#include "configuration.h"
#include "rsid.h"

static size_t wav_frame_count(const std::string& path)
{
	SF_INFO info; memset(&info, 0, sizeof(info));
	SNDFILE* f = sf_open(path.c_str(), SFM_READ, &info);
	if (!f) return 0;
	size_t frames = (size_t)info.frames;
	sf_close(f);
	return frames;
}

int main(int argc, char** argv)
{
	SET_THREAD_ID(FLMAIN_TID);
	if (argc < 2) { fprintf(stderr, "usage: %s <in.wav>\n", argv[0]); return 2; }
	std::string wavpath = argv[1];

	progdefaults.wavSampleRate = 0;
	progdefaults.sample_converter = 0;
	progdefaults.rsid = true;
	progdefaults.rsid_rx_modes.set();

	mfsk* m = new mfsk(MODE_MFSK32);
	active_modem = m;
	bHighSpeed = true;

	if (!ReedSolomon) ReedSolomon = new cRsId;

	RXscard = new SoundNull();
	RXscard->Open(O_RDONLY, m->get_samplerate());
	RXscard->startPlayback(wavpath, 0);

	size_t frames = wav_frame_count(wavpath);
	size_t blocks = frames / SCBLOCKSIZE + 1;
	fprintf(stderr, "[rsid_check] %s: %zu frames, %zu blocks, samplerate=%d, txfreq=%.1f\n",
		wavpath.c_str(), frames, blocks, m->get_samplerate(), m->get_txfreq_woffset());

	std::vector<float> fbuf(SCBLOCKSIZE);

	for (size_t b = 0; b < blocks; b++) {
		bHighSpeed = true;
		size_t n = RXscard->Read(fbuf.data(), SCBLOCKSIZE);
		if (n == 0) n = SCBLOCKSIZE;
		ReedSolomon->receive(fbuf.data(), n);
	}

	RXscard->Close();
	fprintf(stderr, "[rsid_check] done\n");
	return 0;
}
