// SPDX-License-Identifier: GPL-3.0-or-later
#include "mp3_io.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <sndfile.h>
#include <lame/lame.h>
#include <mpg123.h>
#include <samplerate.h>

bool mp3_encode_wav(const std::string& wav_path, const std::string& mp3_path, int cbr_kbps)
{
	SF_INFO info; memset(&info, 0, sizeof(info));
	SNDFILE* in = sf_open(wav_path.c_str(), SFM_READ, &info);
	if (!in) { fprintf(stderr, "mp3_encode: cannot open %s\n", wav_path.c_str()); return false; }

	lame_global_flags* gfp = lame_init();
	lame_set_in_samplerate(gfp, info.samplerate);
	lame_set_out_samplerate(gfp, info.samplerate); // pin - keeps LAME's own resampler out of it at low bitrates
	lame_set_num_channels(gfp, info.channels);
	lame_set_brate(gfp, cbr_kbps);
	lame_set_VBR(gfp, vbr_off); // CBR
	lame_set_quality(gfp, 0);   // best/slowest - same knob ffmpeg's -compression_level maps onto
	if (lame_init_params(gfp) < 0) {
		fprintf(stderr, "mp3_encode: lame_init_params failed (bad samplerate/bitrate combination?)\n");
		lame_close(gfp); sf_close(in);
		return false;
	}

	FILE* out = fopen(mp3_path.c_str(), "wb");
	if (!out) {
		fprintf(stderr, "mp3_encode: cannot open %s for write\n", mp3_path.c_str());
		lame_close(gfp); sf_close(in);
		return false;
	}

	const size_t PCM_CHUNK = 8192;
	std::vector<short> pcm((size_t)PCM_CHUNK * info.channels);
	// Per lame.h's lame_encode_buffer sizing guidance: 1.25x input samples + 7200 bytes headroom.
	std::vector<unsigned char> mp3buf(PCM_CHUNK * 5 / 4 + 7200);

	bool ok = true;
	sf_count_t n;
	while ((n = sf_readf_short(in, pcm.data(), PCM_CHUNK)) > 0) {
		int bytes = (info.channels == 1)
			? lame_encode_buffer(gfp, pcm.data(), pcm.data(), (int)n, mp3buf.data(), (int)mp3buf.size())
			: lame_encode_buffer_interleaved(gfp, pcm.data(), (int)n, mp3buf.data(), (int)mp3buf.size());
		if (bytes < 0) {
			fprintf(stderr, "mp3_encode: lame_encode_buffer failed (%d)\n", bytes);
			ok = false;
			break;
		}
		if (bytes > 0) fwrite(mp3buf.data(), 1, bytes, out);
	}
	sf_close(in);

	if (ok) {
		int flush_bytes = lame_encode_flush(gfp, mp3buf.data(), (int)mp3buf.size());
		if (flush_bytes > 0) fwrite(mp3buf.data(), 1, flush_bytes, out);
	}

	fclose(out);
	lame_close(gfp);
	return ok;
}

bool mp3_decode_to_wav(const std::string& mp3_path, const std::string& wav_path)
{
	if (mpg123_init() != MPG123_OK) {
		fprintf(stderr, "mp3_decode: mpg123_init failed\n");
		return false;
	}

	int err = MPG123_OK;
	mpg123_handle* mh = mpg123_new(nullptr, &err);
	if (!mh) {
		fprintf(stderr, "mp3_decode: mpg123_new failed: %s\n", mpg123_plain_strerror(err));
		mpg123_exit();
		return false;
	}

	// Force signed-16 PCM, whatever channel count the file actually has - no
	// format renegotiation to juggle mid-stream (MPG123_MONO|MPG123_STEREO
	// means "either is fine", not "downmix to mono").
	if (mpg123_open_fixed(mh, mp3_path.c_str(), MPG123_MONO | MPG123_STEREO, MPG123_ENC_SIGNED_16) != MPG123_OK) {
		fprintf(stderr, "mp3_decode: cannot open %s: %s\n", mp3_path.c_str(), mpg123_strerror(mh));
		mpg123_delete(mh); mpg123_exit();
		return false;
	}

	long rate; int channels, encoding;
	if (mpg123_getformat(mh, &rate, &channels, &encoding) != MPG123_OK) {
		fprintf(stderr, "mp3_decode: mpg123_getformat failed: %s\n", mpg123_strerror(mh));
		mpg123_close(mh); mpg123_delete(mh); mpg123_exit();
		return false;
	}

	SF_INFO info; memset(&info, 0, sizeof(info));
	info.samplerate = (int)rate;
	info.channels = channels;
	info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	SNDFILE* out = sf_open(wav_path.c_str(), SFM_WRITE, &info);
	if (!out) {
		fprintf(stderr, "mp3_decode: cannot open %s for write\n", wav_path.c_str());
		mpg123_close(mh); mpg123_delete(mh); mpg123_exit();
		return false;
	}

	std::vector<unsigned char> buf(32768);
	size_t done;
	int ret;
	do {
		ret = mpg123_read(mh, buf.data(), buf.size(), &done);
		if (done > 0) {
			sf_count_t frames = (sf_count_t)(done / sizeof(short) / channels);
			sf_writef_short(out, (const short*)buf.data(), frames);
		}
	} while (ret == MPG123_OK);

	bool ok = (ret == MPG123_DONE);
	if (!ok) fprintf(stderr, "mp3_decode: mpg123_read stopped early: %s\n", mpg123_strerror(mh));

	sf_close(out);
	mpg123_close(mh);
	mpg123_delete(mh);
	mpg123_exit();
	return ok;
}

bool resample_wav(const std::string& src_path, const std::string& dst_path, int target_rate)
{
	SF_INFO info; memset(&info, 0, sizeof(info));
	SNDFILE* in = sf_open(src_path.c_str(), SFM_READ, &info);
	if (!in) { fprintf(stderr, "resample: cannot open %s\n", src_path.c_str()); return false; }

	std::vector<float> in_buf((size_t)info.frames * info.channels);
	sf_count_t in_frames = sf_readf_float(in, in_buf.data(), info.frames);
	sf_close(in);

	double ratio = (double)target_rate / info.samplerate;
	// libsamplerate's docs size the output buffer as input_frames * ratio,
	// plus headroom since src_simple() may generate a few extra frames.
	long out_capacity = (long)std::ceil(in_frames * ratio) + 1024;
	std::vector<float> out_buf((size_t)out_capacity * info.channels);

	SRC_DATA data; memset(&data, 0, sizeof(data));
	data.data_in = in_buf.data();
	data.input_frames = in_frames;
	data.data_out = out_buf.data();
	data.output_frames = out_capacity;
	data.src_ratio = ratio;
	data.end_of_input = 1;

	int err = src_simple(&data, SRC_SINC_BEST_QUALITY, info.channels);
	if (err) {
		fprintf(stderr, "resample: %s\n", src_strerror(err));
		return false;
	}

	SF_INFO out_info = info;
	out_info.samplerate = target_rate;
	out_info.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
	SNDFILE* out = sf_open(dst_path.c_str(), SFM_WRITE, &out_info);
	if (!out) { fprintf(stderr, "resample: cannot open %s for write\n", dst_path.c_str()); return false; }
	sf_writef_float(out, out_buf.data(), data.output_frames_gen);
	sf_close(out);
	return true;
}
