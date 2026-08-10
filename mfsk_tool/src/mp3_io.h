// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <string>

// Direct libmp3lame/libmpg123 bridge - replaces the ffmpeg shellouts that
// used to live in amp_pipeline.sh (encode side) and bitrate_sweep.cxx's
// mp3_roundtrip() (both sides).

// Encodes a PCM WAV to a CBR MP3 at cbr_kbps, quality=0 (best/slowest -
// the direct-lib equivalent of ffmpeg's `-compression_level 0` for
// libmp3lame, which maps 1:1 onto lame_set_quality()). Output sample rate
// is pinned to the input WAV's rate so LAME's own resampler never kicks in
// underneath a low bitrate. Returns true on success.
bool mp3_encode_wav(const std::string& wav_path, const std::string& mp3_path, int cbr_kbps);

// Decodes an MP3 to a 16-bit PCM WAV via libmpg123, at the MP3's native
// channel count and sample rate. Returns true on success.
bool mp3_decode_to_wav(const std::string& mp3_path, const std::string& wav_path);

// Resamples a PCM WAV to target_rate via libsamplerate (SRC_SINC_BEST_QUALITY
// - same converter fldigi's own real-soundcard record/playback path uses,
// see soundcard/sound.cxx write_file()/read_file()). The modem itself only
// runs at its fixed native rate (MFSKSampleRate == 8000, see fldigi's
// mfsk.h), so this is how encode() gets a WAV at any other rate before
// mp3_encode_wav(). Decode needs no counterpart: fldigi's own playback path
// (RXscard, driven by SoundBase::read_file()) already resamples an
// arbitrary source rate down to the modem's native rate on the fly.
// Returns true on success.
bool resample_wav(const std::string& src_path, const std::string& dst_path, int target_rate);
