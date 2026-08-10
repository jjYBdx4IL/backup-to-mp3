#ifndef OFDMFLEX_MP3_H
#define OFDMFLEX_MP3_H

#include <stdint.h>
#include <stdio.h>
#include <lame/lame.h>

// case-insensitive ".mp3" suffix check -- callers use this to decide
// whether -o/-i should be treated as mp3 instead of the tool's native
// wav carrier format
int mp3_has_suffix(const char * _filename);

// ---------------------------------------------------------------------
// decode: mp3 -> mono float PCM via libmp3lame's built-in decoder
// (hip_decode); mirrors wav_read_mono()'s signature so callers don't
// care which carrier format they're reading
// ---------------------------------------------------------------------
float * mp3_read_mono(const char * _filename, uint32_t * _num_samples, uint32_t * _sample_rate);

// ---------------------------------------------------------------------
// encode: streaming mono float PCM -> mp3, mirrors wavwriter_s's usage
// (open once, write repeatedly, close once)
// ---------------------------------------------------------------------
typedef struct {
    FILE *              fid;
    lame_global_flags * lame;
} mp3writer_s;

void mp3writer_open(mp3writer_s * _w, const char * _filename, uint32_t _sample_rate, unsigned int _quality);
void mp3writer_write(mp3writer_s * _w, float * _buf, unsigned int _n);
void mp3writer_close(mp3writer_s * _w);

#endif
