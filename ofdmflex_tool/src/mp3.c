#include "mp3.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

int mp3_has_suffix(const char * _filename)
{
    static const char suffix[] = ".mp3";
    size_t slen = strlen(_filename), suflen = strlen(suffix);
    if (suflen > slen)
        return 0;
    size_t i;
    for (i=0; i<suflen; i++) {
        if (tolower((unsigned char)_filename[slen-suflen+i]) != suffix[i])
            return 0;
    }
    return 1;
}

// decode exactly one mp3 frame's worth of samples (<=1152), pulling more
// compressed bytes from _mp3data as needed; mirrors the reference decode
// loop in lame's own frontend (frontend/get_audio.c: lame_decode_fromfile).
// returns <=0 once both the input and the decoder's internal buffer are
// exhausted (treated as "done", same as upstream -- hip_decode1_headers
// doesn't distinguish clean EOF from a truncated stream here)
static int mp3_decode_next_frame(hip_t _hip, const unsigned char * _mp3data, size_t _mp3_size,
                                  size_t * _pos, short _pcm_l[1152], short _pcm_r[1152],
                                  mp3data_struct * _info)
{
    // drain anything already buffered inside the decoder before reading more
    int ret = hip_decode1_headers(_hip, (unsigned char*)_mp3data + *_pos, 0, _pcm_l, _pcm_r, _info);
    if (ret != 0)
        return ret;

    for (;;) {
        size_t remaining = _mp3_size - *_pos;
        size_t n = remaining < 1024 ? remaining : 1024;
        ret = hip_decode1_headers(_hip, (unsigned char*)_mp3data + *_pos, n, _pcm_l, _pcm_r, _info);
        *_pos += n;
        if (ret != 0 || n == 0)
            return ret;
    }
}

float * mp3_read_mono(const char * _filename, uint32_t * _num_samples, uint32_t * _sample_rate)
{
    FILE * fid = fopen(_filename, "rb");
    if (fid == NULL) {
        fprintf(stderr, "error: could not open '%s' for reading\n", _filename);
        exit(1);
    }
    fseek(fid, 0, SEEK_END);
    long mp3_size = ftell(fid);
    rewind(fid);
    if (mp3_size <= 0) {
        fprintf(stderr, "error: '%s' is empty\n", _filename);
        exit(1);
    }
    unsigned char * mp3data = (unsigned char*) malloc(mp3_size);
    if (fread(mp3data, 1, mp3_size, fid) != (size_t)mp3_size) {
        fprintf(stderr, "error: '%s': could not read file\n", _filename);
        exit(1);
    }
    fclose(fid);

    hip_t hip = hip_decode_init();
    if (hip == NULL) {
        fprintf(stderr, "error: libmp3lame: could not initialize decoder\n");
        exit(1);
    }

    uint32_t cap = 65536, len = 0;
    float * out = (float*) malloc(cap*sizeof(float));
    uint32_t sample_rate = 0;
    int mono_checked = 0;
    short pcm_l[1152], pcm_r[1152];
    mp3data_struct info;

    size_t pos = 0;
    for (;;) {
        int nout = mp3_decode_next_frame(hip, mp3data, (size_t)mp3_size, &pos, pcm_l, pcm_r, &info);
        if (nout <= 0)
            break; // clean EOF or truncated/invalid stream -- either way, nothing more to decode

        if (info.header_parsed && sample_rate == 0)
            sample_rate = (uint32_t) info.samplerate;
        if (info.header_parsed && !mono_checked) {
            mono_checked = 1;
            if (info.stereo != 1) {
                fprintf(stderr, "error: '%s': expected a mono mp3 (this tool only writes mono), "
                        "got %d channel(s)\n", _filename, info.stereo);
                exit(1);
            }
        }

        if (len + (uint32_t)nout > cap) {
            while (len + (uint32_t)nout > cap)
                cap *= 2;
            out = (float*) realloc(out, cap*sizeof(float));
        }
        int k;
        for (k=0; k<nout; k++)
            out[len++] = pcm_l[k] / 32768.0f;
    }

    hip_decode_exit(hip);
    free(mp3data);

    if (sample_rate == 0) {
        fprintf(stderr, "error: '%s': could not find a valid mp3 header\n", _filename);
        exit(1);
    }

    *_num_samples = len;
    *_sample_rate = sample_rate;
    return out;
}

static const int MP3_QUALITY_BITRATE_KBPS[10] = {320,256,224,192,160,128,112,96,80,64};

void mp3writer_open(mp3writer_s * _w, const char * _filename, uint32_t _sample_rate, unsigned int _quality)
{
    _w->fid = fopen(_filename, "wb");
    if (_w->fid == NULL) {
        fprintf(stderr, "error: could not open '%s' for writing\n", _filename);
        exit(1);
    }

    unsigned int q = _quality > 9 ? 9 : _quality;

    _w->lame = lame_init();
    lame_set_in_samplerate(_w->lame, (int)_sample_rate);
    lame_set_num_channels(_w->lame, 1);
    lame_set_mode(_w->lame, MONO);
    lame_set_VBR(_w->lame, vbr_off);
    lame_set_brate(_w->lame, MP3_QUALITY_BITRATE_KBPS[q]);
    if (lame_init_params(_w->lame) < 0) {
        fprintf(stderr, "error: libmp3lame: could not initialize encoder\n");
        exit(1);
    }
}

void mp3writer_write(mp3writer_s * _w, float * _buf, unsigned int _n)
{
    if (_n == 0)
        return;

    // standard libmp3lame sizing recommendation for the worst-case output
    int mp3buf_size = (int)(1.25 * _n) + 7200;
    unsigned char * mp3buf = (unsigned char*) malloc(mp3buf_size);
    int bytes = lame_encode_buffer_ieee_float(_w->lame, _buf, NULL, (int)_n, mp3buf, mp3buf_size);
    if (bytes < 0) {
        fprintf(stderr, "error: libmp3lame: encode failed (code %d)\n", bytes);
        exit(1);
    }
    if (bytes > 0)
        fwrite(mp3buf, 1, bytes, _w->fid);
    free(mp3buf);
}

void mp3writer_close(mp3writer_s * _w)
{
    unsigned char mp3buf[7200];
    int bytes = lame_encode_flush(_w->lame, mp3buf, sizeof(mp3buf));
    if (bytes > 0)
        fwrite(mp3buf, 1, bytes, _w->fid);
    lame_close(_w->lame);
    fclose(_w->fid);
}
