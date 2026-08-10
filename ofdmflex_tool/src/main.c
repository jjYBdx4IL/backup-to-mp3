char __docstr__[] = "Transmit an arbitrary file over OFDM (ofdmflexframegen/ofdmflexframesync), "
                     "captured to/read from a genuinely mono, real-valued WAV file (or, if -o/-i "
                     "ends in .mp3, an mp3 encoded/decoded on the fly via libmp3lame). The complex "
                     "OFDM baseband is oversampled and mixed up to a carrier frequency on encode "
                     "(quadrature upconversion); decode mixes back down, filters, and decimates "
                     "back to baseband before demodulating -- same idea real radios use to put a "
                     "complex baseband signal on a real wire/antenna/speaker. "
                     "Two modes: 'encode' modulates a file to a wav/mp3; 'decode' demodulates a "
                     "wav/mp3 back to a file. The file is split into fixed-size blocks, each sent as its "
                     "own OFDM frame; a dedicated metadata frame (file size, name, sha256) is "
                     "sent twice at the start of each transmission. The whole transmission "
                     "(metadata + all blocks) is repeated several times for redundancy, with a "
                     "silence gap around each repetition.\n"
                     "--preset <name> applies a named bundle of parameter values (still "
                     "overridable by any flag placed after it on the command line); "
                     "--list-presets shows what's available.\n"
                     "Usage: ofdmflex_tool <encode|decode> [options]";

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "liquid.h"
#include "liquid.argparse.h"

#include <openssl/evp.h>

#include "mp3.h"

#define TX_GAIN 0.2f

// ---------------------------------------------------------------------
// SHA-256 (single-shot, whole buffer in memory) via OpenSSL's libcrypto
// ---------------------------------------------------------------------
void sha256(const unsigned char * data, size_t len, unsigned char digest[32])
{
    EVP_Digest(data, len, digest, NULL, EVP_sha256(), NULL);
}

void sha256_hex(const unsigned char digest[32], char hex[65])
{
    int i;
    for (i=0; i<32; i++)
        sprintf(hex+2*i, "%02x", digest[i]);
    hex[64] = '\0';
}

// ---------------------------------------------------------------------
// minimal streaming WAV writer: genuinely mono, 32-bit IEEE-float
// ---------------------------------------------------------------------
typedef struct {
    FILE *   fid;
    uint32_t num_samples;
} wavwriter_s;

void wavwriter_open(wavwriter_s * _w, const char * _filename, uint32_t _sample_rate)
{
    _w->fid         = fopen(_filename, "wb");
    _w->num_samples = 0;
    if (_w->fid == NULL) {
        fprintf(stderr, "error: could not open '%s' for writing\n", _filename);
        exit(1);
    }

    uint16_t num_channels    = 1;  // genuinely mono
    uint16_t bits_per_sample = 32; // 32-bit float
    uint16_t audio_format    = 3;  // IEEE float
    uint32_t byte_rate       = _sample_rate * num_channels * bits_per_sample/8;
    uint16_t block_align     = num_channels * bits_per_sample/8;
    uint32_t fmt_size        = 16;
    uint32_t zero32          = 0;

    fwrite("RIFF",          1, 4, _w->fid);
    fwrite(&zero32,         4, 1, _w->fid); // placeholder: overall chunk size
    fwrite("WAVE",          1, 4, _w->fid);

    fwrite("fmt ",          1, 4, _w->fid);
    fwrite(&fmt_size,       4, 1, _w->fid);
    fwrite(&audio_format,   2, 1, _w->fid);
    fwrite(&num_channels,   2, 1, _w->fid);
    fwrite(&_sample_rate,   4, 1, _w->fid);
    fwrite(&byte_rate,      4, 1, _w->fid);
    fwrite(&block_align,    2, 1, _w->fid);
    fwrite(&bits_per_sample,2, 1, _w->fid);

    fwrite("data",          1, 4, _w->fid);
    fwrite(&zero32,         4, 1, _w->fid); // placeholder: data chunk size
}

void wavwriter_write(wavwriter_s * _w, float * _buf, unsigned int _n)
{
    fwrite(_buf, sizeof(float), _n, _w->fid);
    _w->num_samples += _n;
}

void wavwriter_close(wavwriter_s * _w)
{
    uint32_t data_bytes = _w->num_samples * sizeof(float);
    uint32_t chunk_size = 36 + data_bytes;

    fseek(_w->fid, 4, SEEK_SET);
    fwrite(&chunk_size, 4, 1, _w->fid);

    fseek(_w->fid, 40, SEEK_SET);
    fwrite(&data_bytes, 4, 1, _w->fid);

    fclose(_w->fid);
}

#define WAVE_FORMAT_PCM        1
#define WAVE_FORMAT_IEEE_FLOAT 3
#define WAVE_FORMAT_EXTENSIBLE 0xFFFE

// decode one sample (PCM 8/16/24/32-bit or IEEE float 32/64-bit) to a float in [-1,1]
static float wav_decode_sample(const unsigned char * _p, uint16_t _bytes_per_sample, uint16_t _audio_format)
{
    if (_audio_format == WAVE_FORMAT_IEEE_FLOAT) {
        if (_bytes_per_sample == 8) {
            double v; memcpy(&v, _p, 8);
            return (float) v;
        }
        float v; memcpy(&v, _p, 4);
        return v;
    }
    // PCM
    switch (_bytes_per_sample) {
    case 1:
        return ((int)_p[0] - 128) / 128.0f;
    case 2: {
        int16_t v; memcpy(&v, _p, 2);
        return v / 32768.0f;
    }
    case 3: {
        int32_t v = (int32_t)(_p[0] | (_p[1]<<8) | (_p[2]<<16));
        if (v & 0x00800000) v |= 0xFF000000; // sign-extend
        return v / 8388608.0f;
    }
    default: { // 4
        int32_t v; memcpy(&v, _p, 4);
        return v / 2147483648.0f;
    }
    }
}

// read a WAV file of (almost) any common sample format/channel count back
// into a mono, real-valued float buffer: PCM 8/16/24/32-bit or IEEE float
// 32/64-bit, WAVE_FORMAT_EXTENSIBLE, any channel count (downmixed to mono
// by averaging)
float * wav_read_mono(const char * _filename, uint32_t * _num_samples, uint32_t * _sample_rate)
{
    FILE * fid = fopen(_filename, "rb");
    if (fid == NULL) {
        fprintf(stderr, "error: could not open '%s' for reading\n", _filename);
        exit(1);
    }

    char tag[4];
    uint32_t sz;
    if (fread(tag,1,4,fid)!=4 || memcmp(tag,"RIFF",4)!=0) {
        fprintf(stderr, "error: '%s' is not a valid RIFF file\n", _filename);
        exit(1);
    }
    fread(&sz,4,1,fid); // overall chunk size, unused
    if (fread(tag,1,4,fid)!=4 || memcmp(tag,"WAVE",4)!=0) {
        fprintf(stderr, "error: '%s' is not a valid WAVE file\n", _filename);
        exit(1);
    }

    uint16_t audio_format=0, num_channels=0, bits_per_sample=0;
    uint32_t sample_rate=0;
    unsigned char * data = NULL;
    uint32_t data_bytes = 0;

    while (fread(tag,1,4,fid)==4) {
        if (fread(&sz,4,1,fid)!=1)
            break;

        if (memcmp(tag,"fmt ",4)==0) {
            uint32_t byte_rate;
            uint16_t block_align;
            fread(&audio_format,    2,1,fid);
            fread(&num_channels,    2,1,fid);
            fread(&sample_rate,     4,1,fid);
            fread(&byte_rate,       4,1,fid);
            fread(&block_align,     2,1,fid);
            fread(&bits_per_sample, 2,1,fid);
            uint32_t consumed = 16;
            if (audio_format == WAVE_FORMAT_EXTENSIBLE && sz >= 16 + 2 + 2 + 4 + 16) {
                uint16_t cb_size, valid_bits;
                uint32_t channel_mask;
                unsigned char guid[16];
                fread(&cb_size,     2,1,fid);
                fread(&valid_bits,  2,1,fid);
                fread(&channel_mask,4,1,fid);
                fread(guid,         1,16,fid);
                (void) valid_bits;
                (void) channel_mask;
                audio_format = guid[0] | (guid[1] << 8); // actual sub-format code
                consumed += 2 + 2 + 4 + 16;
            }
            fseek(fid, sz - consumed, SEEK_CUR); // skip any remaining fmt bytes
        } else if (memcmp(tag,"data",4)==0) {
            data_bytes = sz;
            data = (unsigned char*) malloc(data_bytes);
            if (fread(data,1,data_bytes,fid) != data_bytes) {
                fprintf(stderr, "error: '%s': truncated data chunk\n", _filename);
                exit(1);
            }
        } else {
            fseek(fid, sz + (sz & 1), SEEK_CUR);
        }
    }
    fclose(fid);

    if (data == NULL) {
        fprintf(stderr, "error: '%s': no 'data' chunk found\n", _filename);
        exit(1);
    }
    int format_ok = (audio_format == WAVE_FORMAT_PCM        && (bits_per_sample==8 || bits_per_sample==16 || bits_per_sample==24 || bits_per_sample==32))
                  || (audio_format == WAVE_FORMAT_IEEE_FLOAT && (bits_per_sample==32 || bits_per_sample==64));
    if (!format_ok || num_channels == 0) {
        fprintf(stderr, "error: '%s': unsupported wav format=%u channels=%u bits=%u "
                "(supported: PCM 8/16/24/32-bit or IEEE float 32/64-bit, any channel count)\n",
                _filename, audio_format, num_channels, bits_per_sample);
        free(data);
        exit(1);
    }

    uint16_t bytes_per_sample = bits_per_sample / 8;
    uint32_t frame_bytes      = (uint32_t)bytes_per_sample * num_channels;
    uint32_t num_frames       = frame_bytes ? data_bytes / frame_bytes : 0;

    float * mono = (float*) malloc(num_frames * sizeof(float));
    uint32_t i; uint16_t c;
    for (i=0; i<num_frames; i++) {
        const unsigned char * frame = data + (size_t)i*frame_bytes;
        float sum = 0.0f;
        for (c=0; c<num_channels; c++)
            sum += wav_decode_sample(frame + (size_t)c*bytes_per_sample, bytes_per_sample, audio_format);
        mono[i] = sum / num_channels;
    }
    free(data);

    *_num_samples = num_frames;
    *_sample_rate = sample_rate;
    return mono;
}

// dispatch on filename: read the carrier file as mp3 or wav, whichever it is
float * read_mono_auto(const char * _filename, uint32_t * _num_samples, uint32_t * _sample_rate)
{
    if (mp3_has_suffix(_filename))
        return mp3_read_mono(_filename, _num_samples, _sample_rate);
    return wav_read_mono(_filename, _num_samples, _sample_rate);
}

// ---------------------------------------------------------------------
// carrier output: mono 32-bit float wav, or (if the filename ends in
// .mp3) mp3 encoded on the fly via libmp3lame -- lets the transmitted
// audio go straight onto lossy storage without a separate conversion
// pass, which is the whole point of this project
// ---------------------------------------------------------------------
typedef struct {
    int          is_mp3;
    uint32_t     num_samples;
    wavwriter_s  wav;
    mp3writer_s  mp3;
} carrierwriter_s;

void carrierwriter_open(carrierwriter_s * _cw, const char * _filename, uint32_t _sample_rate,
                         unsigned int _mp3_quality)
{
    _cw->is_mp3      = mp3_has_suffix(_filename);
    _cw->num_samples = 0;

    if (_cw->is_mp3)
        mp3writer_open(&_cw->mp3, _filename, _sample_rate, _mp3_quality);
    else
        wavwriter_open(&_cw->wav, _filename, _sample_rate);
}

void carrierwriter_write(carrierwriter_s * _cw, float * _buf, unsigned int _n)
{
    _cw->num_samples += _n;
    if (_cw->is_mp3)
        mp3writer_write(&_cw->mp3, _buf, _n);
    else
        wavwriter_write(&_cw->wav, _buf, _n);
}

void carrierwriter_close(carrierwriter_s * _cw)
{
    if (_cw->is_mp3)
        mp3writer_close(&_cw->mp3);
    else
        wavwriter_close(&_cw->wav);
}

// ---------------------------------------------------------------------
// tx IF chain: interpolate complex baseband, mix up to a carrier, keep
// only the real part -- the quadrature-upconversion step that turns a
// complex baseband signal into something that could ride a real wire
// ---------------------------------------------------------------------
#define NATIVE_BUF_LEN (256)

typedef struct {
    firinterp_crcf         interp;
    nco_crcf                nco;
    unsigned int             oversample;
    liquid_float_complex * interp_scratch; // [NATIVE_BUF_LEN*oversample]
    float *                 mono_scratch;   // [NATIVE_BUF_LEN*oversample]
    uint32_t                 sample_pos;     // native samples emitted so far
} txchain_s;

void txchain_init(txchain_s * _tx, unsigned int _oversample, unsigned int _delay,
                   float _atten, float _carrier, float _final_rate)
{
    _tx->oversample     = _oversample;
    _tx->interp         = firinterp_crcf_create_kaiser(_oversample, _delay, _atten);
    _tx->nco            = nco_crcf_create(LIQUID_VCO);
    _tx->sample_pos     = 0;
    nco_crcf_set_frequency(_tx->nco, 2.0f*M_PI*_carrier/_final_rate);

    unsigned int n = NATIVE_BUF_LEN * _oversample;
    _tx->interp_scratch = (liquid_float_complex*) malloc(n*sizeof(liquid_float_complex));
    _tx->mono_scratch   = (float*) malloc(n*sizeof(float));
}

// _n must be <= NATIVE_BUF_LEN
void txchain_write(txchain_s * _tx, carrierwriter_s * _wav, liquid_float_complex * _buf, unsigned int _n)
{
    firinterp_crcf_execute_block(_tx->interp, _buf, _n, _tx->interp_scratch);

    unsigned int total = _n * _tx->oversample;
    unsigned int j;
    for (j=0; j<total; j++) {
        liquid_float_complex passband;
        nco_crcf_mix_up(_tx->nco, _tx->interp_scratch[j], &passband);
        nco_crcf_step(_tx->nco);
        _tx->mono_scratch[j] = crealf(passband) * TX_GAIN;
        if (!(_tx->mono_scratch[j] > -1.0f && _tx->mono_scratch[j] < 1.0f)) {
            fprintf(stderr, "txchain_write: sample out of range: %f\n", _tx->mono_scratch[j]);
            assert(0);
        }
    }
    carrierwriter_write(_wav, _tx->mono_scratch, total);
    _tx->sample_pos += _n;
}

void txchain_write_gap(txchain_s * _tx, carrierwriter_s * _wav, uint32_t _n)
{
    liquid_float_complex zero[NATIVE_BUF_LEN];
    unsigned int i;
    for (i=0; i<NATIVE_BUF_LEN; i++)
        zero[i] = 0.0f;
    while (_n > 0) {
        unsigned int m = _n > NATIVE_BUF_LEN ? NATIVE_BUF_LEN : _n;
        txchain_write(_tx, _wav, zero, m);
        _n -= m;
    }
}

void txchain_destroy(txchain_s * _tx)
{
    firinterp_crcf_destroy(_tx->interp);
    nco_crcf_destroy(_tx->nco);
    free(_tx->interp_scratch);
    free(_tx->mono_scratch);
}

// ---------------------------------------------------------------------
// rx IF chain: mix a real (Q=0) sample down from the carrier, decimate
// back to the native OFDM sample rate (the decimator's own lowpass
// rejects the image left over from the real-valued mix), correcting for
// the amplitude the real<->complex mixing splits between images
// ---------------------------------------------------------------------
typedef struct {
    firdecim_crcf decim;
    nco_crcf      nco;
    unsigned int  oversample;
    liquid_float_complex * mix_scratch; // [oversample]
} rxchain_s;

void rxchain_init(rxchain_s * _rx, unsigned int _oversample, unsigned int _delay,
                   float _atten, float _carrier, float _final_rate)
{
    _rx->oversample = _oversample;
    _rx->decim      = firdecim_crcf_create_kaiser(_oversample, _delay, _atten);
    firdecim_crcf_set_scale(_rx->decim, 2.0f); // compensate for real-signal image split
    _rx->nco        = nco_crcf_create(LIQUID_VCO);
    nco_crcf_set_frequency(_rx->nco, 2.0f*M_PI*_carrier/_final_rate);
    _rx->mix_scratch = (liquid_float_complex*) malloc(_oversample*sizeof(liquid_float_complex));
}

// consumes exactly 'oversample' real input samples, returns one native
// baseband complex sample
liquid_float_complex rxchain_step(rxchain_s * _rx, const float * _mono)
{
    unsigned int j;
    for (j=0; j<_rx->oversample; j++) {
        liquid_float_complex cin = _mono[j];
        nco_crcf_mix_down(_rx->nco, cin, &_rx->mix_scratch[j]);
        nco_crcf_step(_rx->nco);
    }
    liquid_float_complex out;
    firdecim_crcf_execute(_rx->decim, _rx->mix_scratch, &out);
    return out;
}

void rxchain_destroy(rxchain_s * _rx)
{
    firdecim_crcf_destroy(_rx->decim);
    nco_crcf_destroy(_rx->nco);
    free(_rx->mix_scratch);
}

// ---------------------------------------------------------------------
// chronological event trace, shared by encode and decode
// ---------------------------------------------------------------------
void trace_print(bool _enabled, uint32_t _native_pos, uint32_t _native_rate, const char * _fmt, ...)
{
    if (!_enabled)
        return;
    printf("[trace] t=+%8.3fs (sample %9u) ", (float)_native_pos/(float)_native_rate, _native_pos);
    va_list ap;
    va_start(ap, _fmt);
    vprintf(_fmt, ap);
    va_end(ap);
    printf("\n");
}

// format the 8-byte frame header as space-separated hex, e.g. "01 00 00 00 05 00 00 00"
void header_hex(const unsigned char * _header, char _out[24])
{
    int i;
    for (i=0; i<8; i++)
        sprintf(_out + 3*i, i<7 ? "%02x " : "%02x", _header[i]);
}

// ---------------------------------------------------------------------
// frame header (8 bytes): byte[0] = frame type, bytes[1..4] = chunk
// index (data frames only), bytes[5..7] reserved
// ---------------------------------------------------------------------
#define FRAME_TYPE_META (0)
#define FRAME_TYPE_DATA (1)

// metadata frame payload: file_size(8) + sha256(32) + filename_len(1) + filename(<=255)
#define META_FILENAME_MAX (255)

// ---------------------------------------------------------------------
// receiver state
// ---------------------------------------------------------------------
typedef struct {
    int              received;
    unsigned int     count;   // how many times this chunk was decoded successfully (redundancy)
    unsigned int     len;
    unsigned char *  data;
} chunkslot_s;

typedef struct {
    // metadata, learned from a FRAME_TYPE_META frame
    int           meta_received;
    unsigned int  meta_count;
    uint64_t      file_size;
    unsigned char sha256sum[32];
    char          filename[META_FILENAME_MAX+1];

    // data chunks, grown by chunk index as frames arrive (independent of
    // whether metadata has been seen yet)
    chunkslot_s * chunks;
    uint32_t      chunks_cap;
    uint32_t      chunks_max; // highest touched index + 1

    uint32_t      block_size;
    unsigned int  num_received; // distinct chunk indices received

    unsigned int  header_invalid;
    unsigned int  payload_invalid;

    int           stats_mode;
    long          expected_chunks; // -1 until known from metadata

    // --trace
    bool          trace;
    uint32_t      sample_rate; // native rate, for trace timestamps
    uint32_t      cur_pos;     // native sample position of the current demod block
} rxstate_s;

void rx_ensure_capacity(rxstate_s * _rx, uint32_t _index)
{
    if (_index < _rx->chunks_cap)
        return;
    uint32_t new_cap = _rx->chunks_cap == 0 ? 64 : _rx->chunks_cap;
    while (new_cap <= _index)
        new_cap *= 2;
    _rx->chunks = (chunkslot_s*) realloc(_rx->chunks, new_cap*sizeof(chunkslot_s));
    memset(_rx->chunks + _rx->chunks_cap, 0, (new_cap - _rx->chunks_cap)*sizeof(chunkslot_s));
    _rx->chunks_cap = new_cap;
}

int callback(unsigned char *  _header,
             int              _header_valid,
             unsigned char *  _payload,
             unsigned int     _payload_len,
             int              _payload_valid,
             framesyncstats_s _stats,
             void *           _userdata)
{
    rxstate_s * rx = (rxstate_s*) _userdata;

    char hdrhex[24];
    header_hex(_header, hdrhex);

    if (!_header_valid) {
        rx->header_invalid++;
        trace_print(rx->trace, rx->cur_pos, rx->sample_rate, "[----] header invalid  rssi=%6.2fdB  header=%s", _stats.rssi, hdrhex);
        return 0;
    }

    unsigned char frame_type = _header[0];

    if (frame_type == FRAME_TYPE_META) {
        trace_print(rx->trace, rx->cur_pos, rx->sample_rate, "[meta] rssi=%6.2fdB evm=%6.2fdB valid=%s  header=%s",
                    _stats.rssi, _stats.evm, _payload_valid ? "yes" : "no ", hdrhex);
        if (!_payload_valid || _payload_len < 8+32+1) {
            rx->payload_invalid++;
            return 0;
        }
        uint64_t fsz = 0;
        unsigned int i;
        for (i=0; i<8; i++)
            fsz = (fsz<<8) | _payload[i];
        rx->file_size = fsz;
        memcpy(rx->sha256sum, _payload+8, 32);
        unsigned char fnlen = _payload[40];
        if (fnlen > META_FILENAME_MAX)
            fnlen = META_FILENAME_MAX;
        if (41u+fnlen <= _payload_len) {
            memcpy(rx->filename, _payload+41, fnlen);
            rx->filename[fnlen] = '\0';
        }
        rx->meta_received = 1;
        rx->meta_count++;
        if (rx->expected_chunks < 0)
            rx->expected_chunks = (long)((rx->file_size + rx->block_size - 1) / rx->block_size);
        return 0;
    }

    // data frame
    uint32_t chunk_index = (_header[1]<<24)|(_header[2]<<16)|(_header[3]<<8)|_header[4];
    trace_print(rx->trace, rx->cur_pos, rx->sample_rate, "[data] chunk %6u  rssi=%6.2fdB evm=%6.2fdB valid=%s len=%u  header=%s",
                chunk_index, _stats.rssi, _stats.evm, _payload_valid ? "yes" : "no ", _payload_len, hdrhex);

    if (!_payload_valid) {
        rx->payload_invalid++;
        return 0;
    }

    rx_ensure_capacity(rx, chunk_index);
    chunkslot_s * slot = &rx->chunks[chunk_index];
    slot->count++;
    if (!slot->received) {
        slot->received = 1;
        slot->len      = _payload_len;
        slot->data      = (unsigned char*) malloc(_payload_len);
        memcpy(slot->data, _payload, _payload_len);
        rx->num_received++;
    }
    if (chunk_index+1 > rx->chunks_max)
        rx->chunks_max = chunk_index+1;

    return 0;
}

const char * basename_of(const char * _path)
{
    const char * slash = strrchr(_path, '/');
    return slash ? slash+1 : _path;
}

// ---------------------------------------------------------------------
// named presets: canned bundles of encode/decode parameter values,
// selected with --preset <name>. Applied by splicing the preset's flags
// into argv ahead of the user's own, so getopt's usual "last one wins"
// behavior means any flag the user places after --preset still overrides
// the preset's value for that same option.
// ---------------------------------------------------------------------
#define PRESET_MAX_ARGS (24) // flag/value tokens, NULL-terminated

typedef struct {
    const char * name;
    const char * description;
    const char * args[PRESET_MAX_ARGS];
} preset_s;

static const preset_s presets[] = {
    {
        "phone1",
        "narrowband/robust profile tuned for a phone-handset acoustic link",
        { "-C","64", "-T","12", "-b","1024", "-M","192", "-g","3", "-G","0",
          "-r","6000", "-c","rs8", "-k","v27", "-q","2", "-x","8", NULL }
    },
};
#define NUM_PRESETS ((int)(sizeof(presets)/sizeof(presets[0])))

const preset_s * preset_find(const char * _name)
{
    int p;
    for (p=0; p<NUM_PRESETS; p++) {
        if (strcmp(presets[p].name, _name)==0)
            return &presets[p];
    }
    return NULL;
}

void preset_list(void)
{
    printf("available presets (--preset <name>):\n");
    int p;
    for (p=0; p<NUM_PRESETS; p++) {
        printf("  %-10s %s\n", presets[p].name, presets[p].description);
        printf("  %-10s ", "");
        int k;
        for (k=0; presets[p].args[k] != NULL; k++)
            printf("%s ", presets[p].args[k]);
        printf("\n");
    }
}

// print every parameter value actually in effect for this run (after
// defaults, any --preset, and any explicit flags have all been applied)
void print_params(const char * _mode, const char * _preset_name,
                   const char * _input, const char * _output,
                   unsigned int _M, unsigned int _cp_len, unsigned int _taper_len, unsigned int _block_size,
                   const char * _mod, const char * _crc, const char * _fs0, const char * _fs1,
                   unsigned int _repetitions, float _gap, float _header_gap,
                   unsigned int _samplerate, unsigned int _oversample, float _carrier,
                   unsigned int _filt_delay, float _filt_atten, unsigned int _mp3_quality)
{
    if (_preset_name)
        printf("mode         : %s  (preset: %s)\n", _mode, _preset_name);
    else
        printf("mode         : %s\n", _mode);
    printf("io           : -i '%s'  -o '%s'\n", _input, _output);
    printf("ofdm         : -M %u  -C %u  -T %u  -b %u\n", _M, _cp_len, _taper_len, _block_size);
    // taper overlap-adds onto the cyclic prefix rather than shortening the
    // symbol, so the native symbol period is exactly M+cp_len samples
    unsigned int symbol_len = _M + _cp_len;
    float symbol_rate = (float)_samplerate / (float)symbol_len;
    printf("symbol rate  : %u samples/symbol (M+C)  ->  %.2f symbols/s theoretical, %.2f ms/symbol\n",
           symbol_len, symbol_rate, 1000.0f*(float)symbol_len/(float)_samplerate);
    float cp_ms = 1000.0f*(float)_cp_len/(float)_samplerate;
    printf("cyclic prefix: -C %u samples = %.2f ms  ->  multipath echoes/delay spread up to ~%.2f ms are absorbed without inter-symbol interference; longer echoes bleed into the next symbol\n",
           _cp_len, cp_ms, cp_ms);
    printf("coding       : -m %s  -v %s  -c %s  -k %s\n", _mod, _crc, _fs0, _fs1);
    printf("timing       : -R %u  -g %g  -G %g\n", _repetitions, _gap, _header_gap);
    printf("rf           : -r %u  -x %u  -F %g%s  -I %u  -A %g\n",
           _samplerate, _oversample, _carrier, _carrier<=0.0f ? " (auto)" : "", _filt_delay, _filt_atten);
    printf("mp3          : -q %u\n", _mp3_quality);
}

int main(int argc, char *argv[])
{
    // --list-presets works regardless of mode/other args
    int a;
    for (a=1; a<argc; a++) {
        if (strcmp(argv[a], "--list-presets")==0) {
            preset_list();
            return 0;
        }
    }

    if (argc < 2 || (strcmp(argv[1],"encode")!=0 && strcmp(argv[1],"decode")!=0)) {
        fprintf(stderr, "%s\n", __docstr__);
        return -1;
    }
    int is_encode = strcmp(argv[1],"encode")==0;

    // liquid_argparse only understands short -x options; pull the mode
    // token out and handle long flags manually before handing the rest to
    // the normal parser. --preset is expanded here too, into the preset's
    // own short flags, spliced in ahead of whatever follows -- so a flag
    // the user gives after --preset still wins (getopt applies options in
    // argv order, last one for a given letter sets the final value).
    char ** argv2 = (char**) malloc((argc + argc*PRESET_MAX_ARGS)*sizeof(char*));
    int argc2 = 0;
    argv2[argc2++] = argv[0];
    bool stats_mode = false;
    bool trace_mode = false;
    const char * preset_name = NULL;
    struct { const char * flag; bool * out; } long_flags[] = {
        { "--stats", &stats_mode },
        { "--trace", &trace_mode },
    };
    int i;
    for (i=2; i<argc; i++) {
        if (strcmp(argv[i], "--preset")==0) {
            if (i+1 >= argc) {
                fprintf(stderr, "error: --preset requires a name (see --list-presets)\n");
                free(argv2);
                return -1;
            }
            const preset_s * pr = preset_find(argv[i+1]);
            if (pr == NULL) {
                fprintf(stderr, "error: unknown preset '%s' (see --list-presets)\n", argv[i+1]);
                free(argv2);
                return -1;
            }
            preset_name = argv[i+1];
            int k;
            for (k=0; pr->args[k] != NULL; k++)
                argv2[argc2++] = (char*) pr->args[k];
            i++; // also consume the preset name
            continue;
        }

        unsigned int k;
        int matched = 0;
        for (k=0; k<sizeof(long_flags)/sizeof(long_flags[0]); k++) {
            if (strcmp(argv[i], long_flags[k].flag)==0) {
                *long_flags[k].out = true;
                matched = 1;
                break;
            }
        }
        if (!matched)
            argv2[argc2++] = argv[i];
    }

    liquid_argparse_init(__docstr__);
    liquid_argparse_add(char*,    input,          "", 'i', "input file (encode: data to send; decode: wav/mp3 to receive)", NULL);
    liquid_argparse_add(char*,    output,         "", 'o', "output file (encode: wav/mp3 to write, by extension; decode: reconstructed data file)", NULL);
    liquid_argparse_add(unsigned, block_size,     64, 'b', "payload bytes per OFDM data frame (max 9216); must match between encode/decode", NULL);
    liquid_argparse_add(unsigned, M,              64, 'M', "number of subcarriers", NULL);
    liquid_argparse_add(unsigned, cp_len,         16, 'C', "cyclic prefix length", NULL);
    liquid_argparse_add(unsigned, taper_len,       4, 'T', "taper length", NULL);
    liquid_argparse_add(char*,    mod,        "qpsk", 'm', "[encode] modulation scheme", liquid_argparse_modem);
    liquid_argparse_add(char*,    crc,       "crc32", 'v', "[encode] CRC scheme", liquid_argparse_crc);
    liquid_argparse_add(char*,    fs0,        "none", 'c', "[encode] FEC scheme (inner)", liquid_argparse_fec);
    liquid_argparse_add(char*,    fs1,        "none", 'k', "[encode] FEC scheme (outer)", liquid_argparse_fec);
    liquid_argparse_add(unsigned, repetitions,     3, 'R', "[encode] number of times to repeat the whole transmission", NULL);
    liquid_argparse_add(float,    gap,          1.0f, 'g', "[encode] silence gap [seconds] before/after each transmission", NULL);
    liquid_argparse_add(float,    header_gap,   1.0f, 'G', "[encode] silence gap [seconds]: between the two repeated metadata frames, and again before the first data frame", NULL);
    liquid_argparse_add(unsigned, samplerate,   8000, 'r', "native OFDM sample rate [Hz] (final wav rate = samplerate * oversample); must match between encode/decode -- on decode, input is resampled to this rate if the file's own rate differs", NULL);
    liquid_argparse_add(unsigned, oversample,      4, 'x', "interpolation/decimation factor for IF upconversion; must match between encode/decode", NULL);
    liquid_argparse_add(float,    carrier,      0.0f, 'F', "carrier frequency [Hz]; must match between encode/decode (0 = auto: native samplerate)", NULL);
    liquid_argparse_add(unsigned, filt_delay,      5, 'I', "interp/decim filter delay [symbols]", NULL);
    liquid_argparse_add(float,    filt_atten,  60.0f, 'A', "interp/decim filter stop-band attenuation [dB]", NULL);
    liquid_argparse_add(unsigned, mp3_quality,     2, 'q', "[encode] mp3 quality/bitrate index, 0 (best, 320kbps) - 9 (worst, 64kbps); only used when -o ends in .mp3", NULL);
    liquid_argparse_parse(argc2, argv2);
    free(argv2);

    if (strlen(input)==0 || strlen(output)==0) {
        fprintf(stderr, "error: both -i <input> and -o <output> are required\n");
        return -1;
    }
    if (block_size == 0 || block_size > LIQUID_MAX_PAYLOAD_LEN) {
        fprintf(stderr, "error: -b must be between 1 and %u\n", LIQUID_MAX_PAYLOAD_LEN);
        return -1;
    }
    if (oversample < 2) {
        fprintf(stderr, "error: -x (oversample) must be at least 2\n");
        return -1;
    }

    print_params(is_encode ? "encode" : "decode", preset_name, input, output,
                 M, cp_len, taper_len, block_size, mod, crc, fs0, fs1,
                 repetitions, gap, header_gap, samplerate, oversample, carrier,
                 filt_delay, filt_atten, mp3_quality);

    unsigned int buf_len = NATIVE_BUF_LEN;
    liquid_float_complex buf[NATIVE_BUF_LEN];

    if (is_encode) {
        modulation_scheme ms    = liquid_getopt_str2mod(mod);
        crc_scheme        check = liquid_getopt_str2crc(crc);
        fec_scheme        fec0  = liquid_getopt_str2fec(fs0);
        fec_scheme        fec1  = liquid_getopt_str2fec(fs1);

        if (carrier <= 0.0f)
            carrier = (float)samplerate;
        float final_rate = (float)samplerate * (float)oversample;
        if (carrier - samplerate/2.0f < 0.0f || carrier + samplerate/2.0f > final_rate/2.0f) {
            fprintf(stderr, "error: carrier=%g Hz doesn't leave enough headroom for the OFDM "
                    "bandwidth (~%u Hz) under the oversampled Nyquist (%g Hz); increase -x or "
                    "adjust -F\n", carrier, samplerate, final_rate/2.0f);
            return -1;
        }

        FILE * fid = fopen(input, "rb");
        if (fid == NULL) {
            fprintf(stderr, "error: could not open '%s' for reading\n", input);
            return -1;
        }
        fseek(fid, 0, SEEK_END);
        long file_size = ftell(fid);
        rewind(fid);
        if (file_size <= 0) {
            fprintf(stderr, "error: '%s' is empty\n", input);
            return -1;
        }
        unsigned char * tx_data = (unsigned char*) malloc(file_size);
        if (fread(tx_data,1,file_size,fid) != (size_t)file_size) {
            fprintf(stderr, "error: '%s': could not read file\n", input);
            return -1;
        }
        fclose(fid);

        unsigned char digest[32];
        sha256(tx_data, file_size, digest);
        char digest_hex[65];
        sha256_hex(digest, digest_hex);

        const char * fname = basename_of(input);
        unsigned char fnlen = (unsigned char)(strlen(fname) > META_FILENAME_MAX ? META_FILENAME_MAX : strlen(fname));

        unsigned char meta_payload[8+32+1+META_FILENAME_MAX];
        uint64_t fsz64 = (uint64_t)file_size;
        for (i=0; i<8; i++)
            meta_payload[i] = (fsz64 >> (8*(7-i))) & 0xff;
        memcpy(meta_payload+8, digest, 32);
        meta_payload[40] = fnlen;
        memcpy(meta_payload+41, fname, fnlen);
        unsigned int meta_len = 41 + fnlen;

        uint32_t num_chunks = (uint32_t)((file_size + block_size - 1) / block_size);

        printf("modulation   : %s\n", mod);
        printf("input        : '%s' (%ld bytes)\n", input, file_size);
        printf("sha256       : %s\n", digest_hex);
        printf("blocks       : %u x %u bytes\n", num_chunks, block_size);
        printf("repetitions  : %u\n", repetitions);
        printf("gap          : %.3f s (header-repeat gap: %.3f s)\n", gap, header_gap);
        printf("carrier      : %g Hz (oversample x%u, native rate %u Hz, wav rate %g Hz)\n",
               carrier, oversample, samplerate, final_rate);

        ofdmflexframegenprops_s fgprops;
        ofdmflexframegenprops_init_default(&fgprops);
        fgprops.check      = check;
        fgprops.fec0       = fec0;
        fgprops.fec1       = fec1;
        fgprops.mod_scheme = ms;
        ofdmflexframegen fg = ofdmflexframegen_create(M, cp_len, taper_len, NULL, &fgprops);

        carrierwriter_s wav;
        carrierwriter_open(&wav, output, (uint32_t)final_rate, mp3_quality);

        txchain_s tx;
        txchain_init(&tx, oversample, filt_delay, filt_atten, carrier, final_rate);

        uint32_t gap_samples        = (uint32_t)(gap * samplerate);
        uint32_t header_gap_samples = (uint32_t)(header_gap * samplerate);

        unsigned char header[8];
        trace_print(trace_mode, tx.sample_pos, samplerate, "[gap ] %.3fs silence (leading)", gap);
        txchain_write_gap(&tx, &wav, gap_samples);

        unsigned int rep;
        for (rep=0; rep<repetitions; rep++) {
            // metadata frame, sent twice with a gap in between
            unsigned int copy;
            for (copy=0; copy<2; copy++) {
                memset(header, 0, 8);
                header[0] = FRAME_TYPE_META;
                char hdrhex[24];
                header_hex(header, hdrhex);
                trace_print(trace_mode, tx.sample_pos, samplerate, "[meta] (%u/2) rep %u/%u  header=%s", copy+1, rep+1, repetitions, hdrhex);
                ofdmflexframegen_assemble(fg, header, meta_payload, meta_len);
                int last_symbol = 0;
                while (!last_symbol) {
                    last_symbol = ofdmflexframegen_write(fg, buf, buf_len);
                    txchain_write(&tx, &wav, buf, buf_len);
                }
                if (copy==0) {
                    trace_print(trace_mode, tx.sample_pos, samplerate, "[gap ] %.3fs silence (header-repeat)", header_gap);
                    txchain_write_gap(&tx, &wav, header_gap_samples);
                }
            }

            trace_print(trace_mode, tx.sample_pos, samplerate, "[gap ] %.3fs silence (after header, before data)", header_gap);
            txchain_write_gap(&tx, &wav, header_gap_samples);

            // data frames
            uint32_t c;
            for (c=0; c<num_chunks; c++) {
                size_t offset    = (size_t)c * block_size;
                size_t chunk_len = file_size - offset;
                if (chunk_len > block_size)
                    chunk_len = block_size;

                memset(header, 0, 8);
                header[0] = FRAME_TYPE_DATA;
                header[1] = (c >> 24) & 0xff;
                header[2] = (c >> 16) & 0xff;
                header[3] = (c >>  8) & 0xff;
                header[4] = (c      ) & 0xff;
                char hdrhex[24];
                header_hex(header, hdrhex);
                trace_print(trace_mode, tx.sample_pos, samplerate, "[data] chunk %6u rep %u/%u  header=%s", c, rep+1, repetitions, hdrhex);

                ofdmflexframegen_assemble(fg, header, tx_data + offset, chunk_len);
                int last_symbol = 0;
                while (!last_symbol) {
                    last_symbol = ofdmflexframegen_write(fg, buf, buf_len);
                    txchain_write(&tx, &wav, buf, buf_len);
                }
            }

            trace_print(trace_mode, tx.sample_pos, samplerate, "[gap ] %.3fs silence (end of rep %u/%u)", gap, rep+1, repetitions);
            txchain_write_gap(&tx, &wav, gap_samples);
        }

        uint32_t total_samples = wav.num_samples;
        carrierwriter_close(&wav);
        txchain_destroy(&tx);

        float playback_secs = (float)total_samples/final_rate;
        uint64_t gross_bytes = (uint64_t)file_size * repetitions;
        printf("output       : '%s' (%u samples, %.2f s @ %g Hz)\n",
               output, total_samples, playback_secs, final_rate);
        printf("wire speed   : %.2f bytes/s net (%ld payload bytes delivered / %.2f s playback)\n",
               (float)file_size/playback_secs, file_size, playback_secs);
        printf("             : %.2f bytes/s gross (%llu bytes actually sent, incl. %u repetitions, "
               "not adjusted for redundancy / %.2f s playback)\n",
               (float)gross_bytes/playback_secs, (unsigned long long)gross_bytes, repetitions, playback_secs);

        free(tx_data);
        ofdmflexframegen_destroy(fg);
        return 0;
    }

    // ---------------- decode ----------------
    uint32_t num_mono_samples, wav_rate_u;
    float * mono = read_mono_auto(input, &num_mono_samples, &wav_rate_u);
    printf("input        : '%s' (%u samples, %.2f s @ %u Hz)\n",
           input, num_mono_samples, (float)num_mono_samples/(float)wav_rate_u, wav_rate_u);

    // the captured file's sample rate is whatever the recording device used,
    // which won't generally match samplerate*oversample (the rate the
    // encoder actually wrote); resample so the rest of decode can assume
    // that invariant, same as it always could when fed encode's own output
    uint32_t final_rate_u = samplerate * oversample;
    if (wav_rate_u != final_rate_u) {
        float resamp_rate = (float)final_rate_u / (float)wav_rate_u;
        msresamp_rrrf q = msresamp_rrrf_create(resamp_rate, filt_atten);
        unsigned int out_len = msresamp_rrrf_get_num_output(q, num_mono_samples);
        float * resampled = (float*) malloc(out_len * sizeof(float));
        unsigned int num_written = 0;
        msresamp_rrrf_execute(q, mono, num_mono_samples, resampled, &num_written);
        msresamp_rrrf_destroy(q);
        free(mono);
        mono = resampled;
        num_mono_samples = num_written;
        printf("resampling   : %u Hz -> %u Hz (rate %.6fx, native rate/-r=%u/-x=%u expects %u Hz) -> %u samples\n",
               wav_rate_u, final_rate_u, resamp_rate, samplerate, oversample, final_rate_u, num_mono_samples);
    }
    float final_rate = (float)final_rate_u;
    uint32_t native_rate = final_rate_u / oversample;

    if (carrier <= 0.0f)
        carrier = (float)native_rate;
    if (carrier - native_rate/2.0f < 0.0f || carrier + native_rate/2.0f > final_rate/2.0f) {
        fprintf(stderr, "error: carrier=%g Hz doesn't leave enough headroom under the wav's "
                "Nyquist (%g Hz) given oversample=%u; check -F/-x match the encoder\n",
                carrier, final_rate/2.0f, oversample);
        return -1;
    }

    uint32_t num_native_samples = num_mono_samples / oversample;

    rxstate_s rx;
    memset(&rx, 0, sizeof(rx));
    rx.block_size       = block_size;
    rx.expected_chunks  = -1;
    rx.stats_mode       = stats_mode;
    rx.trace            = trace_mode;
    rx.sample_rate      = native_rate;

    ofdmflexframesync fs = ofdmflexframesync_create(M, cp_len, taper_len, NULL, callback, (void*)&rx);

    // bool is_ask = strstr(mod, "ask") == mod;
    // is_ask = 0;
    // agc_crcf agc = (void*)0;
    // if(is_ask) {
    //     agc = agc_crcf_create();
    //     agc_crcf_set_bandwidth(agc, 1e-2f); // Adjust bandwidth rate (1e-2 to 1e-3 is typical)
    // }

    rxchain_s rxc;
    rxchain_init(&rxc, oversample, filt_delay, filt_atten, carrier, final_rate);

    printf("modulation   : %s\n", mod);
    printf("carrier      : %g Hz (oversample x%u, native rate %u Hz)\n", carrier, oversample, native_rate);
    printf("decoding     : %s\n", stats_mode ? "full pass (--stats)" : "stop once all blocks are recovered");
    //printf("agc          : %s\n", is_ask ? "enabled" : "disabled");

    uint32_t pos;
    for (pos=0; pos<num_native_samples; pos+=buf_len) {
        unsigned int n = num_native_samples - pos;
        if (n > buf_len)
            n = buf_len;

        // if (is_ask) {
        //     unsigned int j;
        //     for (j=0; j<n; j++) {
        //         liquid_float_complex c = rxchain_step(&rxc, mono + (size_t)(pos+j)*oversample);
        //         agc_crcf_execute(agc, c, &buf[j]);
        //     }
        // } else {
            unsigned int j;
            for (j=0; j<n; j++)
                buf[j] = rxchain_step(&rxc, mono + (size_t)(pos+j)*oversample);
        // }

        rx.cur_pos = pos;
        ofdmflexframesync_execute(fs, buf, n);

        if (!stats_mode && rx.expected_chunks >= 0 &&
            (long)rx.num_received >= rx.expected_chunks)
            break;
    }
    free(mono);
    rxchain_destroy(&rxc);
    // if (agc)
    //     agc_crcf_destroy(agc);

    printf("\n");
    if (!rx.meta_received) {
        fprintf(stderr, "warning: no metadata frame recovered -- file size/name/checksum unknown\n");
        fprintf(stderr, "         reconstructing best-effort from %u observed block(s)\n", rx.chunks_max);
    }

    uint64_t   out_size;
    long       total_chunks;
    if (rx.meta_received) {
        total_chunks = rx.expected_chunks;
        out_size     = rx.file_size;
        printf("filename     : '%s'\n", rx.filename);
        printf("file size    : %llu bytes\n", (unsigned long long)rx.file_size);
    } else {
        total_chunks = rx.chunks_max;
        out_size     = (uint64_t)total_chunks * block_size;
    }

    unsigned char * outbuf = (unsigned char*) calloc(out_size > 0 ? out_size : 1, 1);
    for (i=0; i<total_chunks && (uint32_t)i<rx.chunks_max; i++) {
        chunkslot_s * slot = &rx.chunks[i];
        if (!slot->received)
            continue;
        size_t offset = (size_t)i * block_size;
        size_t n       = slot->len;
        if (offset >= out_size)
            continue;
        if (offset + n > out_size)
            n = out_size - offset;
        memcpy(outbuf + offset, slot->data, n);
    }

    FILE * fout = fopen(output, "wb");
    if (fout == NULL) {
        fprintf(stderr, "error: could not open '%s' for writing\n", output);
        return -1;
    }
    fwrite(outbuf, 1, out_size, fout);
    fclose(fout);

    int checksum_ok = 0;
    if (rx.meta_received) {
        unsigned char actual[32];
        sha256(outbuf, out_size, actual);
        checksum_ok = memcmp(actual, rx.sha256sum, 32)==0;

        char expect_hex[65], actual_hex[65];
        sha256_hex(rx.sha256sum, expect_hex);
        sha256_hex(actual, actual_hex);
        printf("sha256       : expected %s\n", expect_hex);
        printf("             : actual   %s (%s)\n", actual_hex, checksum_ok ? "MATCH" : "MISMATCH");
    }

    printf("blocks       : %u / %ld recovered\n", rx.num_received, total_chunks);
    printf("output       : '%s' (%llu bytes)\n", output, (unsigned long long)out_size);
    printf("result       : %s\n",
           (rx.meta_received && checksum_ok && (long)rx.num_received==total_chunks)
               ? "PASS (checksum verified)" : "FAIL (incomplete or unverified)");

    if (stats_mode) {
        printf("\n");
        printf("redundancy statistics (over %ld blocks):\n", total_chunks);

        unsigned int max_count = 0;
        for (i=0; i<total_chunks && (uint32_t)i<rx.chunks_max; i++)
            if (rx.chunks[i].count > max_count)
                max_count = rx.chunks[i].count;

        unsigned int * hist = (unsigned int*) calloc(max_count+1, sizeof(unsigned int));
        for (i=0; i<total_chunks; i++) {
            unsigned int cnt = ((uint32_t)i < rx.chunks_max) ? rx.chunks[i].count : 0;
            hist[cnt]++;
        }
        int level;
        for (level=(int)max_count; level>=0; level--) {
            printf("  received %2dx : %6u block%s%s\n", level, hist[level],
                   hist[level]==1 ? "" : "s", level==0 ? "  (lost)" : "");
        }
        free(hist);

        printf("\n");
        printf("metadata frame received : %u time%s\n", rx.meta_count, rx.meta_count==1 ? "" : "s");
        printf("frames w/ invalid header: %u\n", rx.header_invalid);
        printf("frames w/ invalid payload: %u\n", rx.payload_invalid);
    }

    for (i=0; i<rx.chunks_max; i++)
        free(rx.chunks[i].data);
    free(rx.chunks);
    free(outbuf);
    ofdmflexframesync_destroy(fs);

    return (rx.meta_received && checksum_ok && (long)rx.num_received==total_chunks) ? 0 : 1;
}
