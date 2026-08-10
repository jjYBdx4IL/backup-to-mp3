char __docstr__[] = "Apply channel_cccf impairments (AWGN, carrier offset, multipath, "
                     "shadowing) to a mono WAV file. Since channel_cccf only operates on "
                     "complex samples, each real input sample is promoted to complex (Q=0) "
                     "before the channel is applied, and only the real part of the result is "
                     "kept -- exactly the same embed/extract trick used by "
                     "ofdmflex_tool's IF upconversion, and physically sound for "
                     "AWGN (equivalent to real AWGN) and carrier offset (equivalent to a small "
                     "residual carrier drift). It's an approximation for multipath (a real "
                     "multipath channel is a real-tap echo effect; channel_cccf's complex taps "
                     "are a reasonable but not exact stand-in). With no impairment flags given, "
                     "this is a pass-through copy.";

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include "liquid.h"
#include "liquid.argparse.h"

// ---------------------------------------------------------------------
// mono, 32-bit float WAV read/write (same format as ofdmflex_tool)
// ---------------------------------------------------------------------
void wav_write_mono(const char * _filename, const float * _buf, uint32_t _n, uint32_t _sample_rate)
{
    FILE * fid = fopen(_filename, "wb");
    if (fid == NULL) {
        fprintf(stderr, "error: could not open '%s' for writing\n", _filename);
        exit(1);
    }

    uint16_t num_channels    = 1;
    uint16_t bits_per_sample = 32;
    uint16_t audio_format    = 3; // IEEE float
    uint32_t byte_rate       = _sample_rate * num_channels * bits_per_sample/8;
    uint16_t block_align     = num_channels * bits_per_sample/8;
    uint32_t fmt_size        = 16;
    uint32_t data_bytes      = _n * sizeof(float);
    uint32_t chunk_size      = 36 + data_bytes;

    fwrite("RIFF",          1, 4, fid);
    fwrite(&chunk_size,     4, 1, fid);
    fwrite("WAVE",          1, 4, fid);

    fwrite("fmt ",          1, 4, fid);
    fwrite(&fmt_size,       4, 1, fid);
    fwrite(&audio_format,   2, 1, fid);
    fwrite(&num_channels,   2, 1, fid);
    fwrite(&_sample_rate,   4, 1, fid);
    fwrite(&byte_rate,      4, 1, fid);
    fwrite(&block_align,    2, 1, fid);
    fwrite(&bits_per_sample,2, 1, fid);

    fwrite("data",          1, 4, fid);
    fwrite(&data_bytes,     4, 1, fid);
    fwrite(_buf,            sizeof(float), _n, fid);

    fclose(fid);
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

int main(int argc, char *argv[])
{
    // liquid_argparse only understands short -x options; pull the four
    // impairment enable-toggles out manually before handing the rest to
    // the normal parser (same pattern used in ofdmflex_tool)
    char ** argv2 = (char**) malloc((argc+1)*sizeof(char*));
    int argc2 = 0;
    argv2[argc2++] = argv[0];
    bool en_awgn = false, en_cfo = false, en_multipath = false, en_shadowing = false;
    struct { const char * flag; bool * out; } long_flags[] = {
        { "--awgn",       &en_awgn },
        { "--cfo",        &en_cfo },
        { "--multipath",  &en_multipath },
        { "--shadowing",  &en_shadowing },
    };
    int i;
    for (i=1; i<argc; i++) {
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
    liquid_argparse_add(char*,    input,          "", 'i', "input wav (mono)", NULL);
    liquid_argparse_add(char*,    output,         "", 'o', "output wav (mono)", NULL);
    liquid_argparse_add(float,    noise_floor,   -60, '0', "[--awgn] noise floor [dB]", NULL);
    liquid_argparse_add(float,    snr,            30, 's', "[--awgn] signal-to-noise ratio [dB]", NULL);
    liquid_argparse_add(float,    cfo_freq,     0.0f, 'f', "[--cfo] carrier frequency offset [radians/sample]", NULL);
    liquid_argparse_add(float,    cfo_phase,    0.0f, 'p', "[--cfo] carrier phase offset [radians]", NULL);
    liquid_argparse_add(unsigned, hc_len,          5, 'H', "[--multipath] number of (random) channel taps", NULL);
    liquid_argparse_add(float,    sigma,        1.0f, 'y', "[--shadowing] log-normal shadowing std dev [dB]", NULL);
    liquid_argparse_add(float,    fd,           0.1f, 'D', "[--shadowing] Doppler frequency, range (0,0.5) exclusive", NULL);
    liquid_argparse_parse(argc2, argv2);
    free(argv2);

    if (strlen(input)==0 || strlen(output)==0) {
        fprintf(stderr, "error: both -i <input> and -o <output> are required\n");
        return -1;
    }

    uint32_t num_samples, sample_rate;
    float * mono_in = wav_read_mono(input, &num_samples, &sample_rate);

    channel_cccf channel = channel_cccf_create();

    if (en_awgn) {
        if (channel_cccf_add_awgn(channel, noise_floor, snr) != LIQUID_OK) {
            fprintf(stderr, "error: invalid --awgn parameters (noise_floor=%g, snr=%g)\n", noise_floor, snr);
            return -1;
        }
    }
    if (en_cfo) {
        if (channel_cccf_add_carrier_offset(channel, cfo_freq, cfo_phase) != LIQUID_OK) {
            fprintf(stderr, "error: invalid --cfo parameters (freq=%g, phase=%g)\n", cfo_freq, cfo_phase);
            return -1;
        }
    }
    if (en_multipath) {
        if (channel_cccf_add_multipath(channel, NULL, hc_len) != LIQUID_OK) {
            fprintf(stderr, "error: invalid --multipath parameters (hc_len=%u)\n", hc_len);
            return -1;
        }
    }
    if (en_shadowing) {
        if (channel_cccf_add_shadowing(channel, sigma, fd) != LIQUID_OK) {
            fprintf(stderr, "error: invalid --shadowing parameters (sigma=%g, fd=%g; fd must be in (0,0.5))\n", sigma, fd);
            return -1;
        }
    }

    printf("input        : '%s' (%u samples @ %u Hz)\n", input, num_samples, sample_rate);
    printf("impairments  : %s%s%s%s%s\n",
           en_awgn ? "awgn " : "", en_cfo ? "cfo " : "",
           en_multipath ? "multipath " : "", en_shadowing ? "shadowing " : "",
           (en_awgn||en_cfo||en_multipath||en_shadowing) ? "" : "(none -- pass-through)");
    channel_cccf_print(channel);

    liquid_float_complex * cbuf = (liquid_float_complex*) malloc(num_samples*sizeof(liquid_float_complex));
    for (i=0; (uint32_t)i<num_samples; i++)
        cbuf[i] = mono_in[i]; // promote real -> complex, Q=0
    free(mono_in);

    channel_cccf_execute_block(channel, cbuf, num_samples, cbuf);

    float * mono_out = (float*) malloc(num_samples*sizeof(float));
    for (i=0; (uint32_t)i<num_samples; i++)
        mono_out[i] = crealf(cbuf[i]); // keep only the real part
    free(cbuf);

    wav_write_mono(output, mono_out, num_samples, sample_rate);
    printf("output       : '%s' (%u samples @ %u Hz)\n", output, num_samples, sample_rate);

    free(mono_out);
    channel_cccf_destroy(channel);
    return 0;
}
