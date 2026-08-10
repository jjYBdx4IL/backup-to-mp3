# mfsk_tool

> **Disclaimer:** provided as-is, no warranty. This is AI slop — developed by
> Claude (Sonnet 5). Review before trusting it with anything that matters.

Encodes/decodes small files (e.g. a 7z backup archive) to/from MP3, using
fldigi's real MFSK32/MFSK128L modem code linked in headlessly (no GUI, no
fldigi process). Built to get backups onto a Huawei D2 watch, which only
accepts MP3 uploads and can't be downloaded from — the MP3 is decoded back
via a real AMP-2 (FLAMP-compatible) transmission, including an acoustic
airgap simulation to test speaker/mic round-trip recovery.

See `LICENSE` / `NOTICE` for licensing (GPL-3.0-or-later — required by
static linking against fldigi/FLAMP).

## REQUIREMENTS

Optimally you build this on Debian 13/trixie, but most/any Linux distro should do.

Besides what `fltk-config`/`cmake/CheckSystemDeps.cmake` already checks for,
MP3 encode/decode link directly against `libmp3lame` and `libmpg123` (no
`ffmpeg` process involved):

    sudo apt-get install libmp3lame-dev libmpg123-dev

## USAGE

    mfsk_tool encode [--bitrate KBPS] [--samplerate HZ] [--no-verify] <input-file> <output.wav|output.mp3>
    mfsk_tool decode <input.wav|input.mp3> <output-file>

MP3 vs. WAV is chosen by the output/input file extension. MP3 is encoded
CBR with LAME quality=0 (best/slowest - the direct-lib equivalent of
ffmpeg's `-compression_level 0` for libmp3lame); `--bitrate` sets the CBR
bitrate in kbit/s (default 112). The modem itself only ever runs at its
fixed native rate (8kHz), so MP3 output is resampled up to `--samplerate`
(default 32kHz, via libsamplerate - some MP3 playback targets, including
the Huawei D2 watch this tool was built for, don't reliably accept 8kHz
MP3s) before compressing. WAV output is always written at the modem's
native 8kHz, unaffected by `--samplerate`; decode auto-detects and handles
whatever sample rate its input actually is.

Encoding to `.mp3` verifies by default: it decodes the finished MP3 back
and requires a byte-exact match against `<input-file>`, exiting nonzero
(and leaving no stray temp files) if it doesn't match. Skip with
`--no-verify`.

## LIST OF VARIOUS USEFUL COMMANDS

```
rm -rf b && cmake -S . -B b -DCMAKE_BUILD_TYPE=Release && cd b && cmake --build . -j && ctest . -j && cmake --install .
```
