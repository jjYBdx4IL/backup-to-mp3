# ofdmflex_tool

> **Disclaimer:** provided as-is, no warranty. This is AI slop — developed by
> Claude (Sonnet 5). Review before trusting it with anything that matters.

Transmits an arbitrary file over OFDM (liquid-dsp's `ofdmflexframegen` /
`ofdmflexframesync`), captured to/read from a genuinely mono, real-valued
WAV file. The complex OFDM baseband is oversampled and mixed up to a
carrier frequency on encode (quadrature upconversion); decode mixes back
down, filters, and decimates back to baseband before demodulating — the
same trick real radios use to put a complex baseband signal on a real
wire/antenna/speaker.

If `-o` (encode) or `-i` (decode) ends in `.mp3`, that file is
encoded/decoded as mp3 on the fly via libmp3lame instead of wav — no
separate conversion pass needed to get the transmission onto mp3-only
storage. `-q` sets the mp3 quality/bitrate (encode only, 0=best/320kbps
.. 9=worst/64kbps).

`channel_wav` is a companion tool: it applies `channel_cccf` impairments
(AWGN, carrier offset, multipath, shadowing) to a mono WAV file, for
testing how well a round-trip survives a noisy channel.

liquid-dsp itself is fetched and built via CPM.cmake (MIT-licensed).
OpenSSL (libcrypto) and libmp3lame are pulled from the system instead, so
their `-dev` packages (`libssl-dev`, `libmp3lame-dev` on Debian/Ubuntu)
need to already be installed.

## USAGE

```
ofdmflex_tool encode -i <input-file>       -o <output.wav|output.mp3> [options]
ofdmflex_tool decode -i <input.wav|.mp3>   -o <output-file>           [options]
channel_wav        -i <input.wav>  -o <output.wav>  [--awgn|--cfo|--multipath|--shadowing ...]
```

Run either tool with `-h` for the full option list (modulation, FEC,
block size, sample rate, channel impairment parameters, ...). The file is
split into fixed-size blocks, each sent as its own OFDM frame, preceded
by a metadata frame (size/name/sha256); the whole transmission repeats
several times for redundancy. Decode is self-verifying against that
metadata.

`ofdmflex_tool` also prints every parameter value actually in effect
(defaults, preset, explicit flags all resolved) at the start of each
encode/decode run. `--preset <name>` applies a named bundle of parameter
values, e.g. `ofdmflex_tool encode -i file -o out.wav --preset phone1`;
any flag given after `--preset` on the command line still overrides that
preset's value for the same option. `--list-presets` prints the available
presets and the exact flag values each one sets.

Available presets:

| name     | params                                                              | notes |
|----------|-----------------------------------------------------------------------|-------|
| `phone1` | `-C 64 -T 12 -b 1024 -M 192 -g 0 -G 0 -r 6000 -c rs8 -k v27 -q 2 -x 8` | narrowband/robust profile tuned for a phone-handset acoustic link |

`tests/roundtrip.sh` drives both tools end-to-end (random or given input,
optional channel-noise and/or mp3 lossy stage, decode, and an independent
`cmp` check) — run it directly with `-h` for its own options, or via
`ctest` for the smoke-test wiring.

### Real-channel replica

`tests/ir-custom-channel-48k.wav` is a real impulse response measured off
a live custom channel (log-sweep + deconvolution, via a `channel_probe.py`
capture), not a synthetic one — `tests/extract_channel_ir.py --probe-dir
<capture dir>` regenerates it from a fresh capture, printing the matching
noise floor and clock drift to feed back in. The `roundtrip_custom_channel`
ctest case combines it with that capture's measured receiver noise floor
and TX/RX clock drift (`--post-ir-noise-floor`, `--drift-ppm`) plus the
encode parameters found to be the fastest ones that still decode reliably
against it — see `roundtrip.sh -h` for those flags plus `--ir-preserve-gain`
(keeps a real IR's absolute gain instead of ffmpeg's `afir` auto-normalizing
it away, so the noise floor stays meaningful relative to signal level).

## BUILD

```
cmake -S . -B b -DCMAKE_BUILD_TYPE=Release && cmake --build b -j && ctest --test-dir b
```

Installs to `~/.local/bin` (or `/usr/local/bin` as root) via
`cmake --install b`.

## LICENSE

Somewhat complicated. Read Liquid DSP library's license comments. The build output is also linked against other libs like libmp3lame and openssl.

The source code in this repository is MIT licensed.
