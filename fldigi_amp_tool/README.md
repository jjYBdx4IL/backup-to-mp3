# fldigi_amp_harness

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

