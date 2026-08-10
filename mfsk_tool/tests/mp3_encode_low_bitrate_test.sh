#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CTest driver: encode the same fixture as mp3_encode_default_bitrate_test.sh
# to .mp3, but at a deliberately too-low CBR bitrate (8kbit/s - bitrate_sweep
# found MFSK128L block recovery collapsing to ~49% at 8kbps even without an
# airgap simulation, and this drops blocks empirically here too, even with
# the real 3x-repeat/CRC AMP-2 protocol on top of the raw channel), and
# require the whole command to FAIL (nonzero exit) - encode's built-in
# verify pass (decode the MP3 back, byte-compare against the input) should
# catch the resulting block loss and abort before ever reporting success.
#
# Pins --samplerate 8000 (the modem's native rate, matching bitrate_sweep's
# own measurement above and this test's pre-resampling behavior): at the
# tool's new 32kHz MP3 default, the same 8kbit/s budget survives verify
# instead of failing it - not a bug, just LAME's psychoacoustic bit
# allocation concentrating those bits on the modem's actual (low, mostly
# sub-2kHz) occupied band once it's a small fraction of a wider 16kHz
# Nyquist. That's a real quality difference, not a case this test can catch
# without pinning the rate its bitrate collapse was measured at.
set -uo pipefail  # deliberately no -e: this test's whole point is a nonzero exit

[ $# -eq 3 ] || { echo "usage: $0 <mfsk_tool> <src-dir> <workdir>" >&2; exit 2; }
TOOL="$1"
SRC_DIR="$2"
WORKDIR="$3"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
archive="$WORKDIR/test_archive.tar.gz"
mp3="$WORKDIR/out.mp3"

tar czf "$archive" -C "$SRC_DIR" rsid_check.cxx

if "$TOOL" encode --bitrate 8 --samplerate 8000 "$archive" "$mp3"; then
	echo "FAIL: encode --bitrate 8 --samplerate 8000 succeeded; expected it to fail (8kbit/s @ native rate should not survive verify)" >&2
	exit 1
fi
echo "OK: encode --bitrate 8 --samplerate 8000 failed as expected (nonzero exit)"
