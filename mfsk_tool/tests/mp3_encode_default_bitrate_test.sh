#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CTest driver: encode <input> straight to .mp3 at the default CBR bitrate
# (112kbit/s - see main.cxx's DEFAULT_MP3_KBPS) and require the whole
# command to succeed (exit 0). "Succeed" here means encode's own built-in
# verify pass passed - it decodes the finished MP3 back and requires a
# byte-exact match against the input - not just that a file got written.
# See mp3_encode_low_bitrate_test.sh for the same check at a bitrate too
# low to survive it.
set -euo pipefail

[ $# -eq 3 ] || { echo "usage: $0 <mfsk_tool> <src-dir> <workdir>" >&2; exit 2; }
TOOL="$1"
SRC_DIR="$2"
WORKDIR="$3"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
archive="$WORKDIR/test_archive.tar.gz"
mp3="$WORKDIR/out.mp3"

# Same ~1KB on-the-fly fixture as roundtrip_generated_archive_test.sh - real,
# deterministic binary content (tar header + gzip framing), closer to the
# real 7z-archive use case than a tiny text fixture.
tar czf "$archive" -C "$SRC_DIR" rsid_check.cxx

"$TOOL" encode "$archive" "$mp3"
