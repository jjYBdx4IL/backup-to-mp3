#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CTest driver: same digital round-trip check as roundtrip_test.sh, but generates
# its own ~1KB binary test archive on the fly instead of depending on a checked-in
# fixture — tar czf over one of the harness's own source files, standing in for the
# real-world "tar/7z archive" use case with real, deterministic binary content (tar
# header + gzip framing) instead of a fixture file committed just for this test.
set -euo pipefail

[ $# -eq 3 ] || { echo "usage: $0 <fldigi_amp_tool> <src-dir> <workdir>" >&2; exit 2; }
TOOL="$1"
SRC_DIR="$2"
WORKDIR="$3"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
archive="$WORKDIR/test_archive.tar.gz"
wav="$WORKDIR/out.wav"
decoded="$WORKDIR/decoded"

tar czf "$archive" -C "$SRC_DIR" rsid_check.cxx

"$TOOL" encode "$archive" "$wav"
"$TOOL" decode "$wav" "$decoded"
cmp "$archive" "$decoded"
