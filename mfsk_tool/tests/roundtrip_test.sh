#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CTest driver: encode <input> to WAV, decode it back, and require a byte-exact
# match. This is the digital (no airgap simulation) round-trip check from
# REQUIREMENTS.md's verification item #1.
set -euo pipefail

[ $# -eq 3 ] || { echo "usage: $0 <mfsk_tool> <input-file> <workdir>" >&2; exit 2; }
TOOL="$1"
INPUT="$2"
WORKDIR="$3"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
wav="$WORKDIR/out.wav"
decoded="$WORKDIR/decoded"

"$TOOL" encode "$INPUT" "$wav"
"$TOOL" decode "$wav" "$decoded"
cmp "$INPUT" "$decoded"
