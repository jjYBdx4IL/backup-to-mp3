#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# CTest driver: encode a file (which writes real TxID/RSID bursts), then confirm
# fldigi's own RSID receiver recognizes them via rsid_check (see task #6/rsid_check
# in SUMMARY.md's engineering log). rsid_check doesn't itself fail on a miss, so
# this only checks it runs cleanly end-to-end over a real encoded WAV.
set -euo pipefail

[ $# -eq 4 ] || { echo "usage: $0 <mfsk_tool> <rsid_check> <input-file> <workdir>" >&2; exit 2; }
TOOL="$1"
RSID_CHECK="$2"
INPUT="$3"
WORKDIR="$4"

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"
wav="$WORKDIR/out.wav"

"$TOOL" encode "$INPUT" "$wav"
"$RSID_CHECK" "$wav"
