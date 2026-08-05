#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# End-to-end 7z <-> MP3 pipeline: fldigi_amp_tool (WAV) + ffmpeg (MP3).
# MP3 target: 112kbit/s CBR, matching the modem's native 8kHz so ffmpeg's
# own resample replaces fldigi's much slower internal libsamplerate path
# (both are correct - see SUMMARY.md - this is purely a speed choice).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="$HERE/build-cmake/fldigi_amp_tool"

usage() {
	echo "usage: $0 encode <input-file> <output.mp3>" >&2
	echo "       $0 decode <input.mp3> <output-file>" >&2
	exit 2
}

[ $# -eq 3 ] || usage

case "$1" in
	encode)
		tmpwav="$(mktemp --suffix=.wav)"
		trap 'rm -f "$tmpwav"' EXIT
		"$TOOL" encode "$2" "$tmpwav"
		"$TOOL" checksilence "$tmpwav"
		ffmpeg -y -loglevel error -i "$tmpwav" -ar 8000 -b:a 112k -c:a libmp3lame "$3"
		;;
	decode)
		tmpwav="$(mktemp --suffix=.wav)"
		trap 'rm -f "$tmpwav"' EXIT
		ffmpeg -y -loglevel error -i "$2" -ar 8000 "$tmpwav"
		"$TOOL" decode "$tmpwav" "$3"
		;;
	*)
		usage
		;;
esac
