#!/usr/bin/env bash
#
# Round-trip test for ofdmflex_tool: encodes a file to a genuinely mono wav
# via OFDM + IF upconversion, optionally runs it through a channel-noise
# stage (channel_wav) and/or an mp3 encode/decode pass, then decodes it
# back and reports the result.
#
# The C tool's decode step is self-verifying (it carries a metadata frame
# with the original file size/name/sha256, repeated redundantly along with
# the data), so its own PASS/FAIL + checksum check is authoritative; this
# script also does an independent `cmp` against the original input as a
# sanity check that the two never disagree.
#
# Binaries default to a `b/` build dir (cmake -S . -B b) next to this
# tool's root; override with OFDMFLEX_TOOL_BIN / CHANNEL_WAV_BIN env vars
# (set automatically when run via ctest).

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${OFDMFLEX_TOOL_BIN:-./ofdmflex_tool}"
CHBIN="${CHANNEL_WAV_BIN:-./channel_wav}"
IR_FILE="${IR_FILE:-$ROOT/tests/ir-48k.wav}"

die() { echo "error: $*" 1>&2; exit 1; }

if ! test -x "$BIN"; then die "run this from the build directory"; fi

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

  -i FILE      input file to round-trip (default: generated random data)
  -n BYTES     size of generated random input, if -i not given (default: 4096)
  --ofdm-params "STRING"  ofdmflex_tool's own encode/decode flags -- -m/-M/-C/
               -T/-b/-r/-x/-F/-c/-k/-R/-g/-G/--preset -- given verbatim to
               both the encode and decode invocations, e.g.:
               --ofdm-params "--preset phone1 -g 0.5 -G 0.5"
               This script doesn't parse or default any of these itself;
               omitted flags fall through to ofdmflex_tool's own defaults
               (see ofdmflex_tool -h / --list-presets). Encode-only flags
               (-R/-g/-G/-c/-k) are harmless no-ops when decode also
               receives them verbatim.
  --trace      pass --trace through to encode and decode
  --stats      pass --stats through to decode
  --ir                    apply impulse response
  --ir-preserve-gain      don't let ffmpeg's afir auto-normalize the IR's gain --
                          use this for a real measured IR (e.g.
                          ir-custom-channel-48k.wav) where the kernel's absolute
                          amplitude carries the channel's true attenuation and
                          must survive to be consistent with --post-ir-noise-floor;
                          leave off for a synthetic/generic IR where only the
                          frequency-selectivity shape matters
  --drift-ppm PPM         resample as if TX/RX sample clocks differ by PPM
                          (applied right after encode, before any other stage --
                          e.g. -107 for a receiver clock ~107ppm slow)
  --post-ir-noise-floor DB  add AWGN at an absolute level [dBFS], after the IR
                            stage rather than before it like --channel-awgn --
                            models receiver-side (mic/ADC) noise that doesn't
                            get attenuated by the channel's own path loss

  --mp3               round-trip the wav through mp3 encode/decode (lossy, CBR)
  -q QUALITY          mp3 quality/bitrate index, 0 (best, 320kbps) - 9 (worst, 64kbps) (default: 2)
  --mp3-engine ENGINE ffmpeg|internal -- ffmpeg shells out to libmp3lame via ffmpeg
                      for an intermediate wav; internal has the tool encode straight
                      to .mp3 and decode straight from it, via its own built-in
                      libmp3lame support, no ffmpeg needed. internal cannot be
                      combined with --channel (no intermediate wav to inject into).
                      (default: ffmpeg)

  --channel                  enable the channel-noise stage (channel_wav)
  --channel-order ORDER      before|after -- position relative to mp3 (default: after)
  --channel-awgn             enable AWGN
  --channel-noise-floor DB   AWGN noise floor [dB]
  --channel-snr DB           AWGN signal-to-noise ratio [dB]
  --channel-cfo              enable carrier offset
  --channel-freq RAD         carrier offset frequency [rad/sample]
  --channel-phase RAD        carrier offset phase [rad]
  --channel-multipath        enable multipath fading
  --channel-hlen N           multipath channel tap count
  --channel-shadowing        enable log-normal shadowing
  --channel-sigma DB         shadowing std dev [dB]
  --channel-fd FD            shadowing Doppler frequency, (0,0.5) exclusive

  -k, --keep   keep the working directory instead of deleting it on exit
  -h, --help   show this help
EOF
}

input_file=""
gen_size=4096
ofdm_params=""
trace=0
stats=0
apply_ir=0
ir_preserve_gain=0
drift_ppm=""
post_ir_noise_floor=""

run_mp3=0
mp3_quality=2
mp3_engine=ffmpeg

run_channel=0
channel_order=after
ch_awgn=0; ch_noise_floor=""; ch_snr=""
ch_cfo=0; ch_freq=""; ch_phase=""
ch_multipath=0; ch_hlen=""
ch_shadowing=0; ch_sigma=""; ch_fd=""

keep=0

while [ $# -gt 0 ]; do
    case "$1" in
        -i) input_file="$2"; shift 2 ;;
        -n) gen_size="$2"; shift 2 ;;
        --ofdm-params) ofdm_params="$2"; shift 2 ;;
        --trace) trace=1; shift ;;
        --stats) stats=1; shift ;;
        --ir) apply_ir=1; shift ;;
        --ir-preserve-gain) ir_preserve_gain=1; shift ;;
        --drift-ppm) drift_ppm="$2"; shift 2 ;;
        --post-ir-noise-floor) post_ir_noise_floor="$2"; shift 2 ;;

        --mp3) run_mp3=1; shift ;;
        -q) mp3_quality="$2"; shift 2 ;;
        --mp3-engine) mp3_engine="$2"; shift 2 ;;

        --channel) run_channel=1; shift ;;
        --channel-order) channel_order="$2"; shift 2 ;;
        --channel-awgn) ch_awgn=1; shift ;;
        --channel-noise-floor) ch_noise_floor="$2"; shift 2 ;;
        --channel-snr) ch_snr="$2"; shift 2 ;;
        --channel-cfo) ch_cfo=1; shift ;;
        --channel-freq) ch_freq="$2"; shift 2 ;;
        --channel-phase) ch_phase="$2"; shift 2 ;;
        --channel-multipath) ch_multipath=1; shift ;;
        --channel-hlen) ch_hlen="$2"; shift 2 ;;
        --channel-shadowing) ch_shadowing=1; shift ;;
        --channel-sigma) ch_sigma="$2"; shift 2 ;;
        --channel-fd) ch_fd="$2"; shift 2 ;;

        -k|--keep) keep=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1' (see -h)" ;;
    esac
done

case "$channel_order" in
    before|after) ;;
    *) die "--channel-order must be 'before' or 'after'" ;;
esac

case "$mp3_engine" in
    ffmpeg|internal) ;;
    *) die "--mp3-engine must be 'ffmpeg' or 'internal'" ;;
esac

# CBR bitrate [kbps] per -q index (0=best/highest bitrate .. 9=worst/lowest),
# using standard MPEG-1 Layer III bitrates -- matches the tool's own internal
# libmp3lame quality table, and roughly tracks LAME's V0-V9 VBR average
# bitrates for the ffmpeg path
mp3_bitrates=(320 256 224 192 160 128 112 96 80 64)

if [ "$run_mp3" -eq 1 ]; then
    if ! [[ "$mp3_quality" =~ ^[0-9]$ ]]; then
        die "-q must be an integer 0-9 (mp3 quality/bitrate index)"
    fi
    if [ "$mp3_engine" = "internal" ] && [ "$run_channel" -eq 1 ]; then
        die "--mp3-engine internal cannot be combined with --channel -- the internal" \
            "path has the tool encode straight to .mp3 with no intermediate wav for" \
            "the channel stage to operate on; use --mp3-engine ffmpeg instead"
    fi
fi

echo "== configuration =="
if [ -n "$input_file" ]; then
    echo "input        : $input_file"
else
    echo "input        : (generated, $gen_size random bytes)"
fi
echo "ofdm params  : ${ofdm_params:-(none given -- ofdmflex_tool defaults apply)}"
echo "trace        : $([ "$trace" -eq 1 ] && echo on || echo off)"
echo "stats        : $([ "$stats" -eq 1 ] && echo on || echo off)"
echo "keep workdir : $([ "$keep" -eq 1 ] && echo yes || echo no)"
[ -n "$drift_ppm" ] && echo "clock drift  : ${drift_ppm} ppm"
[ "$apply_ir" -eq 1 ] && echo "impulse resp : $IR_FILE$([ "$ir_preserve_gain" -eq 1 ] && echo " (gain preserved)")"
[ -n "$post_ir_noise_floor" ] && echo "post-ir noise: ${post_ir_noise_floor} dBFS"

if [ "$run_mp3" -eq 1 ]; then
    echo "mp3 stage    : enabled (quality $mp3_quality, engine: $mp3_engine)"
else
    echo "mp3 stage    : disabled"
fi

if [ "$run_channel" -eq 1 ]; then
    echo "channel stage: enabled (order: $channel_order)"
    [ "$ch_awgn" -eq 1 ]      && echo "  awgn       : noise_floor=${ch_noise_floor:-default} snr=${ch_snr:-default}"
    [ "$ch_cfo" -eq 1 ]       && echo "  cfo        : freq=${ch_freq:-default} phase=${ch_phase:-default}"
    [ "$ch_multipath" -eq 1 ] && echo "  multipath  : hlen=${ch_hlen:-default}"
    [ "$ch_shadowing" -eq 1 ] && echo "  shadowing  : sigma=${ch_sigma:-default} fd=${ch_fd:-default}"
    if [ "$ch_awgn" -eq 0 ] && [ "$ch_cfo" -eq 0 ] && [ "$ch_multipath" -eq 0 ] && [ "$ch_shadowing" -eq 0 ]; then
        echo "  (none of --channel-awgn/-cfo/-multipath/-shadowing given -- pass-through)"
    fi
else
    echo "channel stage: disabled"
fi
echo

# the tools must already be built (cmake -S . -B b && cmake --build b)
[ -x "$BIN" ] || die "'$BIN' not found -- build it first (cmake -S . -B b && cmake --build b)"
if [ "$run_channel" -eq 1 ]; then
    [ -x "$CHBIN" ] || die "'$CHBIN' not found -- build it first (cmake -S . -B b && cmake --build b)"
fi

work="$(mktemp -d)"
cleanup() {
    if [ "$keep" -eq 1 ]; then
        echo "working directory kept at: $work"
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

# input data
if [ -n "$input_file" ]; then
    [ -f "$input_file" ] || die "input file '$input_file' not found"
    cp "$input_file" "$work/input.bin"
    input_desc="$input_file"
else
    head -c "$gen_size" /dev/urandom > "$work/input.bin"
    input_desc="(generated, $gen_size random bytes)"
fi
in_size=$(stat -c%s "$work/input.bin" 2>/dev/null || stat -f%z "$work/input.bin")

echo "== encode =="
echo "input        : $input_desc ($in_size bytes)"
echo

internal_mp3=0
[ "$run_mp3" -eq 1 ] && [ "$mp3_engine" = "internal" ] && internal_mp3=1

if [ "$internal_mp3" -eq 1 ]; then
    carrier_out="$work/wave.mp3"
else
    carrier_out="$work/wave.wav"
fi

# split --ofdm-params into an argv array via `read`, not bare unquoted
# expansion, so glob characters in the string (if any) aren't expanded
ofdm_arr=()
[ -n "$ofdm_params" ] && IFS=' ' read -ra ofdm_arr <<< "$ofdm_params"

encode_args=(encode -i "$work/input.bin" -o "$carrier_out" "${ofdm_arr[@]}")
[ "$trace" -eq 1 ] && encode_args+=(--trace)
[ "$internal_mp3" -eq 1 ] && encode_args+=(-q "$mp3_quality")

echo "$BIN" "${encode_args[@]}"
"$BIN" "${encode_args[@]}" || die "encode failed"

stage_input="$carrier_out"

# the wav rate ofdmflex_tool actually encoded at isn't known to this script
# any more (it's whatever --ofdm-params -r/-x/--preset resolved to inside
# the tool) -- detect it from the encoded file itself, lazily, only if a
# stage that needs it (drift/mp3-ffmpeg/ir) actually runs
final_rate=""
detect_final_rate() {
    [ -n "$final_rate" ] && return
    final_rate=$(ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of csv=p=0 "$carrier_out" 2>/dev/null | head -n1)
    [ -n "$final_rate" ] || die "could not detect the sample rate ofdmflex_tool encoded at (ffprobe probe of '$carrier_out' failed -- is ffprobe installed?)"
}

if [ "$internal_mp3" -eq 1 ]; then
    mp3_bitrate_kbps="${mp3_bitrates[$mp3_quality]}"
    mp3_size=$(stat -c%s "$carrier_out" 2>/dev/null || stat -f%z "$carrier_out")
    echo
    echo "== mp3 stage (internal, lossy, CBR) =="
    echo "encoded straight to $(basename "$carrier_out") ($mp3_size bytes, quality $mp3_quality = ${mp3_bitrate_kbps}kbps CBR) via the tool's built-in libmp3lame path (no ffmpeg)"
fi

if [ -n "$drift_ppm" ]; then
    [ "$internal_mp3" -eq 1 ] && die "--drift-ppm cannot be combined with --mp3-engine internal -- no intermediate wav for it to operate on"
    echo
    echo "== clock drift (${drift_ppm} ppm) =="
    drift_factor=$(LC_NUMERIC=C awk -v p="$drift_ppm" 'BEGIN{printf "%.9f", 1.0 + p*1e-6}')
    detect_final_rate
    sox "$stage_input" -r "$final_rate" "$work/stage_drift_out.wav" speed "$drift_factor" || die "drift stage failed"
    stage_input="$work/stage_drift_out.wav"
fi

run_mp3_stage() {
    echo
    echo "== mp3 stage (lossy, CBR) =="

    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "SKIPPED (ffmpeg not found)"
        return
    fi
    mp3_bitrate_kbps="${mp3_bitrates[$mp3_quality]}"

    ffmpeg -y -loglevel error \
        -i "$stage_input" \
        -codec:a libmp3lame -b:a "${mp3_bitrate_kbps}k" -compression_level 0 \
        "$work/stage_mp3.mp3" || die "mp3 encode failed"

    mp3_size=$(stat -c%s "$work/stage_mp3.mp3" 2>/dev/null || stat -f%z "$work/stage_mp3.mp3")

    detect_final_rate
    ffmpeg -y -loglevel error \
        -i "$work/stage_mp3.mp3" \
        -ar "$final_rate" -ac 1 -acodec pcm_f32le \
        "$work/stage_mp3_out.wav" || die "mp3 decode failed"

    echo "$(basename "$stage_input") -> stage_mp3.mp3 ($mp3_size bytes, quality $mp3_quality = ${mp3_bitrate_kbps}kbps CBR) -> stage_mp3_out.wav"

    if [ "$stats" -eq 1 ] && command -v ffprobe >/dev/null 2>&1; then
        mp3_duration=$(ffprobe -v quiet -show_entries format=duration -of csv="p=0" "$work/stage_mp3.mp3")
        if [ -n "$mp3_duration" ]; then
            effective_kbps=$(LC_NUMERIC=C awk -v b="$mp3_size" -v d="$mp3_duration" 'BEGIN{printf "%.1f", (b*8)/d/1000}')
            echo "effective mp3 bitrate: ${effective_kbps} kbps (measured: ${mp3_size} bytes / ${mp3_duration}s)"
        fi
    fi

    stage_input="$work/stage_mp3_out.wav"
}

run_channel_stage() {
    echo
    echo "== channel-noise stage =="

    channel_args=(-i "$stage_input" -o "$work/stage_channel_out.wav")
    [ "$ch_awgn" -eq 1 ]      && channel_args+=(--awgn)
    [ -n "$ch_noise_floor" ]  && channel_args+=(-0 "$ch_noise_floor")
    [ -n "$ch_snr" ]          && channel_args+=(-s "$ch_snr")
    [ "$ch_cfo" -eq 1 ]       && channel_args+=(--cfo)
    [ -n "$ch_freq" ]         && channel_args+=(-f "$ch_freq")
    [ -n "$ch_phase" ]        && channel_args+=(-p "$ch_phase")
    [ "$ch_multipath" -eq 1 ] && channel_args+=(--multipath)
    [ -n "$ch_hlen" ]         && channel_args+=(-H "$ch_hlen")
    [ "$ch_shadowing" -eq 1 ] && channel_args+=(--shadowing)
    [ -n "$ch_sigma" ]        && channel_args+=(-y "$ch_sigma")
    [ -n "$ch_fd" ]           && channel_args+=(-D "$ch_fd")

    "$CHBIN" "${channel_args[@]}" || die "channel stage failed"
    stage_input="$work/stage_channel_out.wav"
}

run_mp3_ffmpeg=0
[ "$run_mp3" -eq 1 ] && [ "$mp3_engine" = "ffmpeg" ] && run_mp3_ffmpeg=1

if [ "$run_channel" -eq 1 ] || [ "$run_mp3_ffmpeg" -eq 1 ]; then
    if [ "$channel_order" = "before" ]; then
        [ "$run_channel" -eq 1 ]    && run_channel_stage
        [ "$run_mp3_ffmpeg" -eq 1 ] && run_mp3_stage
    else
        [ "$run_mp3_ffmpeg" -eq 1 ] && run_mp3_stage
        [ "$run_channel" -eq 1 ]    && run_channel_stage
    fi
fi

if [ "$apply_ir" -eq 1 ]; then
    echo
    echo "== apply impulse response =="
    afir_filter="afir"
    [ "$ir_preserve_gain" -eq 1 ] && afir_filter="afir=gtype=none:irnorm=-1"
    detect_final_rate
    ffmpeg -y -loglevel error \
        -i "$stage_input" -i "$IR_FILE" \
        -filter_complex "$afir_filter" \
        -c:a pcm_f32le -ar "$final_rate" \
        "$work/stage_ir_out.wav" || die "impulse response stage failed"
    stage_input="$work/stage_ir_out.wav"
fi

if [ -n "$post_ir_noise_floor" ]; then
    echo
    echo "== post-channel receiver noise floor (${post_ir_noise_floor} dBFS) =="
    [ -x "$CHBIN" ] || die "'$CHBIN' not found -- build it first (cmake -S . -B b && cmake --build b)"
    # channel_cccf_add_awgn() applies gain gamma=10^((snr+noise_floor)/20) before
    # adding noise of std 10^(noise_floor/20); setting snr = -noise_floor makes
    # gamma == 1 (no signal rescale), leaving only the absolute noise addition --
    # unlike --channel-awgn (applied before --ir), this happens after the channel
    # path, like a real receiver's own noise floor
    neg_snr=$(LC_NUMERIC=C awk -v n="$post_ir_noise_floor" 'BEGIN{printf "%.6f", -n}')
    "$CHBIN" -i "$stage_input" -o "$work/stage_postirnoise_out.wav" \
        --awgn -0 "$post_ir_noise_floor" -s "$neg_snr" || die "post-ir noise stage failed"
    stage_input="$work/stage_postirnoise_out.wav"
fi

echo
echo "== decode =="
echo

decode_args=(decode -i "$stage_input" -o "$work/output.bin" "${ofdm_arr[@]}")
[ "$trace" -eq 1 ] && decode_args+=(--trace)
[ "$stats" -eq 1 ] && decode_args+=(--stats)

echo "$BIN" "${decode_args[@]}"
"$BIN" "${decode_args[@]}"
decode_status=$?

echo
if cmp -s "$work/input.bin" "$work/output.bin"; then
    cmp_result="identical"
else
    cmp_result="DIFFERS"
fi
echo "independent cmp check: input vs output -- $cmp_result"

echo
if [ "$decode_status" -eq 0 ]; then
    echo "overall: PASS"
else
    echo "overall: FAIL"
fi
exit $decode_status
