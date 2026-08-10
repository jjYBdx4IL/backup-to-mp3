#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulate an average speaker -> air -> microphone acoustic path over a
WAV file, per REQUIREMENTS.md's airgap verification step: frequency
response, nonlinearity, noise, plus (per follow-up request) random
dropouts, cutouts, short-duration stutter-repeats, and echo.

No scipy in this environment - the bandpass filter is a hand-built
windowed-sinc FIR (standard DSP technique, numpy-only).
"""
import argparse
import numpy as np
import soundfile as sf


def design_bandpass_fir(samplerate, low_hz, high_hz, numtaps):
    # Windowed-sinc bandpass: high-pass sinc minus low-pass sinc, Hamming windowed.
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    def sinc_lp(cutoff):
        h = 2 * cutoff / samplerate * np.sinc(2 * cutoff / samplerate * n)
        return h
    h = sinc_lp(high_hz) - sinc_lp(low_hz)
    h *= np.hamming(numtaps)
    h /= np.sum(h)  # unity gain at passband center-ish
    return h


def apply_fir(x, h):
    return np.convolve(x, h, mode="same")


def soft_clip(x, drive):
    # tanh soft-clip, normalized so drive=0 is a no-op.
    if drive <= 0:
        return x
    peak = np.max(np.abs(x)) + 1e-9
    xn = x / peak
    y = np.tanh(drive * xn) / np.tanh(drive)
    return y * peak


def add_echo(x, samplerate, delay_ms, gain):
    if gain <= 0:
        return x
    delay = int(samplerate * delay_ms / 1000.0)
    y = x.copy()
    y[delay:] += gain * x[:-delay]
    return y


def add_noise(x, snr_db):
    if snr_db is None:
        return x
    sig_power = np.mean(x.astype(np.float64) ** 2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def apply_glitches(x, samplerate, rng, dropout_rate_per_min, cutout_rate_per_min,
                    repeat_rate_per_min, min_ms, max_ms, log):
    duration_s = len(x) / samplerate
    out = []
    pos = 0
    n = len(x)

    def rate_to_count(rate_per_min):
        expected = rate_per_min * duration_s / 60.0
        return rng.poisson(expected)

    events = []
    for _ in range(rate_to_count(dropout_rate_per_min)):
        events.append(("dropout", rng.uniform(0, duration_s)))
    for _ in range(rate_to_count(cutout_rate_per_min)):
        events.append(("cutout", rng.uniform(0, duration_s)))
    for _ in range(rate_to_count(repeat_rate_per_min)):
        events.append(("repeat", rng.uniform(0, duration_s)))
    events.sort(key=lambda e: e[1])

    cursor = 0
    for kind, t in events:
        start = int(t * samplerate)
        if start < cursor or start >= n:
            continue
        dur = int(rng.uniform(min_ms, max_ms) / 1000.0 * samplerate)
        end = min(start + dur, n)

        out.append(x[cursor:start])
        if kind == "dropout":
            # signal drops to near-silence (buffer underrun / signal loss)
            out.append(x[start:end] * 0.02)
            log.append(f"dropout  at {start/samplerate:7.3f}s dur={dur/samplerate*1000:6.1f}ms")
        elif kind == "cutout":
            # segment is spliced out entirely - a true timing discontinuity
            log.append(f"cutout   at {start/samplerate:7.3f}s dur={dur/samplerate*1000:6.1f}ms (removed)")
        elif kind == "repeat":
            # short stutter: the preceding snippet plays again (buffer-replay glitch)
            src_start = max(0, start - dur)
            out.append(x[start:end])
            out.append(x[src_start:start])
            log.append(f"repeat   at {start/samplerate:7.3f}s dur={dur/samplerate*1000:6.1f}ms (snippet replayed)")
        cursor = end

    out.append(x[cursor:])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_wav")
    ap.add_argument("output_wav")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--low-hz", type=float, default=300.0, help="speaker/mic passband low edge")
    ap.add_argument("--high-hz", type=float, default=3400.0, help="speaker/mic passband high edge")
    ap.add_argument("--taps", type=int, default=257)
    ap.add_argument("--clip-drive", type=float, default=1.5, help="0 disables nonlinearity")
    ap.add_argument("--echo-delay-ms", type=float, default=35.0)
    ap.add_argument("--echo-gain", type=float, default=0.15, help="0 disables echo")
    ap.add_argument("--snr-db", type=float, default=20.0, help="omit/negative disables noise")
    ap.add_argument("--dropout-per-min", type=float, default=2.0)
    ap.add_argument("--cutout-per-min", type=float, default=1.0)
    ap.add_argument("--repeat-per-min", type=float, default=1.0)
    ap.add_argument("--glitch-min-ms", type=float, default=15.0)
    ap.add_argument("--glitch-max-ms", type=float, default=120.0)
    ap.add_argument("--no-glitches", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    x, samplerate = sf.read(args.input_wav, dtype="float64")
    if x.ndim > 1:
        x = x.mean(axis=1)
    print(f"[airgap] loaded {args.input_wav}: {len(x)} samples @ {samplerate}Hz "
          f"({len(x)/samplerate:.1f}s)")

    h = design_bandpass_fir(samplerate, args.low_hz, args.high_hz, args.taps)
    x = apply_fir(x, h)
    print(f"[airgap] applied bandpass {args.low_hz:.0f}-{args.high_hz:.0f}Hz "
          f"({args.taps}-tap FIR, simulates small speaker+mic frequency response)")

    x = soft_clip(x, args.clip_drive)
    print(f"[airgap] applied soft-clip nonlinearity, drive={args.clip_drive}")

    x = add_echo(x, samplerate, args.echo_delay_ms, args.echo_gain)
    print(f"[airgap] applied echo: delay={args.echo_delay_ms}ms gain={args.echo_gain}")

    if args.snr_db is not None and args.snr_db > 0:
        x = add_noise(x, args.snr_db)
        print(f"[airgap] applied additive noise at {args.snr_db:.1f}dB SNR")

    log = []
    if not args.no_glitches:
        x = apply_glitches(x, samplerate, rng, args.dropout_per_min, args.cutout_per_min,
                            args.repeat_per_min, args.glitch_min_ms, args.glitch_max_ms, log)
        print(f"[airgap] injected {len(log)} glitch event(s):")
        for line in log:
            print(f"[airgap]   {line}")

    peak = np.max(np.abs(x)) + 1e-9
    if peak > 0.98:
        x = x / peak * 0.98
    x16 = np.clip(x * 32767, -32768, 32767).astype(np.int16)
    sf.write(args.output_wav, x16, samplerate, subtype="PCM_16")
    print(f"[airgap] wrote {args.output_wav}: {len(x16)/samplerate:.1f}s")


if __name__ == "__main__":
    main()
