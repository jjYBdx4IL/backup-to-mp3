#!/usr/bin/env python3
"""
Regenerates tests/ir-custom-channel-48k.wav from a channel_probe.py capture
directory (see the channel_probe.py tool alongside rt.py in the project
that talks to the actual custom channel).

This re-derives the *complex* deconvolved impulse response from the raw
dry/wet sweep recording -- channel_probe.py's own results.json only stores
a magnitude-vs-time summary (for plotting), not the actual real-valued
kernel, so the deconvolution is repeated here rather than read back. The
math mirrors channel_probe.py's analyze_sweep() exactly.

Usage:
    python3 extract_channel_ir.py --probe-dir /path/to/channel_probe_out/<timestamp> \
        --out ir-custom-channel-48k.wav

Requires the probe dir to contain dry_probe.wav, wet_probe.wav, and
results.json (all written by channel_probe.py).
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
from scipy.signal import correlate

SR = 48000


def extract(probe_dir, pre_ms=2.0, post_ms=60.0, expected_offset_s=1.0):
    with open(os.path.join(probe_dir, "results.json")) as f:
        results = json.load(f)
    layout = results["probe_layout"]

    dry, sr_d = sf.read(os.path.join(probe_dir, "dry_probe.wav"))
    wet, sr_w = sf.read(os.path.join(probe_dir, "wet_probe.wav"))
    if sr_d != SR or sr_w != SR:
        raise ValueError(f"expected {SR} Hz captures, got dry={sr_d} wet={sr_w}")
    if dry.ndim > 1:
        dry = dry[:, 0]
    if wet.ndim > 1:
        wet = wet[:, 0]

    d0 = int(layout["sweep_start"] * SR)
    d1 = int(layout["sweep_end"] * SR)
    ref = dry[d0:d1]
    N = len(ref)

    search_center = int((d0 / SR + expected_offset_s) * SR)
    margin = int(0.5 * SR)
    lo = max(0, search_center - margin)
    hi = min(len(wet), search_center + margin + N)
    wet_search = wet[lo:hi]
    if len(wet_search) < N:
        raise ValueError("not enough wet audio to search for sweep alignment")

    corr = correlate(wet_search, ref, mode="valid")
    start = lo + int(np.argmax(corr))

    out_al = wet[start:start + N]
    if len(out_al) < N:
        out_al = np.pad(out_al, (0, N - len(out_al)))

    X = np.fft.rfft(ref)
    Y = np.fft.rfft(out_al)

    # deconvolve (regularized) -- H is the true complex transfer function,
    # gain and phase included, not just the |H| that results.json keeps
    Xmag2 = np.abs(X) ** 2
    eps = np.max(Xmag2) * 1e-4
    H = (Y * np.conj(X)) / (Xmag2 + eps)

    Xdb = 20 * np.log10(np.abs(X) + 1e-20)
    reliable = Xdb > (Xdb.max() - 40)
    Hclean = np.where(reliable, H, 0)

    h_ir = np.fft.irfft(Hclean, n=N)
    h_ir = np.fft.fftshift(h_ir)
    center = N // 2
    peak_search_span = int(0.05 * SR)
    peak_idx_local = int(np.argmax(np.abs(h_ir[center - peak_search_span: center + peak_search_span])))
    peak_idx = center - peak_search_span + peak_idx_local

    pre = int(pre_ms / 1000 * SR)
    post = int(post_ms / 1000 * SR)
    kernel = h_ir[peak_idx - pre:peak_idx + post].astype(np.float32)

    tone = results.get("tone_analysis", {})
    sweep = results.get("sweep_analysis", {})
    return kernel, {
        "source_probe_dir": os.path.abspath(probe_dir),
        "source_timestamp": results.get("timestamp"),
        "noise_floor_dbfs": tone.get("silence_dbfs"),
        "clock_drift_ppm": tone.get("freq_shift_ppm"),
        "delay_spread_ms_40db": sweep.get("delay_spread_ms_40db"),
        "gain_db_at_1khz": tone.get("gain_db"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-dir", required=True, help="a channel_probe_out/<timestamp> directory")
    ap.add_argument("--out", default="ir-custom-channel-48k.wav")
    ap.add_argument("--pre-ms", type=float, default=2.0, help="kernel margin before the peak (numerical leakage)")
    ap.add_argument("--post-ms", type=float, default=60.0, help="kernel length after the peak (multipath tail)")
    args = ap.parse_args()

    kernel, meta = extract(args.probe_dir, args.pre_ms, args.post_ms)
    sf.write(args.out, kernel, SR, subtype="FLOAT")
    print(f"wrote {args.out} ({len(kernel)} samples, {len(kernel)/SR*1000:.1f} ms @ {SR} Hz)")
    print(f"source probe dir : {meta['source_probe_dir']}")
    print(f"captured         : {meta['source_timestamp']}")
    print(f"noise floor      : {meta['noise_floor_dbfs']:.2f} dBFS" if meta['noise_floor_dbfs'] is not None else "noise floor      : n/a")
    print(f"clock drift      : {meta['clock_drift_ppm']:.2f} ppm" if meta['clock_drift_ppm'] is not None else "clock drift      : n/a")
    print(f"delay spread     : {meta['delay_spread_ms_40db']:.2f} ms (-40dB)" if meta['delay_spread_ms_40db'] is not None else "delay spread     : n/a")
    print("(feed noise floor / clock drift into roundtrip.sh's --post-ir-noise-floor / --drift-ppm)")
