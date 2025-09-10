#!/usr/bin/env python3
"""
Stream simulated EEG samples to your Flask server in real time.

Server:
  python rd_taichi_eeg_classify.py --fs 256 --buffer-s 8 --port 5000

Client (this script):
  python eeg_stream_client.py --host 127.0.0.1 --port 5000 --fs 256 --chunk 64 --duration 120 --demo-profile

POSTs JSON to: http://{host}:{port}/ingest
Body: {"samples": [float, ...]}
"""

import argparse
import time
import math
from typing import List, Tuple
import random


import numpy as np
import requests


def demo_profile_segments() -> List[Tuple[float, float]]:
    """
    Returns a repeating profile of (duration_seconds, hf_weight) segments.
    hf_weight in [0..1] controls how much high-frequency power to add.
    """
    fuktuple = [
        (20.0, 0.04),
        (20.0, 0.05),  # CALM: mostly LF (alpha-ish ~10 Hz)
        (20.0, 0.25),  # MOD-STRESS
        (15.0, 0.26),
        (16.0, 0.27),
        (20.0, 0.55),  # HIGH-STRESS
        (20.0, 0.90),  # EXTREME-STRESS: mostly HF
    ]
    fuklist = list(fuktuple)
    random.shuffle(fuklist)
    fuktuple = tuple(fuklist)
    return fuktuple


def make_chunk(
    fs: float,
    start_sample_idx: int,
    n: int,
    mode: str,
    lf_freq: float,
    hf_freq: float,
    noise_std: float,
    t0: float,
    profile: List[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Generate a chunk of EEG-like samples.

    mode:
      - "demo": uses a time-based profile of HF/LF blends
      - "fixed": fixed blend using hf_weight from profile[0][1] if provided, else 0.2
    """
    # Time vector for this chunk
    idx = np.arange(start_sample_idx, start_sample_idx + n, dtype=np.float64)
    t = idx / fs

    if mode == "demo" and profile:
        # Determine which segment we are in based on wall time elapsed
        elapsed = time.perf_counter() - t0
        cycle = sum(seg[0] for seg in profile)
        in_cycle = elapsed % cycle
        acc = 0.0
        hf_w = profile[-1][1]
        for dur, w in profile:
            if in_cycle < acc + dur:
                hf_w = w
                break
            acc += dur
    else:
        hf_w = profile[0][1] if (profile and len(profile) > 0) else 0.2

    lf = np.sin(2.0 * math.pi * lf_freq * t, dtype=np.float64)  # 8–12 Hz alpha-ish
    hf = np.sin(2.0 * math.pi * hf_freq * t + 0.7, dtype=np.float64)  # 20–35 Hz beta/gamma-ish
    signal = (1.0 - hf_w) * lf + hf_w * hf

    if noise_std > 0:
        signal = signal + np.random.normal(0.0, noise_std, size=signal.shape)

    # Optional mild amplitude limiting to look realistic(ish)
    np.clip(signal, -3.0, 3.0, out=signal)
    return signal.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Flask server host")
    ap.add_argument("--port", type=int, default=5000, help="Flask server port")
    ap.add_argument("--fs", type=float, default=256.0, help="Sample rate (Hz)")
    ap.add_argument("--chunk", type=int, default=64, help="Samples per POST")
    ap.add_argument("--duration", type=float, default=60.0, help="Seconds to stream (<=0 = infinite)")
    ap.add_argument("--noise-std", type=float, default=0.05, help="Gaussian noise std dev")
    ap.add_argument("--lf-freq", type=float, default=10.0, help="Low-frequency tone (Hz)")
    ap.add_argument("--hf-freq", type=float, default=25.0, help="High-frequency tone (Hz)")
    ap.add_argument("--demo-profile", action="store_true", help="Cycle CALM→MOD→HIGH→EXTREME automatically")
    ap.add_argument("--fixed-hf", type=float, default=0.2, help="HF weight if not using --demo-profile (0..1)")
    ap.add_argument("--timeout", type=float, default=2.0, help="HTTP request timeout (s)")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/ingest"
    print(f"Streaming to {url} at fs={args.fs} Hz, chunk={args.chunk} samples... (Ctrl-C to stop)")

    # Profile configuration
    profile = demo_profile_segments() if args.demo_profile else [(float("inf"), float(np.clip(args.fixed_hf, 0.0, 1.0)))]
    mode = "demo" if args.demo_profile else "fixed"

    # Real-time pacing
    chunk_period = args.chunk / args.fs  # seconds per chunk
    next_send = time.perf_counter() + chunk_period
    start_sample = 0
    t0 = time.perf_counter()
    end_time = t0 + (args.duration if args.duration > 0 else 10**12)

    try:
        while True:
            now = time.perf_counter()
            if now >= end_time:
                print("Done (duration reached).")
                break

            # Generate next chunk
            chunk = make_chunk(
                fs=args.fs,
                start_sample_idx=start_sample,
                n=args.chunk,
                mode=mode,
                lf_freq=args.lf_freq,
                hf_freq=args.hf_freq,
                noise_std=args.noise_std,
                t0=t0,
                profile=profile,
            )
            start_sample += args.chunk

            # Send
            try:
                r = requests.post(url, json={"samples": chunk.tolist()}, timeout=args.timeout)
                if r.status_code != 200:
                    print(f"[WARN] Bad status {r.status_code}: {r.text[:200]}...")
            except Exception as e:
                print(f"[ERROR] POST failed: {e}")

            # Pace to real time (account for time spent)
            next_send += chunk_period
            sleep_for = next_send - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We’re behind; reset schedule to avoid drift
                next_send = time.perf_counter()

    except KeyboardInterrupt:
        print("\nInterrupted, stopping.")


if __name__ == "__main__":
    main()
