"""
rd_live_matplotlib_eeg.py
-------------------------
Reaction–Diffusion (Gray–Scott) demo with F and k driven by EEG text files.
- Loads one or more text files containing a single channel of EEG samples.
- Computes an arousal index = HF/LF power (HF: 13–40 Hz, LF: 1–8 Hz) on a rolling window.
- Maps arousal to F (lower with arousal) and k (higher with arousal).
- Visualizes the simulation with matplotlib.

Usage (example):
    python rd_live_matplotlib_eeg.py --files 1_horror_movie_data_filtered.txt \
        --fs 256 --speed 1.0 --chunk 32

Press keys 1..9 to switch between up to 9 loaded files.
Press 'q' to quit.

Note: This is a minimal, dependency-free implementation (NumPy + Matplotlib only).
"""

import argparse
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

#my code
from eeg2rd import EEGtoRDMapper
from eeg_filereader import OfflineEEGFeeder

def laplacian_periodic(A):
    # 5-point stencil with periodic boundary conditions
    return (
        -4 * A
        + np.roll(A, 1, 0) + np.roll(A, -1, 0)
        + np.roll(A, 1, 1) + np.roll(A, -1, 1)
    )

def rd_step(U, V, Du, Dv, F, k, dt):
    Lu = laplacian_periodic(U)
    Lv = laplacian_periodic(V)
    UVV = U * V * V
    U += (Du * Lu - UVV + F * (1 - U)) * dt
    V += (Dv * Lv + UVV - (F + k) * V) * dt
    # clamp for stability
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)

def init_fields(n=192, seed=0):
    rng = np.random.default_rng(seed)
    U = np.ones((n, n), dtype=np.float32)
    V = np.zeros((n, n), dtype=np.float32)

    # add a few random rectangles of V to kick things off
    for _ in range(4):
        x, y = rng.integers(0, n//2, size=2)
        w, h = rng.integers(n//12, n//6, size=2)
        U[y:y+h, x:x+w] = 0.50 + 0.02 * rng.standard_normal((h, w))
        V[y:y+h, x:x+w] = 0.25 + 0.02 * rng.standard_normal((h, w))
    U = np.clip(U, 0.0, 1.0)
    V = np.clip(V, 0.0, 1.0)
    return U, V

# --------------- Main program ---------------

def main():
    ap = argparse.ArgumentParser()
    #ap.add_argument("--files", nargs="+", required=True, help="EEG text files")
    ap.add_argument("--fs", type=float, default=256.0, help="Sampling rate (Hz)")
    ap.add_argument("--chunk", type=int, default=32, help="Samples per update")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument("--size", type=int, default=192, help="RD grid size (NxN)")
    ap.add_argument("--Du", type=float, default=0.16, help="Diffusion U")
    ap.add_argument("--Dv", type=float, default=0.08, help="Diffusion V")
    ap.add_argument("--dt", type=float, default=1.0, help="Integration dt")
    ap.add_argument("--win", type=float, default=4.0, help="EEG PSD window (s)")
    args = ap.parse_args()

    filename = "eeg_files/1_horror_movie_data_filtered.txt"
    feeder = OfflineEEGFeeder(filename, fs=args.fs, chunk_size=args.chunk, speed=args.speed, loop=True)
    mapper = EEGtoRDMapper(fs_hz=args.fs, win_seconds=args.win)

    U, V = init_fields(n=args.size)

    plt.figure("RD driven by EEG (F↓, k↑ with arousal)")
    img = plt.imshow(U, interpolation="nearest", vmin=0, vmax=1)
    plt.axis("off")
    txt = plt.text(5, 10, "", fontsize=9, color="white", bbox=dict(facecolor="black", alpha=0.5, pad=2))

    current_track = 0
    print(f"Loaded {feeder.n_tracks} EEG file(s). Press 1..9 to switch tracks.")

    def on_key(event):
        nonlocal current_track
        if event.key == "q":
            plt.close("all")
        elif event.key.isdigit():
            i = int(event.key) - 1
            if 0 <= i < feeder.n_tracks:
                current_track = i
                feeder.set_track(i)
                print(f"Switched to track {i+1}: {args.files[i]}")

    plt.gcf().canvas.mpl_connect("key_press_event", on_key)

    # main loop
    while plt.fignum_exists(plt.gcf().number):
        feeder.step_once()
        eeg_buf = feeder.get_buffer()
        params = mapper.update(eeg_buf)

        # RD params from EEG
        F, k = params["F"], params["k"]

        # advance several micro-steps per visual frame for smoother patterns
        for _ in range(8):
            rd_step(U, V, args.Du, args.Dv, F, k, args.dt)

        img.set_data(U)  # visualize U field
        label = f"Track:{current_track+1}/{feeder.n_tracks}  F={F:.4f}  k={k:.4f}  HF/LF={params['arousal_raw'] if params['arousal_raw'] is not None else float('nan'):.3f}"
        txt.set_text(label)
        plt.pause(0.001)
        feeder.sleep_dt()

if __name__ == "__main__":
    main()
