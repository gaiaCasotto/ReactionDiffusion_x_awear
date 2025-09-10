
# t_rd_multi.py
#
# One-window, multi-simulation version of your RD + EEG-driven color/params sketch.
# Usage:
#   python t_rd_colors_multi.py eeg1.txt eeg2.txt eeg3.txt ...
#
# For each EEG .txt you pass, we run one Gray–Scott simulation driven by a
# LiveArousalClassifier over that file's stream. All sims are tiled into a single
# Taichi window. The rendering shader + easing and stress palette are adapted from
# your original t_rd_colors.py.
#
# Requirements: taichi>=1.6, numpy, and your eeg_filereader module available.

import time
import math
import random
from pathlib import Path
import sys
from dataclasses import dataclass

import numpy as np
import taichi as ti

# your module
from eeg_filereader import OfflineEEGFeeder, LiveArousalClassifier




# =========================
# Config
# =========================
@dataclass
class RDConfig:
    tile_res: int = 448          # resolution of each simulation tile (square)
    steps_per_frame: int = 60    # RD substeps per frame (like your 'steps' slider)
    tau_secs: float = 3.0        # easing time-constant for params & color
    rng_seed: int = 123
    # default/base params used per state (same spirit as original)
    calm:  tuple = (0.107, 0.056, 1.50, 0.591)
    mod:   tuple = (0.107, 0.056, 1.50, 0.591)
    high:  tuple = (0.024, 0.058, 1.050, 0.28)
    xtrm:  tuple = (0.025, 0.058, 1.050, 0.50)

CFG = RDConfig()

# small LFO wobble values (copied from your original intent)
AMP_F, AMP_K  = 0.0010, 0.001
AMP_DA, AMP_DB = 0.015, 0.010
FREQ_F, FREQ_K, FREQ_DA, FREQ_DB = 0.03, 0.021, 0.017, 0.026

# clamp utility
@ti.func
def clamp(x, lo, hi):
    return ti.min(hi, ti.max(lo, x))

# =========================
# Taichi setup (vectorized over N sims)
# =========================


try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)


# Scalar constants
DT = 0.5  # match your original dt

# Dynamic fields sized after we know N and tile_res
A = ti.field(dtype=ti.f32)  # U
B = ti.field(dtype=ti.f32)  # V
_dA = ti.field(dtype=ti.f32)
_dB = ti.field(dtype=ti.f32)

# per-sim parameters & targets: [N]
F      = ti.field(dtype=ti.f32)
K      = ti.field(dtype=ti.f32)
Da     = ti.field(dtype=ti.f32)
Db     = ti.field(dtype=ti.f32)
F_tgt  = ti.field(dtype=ti.f32)
K_tgt  = ti.field(dtype=ti.f32)
Da_tgt = ti.field(dtype=ti.f32)
Db_tgt = ti.field(dtype=ti.f32)

# smoothed stress mix (0..1) per-sim and its target
stress_mix   = ti.field(dtype=ti.f32)
stress_tgt   = ti.field(dtype=ti.f32)

# window-sized render target (NDArray for fast blit)
# we'll create a NumPy array and set it via canvas.set_image()


def alloc_fields(n_sims: int, tile: int):
    # 3D grids for the RD arrays
    ti.root.dense(ti.ijk, (n_sims, tile, tile)).place(A, B, _dA, _dB)   #this says: “For each of the N simulations (n_sims), 
                                                                                    #allocate a full (tile × tile) reaction–diffusion grid, 
                                                                                    #and advance them all in parallel.”
    # 1D per-sim parameter vectors
    ti.root.dense(ti.i, n_sims).place(
        F, K, Da, Db,
        F_tgt, K_tgt, Da_tgt, Db_tgt,
        stress_mix, stress_tgt
    )


@ti.func
def lap(f: ti.template(), s: int, x: int, y: int) -> ti.f32:
    return 0.25 * (f[s, x+1, y] + f[s, x-1, y] + f[s, x, y+1] + f[s, x, y-1] - 4.0 * f[s, x, y])


@ti.kernel
def initialize_disks(n_sims: int, tile: int, seed_sz: int):
    for s, i, j in A:
        if i < tile and j < tile and s < n_sims:
            A[s, i, j] = 0.0
            B[s, i, j] = 0.0
    # fill A=1 inside borders
    for s in range(n_sims):
        for i, j in ti.ndrange((1, tile-1), (1, tile-1)):
            A[s, i, j] = 1.0
    # seed a gaussian blob of B at random offsets per sim
    for s in range(n_sims):
        cx = tile // 2 + (s * 97) % 31 - 15
        cy = tile // 2 + (s * 53) % 41 - 20
        for di, dj in ti.ndrange(seed_sz, seed_sz):
            x = cx + di - seed_sz//2
            y = cy + dj - seed_sz//2
            if 1 <= x < tile-1 and 1 <= y < tile-1:
                r2 = (di - seed_sz//2) * (di - seed_sz//2) + (dj - seed_sz//2) * (dj - seed_sz//2)
                B[s, x, y] = ti.exp(-0.1 * r2)


@ti.kernel
def simulate(n_sims: int, tile: int):
    for s in range(n_sims):
        for i, j in ti.ndrange((1, tile-1), (1, tile-1)):
            a = A[s, i, j]
            b = B[s, i, j]
            La = lap(A, s, i, j)
            Lb = lap(B, s, i, j)
            _dA[s, i, j] = Da[s] * La - a * b * b + F[s] * (1.0 - a)
            _dB[s, i, j] = Db[s] * Lb + a * b * b - (K[s] + F[s]) * b
    for s in range(n_sims):
        for i, j in ti.ndrange((1, tile-1), (1, tile-1)):
            A[s, i, j] += _dA[s, i, j] * DT
            B[s, i, j] += _dB[s, i, j] * DT


@ti.func
def smoothstep(x: ti.f32, e0: ti.f32, e1: ti.f32) -> ti.f32:
    t = clamp((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def lerp(a: ti.f32, b: ti.f32, t: ti.f32) -> ti.f32:
    return a + t * (b - a)

@ti.func
def lerp3(a: ti.types.vector(3, ti.f32), b: ti.types.vector(3, ti.f32), t: ti.f32):
    return a + t * (b - a)


@ti.kernel
def ease_all(n_sims: int, alpha: ti.f32):
    for s in range(n_sims):
        F[s]  += alpha * (F_tgt[s]  - F[s])
        K[s]  += alpha * (K_tgt[s]  - K[s])
        Da[s] += alpha * (Da_tgt[s] - Da[s])
        Db[s] += alpha * (Db_tgt[s] - Db[s])
        stress_mix[s] = (1.0 - alpha) * stress_mix[s] + alpha * stress_tgt[s]


@ti.kernel
def blit_render(n_sims: int, tile: int, rows: int, cols: int, out_img: ti.types.ndarray(dtype=ti.f32, ndim=3)):
    for s in range(n_sims):
        r = s // cols
        c = s % cols
        ox = r * tile
        oy = c * tile
        for i, j in ti.ndrange(tile, tile):
            t = B[s, i, j]
            # fake gradient for lighting
            gx = B[s, min(i + 1, tile - 1), j] - B[s, max(i - 1, 0), j]
            gy = B[s, i, min(j + 1, tile - 1)] - B[s, i, max(j - 1, 0)]
            norm = ti.Vector([-gx, -gy, 1.0]).normalized()
            light_dir  = ti.Vector([0.5, 1.0, 12.0]).normalized()
            illum      = clamp(light_dir.dot(norm), 0.0, 1.0)
            spec_illum = smoothstep(illum, 0.99, 0.999)

            s_mix = stress_mix[s]

            # EXTREME palette (warm)
            deep_red   = ti.Vector([0.35, 0.00, 0.05])
            hot_orange = ti.Vector([1.00, 0.35, 0.00])
            bright_yel = ti.Vector([1.00, 0.85, 0.20])
            th = ti.pow(t, 0.6)
            col_ext = (1.0 - th) * deep_red + th * hot_orange
            col_ext = (1.0 - th) * col_ext + th * bright_yel
            spec_ext = ti.Vector([1.0, 0.8, 0.3])
            amb_ext  = deep_red * 0.10

            # CALM palette (cool)
            col_calm  = ti.Vector([t, t, 0.8 - t])
            spec_calm = ti.Vector([1.0, 1.0, 1.0])
            amb_calm  = ti.Vector([0.0, 0.0, 0.0])

            col        = lerp3(col_calm,  col_ext,  s_mix)
            spec_color = lerp3(spec_calm, spec_ext, s_mix)
            ambient    = lerp3(amb_calm,  amb_ext,  s_mix)

            rgb = ambient + illum * col + spec_illum * spec_color * lerp(0.5, 0.3, s_mix)
            out_img[ox + i, oy + j, 0] = rgb[0]
            out_img[ox + i, oy + j, 1] = rgb[1]
            out_img[ox + i, oy + j, 2] = rgb[2]


# =========================
# Layout utilities
# =========================

def best_grid(n: int):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

# =========================
# Main
# =========================

def main(argv_files):
    eeg_files = [str(p) for p in argv_files if str(p).lower().endswith('.txt')]
    if len(eeg_files) == 0:
        print('Usage: python t_rd_colors_multi.py eeg1.txt [eeg2.txt ...]')
        sys.exit(1)

    N = len(eeg_files)
    tile = CFG.tile_res
    rows, cols = best_grid(N)

    rng = np.random.default_rng(CFG.rng_seed)

    # Allocate fields
    alloc_fields(N, tile)

    # Initialize seeds and default params/targets
    initialize_disks(N, tile, seed_sz=20)

    # defaults: start from calm values
    for s in range(N):
        F[s] = F_tgt[s] = CFG.calm[0]
        K[s] = K_tgt[s] = CFG.calm[1]
        Da[s] = Da_tgt[s] = CFG.calm[2]
        Db[s] = Db_tgt[s] = CFG.calm[3]
        stress_mix[s] = 0.0
        stress_tgt[s] = 0.0

    # Build feeders & classifiers per sim
    fs = 256.0
    feeders = []
    clfs = []
    for path in eeg_files:
        feeders.append(OfflineEEGFeeder([path], fs=fs, chunk=32, speed=1.0, loop=True, buffer_s=8.0))
        clfs.append(LiveArousalClassifier(fs=fs, lf=(4, 12), hf=(13, 40), win_s=4.0))

    # Window
    win_h, win_w = rows * tile, cols * tile
    window = ti.ui.Window('RD (EEG-driven) — tiled', (win_w, win_h))
    canvas = window.get_canvas()

    # Output buffer
    out = np.zeros((win_h, win_w, 3), dtype=np.float32)

    t0 = time.perf_counter()
    last = t0

    while window.running:
        # Time + easing alpha
        now = time.perf_counter()
        dt_wall = now - last
        last = now
        alpha = 1.0 - math.exp(-dt_wall / CFG.tau_secs)

        # Per-sim EEG update → targets
        for s in range(N):
            feeders[s].step_once()
            state, ratio, changed = clfs[s].update(feeders[s].get_buffer())

            # choose base params by state (copied mapping)
            if state == 'CALM':
                base_f, base_k, base_Da, base_Db = CFG.calm
                s_tgt = 0.0
            elif state == 'MOD-STRESS':
                base_f, base_k, base_Da, base_Db = CFG.mod
                s_tgt = 0.33
            elif state == 'HIGH-STRESS':
                base_f, base_k, base_Da, base_Db = CFG.high
                s_tgt = 0.66
            else:  # EXTREME-STRESS
                base_f, base_k, base_Da, base_Db = CFG.xtrm
                s_tgt = 1.0

            # LFO wobble when close to target
            close = (abs(float(F[s])  - base_f)  < 0.002 and
                     abs(float(K[s])  - base_k)  < 0.002 and
                     abs(float(Da[s]) - base_Da) < 0.02  and
                     abs(float(Db[s]) - base_Db) < 0.02)
            if close:
                t = now - t0
                base_f  += AMP_F  * math.sin(2 * math.pi * FREQ_F  * t)
                base_k  += AMP_K  * math.sin(2 * math.pi * FREQ_K  * t + 1.3)
                base_Da += AMP_DA * math.sin(2 * math.pi * FREQ_DA * t + 2.1)
                base_Db += AMP_DB * math.sin(2 * math.pi * FREQ_DB * t + 0.4)

            # clamp to safe ranges
            base_f  = float(np.clip(base_f,  0.002, 0.120))
            base_k  = float(np.clip(base_k,  0.014, 0.070))
            base_Da = float(np.clip(base_Da, 0.100, 2.000))
            base_Db = float(np.clip(base_Db, 0.100, 2.000))

            # set targets
            F_tgt[s], K_tgt[s], Da_tgt[s], Db_tgt[s] = base_f, base_k, base_Da, base_Db
            stress_tgt[s] = s_tgt

        # Apply easing on-device
        ease_all(N, alpha)

        # Simulate
        for _ in range(CFG.steps_per_frame):
            simulate(N, tile)

        # Render/blit
        blit_render(N, tile, rows, cols, out)
        canvas.set_image(out)
        window.show()


if __name__ == '__main__':
    main(sys.argv[1:])
