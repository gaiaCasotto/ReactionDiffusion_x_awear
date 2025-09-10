# t_rd_tiled5.py
# One window, five Gray–Scott RD simulations tiled.
# Usage:
#   python t_rd_tiled5.py --tile_n 384 --rows 2 --cols 3 --steps 1 --vsync 0
#
# Notes:
# - Defaults choose a 2x3 grid; only 5 tiles are used (last cell stays dark).
# - Per-tile colors and parameters are varied for visual diversity.
# - Press 'r' to reseed all tiles, 'p' to pause/resume.

import argparse
import random
import time
import taichi as ti


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile_n", type=int, default=384, help="Resolution of each tile (pixels per side).")
    p.add_argument("--rows", type=int, default=2, help="Grid rows (use 2 for five tiles).")
    p.add_argument("--cols", type=int, default=3, help="Grid cols (use 3 for five tiles).")
    p.add_argument("--steps", type=int, default=1, help="Simulation steps per frame (1-3 is typical).")
    p.add_argument("--vsync", type=int, default=0, help="1=enable vsync, 0=disable.")
    p.add_argument("--cpu", action="store_true", help="Force CPU backend (for testing).")
    return p.parse_args()


def main():
    args = parse_args()
    K = 5  # number of tiles/simulations

    if args.rows * args.cols < K:
        raise SystemExit(f"Grid too small ({args.rows}x{args.cols}) for {K} tiles. Increase rows/cols.")

    arch = ti.cpu if args.cpu else (ti.gpu if ti._lib.core.with_cuda() else ti.cpu)
    ti.init(arch=arch, debug=False, kernel_profiler=False)

    N = args.tile_n
    ROWS, COLS = args.rows, args.cols
    W, H = COLS * N, ROWS * N

    # --- Gray–Scott parameters ---
    DIFF_U = 0.16
    DIFF_V = 0.08

    # Per-tile feed/kill and colors
    feed = ti.field(dtype=ti.f32, shape=K)
    kill = ti.field(dtype=ti.f32, shape=K)
    color_a = ti.Vector.field(3, dtype=ti.f32, shape=K)
    color_b = ti.Vector.field(3, dtype=ti.f32, shape=K)

    # State fields: [k, i, j]
    u = ti.field(dtype=ti.f32, shape=(K, N, N))
    v = ti.field(dtype=ti.f32, shape=(K, N, N))
    du = ti.field(dtype=ti.f32, shape=(K, N, N))
    dv = ti.field(dtype=ti.f32, shape=(K, N, N))

    # Output image
    img = ti.Vector.field(3, dtype=ti.f32, shape=(H, W))

    @ti.kernel
    def init_fields():
        for k, i, j in u:
            u[k, i, j] = 1.0
            v[k, i, j] = 0.0

    @ti.func
    def lap(f, k, i, j):
        return (f[k, i, j] * -1.0 +
                0.2 * (f[k, i+1, j] + f[k, i-1, j] + f[k, i, j+1] + f[k, i, j-1]) +
                0.05 * (f[k, i+1, j+1] + f[k, i+1, j-1] + f[k, i-1, j+1] + f[k, i-1, j-1]))

    @ti.kernel
    def step_once():
        for k, i, j in u:
            lap_u = lap(u, k, i, j)
            lap_v = lap(v, k, i, j)
            uvv = u[k, i, j] * v[k, i, j] * v[k, i, j]
            du[k, i, j] = DIFF_U * lap_u - uvv + feed[k] * (1.0 - u[k, i, j])
            dv[k, i, j] = DIFF_V * lap_v + uvv - (kill[k] + feed[k]) * v[k, i, j]
        for k, i, j in u:
            u[k, i, j] += du[k, i, j]
            v[k, i, j] += dv[k, i, j]

    @ti.kernel
    def inject_circle(k: ti.i32, cx: ti.i32, cy: ti.i32, r: ti.i32, uval: ti.f32, vval: ti.f32):
        for i, j in ti.ndrange(N, N):
            dx = i - cx
            dy = j - cy
            if dx*dx + dy*dy <= r*r:
                u[k, i, j] = uval
                v[k, i, j] = vval

    @ti.kernel
    def render():
        for y, x in img:
            img[y, x] = ti.Vector([0.03, 0.04, 0.06])  # dark background

        for k in range(K):
            row = k // COLS
            col = k % COLS
            ox = col * N
            oy = row * N
            for i, j in ti.ndrange(N, N):
                val = ti.math.clamp(v[k, i, j] * 5.0, 0.0, 1.0)
                c1 = color_a[k] * val + (1.0 - val) * ti.Vector([0.03, 0.04, 0.06])
                c2 = color_b[k] * val + (1.0 - val) * ti.Vector([0.03, 0.04, 0.06])
                mix = 0.5
                img[oy + i, ox + j] = c1 * (1.0 - mix) + c2 * mix

    # --- Randomize per-tile parameters/colors ---
    rng = random.Random(1234)
    palette = [
        ((0.2, 0.5, 1.0), (1.0, 0.3, 0.6)),   # blue→magenta
        ((0.7, 1.0, 0.7), (0.2, 0.8, 0.6)),   # mint→teal
        ((1.0, 0.8, 0.4), (1.0, 0.4, 0.1)),   # gold→orange
        ((0.8, 0.6, 1.0), (0.3, 0.2, 0.9)),   # lilac→violet
        ((0.9, 0.9, 0.9), (0.3, 0.7, 1.0)),   # silver→sky
    ]
    feeds = [0.030, 0.034, 0.037, 0.040, 0.042]
    kills = [0.060, 0.061, 0.062, 0.063, 0.064]

    for k in range(K):
        a, b = palette[k % len(palette)]
        color_a[k] = a
        color_b[k] = b
        feed[k] = feeds[k % len(feeds)]
        kill[k] = kills[k % len(kills)]

    init_fields()

    for k in range(K):
        for _ in range(6):
            cx = rng.randrange(N//6, N - N//6)
            cy = rng.randrange(N//6, N - N//6)
            r = rng.randrange(6, 18)
            inject_circle(k, cx, cy, r, 0.50, 0.25)

    window = ti.ui.Window("RD Tiled x5 (single window)", (W, H), vsync=bool(args.vsync))
    canvas = window.get_canvas()

    paused = False

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'p':
                paused = not paused
            if window.event.key == 'r':
                init_fields()
                for k in range(K):
                    for _ in range(6):
                        cx = rng.randrange(N//6, N - N//6)
                        cy = rng.randrange(N//6, N - N//6)
                        r = rng.randrange(6, 18)
                        inject_circle(k, cx, cy, r, 0.50, 0.25)

        if not paused:
            for _ in range(max(1, args.steps)):
                step_once()

        render()
        canvas.set_image(img)
        window.show()


if __name__ == "__main__":
    main()

