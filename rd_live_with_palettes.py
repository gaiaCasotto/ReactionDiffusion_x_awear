#!/usr/bin/env python3
"""
rd_live_matplotlib_palettes.py — Real-time Gray–Scott with changeable color schemes
Controls:
  • Click/drag: inject V (stir the system)
  • Sliders: F, k, Du, Dv, dt
  • SPACE: pause/resume   R: reset   S: save PNG   Q/Esc: quit
  • C: cycle color map    M: cycle display mode (V → U → U-V)
  • Radio buttons (right): pick a specific colormap

Requirements: numpy, matplotlib
Run: python rd_live_matplotlib_palettes.py --n 256
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import time

CMAPS = ["magma", "inferno", "plasma", "viridis", "cividis", "twilight", "cubehelix", "Greys"]
MODES = ["V", "U", "U-V"]

def laplacian(Z):
    return (-4*Z
            + np.roll(Z, 1, 0) + np.roll(Z, -1, 0)
            + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))

def step(U, V, Du, Dv, F, k, dt):
    Lu = laplacian(U)
    Lv = laplacian(V)
    UVV = U*V*V
    dU = Du*Lu - UVV + F*(1-U)
    dV = Dv*Lv + UVV - (F+k)*V
    U += dU*dt; V += dV*dt
    np.clip(U, 0.0, 1.0, out=U); np.clip(V, 0.0, 1.0, out=V)
    return U, V

def init_state(n, seed_size=20, seed_V=0.25, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    U = np.ones((n,n), dtype=np.float32)
    V = np.zeros((n,n), dtype=np.float32)
    U += (rng.random((n,n), dtype=np.float32)-0.5)*0.02
    V += (rng.random((n,n), dtype=np.float32)-0.5)*0.02
    np.clip(U,0,1,out=U); np.clip(V,0,1,out=V)
    c = n//2; s = seed_size
    V[c-s:c+s, c-s:c+s] = seed_V
    U[c-s:c+s, c-s:c+s] = 1.0 - seed_V
    return U, V

def normalize(arr):
    a = arr.astype(np.float32)
    amin, amax = np.min(a), np.max(a)
    if amax - amin < 1e-8:
        return np.zeros_like(a)
    return (a - amin) / (amax - amin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="grid size (n x n)")
    ap.add_argument("--fps", type=float, default=30, help="target frames per second")
    ap.add_argument("--brush", type=int, default=8, help="paint radius (pixels)")
    ap.add_argument("--cmap", type=str, default="magma", choices=CMAPS, help="initial colormap")
    ap.add_argument("--mode", type=str, default="V", choices=MODES, help="initial display mode")
    args = ap.parse_args()

    n = args.n
    U, V = init_state(n)

    Du, Dv = 0.16, 0.08
    F, k   = 0.037, 0.06
    dt     = 1.0

    paused = False
    brush_radius = args.brush
    mouse_down = False
    mouse_pos = (None, None)
    cmap_idx = CMAPS.index(args.cmap)
    mode_idx = MODES.index(args.mode)

    fig = plt.figure(figsize=(8.6,7))
    ax_img = plt.axes([0.05, 0.25, 0.72, 0.7])
    ax_radio = plt.axes([0.80, 0.25, 0.15, 0.7])
    im = ax_img.imshow(V, interpolation="nearest", cmap=CMAPS[cmap_idx])
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"Gray–Scott — Mode: {MODES[mode_idx]}  |  Cmap: {CMAPS[cmap_idx]}")

    radio = RadioButtons(ax_radio, CMAPS, active=cmap_idx)
    def on_radio(label):
        nonlocal cmap_idx
        cmap_idx = CMAPS.index(label)
        im.set_cmap(CMAPS[cmap_idx])
        fig.canvas.draw_idle()
    radio.on_clicked(on_radio)

    ax_F  = plt.axes([0.05, 0.18, 0.72, 0.03])
    ax_k  = plt.axes([0.05, 0.14, 0.72, 0.03])
    ax_Du = plt.axes([0.05, 0.10, 0.72, 0.03])
    ax_Dv = plt.axes([0.05, 0.06, 0.72, 0.03])
    ax_dt = plt.axes([0.05, 0.02, 0.72, 0.03])

    s_F  = Slider(ax_F,  "F (feed)",  0.0, 0.08, valinit=F, valstep=0.001)
    s_k  = Slider(ax_k,  "k (kill)",  0.0, 0.08, valinit=k, valstep=0.001)
    s_Du = Slider(ax_Du, "Du",        0.0, 0.40, valinit=Du, valstep=0.001)
    s_Dv = Slider(ax_Dv, "Dv",        0.0, 0.40, valinit=Dv, valstep=0.001)
    s_dt = Slider(ax_dt, "dt",        0.1, 1.5,  valinit=dt, valstep=0.01)

    def on_press(event):
        nonlocal paused, mouse_down, mouse_pos, U, V, mode_idx, cmap_idx
        if event.inaxes == ax_img and event.button == 1:
            mouse_down = True
            if event.xdata is not None and event.ydata is not None:
                mouse_pos = (int(event.xdata), int(event.ydata))
        if event.key == " ":
            paused = not paused
        elif event.key in ("r", "R"):
            U, V = init_state(n, seed_size=n//12)
        elif event.key in ("s", "S"):
            from datetime import datetime
            path = f"grayscott_{MODES[mode_idx]}_{CMAPS[cmap_idx]}_{datetime.now().strftime('%H%M%S')}.png"
            plt.imsave(path, render_frame(U, V, MODES[mode_idx]), cmap=CMAPS[cmap_idx])
            print("Saved", path)
        elif event.key in ("q", "escape"):
            plt.close("all")
        elif event.key in ("c", "C"):
            cmap_idx = (cmap_idx + 1) % len(CMAPS)
            im.set_cmap(CMAPS[cmap_idx])
        elif event.key in ("m", "M"):
            mode_idx = (mode_idx + 1) % len(MODES)

    def on_release(event):
        nonlocal mouse_down
        mouse_down = False

    def on_motion(event):
        nonlocal mouse_pos
        if mouse_down and event.inaxes == ax_img and event.xdata is not None and event.ydata is not None:
            mouse_pos = (int(event.xdata), int(event.ydata))

    def paint_v(V, pos, r_pix):
        if pos[0] is None: return
        x, y = pos
        x0 = max(0, x-r_pix); x1 = min(n, x+r_pix+1)
        y0 = max(0, y-r_pix); y1 = min(n, y+r_pix+1)
        if x0>=x1 or y0>=y1: return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x)**2 + (yy - y)**2 <= r_pix*r_pix
        V[y0:y1, x0:x1][mask] = np.clip(V[y0:y1, x0:x1][mask] + 0.8, 0.0, 1.0)

    def render_frame(U, V, mode):
            if mode == "V":
                return normalize(V)
            elif mode == "U":
                return normalize(U)
            else:  # "U-V"
                return normalize(U - V)

    cid_press   = fig.canvas.mpl_connect("key_press_event", on_press)
    cid_button  = fig.canvas.mpl_connect("button_press_event", on_press)
    cid_release = fig.canvas.mpl_connect("button_release_event", on_release)
    cid_motion  = fig.canvas.mpl_connect("motion_notify_event", on_motion)

    target_dt = 1.0 / args.fps
    last = time.perf_counter()

    try:
        while plt.fignum_exists(fig.number):
            now = time.perf_counter()
            if now - last < target_dt:
                plt.pause(0.001)
                continue
            last = now

            F  = float(s_F.val);  k  = float(s_k.val)
            Du = float(s_Du.val); Dv = float(s_Dv.val)
            dt = float(s_dt.val)

            ax_img.set_title(f"Gray–Scott — Mode: {MODES[mode_idx]}  |  Cmap: {CMAPS[cmap_idx]}")

            if not paused:
                if mouse_down:
                    paint_v(V, mouse_pos, brush_radius)
                for _ in range(2):
                    U, V = step(U, V, Du, Dv, F, k, dt)

                frame = render_frame(U, V, MODES[mode_idx])
                im.set_data(frame)
                im.set_cmap(CMAPS[cmap_idx])
                fig.canvas.draw_idle()

            plt.pause(0.001)
    finally:
        fig.canvas.mpl_disconnect(cid_press)
        fig.canvas.mpl_disconnect(cid_button)
        fig.canvas.mpl_disconnect(cid_release)
        fig.canvas.mpl_disconnect(cid_motion)

if __name__ == "__main__":
    main()