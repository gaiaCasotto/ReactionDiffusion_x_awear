#!/usr/bin/env python3
"""
Real-time Gray–Scott with palettes + Calm/Agitated snap presets (with smooth morph).
Controls:
  Mouse:    click/drag to inject V
  Sliders:  F, k, Du, Dv, dt (still live!)
  Keys:
    Space = pause/resume
    R     = reset
    S     = save PNG
    C     = cycle colormap
    M     = cycle display mode (V → U → U-V)
    1     = SNAP to Calm preset (or morph if smoothing ON)
    2     = SNAP to Agitated preset (or morph if smoothing ON)
    X     = toggle smoothing ON/OFF (EMA toward target)
    Q/Esc = quit
Run: python rd_live_matplotlib_snap.py --n 256 --fps 30
"""
import argparse, time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# --- guard: ensure a GUI backend ---
if matplotlib.get_backend().lower() == "agg":
    raise RuntimeError(
        "Matplotlib backend is 'Agg' (no GUI). Install/enable a GUI backend (Qt/Tk)."
    )

CMAPS = ["magma","inferno","plasma","viridis","cividis","twilight","cubehelix","Greys"]
MODES  = ["V","U","U-V"]

# Presets (feel free to tweak)
CALM =    dict(F=0.024, k=0.060, Du=0.18, Dv=0.09, dt=0.80, steps_per_frame=1)
AGITATED= dict(F=0.046, k=0.060, Du=0.15, Dv=0.075, dt=1.10, steps_per_frame=3)

def laplacian(Z):
    return (-4*Z + np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))

def step(U,V,Du,Dv,F,k,dt):
    Lu, Lv = laplacian(U), laplacian(V)
    UVV = U*V*V
    U += (Du*Lu - UVV + F*(1-U))*dt
    V += (Dv*Lv + UVV - (F+k)*V)*dt
    np.clip(U,0,1,out=U); np.clip(V,0,1,out=V)
    return U,V

def init_state(n, seed_size=20, seed_V=0.25, noise=0.02):
    rng = np.random.default_rng(0)
    U = np.ones((n,n), np.float32); V = np.zeros((n,n), np.float32)
    if noise>0:
        U += (rng.random((n,n), np.float32)-0.5)*noise
        V += (rng.random((n,n), np.float32)-0.5)*noise
        np.clip(U,0,1,out=U); np.clip(V,0,1,out=V)
    c, s = n//2, seed_size
    V[c-s:c+s, c-s:c+s] = seed_V; U[c-s:c+s, c-s:c+s] = 1.0 - seed_V
    return U,V

def normalize(a):
    a = a.astype(np.float32); lo, hi = a.min(), a.max()
    return np.zeros_like(a) if hi-lo < 1e-8 else (a-lo)/(hi-lo)

def render_frame(U,V,mode):
    return normalize(V) if mode=="V" else normalize(U) if mode=="U" else normalize(U-V)

def apply_preset_to_sliders(preset, s_F, s_k, s_Du, s_Dv, s_dt):
    s_F.set_val(preset["F"])
    s_k.set_val(preset["k"])
    s_Du.set_val(preset["Du"])
    s_Dv.set_val(preset["Dv"])
    s_dt.set_val(preset["dt"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--brush", type=int, default=8)
    ap.add_argument("--cmap", type=str, default="magma", choices=CMAPS)
    ap.add_argument("--mode", type=str, default="V", choices=MODES)
    ap.add_argument("--micro_jitter", type=float, default=0.0, help="per-frame random jitter amplitude")
    args = ap.parse_args()

    n = args.n
    U,V = init_state(n, seed_size=n//12, noise=0.02)

    # Start in a neutral-ish regime (we’ll update from sliders anyway)
    Du,Dv,F,k,dt = 0.16,0.08,0.037,0.060,1.0
    steps_per_frame = 2

    paused = False
    brush_radius = args.brush
    mouse_down = False
    mouse_pos = (None,None)
    cmap_idx = CMAPS.index(args.cmap)
    mode_idx = MODES.index(args.mode)

    # Smoothing state (EMA)
    smoothing_on = True      # toggle with X
    ema_alpha    = 0.15      # smoothing strength per frame
    # Targets we steer toward when presets selected
    target = dict(F=F, k=k, Du=Du, Dv=Dv, dt=dt, steps_per_frame=steps_per_frame)
    current = dict(F=F, k=k, Du=Du, Dv=Dv, dt=dt, steps_per_frame=steps_per_frame)

    fig = plt.figure(figsize=(9.4,7))
    ax_img   = plt.axes([0.05, 0.25, 0.64, 0.7])
    ax_radio = plt.axes([0.72, 0.25, 0.22, 0.7])
    im = ax_img.imshow(render_frame(U,V,MODES[mode_idx]), cmap=CMAPS[cmap_idx], interpolation="nearest")
    ax_img.set_xticks([]); ax_img.set_yticks([])

    radio = RadioButtons(ax_radio, CMAPS, active=cmap_idx)
    def on_radio(label):
        nonlocal cmap_idx
        cmap_idx = CMAPS.index(label)
        im.set_cmap(CMAPS[cmap_idx]); fig.canvas.draw_idle()
    radio.on_clicked(on_radio)

    ax_F  = plt.axes([0.05, 0.18, 0.64, 0.03])
    ax_k  = plt.axes([0.05, 0.14, 0.64, 0.03])
    ax_Du = plt.axes([0.05, 0.10, 0.64, 0.03])
    ax_Dv = plt.axes([0.05, 0.06,  0.64, 0.03])
    ax_dt = plt.axes([0.05, 0.02,  0.64, 0.03])
    s_F  = Slider(ax_F, "F (feed)", 0.0, 0.08, valinit=F, valstep=0.001)
    s_k  = Slider(ax_k, "k (kill)", 0.0, 0.08, valinit=k, valstep=0.001)
    s_Du = Slider(ax_Du,"Du",       0.0, 0.40, valinit=Du, valstep=0.001)
    s_Dv = Slider(ax_Dv,"Dv",       0.0, 0.40, valinit=Dv, valstep=0.001)
    s_dt = Slider(ax_dt,"dt",       0.1, 1.5,  valinit=dt, valstep=0.01)

    def sync_current_from_sliders():
        current["F"]  = float(s_F.val)
        current["k"]  = float(s_k.val)
        current["Du"] = float(s_Du.val)
        current["Dv"] = float(s_Dv.val)
        current["dt"] = float(s_dt.val)

    def on_press(event):
        nonlocal paused, mouse_down, mouse_pos, mode_idx, steps_per_frame, smoothing_on
        if event.inaxes == ax_img and getattr(event, "button", None) == 1:
            mouse_down = True
            if event.xdata is not None and event.ydata is not None:
                mouse_pos = (int(event.xdata), int(event.ydata))

        if event.key == " ":
            paused = not paused
        elif event.key in ("r","R"):
            # Reset with mild noise (neutral)
            nonlocal U,V
            U,V = init_state(n, seed_size=n//12, noise=0.02)
        elif event.key in ("s","S"):
            from datetime import datetime
            path = f"grayscott_{MODES[mode_idx]}_{CMAPS[cmap_idx]}_{datetime.now().strftime('%H%M%S')}.png"
            plt.imsave(path, render_frame(U,V,MODES[mode_idx]), cmap=CMAPS[cmap_idx])
            print("Saved", path)
        elif event.key in ("q","escape"):
            plt.close("all")
        elif event.key in ("c","C"):
            cmap_idx = (cmap_idx + 1) % len(CMAPS); im.set_cmap(CMAPS[cmap_idx])
        elif event.key in ("m","M"):
            mode_idx = (mode_idx + 1) % len(MODES)
        elif event.key == "1":  # Calm
            target.update(CALM)
            if not smoothing_on:
                # snap instantly
                apply_preset_to_sliders(CALM, s_F,s_k,s_Du,s_Dv,s_dt)
                steps_per_frame = CALM["steps_per_frame"]
                sync_current_from_sliders()
            print("→ Target preset: CALM", CALM)
        elif event.key == "2":  # Agitated
            target.update(AGITATED)
            if not smoothing_on:
                apply_preset_to_sliders(AGITATED, s_F,s_k,s_Du,s_Dv,s_dt)
                steps_per_frame = AGITATED["steps_per_frame"]
                sync_current_from_sliders()
            print("→ Target preset: AGITATED", AGITATED)
        elif event.key in ("x","X"):
            smoothing_on = not smoothing_on
            print("Smoothing:", "ON (EMA)" if smoothing_on else "OFF (snap)")

    def on_release(event):
        nonlocal mouse_down
        mouse_down = False

    def on_motion(event):
        nonlocal mouse_pos
        if mouse_down and event.inaxes == ax_img and event.xdata is not None and event.ydata is not None:
            mouse_pos = (int(event.xdata), int(event.ydata))

    def paint_v(V,pos,r):
        if pos[0] is None: return
        x,y = pos; x0=max(0,x-r); x1=min(n,x+r+1); y0=max(0,y-r); y1=min(n,y+r+1)
        if x0>=x1 or y0>=y1: return
        yy,xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx-x)**2 + (yy-y)**2 <= r*r
        V[y0:y1, x0:x1][mask] = np.clip(V[y0:y1, x0:x1][mask] + 0.8, 0, 1)

    fig.canvas.mpl_connect("key_press_event", on_press)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)

    plt.show(block=False)

    target_dt = 1.0 / args.fps
    last = time.perf_counter()
    print("Running… (1: Calm, 2: Agitated, X: toggle smoothing, Space: pause)")

    while plt.fignum_exists(fig.number):
        now = time.perf_counter()
        if now - last < target_dt:
            plt.pause(0.001)
            continue
        last = now

        # Read live slider values as the base "current"
        sync_current_from_sliders()

        # Smooth toward target (EMA) if ON
        if smoothing_on:
            for key in ("F","k","Du","Dv","dt"):
                current[key] = (1-ema_alpha)*current[key] + ema_alpha*target[key]
            # smoothly approach steps_per_frame as integer
            spf = (1-ema_alpha)*current["steps_per_frame"] + ema_alpha*target["steps_per_frame"]
            steps_per_frame = int(round(spf))
            # Apply smoothed values back to sliders so UI reflects the morph
            s_F.set_val(current["F"]); s_k.set_val(current["k"])
            s_Du.set_val(current["Du"]); s_Dv.set_val(current["Dv"]); s_dt.set_val(current["dt"])

        # Update title/status
        mode = MODES[mode_idx]; cmap = CMAPS[cmap_idx]
        status = f"Gray–Scott — Mode: {mode} | Cmap: {cmap} | Steps/frame: {steps_per_frame} | Smooth: {'ON' if smoothing_on else 'OFF'}"
        ax_img.set_title(status)

        if not paused:
            if mouse_down: paint_v(V, mouse_pos, brush_radius)
            for _ in range(max(1, steps_per_frame)):
                U,V = step(U,V, current["Du"], current["Dv"], current["F"], current["k"], current["dt"])

            # Optional micro jitter to keep “nervous” feel if desired
            if args.micro_jitter > 0:
                amp = args.micro_jitter
                U += (np.random.random(U.shape).astype(np.float32)-0.5)*amp
                V += (np.random.random(V.shape).astype(np.float32)-0.5)*amp
                np.clip(U,0,1,out=U); np.clip(V,0,1,out=V)

            frame = render_frame(U,V,mode)
            im.set_data(frame); im.set_cmap(cmap)
            fig.canvas.draw_idle()

        plt.pause(0.001)

if __name__ == "__main__":
    main()