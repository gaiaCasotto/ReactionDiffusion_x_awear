
"""
rd_live_matplotlib.py — Real-time Gray–Scott Reaction–Diffusion (NumPy + Matplotlib)
Controls:
  • Click/drag on the image to inject reagent V (stirs the system).
  • Sliders: to adjust F, k, Du, Dv, dt in real-time.
  • SPACE = Pause/Resume,  R = Reset,  S = Save PNG,  Q/ESC = Quit.

Requirements: numpy, matplotlib
Run: python rd_live_classify_colors.py --n 256 
"""

"""
Gray scott model simulates the way 2 chemicals U and V react with each other.
dU, dV = diffusion rates (spread speed of U and V)
F = feed rate (of u)
k = kill rate (how much V decays)
Different values of F and k give different organic patterns: spots, stripes, waves.
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time

from eeg_classify_palette import (
    OfflineEEGFeeder, LiveArousalClassifier, make_palette
)


def laplacian(Z): #(DIFFUSION fucntion)
    return (-4*Z
            + np.roll(Z, 1, 0) + np.roll(Z, -1, 0)
            + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))

def step(U, V, Du, Dv, F, k, dt):
    Lu = laplacian(U)
    Lv = laplacian(V)
    UVV = U*V*V  #term that turns U into V when enough V is nearby (REACTION)
    dU = Du*Lu - UVV + F*(1-U)
    dV = Dv*Lv + UVV - (F+k)*V
    U += dU*dt; V += dV*dt
    np.clip(U, 0.0, 1.0, out=U); np.clip(V, 0.0, 1.0, out=V)
    return U, V


"""
The system starts with “all food” (U) and one small patch of “predator” (V).
"""
def init_state(n, seed_size=20, seed_V=0.25, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    U = np.ones((n,n), dtype=np.float32) #start with U everywhere
    V = np.zeros((n,n), dtype=np.float32) # no V
    #then add a seed of V in the middle
    U += (rng.random((n,n), dtype=np.float32)-0.5)*0.05 #use 0.05 for agitated, 0.02 for calm
    V += (rng.random((n,n), dtype=np.float32)-0.5)*0.05
    np.clip(U,0,1,out=U); np.clip(V,0,1,out=V)
    #NORMAL SETUP:
    '''
    c = n//2; s = seed_size
    V[c-s:c+s, c-s:c+s] = seed_V
    U[c-s:c+s, c-s:c+s] = 1.0 - seed_V
    '''
    #FOR CALM: make tiny seed
    c , s = n//2 , n//16
    V[c-s:c+s, c-s:c+s] = 0.20
    U[c-s:c+s, c-s:c+s] = 0.80
    #FOR AGITATED: start from a noisier initial state

    return U, V

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="grid size (n x n)")
    ap.add_argument("--fps", type=float, default=30, help="target frames per second")
    ap.add_argument("--brush", type=int, default=8, help="paint radius (pixels)")
    args = ap.parse_args()

    n = args.n
    U, V = init_state(n)

    # Parameters (start in a nice patterning regime)
    '''
    Du, Dv = 0.16, 0.08
    F, k   = 0.037, 0.06
    dt     = 1.0
    '''

    #calm DEFAULTS
    
    Du, Dv = 0.18, 0.09
    F, k   = 0.024, 0.060
    dt     = 0.8
    

    #Stressed, agitated DEFAULTS 
    #(+ speed up evolution in main looop)
    #(+ add some jitter top each frame to keep it nervous)
    Du, Dv = 0.15, 0.075
    F,  k  = 0.046, 0.060
    dt     = 1.1

    paused = False
    brush_radius = args.brush
    mouse_down = False
    mouse_pos = (None, None)

    # --- Matplotlib figure & sliders ---
    plt.figure(figsize=(6,7))
    ax_img = plt.axes([0.05, 0.25, 0.9, 0.7])
    im = ax_img.imshow(V, interpolation="nearest")
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title("Gray–Scott Reaction–Diffusion (V field)")

    # Slider axes
    ax_F  = plt.axes([0.10, 0.18, 0.8, 0.03])
    ax_k  = plt.axes([0.10, 0.14, 0.8, 0.03])
    ax_Du = plt.axes([0.10, 0.10, 0.8, 0.03])
    ax_Dv = plt.axes([0.10, 0.06, 0.8, 0.03])
    ax_dt = plt.axes([0.10, 0.02, 0.8, 0.03])

    s_F  = Slider(ax_F,  "F (feed)",  0.0, 0.08, valinit=F, valstep=0.001)
    s_k  = Slider(ax_k,  "k (kill)",  0.0, 0.08, valinit=k, valstep=0.001)
    s_Du = Slider(ax_Du, "Du",        0.0, 0.40, valinit=Du, valstep=0.001)
    s_Dv = Slider(ax_Dv, "Dv",        0.0, 0.40, valinit=Dv, valstep=0.001)
    s_dt = Slider(ax_dt, "dt",        0.1, 1.5,  valinit=dt, valstep=0.01)


    # --- EEG input (point to your files) ---
    EEG_FILES = [
        "../eeg_files/1_horror_movie_data_filtered.txt",
        "../eeg_files/2_vipassana_data_filtered.txt",
        "../eeg_files/3_hot_tub_data_filtered.txt",
    ]
    EEG_FS = 256.0      # change if different
    feeder = OfflineEEGFeeder(EEG_FILES, fs=EEG_FS, chunk=32, speed=1.0, loop=True, buffer_s=8.0)

    # live HF/LF classifier (threshold ~3.0 based on your data; 20% hysteresis)
    clf = LiveArousalClassifier(fs=EEG_FS, lf=(4,12), hf=(13,40), thr=7.0, hyst=0.2, win_s=4.0)

    # initial palette based on current state
    state, ratio, changed = clf.update(feeder.get_buffer())
    cmap, norm = make_palette(state)

    # your imshow for U — NOTE: fix vmin/vmax via norm to avoid overflow warnings
    im = ax_img.imshow(U, cmap=cmap, norm=norm, interpolation="nearest")
    ax_img.set_title(f"RD — state: {state}  (HF/LF={ratio:.2f})")


    # --- Event handlers ---
    def on_press(event):
        nonlocal paused, mouse_down, mouse_pos, U, V
        if event.inaxes == ax_img:
            mouse_down = True
            mouse_pos = (int(event.xdata), int(event.ydata))
        if event.key == " ":
            paused = not paused
            ax_img.set_title(("Paused — " if paused else "") + "Gray–Scott Reaction–Diffusion (V field)")
            plt.draw()
        elif event.key in ("r", "R"):
            U, V = init_state(n, seed_size=n//12)
        elif event.key in ("s", "S"):
            plt.imsave("grayscott_snapshot.png", V)
            print("Saved grayscott_snapshot.png")
        elif event.key in ("q", "escape"):
            plt.close("all")

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

    cid_press   = im.figure.canvas.mpl_connect("key_press_event", on_press)
    cid_button  = im.figure.canvas.mpl_connect("button_press_event", on_press)
    cid_release = im.figure.canvas.mpl_connect("button_release_event", on_release)
    cid_motion  = im.figure.canvas.mpl_connect("motion_notify_event", on_motion)

    # --- Main loop ---
    target_dt = 1.0 / args.fps
    last = time.perf_counter()

    try:
        while plt.fignum_exists(im.figure.number):
            now = time.perf_counter()
            if now - last < target_dt:
                plt.pause(0.001)
                continue
            last = now

            '''
            # Update params from sliders
            F  = float(s_F.val);  k  = float(s_k.val)
            Du = float(s_Du.val); Dv = float(s_Dv.val)
            dt = float(s_dt.val)
            '''
            feeder.step_once()
            state, ratio, changed = clf.update(feeder.get_buffer())

            # choose RD params based on stress state
            if state == "STRESSED":
                F, k, Du, Dv = 0.052, 0.055, 0.14, 0.07
            else:  # NOT STRESSED
                F, k, Du, Dv = 0.024, 0.060, 0.18, 0.09

            s_F.set_val(F)
            s_k.set_val(k)
            s_Du.set_val(Du)
            s_Dv.set_val(Dv)



            if not paused:
                # Paint V if mouse down
                if mouse_down:
                    paint_v(V, mouse_pos, brush_radius)
                # Do several solver steps per frame for faster evolution
                for _ in range(3): # (1 or 2 for calm) , (3 (4?) for agitated)
                    U, V = step(U, V, Du, Dv, F, k, dt)
                    #------- REMOVE FOR CALM ------- #
                    noise_amp = 0.002
                    U += (np.random.random(U.shape).astype(np.float32) - 0.5) * noise_amp
                    V += (np.random.random(V.shape).astype(np.float32) - 0.5) * noise_amp
                    np.clip(U, 0, 1, out=U); np.clip(V, 0, 1, out=V)

                    if changed:
                        cmap, norm = make_palette(state)
                        im.set_cmap(cmap)
                        im.set_norm(norm)

                im.set_data(V)
                ax_img.set_title(f"RD — {state}   HF/LF={ratio:.2f}   (thr_on={clf.thr_on:.2f}, thr_off={clf.thr_off:.2f})")

                im.figure.canvas.draw_idle()

            plt.pause(0.001)
            feeder.sleep_dt()
    finally:
        im.figure.canvas.mpl_disconnect(cid_press)
        im.figure.canvas.mpl_disconnect(cid_button)
        im.figure.canvas.mpl_disconnect(cid_release)
        im.figure.canvas.mpl_disconnect(cid_motion)

if __name__ == "__main__":
    main()
