# rd_taichi_eeg_classify.py
# Taichi Gray–Scott RD + live EEG HF/LF classifier to switch params and colors.

import time
from sys import argv
from collections import deque
import numpy as np
import taichi as ti

#my own file - reads brainwave files and classifies stressed or not
from eeg_filereader import OfflineEEGFeeder, LiveArousalClassifier

# =========================
# Taichi Reaction–Diffusion
# =========================
programs = [["Unstable mitosis and cell death", 0.057, 0.013,  1.57,  0.509],
    ["Holes emerging and disappearing",  0.055, 0.030, 1.08, 0.952],
    ["Growing disk", 0.056, 0.04, 1.05, 1.050],
    ["Flower folding", 0.056, 0.107, 1.50, 0.591],
    ["Tight Pattern", 0.056, 0.107, 1.90, 0.28],
    ["Dimples", 0.056, 0.1, 0.65, 0.215],
    ["Pattern of dots and lines", 0.066, 0.1, 0.65, 0.215],
    ["Colonies", 0.07, 0.011, 1.98, 0.231],
    ["Seekers", 0.064, 0.060, 1.691, 0.855],
    ["Colonisers", 0.064, 0.060, 1.691, 0.833]]

# --- put near your other Python globals ---
import time, numpy as np
t0 = time.perf_counter()

# tiny wobble amplitudes (tune to taste)
AMP_F, AMP_K  = 0.0010, 0.001
AMP_DA, AMP_DB = 0.015, 0.010
# slow frequencies in Hz (different so they don’t sync)
FREQ_F, FREQ_K, FREQ_DA, FREQ_DB = 0.03, 0.021, 0.017, 0.026

# helper to clamp
def clamp(x, lo, hi): return max(lo, min(hi, x))


@ti.func
def smoothstep(x, edge0, edge1):
    t = (x - edge0) / (edge1 - edge0)
    t = max(0, min(1, t))  # Clamp t to [0, 1]
    return t * t * (3 - 2 * t)

ti.init(arch=ti.gpu)

# Params (fields so kernels see updates immediately)
#Mitosis
dt = 0.5
Da = ti.field(dtype=float, shape=())
Db = ti.field(dtype=float, shape=())
k  = ti.field(dtype=float, shape=())
f  = ti.field(dtype=float, shape=())
steps = ti.field(dtype=int, shape=())
Da[None] = 1.0; 
Db[None] = 0.5; 
k[None] = 0.063; 
f[None] = 0.069;  
steps[None] = 100
#stressed = 0.025, 0.058, 1.050, 0.50


# Targets the sim will ease TOWARD
f_target  = ti.field(dtype=float, shape=())
k_target  = ti.field(dtype=float, shape=())
Da_target = ti.field(dtype=float, shape=())
Db_target = ti.field(dtype=float, shape=())

# Smoothed stress mix (0..1) for visuals: 0 = NOT STRESSED, 1 = STRESSED
stress_mix = ti.field(dtype=float, shape=())  # replaces hard 0/1 flips if you like
stress_mix[None] = 0.0

# Operating parameters
n = 800
TAU_SECS = 3.0 

# Fields
pixelsA = ti.field(dtype=float, shape=(n, n))
pixelsB = ti.field(dtype=float, shape=(n, n))
dA = ti.field(dtype=float, shape=(n, n))
dB = ti.field(dtype=float, shape=(n, n))
shading = ti.field(dtype=float, shape=(n, n))
#render_image = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))
stress_state = ti.field(dtype=ti.i32, shape=())  # 1=STRESSED, 0=NOT

@ti.func
def laplacian(i,j, pixels):
    return 0.25*(pixels[i+1,j] + pixels[i-1,j] + pixels[i,j+1] + pixels[i,j-1] - 4 * pixels[i,j])

@ti.kernel
def initialize():
    for i, j in ti.ndrange(n,n):
        pixelsA[i, j] = 0.0
        pixelsB[i, j] = 0.0
    for _i, _j in ti.ndrange(n-2, n-2):  # Iterate over the first two dimensions
        i, j = (_i+1, _j+1) # Offset by 1 to avoid boundary issues
        pixelsA[i, j] = 1.0
    for _i, _j in ti.ndrange(20, 20):
        i, j = (_i-10, _j-10)
        pixelsB[n//2+i, n//2+j] = ti.exp(-0.1*(i**2+j**2))

@ti.kernel
def simulate():
    for _i, _j in ti.ndrange(n-2, n-2):
        i, j = (_i+1, _j+1)
        # Implement the Gray-Scott model (Karl Sim's version)
        LapA = laplacian(i, j, pixelsA)
        LapB = laplacian(i, j, pixelsB)
        A = pixelsA[i, j]
        B = pixelsB[i, j]
        dA[i,j] = Da[None] * LapA - A * B**2 + f[None] * (1 - A)
        dB[i,j] = Db[None] * LapB + A * B**2 - (k[None] + f[None])*B
    for _i, _j in ti.ndrange(n-2, n-2):
        i, j = (_i+1, _j+1)
        pixelsA[i, j] += dA[i,j] * dt
        pixelsB[i, j] += dB[i,j] * dt


render_image = ti.Vector.field(3, dtype=ti.f32, shape=(n, n))


@ti.kernel
def render_with_shader_kernel(field: ti.template()):
    # Hot/high-contrast when stressed; cool/softer when not
    for i, j in render_image:
        t = field[i, j]
        gx = field[min(i + 1, n - 1), j] - field[max(i - 1, 0), j]
        gy = field[i, min(j + 1, n - 1)] - field[i, max(j - 1, 0)]
        norm = ti.Vector([-gx, -gy, 1.0]).normalized()
        light_dir = ti.Vector([0.5, 1.0, 12.0]).normalized()
        illum = max(0, light_dir.dot(norm))
        spec_illum = smoothstep(illum, 0.99, 0.999)
        white = ti.Vector([1.0, 1.0, 1.0])
        col = ti.Vector([0.4-t, 0.4-t, t])
        render_image[i, j] = illum*col + spec_illum*white*0.5
        '''if stress_state[None] == 1:
            # STRESSED: hotter palette, higher contrast
            base = ti.Vector([t, 0.18, 0.02])  # hot
            col = illum * base * 1.15 + spec_illum * white * 0.7
        else:
            # NOT STRESSED: cooler palette, softer contrast
            base = ti.Vector([0.20 - 0.10*t, 0.40 - 0.15*t, t])  # cool
            col = illum * base * 0.95 + spec_illum * white * 0.4

        # Clamp
        col = ti.max(ti.min(col, 1.0), 0.0)
        render_image[i, j] = col
        '''

        
#easing kernel. We’ll use an exponential approach with time-constant tau
@ti.kernel
def ease_params(alpha: float):
    # Exponential smoothing: x += alpha * (target - x)
    f[None]  += alpha * (f_target[None]  - f[None])
    k[None]  += alpha * (k_target[None]  - k[None])
    Da[None] += alpha * (Da_target[None] - Da[None])
    Db[None] += alpha * (Db_target[None] - Db[None])

@ti.kernel
def ease_stress(alpha: float, target: float):
    stress_mix[None] += alpha * (target - stress_mix[None])
    


def main():
    # ---------- EEG setup ----------
    EEG_FILES = [
        #"../eeg_files/1_horror_movie_data_filtered.txt",
        "../eeg_files/2_vipassana_data_filtered.txt",
        #"../eeg_files/3_hot_tub_data_filtered.txt",
        #"../eeg_files/fake_eeg_stress2calm.txt"
        #"../eeg_files/fake_eeg_longblocks.txt" #stressed first
        #"../eeg_files/fake_eeg_longblocks_calmfirst.txt"
    ]
    EEG_FS = 256.0
    try:
        feeder = OfflineEEGFeeder(EEG_FILES, fs=EEG_FS, chunk=32, speed=1.0, loop=True, buffer_s=8.0)
        clf = LiveArousalClassifier(fs=EEG_FS, lf=(4,12), hf=(13,40), thr=3.0, hyst=0.2, win_s=4.0)
        eeg_available = True #if the file exists, if the reader can read it
    except Exception as e:
        print("EEG feeder disabled:", e)
        eeg_available = False
        feeder = None; clf = None
    
    window = ti.ui.Window("Reaction Diffusion (EEG-driven colors & params)", (n, n))
    canvas = window.get_canvas()
    gui = window.get_gui()

    # Program init from CLI choice (optional)
    on_program = False
    initialize()

    # Live control flags
    current_track = 0
    outer_steps = 0
    last_time = time.perf_counter()
    f_target[None], k_target[None], Da_target[None], Db_target[None] = f[None], k[None], Da[None], Db[None]

    base_f, base_k, base_Da, base_Db = 0.052, 0.055, 0.14, 0.07
    

    while window.running:
        outer_steps += 1
        # ---- EEG update + choose params / color based on state ----
        state = "NOT STRESSED"
        ratio = float('nan')
        now = time.perf_counter()
        dt_wall = now - last_time
        last_time = now

        # Convert tau to a per-frame alpha: alpha = 1 - exp(-dt/tau)
        alpha = 1.0 - np.exp(-dt_wall / TAU_SECS)

        # apply easing
        ease_params(alpha)
        #ease_stress(alpha, stress_target)
        
        feeder.step_once()
        state, ratio, changed = clf.update(feeder.get_buffer())
        now = time.perf_counter()
        t = now - t0
        if state == "STRESSED":
            #f_target[None], k_target[None], Da_target[None], Db_target[None] = 0.025, 0.058, 1.050, 0.50
            base_f, base_k, base_Da, base_Db = 0.025, 0.058, 1.050, 0.50
            stress_state[None] = 1
            #stress_target = 1.0
        else:
            #f_target[None], k_target[None], Da_target[None], Db_target[None] = 0.069, 0.063, 0.80, 0.50
            base_f, base_k, base_Da, base_Db = 0.069, 0.063, 1.00, 0.50
            stress_state[None] = 0
            #stress_target = 0.0

            # only wobble when we're close to the calm target already
            close = (abs(f[None] - base_f)  < 0.002 and
                    abs(k[None] - base_k)  < 0.002 and
                    abs(Da[None]- base_Da) < 0.02  and
                    abs(Db[None]- base_Db) < 0.02)

            if close:
                # slow, tiny LFOs
                base_f  += AMP_F  * np.sin(2*np.pi*FREQ_F  * t)
                base_k  += AMP_K  * np.sin(2*np.pi*FREQ_K  * t + 1.3)
                base_Da += AMP_DA * np.sin(2*np.pi*FREQ_DA * t + 2.1)
                base_Db += AMP_DB * np.sin(2*np.pi*FREQ_DB * t + 0.4)
        
        # clamp to safe ranges you already expose on sliders
        base_f  = clamp(base_f,  0.002, 0.120)
        base_k  = clamp(base_k,  0.014, 0.070)
        base_Da = clamp(base_Da, 0.100, 2.000)
        base_Db = clamp(base_Db, 0.100, 2.000)

        # set TARGETS (let your existing easing move actual fields)
        f_target[None], k_target[None], Da_target[None], Db_target[None] = base_f, base_k, base_Da, base_Db
        '''
        if state == "STRESSED":
            stress_state[None] = 1
            f[None], k[None], Da[None], Db[None] = 0.025, 0.058, 1.050, 0.50 #NB chnage has to be GRADUAL!!! otherwise unstable
        else:
            stress_state[None] = 0
            f[None], k[None], Da[None], Db[None] = 0.069, 0.063, 0.80, 0.50
        '''

        # ---- GUI ----
        
        gui.text(f"File {current_track+1 if eeg_available else 0}")
        #if eeg_available:
        gui.text(f"HF/LF: {ratio:.3f}  |  State: {state}  (thr_on={clf.thr_on:.2f}, thr_off={clf.thr_off:.2f})")

        if not on_program:
            # Sliders
            gui.begin("Controls", 0, 0, 0.25, 0.18)
            k[None] = gui.slider_float("k",  k[None], 0.014, 0.07)
            f[None] = gui.slider_float("f",  f[None], 0.002, 0.12)
            Da[None]= gui.slider_float("Da", Da[None], 0.1,  2.0)
            Db[None]= gui.slider_float("Db", Db[None], 0.1,  2.0)
            steps[None]=gui.slider_int("Steps", steps[None], 1, 300)
            
            if gui.button("Reset"):
                initialize()
            gui.end()

        # ---- simulate ----
        for _ in range(steps[None]):
            simulate()

        # ---- render ----
        render_with_shader_kernel(pixelsB)
        canvas.set_image(render_image)
        window.show()

        #if eeg_available:
         #   feeder.sleep_dt()

if __name__ == "__main__":
    main()
