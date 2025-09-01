# rd_taichi_eeg_classify.py
# Taichi Gray–Scott RD + live EEG HF/LF classifier to switch params and colors.


'''
adding watchdogs so that the patterns never die

'''

import time
from sys import argv
from collections import deque
import numpy as np
import taichi as ti

# =========================
# EEG streaming + classifier
# =========================
def load_eeg_file(path: str) -> np.ndarray:
    with open(path, "r") as f:
        txt = f.read().replace(",", " ")
    arr = np.fromstring(txt, sep=" ")
    arr = arr[np.isfinite(arr)].astype(np.float32)
    if arr.size:
        arr -= np.mean(arr)
    return arr

class OfflineEEGFeeder:
    def __init__(self, paths, fs=256.0, chunk=32, speed=1.0, loop=True, buffer_s=8.0):
        self.fs=float(fs); self.chunk=int(chunk); self.speed=float(speed)
        self.loop=bool(loop); self.paths=list(paths)
        self.tracks=[load_eeg_file(p) for p in paths]
        if not self.tracks: raise ValueError("No EEG files loaded.")
        self.i=0; self.idx=0; self.dt=self.chunk/(self.fs*self.speed)
        self.buf=deque(maxlen=int(buffer_s*self.fs))
    @property
    def n_tracks(self): return len(self.tracks)
    def set_track(self, i): self.i=int(i)%len(self.tracks); self.idx=0; self.buf.clear()
    def step_once(self):
        x=self.tracks[self.i]
        if self.idx>=len(x):
            if self.loop: self.idx=0
            else: return False
        j=min(self.idx+self.chunk, len(x))
        self.buf.extend(x[self.idx:j].tolist()); self.idx=j; return True
    def get_buffer(self):
        return np.array(self.buf, dtype=np.float32) if self.buf else np.array([], np.float32)
    def sleep_dt(self): time.sleep(self.dt)

def compute_psd(buffer: np.ndarray, fs: float, win_s: float):
    if buffer.size==0: return np.array([0.0]), np.array([0.0])
    N=int(win_s*fs); x=buffer[-N:] if buffer.size>=N else buffer
    if x.size < max(16, int(0.25*N)):
        freqs=np.fft.rfftfreq(max(x.size,2), 1.0/fs)
        return freqs, np.zeros_like(freqs, np.float32)
    w=np.hanning(len(x)); X=np.fft.rfft(x*w)
    psd=(np.abs(X)**2)/(fs*np.sum(w**2))
    freqs=np.fft.rfftfreq(len(x), 1.0/fs)
    return freqs, psd.astype(np.float32)

def bandpower(psd, freqs, fmin, fmax):
    idx=(freqs>=fmin)&(freqs<fmax)
    if not np.any(idx): return 0.0
    return float(np.trapz(psd[idx], freqs[idx]))

class LiveArousalClassifier:
    def __init__(self, fs=256.0, lf=(4.0,12.0), hf=(13.0,40.0),
                 thr=3.0, hyst=0.2, win_s=4.0):
        self.fs=fs; self.lf=lf; self.hf=hf
        self.thr_on = float(thr)
        self.thr_off = float(thr)*(1.0-float(hyst))
        self.win_s = float(win_s)
        self.state="NOT STRESSED"
        self.last_ratio=np.nan
    def update(self, buffer):
        freqs, psd = compute_psd(buffer, self.fs, self.win_s)
        lf_p = bandpower(psd, freqs, *self.lf)
        hf_p = bandpower(psd, freqs, *self.hf)
        ratio = (hf_p + 1e-12)/(lf_p + 1e-12) # > 3 é stressato
        prev = self.state
        if self.state=="NOT STRESSED" and ratio>=self.thr_on:
            self.state="STRESSED"
        elif self.state=="STRESSED" and ratio<=self.thr_off:
            self.state="NOT STRESSED"
        self.last_ratio=ratio
        return self.state, ratio, (self.state!=prev)

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


##NEW##
# --- Watchdog stats ---
b_min   = ti.field(dtype=float, shape=())
b_max   = ti.field(dtype=float, shape=())
sumB    = ti.field(dtype=float, shape=())
sumsqB  = ti.field(dtype=float, shape=())
gradSum = ti.field(dtype=float, shape=())   # mean |∇B| proxy


@ti.kernel
def reset_stats():
    b_min[None] = 1e9
    b_max[None] = -1e9
    sumB[None] = 0.0
    sumsqB[None] = 0.0
    gradSum[None] = 0.0

@ti.func
def safe(idx, nmax):
    return max(0, min(nmax-1, idx))

@ti.kernel
def accumulate_stats():
    for i, j in pixelsB:
        v = pixelsB[i, j]
        ti.atomic_min(b_min[None], v)
        ti.atomic_max(b_max[None], v)
        ti.atomic_add(sumB[None], v)
        ti.atomic_add(sumsqB[None], v * v)

        # gradient magnitude (4-neighborhood)
        gx = pixelsB[safe(i+1, n), j] - pixelsB[safe(i-1, n), j]
        gy = pixelsB[i, safe(j+1, n)] - pixelsB[i, safe(j-1, n)]
        g = ti.abs(gx) + ti.abs(gy)
        ti.atomic_add(gradSum[None], g)

@ti.kernel
def reseed_noise(amplitude: float, probability: float):
    """Inject sparse noise so patterns revive but don’t explode."""
    for i, j in pixelsB:
        # Bernoulli-ish sparse mask
        if ti.random(float) < probability:
            pixelsB[i, j] += (ti.random(float) - 0.5) * amplitude



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
        #"../eeg_files/2_vipassana_data_filtered.txt",
        #"../eeg_files/3_hot_tub_data_filtered.txt",
        #"../eeg_files/fake_eeg_stress2calm.txt"
        "../eeg_files/fake_eeg_longblocks.txt"
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
    eeg_mode = True  # toggle with 'E'
    current_track = 0
    outer_steps = 0
    last_time = time.perf_counter()
    f_target[None], k_target[None], Da_target[None], Db_target[None] = f[None], k[None], Da[None], Db[None]


    dead_frames = 0
    PATIENCE = 30          # number of frames that must look "dead" before reseeding
    RANGE_THR = 0.02       # if (max-min) below this, it's too flat
    STD_THR   = 1e-3       # low variance means uniform field
    GRAD_THR  = 1e-3       # weak edges/texture



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

        if state == "STRESSED":
            f_target[None], k_target[None], Da_target[None], Db_target[None] = 0.025, 0.058, 1.050, 0.50
            stress_state[None] = 1
            #stress_target = 1.0
        else:
            f_target[None], k_target[None], Da_target[None], Db_target[None] = 0.069, 0.063, 0.80, 0.50
            stress_state[None] = 0
            #stress_target = 0.0
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

            # ---- evaluate reseed ----
        reset_stats()
        accumulate_stats()

        # bring scalars to Python
        bmin = b_min[None]
        bmax = b_max[None]
        rng  = bmax - bmin

        NN   = float(n * n)
        mean = sumB[None] / NN
        var  = max(0.0, sumsqB[None] / NN - mean * mean)
        std  = var ** 0.5
        gmean = gradSum[None] / NN

        # decide if "about to die"
        is_flat = (rng < RANGE_THR) or (std < STD_THR) or (gmean < GRAD_THR)
        dead_frames = dead_frames + 1 if is_flat else max(0, dead_frames - 1)

        if dead_frames >= PATIENCE:
            # gentle reseed: very sparse, very small noise
            reseed_noise(amplitude=0.02, probability=0.002)
            dead_frames = 0     # reset the counter

        # ---- render ----
        render_with_shader_kernel(pixelsB)
        canvas.set_image(render_image)
        window.show()

       

if __name__ == "__main__":
    main()
