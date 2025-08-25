# eeg_classify_palette.py
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---------- Streaming from .txt ----------
def load_eeg_file(path: str) -> np.ndarray:
    with open(path, "r") as f:
        txt = f.read().replace(",", " ")
    arr = np.fromstring(txt, sep=" ")
    arr = arr[np.isfinite(arr)].astype(np.float32)
    if arr.size: arr -= np.mean(arr)
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

# ---------- PSD + HF/LF ----------
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

# ---------- Live classifier with hysteresis ----------
class LiveArousalClassifier:
    def __init__(self, fs=256.0, lf=(4.0,12.0), hf=(13.0,40.0),
                 thr=7.0, hyst=0.2, win_s=4.0):
        self.fs=fs; self.lf=lf; self.hf=hf
        self.thr_on = thr
        self.thr_off = thr*(1.0-hyst)
        self.win_s = win_s
        self.state="NOT STRESSED"
        self.last_ratio=np.nan

    def update(self, buffer):
        freqs, psd = compute_psd(buffer, self.fs, self.win_s)
        lf_p = bandpower(psd, freqs, *self.lf)
        hf_p = bandpower(psd, freqs, *self.hf)
        ratio = (hf_p + 1e-12)/(lf_p + 1e-12)
        prev = self.state
        if self.state=="NOT STRESSED" and ratio>=self.thr_on:
            self.state="STRESSED"
        elif self.state=="STRESSED" and ratio<=self.thr_off:
            self.state="NOT STRESSED"
        changed = (self.state!=prev)
        self.last_ratio=ratio
        return self.state, ratio, changed

# ---------- Palettes for RD ----------
def make_palette(state, high_contrast_gamma=1.2):
    """
    Returns (cmap, norm) to use with matplotlib.imshow(U, cmap=cmap, norm=norm)
    STRESSED: hotter, high-contrast; NOT STRESSED: cooler, softer contrast.
    """
    if state=="STRESSED":
        cmap = plt.get_cmap("inferno")
        # higher contrast via tighter normalization and light gamma
        norm = mcolors.PowerNorm(gamma=1.0/high_contrast_gamma, vmin=0.0, vmax=1.0, clip=True)
    else:
        cmap = plt.get_cmap("cividis")  # or "viridis"
        norm = mcolors.PowerNorm(gamma=high_contrast_gamma, vmin=0.0, vmax=1.0, clip=True)
    return cmap, norm
