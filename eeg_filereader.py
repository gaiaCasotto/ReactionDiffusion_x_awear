"""
---- NOTES ON EEG ----- # to add on overleaf


EEG frequency bands
Delta: 0.5 – 4 Hz → deep sleep, very low arousal
Theta: 4 – 8 Hz → drowsiness, light sleep, relaxed attention
Alpha: 8 – 12 Hz → relaxed wakefulness, calm but alert
Beta: 13 – 30 Hz → active thinking, alertness, stress, anxiety
Gamma: > 30 Hz → high-level processing, hyperarousal, sometimes linked to stress


Mapping to LF and HF (common in stress research)
LF (Low Frequency)
    Delta + Theta (0.5 – 8 Hz)
    Sometimes includes Alpha (depending on the study)
    Interpreted as relaxation / baseline cognitive state
HF (High Frequency)
    Beta + Gamma (13 – 40+ Hz)
    Interpreted as arousal, stress, cognitive load


    
If HF power ↑ relative to LF → brain is in a high-arousal / stressed / anxious state
If LF power dominates → calmer, more relaxed baseline

"""




# offline_eeg_stream.py
import time
from collections import deque
import numpy as np

AWear_SAMPLE_RATE_HZ = 256.0  # <-- change if your files are a different fs

# ---------------- EEG utilities ----------------

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
    
    '''
    keep in mind:
    1) Ratios are unitless and scale-sensitive: If LF power is very small (close to 0), 
    the ratio can blow up because of division.
    That’s why i add 1e-12 in your code — to avoid divide-by-zero.
    '''
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
