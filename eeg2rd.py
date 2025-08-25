import numpy as np

class EEGtoRDMapper:
    def __init__(self, fs_hz, win_seconds=4.0, ema_alpha=0.2,
                 lf=(1.0,8.0), hf=(13.0,40.0), clamp=(0.2,3.0),
                 F_relaxed=0.060, F_aroused=0.018,
                 k_relaxed=0.045, k_aroused=0.070):
        self.fs = float(fs_hz); self.N = int(win_seconds * self.fs)
        self.alpha = float(ema_alpha)
        self.lf = lf; self.hf = hf; self.clamp = clamp
        self.Fr, self.Fa = float(F_relaxed), float(F_aroused)
        self.kr, self.ka = float(k_relaxed), float(k_aroused)
        self._ema = None

    @staticmethod
    def _bandpower(psd, freqs, fmin, fmax):
        idx = (freqs >= fmin) & (freqs < fmax)
        if not np.any(idx): return 0.0
        return np.trapz(psd[idx], freqs[idx])

    def _normalize(self, r):
        lo, hi = self.clamp
        r = np.clip(r, lo, hi)
        t = (np.log(r) - np.log(lo)) / (np.log(hi) - np.log(lo))
        t = float(np.clip(t, 0.0, 1.0))
        if self._ema is None: self._ema = t
        else: self._ema = (1-self.alpha)*self._ema + self.alpha*t
        return self._ema

    def update(self, eeg_buffer: np.ndarray):
        x = eeg_buffer[-self.N:] if len(eeg_buffer) >= self.N else eeg_buffer
        if x.size < max(8, int(0.5*self.N)):
            t = 0.5
            return dict(F=self.Fr + (self.Fa-self.Fr)*t,
                        k=self.kr + (self.ka-self.kr)*t,
                        ratio=None, t=t)
        w = np.hanning(len(x))
        X = np.fft.rfft(x*w)
        psd = (np.abs(X)**2) / (self.fs*np.sum(w**2))
        freqs = np.fft.rfftfreq(len(x), 1.0/self.fs)
        lf = self._bandpower(psd, freqs, *self.lf)
        hf = self._bandpower(psd, freqs, *self.hf)
        ratio = (hf+1e-12)/(lf+1e-12)
        t = self._normalize(ratio)
        F = self.Fr + (self.Fa-self.Fr)*t
        k = self.kr + (self.ka-self.kr)*t
        return dict(F=float(F), k=float(k), ratio=float(ratio), t=float(t))
