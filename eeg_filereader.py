# offline_eeg_stream.py
import time
from collections import deque
import numpy as np

AWear_SAMPLE_RATE_HZ = 256.0  # <-- change if your files are a different fs

# ---------------- EEG utilities ----------------

def _load_eeg_file(path: str) -> np.ndarray:
    with open(path, "r") as f:
        txt = f.read().replace(",", " ")
    arr = np.fromstring(txt, sep=" ")
    arr = arr[np.isfinite(arr)].astype(np.float32)
    if arr.size > 0:
        arr -= np.mean(arr)
    return arr

class OfflineEEGFeeder:
    def __init__(self, paths, fs=256.0, chunk_size=32, speed=1.0, loop=True, buffer_seconds=6.0):
        self.paths = list(paths)
        self.fs = float(fs)
        self.chunk = int(chunk_size)
        self.speed = float(speed)
        self.loop = bool(loop)
        self.tracks = [_load_eeg_file(p) for p in self.paths]
        if not self.tracks:
            raise ValueError("No EEG files loaded.")
        self.i = 0
        self.idx = 0
        self.dt = self.chunk / (self.fs * self.speed)
        self.buffer = deque(maxlen=int(buffer_seconds * self.fs))

    @property
    def n_tracks(self): return len(self.tracks)

    def set_track(self, i):
        self.i = int(i) % self.n_tracks
        self.idx = 0
        self.buffer.clear()

    def step(self):
        x = self.tracks[self.i]
        if self.idx >= len(x):
            if self.loop:
                self.idx = 0
            else:
                return False
        j = min(self.idx + self.chunk, len(x))
        chunk = x[self.idx:j]
        self.idx = j
        self.buffer.extend(chunk.tolist())
        return True

    def get_buffer(self):
        if not self.buffer: return np.array([], dtype=np.float32)
        return np.array(self.buffer, dtype=np.float32)
