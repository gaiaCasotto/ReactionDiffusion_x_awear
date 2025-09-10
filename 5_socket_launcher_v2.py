# five_socket_runner_v2.py
# Five sockets + robust launcher for t_rd_colors.py (with logs & optional autostart).

import socket
import threading
import subprocess
import sys
import time
import os
from pathlib import Path

HOST = "127.0.0.1"
BASE_PORT = 9601
NUM_SOCKETS = 4
BACKLOG = 1

# --- Configure this if your script is elsewhere ---
PROGRAM_PATH = Path("5_t_rd.py")  # <-- use your actual script here

# We'll build PROGRAM_ARGS dynamically per idx to include the EEG file:
BASE_PROGRAM_ARGS = ["--user", "{idx}"]  # will append ["--eeg", "<file>"] if provided

# Launch behavior
AUTOSTART = True

PROGRAM_CWD = None
PROGRAM_ENV = dict(os.environ)

# Grab EEG files from launcher argv (positional)
EEG_FILES_FROM_CLI = sys.argv[1:]  # e.g. ["../eeg_files/a.txt", "../eeg_files/b.txt", ...]

class SocketRunner(threading.Thread):
    def __init__(self, idx: int):
        super().__init__(daemon=True)
        self.idx = idx
        self.port = BASE_PORT + idx - 1
        self.proc = None
        self.launched = False
        self._stop = threading.Event()

    def launch(self):
        if self.launched:
            return

        if not PROGRAM_PATH.exists():
            print(f"[#{self.idx}] ERROR: Program not found at {PROGRAM_PATH.resolve()}")
            return

        # Build per-instance args
        args = [sys.executable, str(PROGRAM_PATH)]
        # always include base args
        args += [a.format(idx=self.idx) for a in BASE_PROGRAM_ARGS]

        # If we have an EEG file for this idx, append it
        eeg_for_idx = None
        if 1 <= self.idx <= len(EEG_FILES_FROM_CLI):
            eeg_for_idx = EEG_FILES_FROM_CLI[self.idx - 1]
        if eeg_for_idx:
            args += ["--eeg", eeg_for_idx]

        # Logs
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        stdout_path = log_dir / f"user{self.idx}_stdout.log"
        stderr_path = log_dir / f"user{self.idx}_stderr.log"
        stdout_f = open(stdout_path, "w")
        stderr_f = open(stderr_path, "w")

        try:
            self.proc = subprocess.Popen(
                args,
                cwd=PROGRAM_CWD,
                env=PROGRAM_ENV,
                stdout=stdout_f,
                stderr=stderr_f
            )
            print(f"[#{self.idx}] Launched: {' '.join(args)} (pid={self.proc.pid})")
            print(f"[#{self.idx}] Logs -> {stdout_path} / {stderr_path}")
            self.launched = True
        except Exception as e:
            print(f"[#{self.idx}] ERROR launching program: {e}")

    def run(self):
        if AUTOSTART:
            self.launch()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, self.port))
            s.listen(BACKLOG)
            print(f"[#{self.idx}] Listening on {HOST}:{self.port}")
            while not self._stop.is_set():
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                with conn:
                    print(f"[#{self.idx}] Client connected from {addr}")
                    if not self.launched:
                        self.launch()
                    try:
                        while conn.recv(1024):
                            pass
                    except Exception:
                        pass
                    print(f"[#{self.idx}] Client disconnected.")

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

def main():
    print(f"Python: {sys.executable}")
    print(f"Working dir: {Path.cwd()}")
    print(f"Program path: {PROGRAM_PATH.resolve()}")
    if PROGRAM_CWD:
        print(f"Program CWD: {PROGRAM_CWD}")
    if EEG_FILES_FROM_CLI:
        print("EEG files:", EEG_FILES_FROM_CLI)

    runners = [SocketRunner(i) for i in range(1, NUM_SOCKETS + 1)]
    for r in runners:
        r.start()
    print("Five sockets ready on 127.0.0.1:9601..9605")
    print("AUTOSTART =", AUTOSTART)
    print("Press Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")
        for r in runners:
            r.stop()

if __name__ == "__main__":
    main()
