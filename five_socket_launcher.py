
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
NUM_SOCKETS = 5
BACKLOG = 1

# --- Configure this if your script is elsewhere ---
PROGRAM_PATH = Path("t_rd_colors_ac.py")  # absolute or relative path
PROGRAM_ARGS = ["--user", "{idx}"]     # idx expands to 1..5

# Launch behavior
AUTOSTART = True   # True: start all 5 immediately; False: start on first client connect

# Optional: set a working directory for the program (None = current)
PROGRAM_CWD = None  # e.g., str(Path(__file__).parent)

# Environment (inherit + tweaks)
PROGRAM_ENV = dict(os.environ)

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

        args = [sys.executable, str(PROGRAM_PATH)] + [a.format(idx=self.idx) for a in PROGRAM_ARGS]

        # Per-instance log files
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
                    # Keep the socket open until the client closes
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
    # Helpful diagnostics
    print(f"Python: {sys.executable}")
    print(f"Working dir: {Path.cwd()}")
    print(f"Program path: {PROGRAM_PATH.resolve()}")
    if PROGRAM_CWD:
        print(f"Program CWD: {PROGRAM_CWD}")

    runners = [SocketRunner(i) for i in range(1, NUM_SOCKETS + 1)]
    for r in runners:
        r.start()
    print("Five sockets ready on 127.0.0.1:9601..9605")
    print("AUTOSTART =", AUTOSTART, "(windows should appear now if True)")
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





'''

# five_socket_runner.py
# Opens 5 TCP sockets; each launches t_rd_colors.py when a client connects.

import socket
import threading
import subprocess
import sys
import time

HOST = "127.0.0.1"
BASE_PORT = 9601
NUM_SOCKETS = 5
BACKLOG = 1  # one client per socket typical

PROGRAM = [sys.executable, "t_rd_colors.py", "--user", "{idx}"]

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
        args = [a.format(idx=self.idx) for a in PROGRAM]
        try:
            self.proc = subprocess.Popen(args)
            print(f"[#{self.idx}] Launched: {' '.join(args)} (pid={self.proc.pid})")
            self.launched = True
        except Exception as e:
            print(f"[#{self.idx}] ERROR launching program: {e}")

    def run(self):
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
    runners = [SocketRunner(i) for i in range(1, NUM_SOCKETS + 1)]
    for r in runners:
        r.start()
    print("Five sockets active. Connect to 127.0.0.1:9601..9605 to trigger launches.")
    print("Ctrl+C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")
        for r in runners:
            r.stop()

if __name__ == "__main__":
    main()


'''




'''


# five_socket_launcher.py
# Opens 5 TCP sockets; each streams lines from a .txt file.
# On first client connection per socket, starts your rendering program (configurable).

import os
import sys
import socket
import threading
import subprocess
import time
from pathlib import Path
from typing import Optional, List

HOST = "127.0.0.1"
BASE_PORT     = 9501
BACKLOG       = 1
NUM_SOCKETS   = 5
POLL_INTERVAL = 0.25

# Input files (1..5). Create them if missing.
INPUT_FILES: List[str] = [
    "../eeg_files/1_horror_movie_data_filtered.txt",
    "../eeg_files/2_vipassana_data_filtered.txt",
    "../eeg_files/3_hot_tub_data_filtered.txt",
    "../eeg_files/fake_eeg_longblocks.txt",            # stressed first
    "../eeg_files/fake_eeg_longblocks_calmfirst.txt",  # calm first
]
# Command template for your program.
# Use {idx} placeholder for the user/index (1..5).
# Example uses a Taichi script with a --user parameter; change it to your program.
PROGRAM_CMD = [sys.executable, "t_rd_colors.py", "--user", "{idx}"]

# If program needs environment vars, set them here:
PROGRAM_ENV = None  # or dict(os.environ, MYFLAG="1")

class SocketFeeder(threading.Thread):
    def __init__(self, idx: int, input_path: str):
        super().__init__(daemon=True)
        self.idx                              = idx
        self.port                             = BASE_PORT + idx - 1
        self.input_path                       = Path(input_path)
        self.proc: Optional[subprocess.Popen] = None
        self.launched                         = False
        self._stop                            = threading.Event()

    def launch_program(self):
        if self.launched:
            return
        args = [a.format(idx=self.idx) for a in PROGRAM_CMD]
        try:
            self.proc = subprocess.Popen(args, env=PROGRAM_ENV)
            print(f"[#{self.idx}] Launched program: {' '.join(args)} (pid={self.proc.pid})")
            self.launched = True
        except Exception as e:
            print(f"[#{self.idx}] ERROR launching program: {e}")

    def tail_send(self, conn: socket.socket):
        # wait for file to exist
        while not self.input_path.exists() and not self._stop.is_set():
            print(f"[#{self.idx}] Waiting for file: {self.input_path}")
            time.sleep(0.5)
        last_size = 0
        try:
            last_size = self.input_path.stat().st_size
        except Exception:
            last_size = 0
        print(f"[#{self.idx}] Streaming from {self.input_path} -> {conn.getpeername()}")
        while not self._stop.is_set():
            try:
                size = self.input_path.stat().st_size
                if size < last_size:
                    last_size = 0  # truncated
                if size > last_size:
                    with self.input_path.open("rb") as f:
                        f.seek(last_size)
                        chunk = f.read(size - last_size)
                        if chunk:
                            conn.sendall(chunk)
                    last_size = size
                time.sleep(POLL_INTERVAL)
            except (BrokenPipeError, ConnectionResetError):
                print(f"[#{self.idx}] Client disconnected.")
                break
            except Exception as e:
                print(f"[#{self.idx}] Tail/send error: {e}")
                time.sleep(POLL_INTERVAL)

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, self.port))
            s.listen(BACKLOG)
            print(f"[#{self.idx}] Listening on {HOST}:{self.port} (file: {self.input_path})")
            while not self._stop.is_set():
                try:
                    s.settimeout(1.0)
                    conn, addr = s.accept()
                except socket.timeout:
                    continue
                with conn:
                    print(f"[#{self.idx}] Client connected from {addr}")
                    self.launch_program()
                    self.tail_send(conn)

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:
                pass

def main():
    if len(INPUT_FILES) != 5:
        print("ERROR: Need exactly 5 input files in INPUT_FILES.", file=sys.stderr)
        sys.exit(1)
    feeders = [SocketFeeder(i, INPUT_FILES[i-1]) for i in range(1, 6)]
    for f in feeders:
        f.start()
    print("All sockets ready:")
    for i, path in enumerate(INPUT_FILES, 1):
        print(f"  #{i}: {HOST}:{BASE_PORT+i-1}  <-  {path}")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")
        for f in feeders:
            f.stop()

if __name__ == "__main__":
    main()
'''