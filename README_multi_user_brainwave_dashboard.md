
# Multi-User Brainwave Dashboard (1–10 users)

**What it does**
- Displays 1–10 users' brainwaves on one screen.
- Computes alpha/beta ratio per user and triggers a reaction when beta dominates.
- Includes a clean interface to plug in your real EEG devices.
- Optional: launches your Taichi RD visual (`t_rd_colors.py`) per user when triggered.

**Run (simulated data)**
```bash
python multi_user_brainwave_dashboard.py --users 4 --simulate --reaction log
```

**Optional Taichi integration**
```bash
python multi_user_brainwave_dashboard.py --users 3 --simulate --reaction taichi
```
This will try to launch `t_rd_colors.py` in the same folder, passing `--stress <0..1>` and `--user <id>`.

**Hooking up real devices**
- Implement `RealDeviceClient.read_samples()` to yield numpy arrays of samples at `sr` Hz.
- Replace `make_device_clients()` to return `RealDeviceClient` objects.

**Notes**
- Threshold defaults to `alpha/beta < 0.8` meaning beta-dominant -> trigger.
- Cooldown prevents spam triggers (default 3s).
- Window shows ~6s of data, updating ~20 fps.
