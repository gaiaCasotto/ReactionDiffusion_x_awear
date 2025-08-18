#!/usr/bin/env python3
# Gray–Scott Reaction–Diffusion
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def laplacian(Z):
    return (-4*Z
            + np.roll(Z, 1, 0) + np.roll(Z, -1, 0)
            + np.roll(Z, 1, 1) + np.roll(Z, -1, 1))

def grayscott_step(U, V, Du, Dv, F, k, dt):
    Lu = laplacian(U)
    Lv = laplacian(V)
    UVV = U*V*V
    dU = Du*Lu - UVV + F*(1-U)
    dV = Dv*Lv + UVV - (F+k)*V
    U += dU*dt
    V += dV*dt
    np.clip(U, 0.0, 1.0, out=U)
    np.clip(V, 0.0, 1.0, out=V)
    return U, V

def simulate(n=256, steps=200, Du=0.16, Dv=0.08, F=0.037, k=0.06, dt=1.0, seed_size=20, seed_V=0.25):
    rng = np.random.default_rng(0)
    U = np.ones((n,n), dtype=np.float32)
    V = np.zeros((n,n), dtype=np.float32)
    U += (rng.random((n,n), dtype=np.float32)-0.5)*0.02
    V += (rng.random((n,n), dtype=np.float32)-0.5)*0.02
    np.clip(U,0.0,1.0,out=U); np.clip(V,0.0,1.0,out=V)
    s = seed_size; c = n//2
    V[c-s:c+s, c-s:c+s] = seed_V
    U[c-s:c+s, c-s:c+s] = 1.0 - seed_V

    frames = []
    for i in range(steps):
        U, V = grayscott_step(U, V, Du, Dv, F, k, dt)
        if i % 2 == 0:
            frames.append(V.copy())
    return U, V, frames

def save_gif(frames, path="grayscott.gif", duration_ms=50):
    imgs = []
    for arr in frames:
        a = (255*(arr - arr.min())/(arr.ptp()+1e-8)).astype(np.uint8)
        imgs.append(Image.fromarray(a))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0, duration=duration_ms)

if __name__ == "__main__":
    U, V, frames = simulate()
    plt.figure(figsize=(5,5))
    plt.imshow(frames[-1])
    plt.axis("off"); plt.tight_layout()
    plt.show()
    save_gif(frames, "grayscott.gif", duration_ms=50)
