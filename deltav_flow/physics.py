"""Recorded Task 7/8 solver, analytic initial condition and outcome readouts.

Extracted without scientific change from research/DeltaV_Task8_MoLab.py
(`solve_flow`, `initial7`, `physical_outcomes`); see contract.PROVENANCE.
"""
from __future__ import annotations

import time

import numpy as np

from .contract import OUTCOMES


def physical_outcomes(fields, specs=OUTCOMES):
    """Outcomes from fields [..., 4, H, W]. Works for numpy arrays and torch tensors."""
    import torch

    values = []
    for spec in specs:
        if spec["kind"] == "mean_kinetic_energy":
            value = 0.5 * (fields[..., 2, :, :] ** 2 + fields[..., 3, :, :] ** 2).mean(axis=(-2, -1))
        elif spec["kind"] == "pressure_rms":
            p = fields[..., 1, :, :]
            centered = p - p.mean(axis=(-2, -1), keepdims=True)
            if isinstance(p, torch.Tensor):
                value = torch.linalg.vector_norm(centered.flatten(start_dim=-2), dim=-1) / (p.shape[-2] * p.shape[-1]) ** 0.5
            else:
                value = (centered ** 2).mean(axis=(-2, -1)) ** 0.5
        else:
            raise ValueError("Unsupported outcome definition: " + spec["kind"])
        values.append(value)
    return torch.stack(values, dim=-1) if isinstance(fields, torch.Tensor) else np.stack(values, axis=-1)


def solve_flow(controls, n=64, dt=0.005, final_time=2.0, initial_only=False, output_dtype="float32"):
    """Dealiased Fourier vorticity RK4 solver; returns ([density=1, p, u, v], receipt)."""
    nu, force, mix = map(float, controls)
    if n % 2 or nu <= 0 or dt <= 0 or (final_time < 0):
        raise ValueError("Invalid numerical configuration")
    q = np.arange(n) * 2 * np.pi / n
    xx, yy = np.meshgrid(q, q)
    k = np.fft.fftfreq(n, 1 / n)
    kx, ky = np.meshgrid(k, k)
    k2 = kx * kx + ky * ky
    inverse = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 != 0)
    mask = (np.abs(kx) < n / 3) & (np.abs(ky) < n / 3)
    omega = 2 * np.sin(xx) * np.sin(yy) + 5 * mix * np.cos(2 * xx) * np.sin(yy)
    w = np.fft.fft2(omega) * mask
    forcing = np.fft.fft2(force * (np.cos(2 * yy) + 0.7 * np.sin(3 * xx))) * mask

    def velocities(z):
        psi = z * inverse
        return (np.fft.ifft2(1j * ky * psi).real, np.fft.ifft2(-1j * kx * psi).real)

    def rhs(z):
        u, v = velocities(z)
        wx = np.fft.ifft2(1j * kx * z).real
        wy = np.fft.ifft2(1j * ky * z).real
        return (-np.fft.fft2(u * wx + v * wy) * mask - nu * k2 * z + forcing) * mask

    steps = 0 if initial_only else int(np.ceil(final_time / dt))
    step = final_time / max(steps, 1)
    peak_cfl = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        u, v = velocities(w)
        cfl = step * (np.abs(u).max() + np.abs(v).max()) / (2 * np.pi / n)
        peak_cfl = max(peak_cfl, float(cfl))
        if cfl > 0.65 or step * nu * k2[mask].max() > 2.0:
            raise ValueError("Solver stability guard; reduce dt")
        a = rhs(w)
        b = rhs(w + step * a / 2)
        c = rhs(w + step * b / 2)
        d = rhs(w + step * c)
        w = (w + step * (a + 2 * b + 2 * c + d) / 6) * mask
        if not np.isfinite(w).all():
            raise ValueError("Nonfinite reference solver")
    u, v = velocities(w)
    ux = np.fft.ifft2(1j * kx * np.fft.fft2(u)).real
    uy = np.fft.ifft2(1j * ky * np.fft.fft2(u)).real
    vx = np.fft.ifft2(1j * kx * np.fft.fft2(v)).real
    vy = np.fft.ifft2(1j * ky * np.fft.fft2(v)).real
    p = np.fft.ifft2(np.fft.fft2(ux * ux + 2 * uy * vx + vy * vy) * mask * inverse).real
    field = np.stack([np.ones_like(u), p, u, v]).astype(output_dtype)
    return (field, {"seconds": time.perf_counter() - started, "steps": steps, "grid": n, "dt": step,
                    "peak_cfl": peak_cfl, "max_divergence": float(np.abs(ux + vy).max())})


def initial_fields(x) -> np.ndarray:
    """Known analytic initial fields for controls [N, 3] (Task 7 `initial7`). No future solve."""
    return np.stack([solve_flow(v, initial_only=True)[0] for v in np.asarray(x, float)])
