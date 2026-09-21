"""Berry curvature and Chern number for two-band QWZ."""

from __future__ import annotations

import numpy as np

from . import lattice, qwz


def dhat(kx, ky, m: float = 1.0, lam: float = 1.0) -> np.ndarray:
    d = qwz.d_vector(kx, ky, m=m, lam=lam)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-15)
    return d / n


def berry_curvature_mesh(
    m: float = 1.0,
    lam: float = 1.0,
    Nk: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-band valence Berry curvature F = (1/2) d̂ · (∂kx d̂ × ∂ky d̂).

    Returns kx, ky meshes and F[Nk, Nk].
    """
    kx, ky = lattice.bz_mesh(Nk)
    dk = 2.0 * np.pi / Nk
    d = dhat(kx, ky, m=m, lam=lam)

    d_kx_p = np.roll(d, -1, axis=0)
    d_kx_m = np.roll(d, +1, axis=0)
    d_ky_p = np.roll(d, -1, axis=1)
    d_ky_m = np.roll(d, +1, axis=1)

    ddkx = (d_kx_p - d_kx_m) / (2.0 * dk)
    ddky = (d_ky_p - d_ky_m) / (2.0 * dk)
    cross = np.cross(ddkx, ddky)
    # Lower-band curvature for H = d·σ (gives C = sgn(λ) for QWZ at m=1).
    F = -0.5 * np.sum(d * cross, axis=-1)
    return kx, ky, F


def chern_number(m: float = 1.0, lam: float = 1.0, Nk: int = 48) -> int:
    """C = (1/2π) ∫ F d²k, rounded to nearest int. Valence = lower band."""
    _, _, F = berry_curvature_mesh(m=m, lam=lam, Nk=Nk)
    dk = 2.0 * np.pi / Nk
    C = float(np.sum(F) * dk * dk / (2.0 * np.pi))
    return int(np.rint(C))
