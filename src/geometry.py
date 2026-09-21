"""
Cylinder / Möbius gluing and 3D Möbius embedding.

Chart convention
----------------
Lattice sites use 0-based (x, y) with x in {0..Nx-1}, y in {0..Ny-1}.
For the embedded strip, map

    u = 2π x / Nx          (around the hole)
    v = 2 (y / (Ny - 1)) - 1   if Ny > 1 else 0   (width coordinate in [-1, 1])

Normals n̂ = ∂_u r × ∂_v r / |...| are computed in this same chart so that the
Hall vector field h = C(r) n̂(r) uses a matching orientation with the
per-site Bianco–Resta marker chart (minimal-image dx; y flipped across the
Möbius seam when evaluating position operators).
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

BcX = Literal["open", "cylinder", "mobius"]


def hop_x(
    x: int,
    y: int,
    Nx: int,
    Ny: int,
    kind: BcX,
) -> Optional[tuple[int, int]]:
    """
    Destination of a +x hop from (x, y), or None if the bond is absent.

    Möbius seam: (Nx-1, y) -> (0, Ny-1-y).
    """
    x = int(x)
    y = int(y)
    if x < Nx - 1:
        return x + 1, y
    # seam / wrap
    if kind == "open":
        return None
    if kind == "cylinder":
        return 0, y
    if kind == "mobius":
        return 0, Ny - 1 - y
    raise ValueError(f"unknown bc kind: {kind!r}")


def y_neighbor(y: int, Ny: int, dy: int) -> Optional[int]:
    """Open boundary in y: neighbor or None."""
    y2 = int(y) + int(dy)
    if 0 <= y2 < Ny:
        return y2
    return None


def iter_plus_x_bonds(Nx: int, Ny: int, kind: BcX):
    """Yield (x, y, x2, y2) for every existing +x bond."""
    for x in range(Nx):
        for y in range(Ny):
            nxt = hop_x(x, y, Nx, Ny, kind)
            if nxt is not None:
                x2, y2 = nxt
                yield x, y, x2, y2


def iter_plus_y_bonds(Nx: int, Ny: int):
    """Yield (x, y, x, y+1) for open +y bonds."""
    for x in range(Nx):
        for y in range(Ny - 1):
            yield x, y, x, y + 1


def lattice_to_uv(x, y, Nx: int, Ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Map lattice coordinates to embedding parameters (u, v)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    u = 2.0 * np.pi * x / float(Nx)
    if Ny <= 1:
        v = np.zeros_like(y)
    else:
        v = 2.0 * (y / float(Ny - 1)) - 1.0
    return u, v


def embed_mobius(
    u,
    v,
    R: float = 1.0,
    w: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Möbius embedding.

    r(u,v) with u in [0, 2π), v in [-1, 1], radius R, half-width scale w.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    half = 0.5 * u
    X = (R + w * v * np.cos(half)) * np.cos(u)
    Y = (R + w * v * np.cos(half)) * np.sin(u)
    Z = w * v * np.sin(half)
    return X, Y, Z


def _mobius_partials(u, v, R: float, w: float):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    half = 0.5 * u
    c_h, s_h = np.cos(half), np.sin(half)
    c_u, s_u = np.cos(u), np.sin(u)
    rho = R + w * v * c_h

    # ∂r/∂u
    dr_du = np.stack(
        [
            -rho * s_u + (-0.5 * w * v * s_h) * c_u,
            rho * c_u + (-0.5 * w * v * s_h) * s_u,
            0.5 * w * v * c_h,
        ],
        axis=-1,
    )
    # ∂r/∂v
    dr_dv = np.stack(
        [
            w * c_h * c_u,
            w * c_h * s_u,
            w * s_h,
        ],
        axis=-1,
    )
    return dr_du, dr_dv


def normals_mobius(
    u,
    v,
    R: float = 1.0,
    w: float = 0.35,
) -> np.ndarray:
    """Unit normals n̂ = (∂u × ∂v) / |...| with shape (..., 3)."""
    dr_du, dr_dv = _mobius_partials(u, v, R, w)
    n = np.cross(dr_du, dr_dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-15)
    return n / norm


def embed_lattice_mobius(
    Nx: int,
    Ny: int,
    R: float = 1.0,
    w: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Embed all lattice sites; returns X,Y,Z, normals each shaped (Nx, Ny) or (Nx,Ny,3).
    """
    xs = np.arange(Nx)[:, None]
    ys = np.arange(Ny)[None, :]
    u, v = lattice_to_uv(xs, ys, Nx, Ny)
    X, Y, Z = embed_mobius(u, v, R=R, w=w)
    n = normals_mobius(u, v, R=R, w=w)
    return X, Y, Z, n
