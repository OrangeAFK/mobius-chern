"""Qi–Wu–Zhang Hamiltonian: k-space, hoppings, λ(x), real-space builds."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
from scipy import sparse

from . import geometry, lattice

PhaseMap = Mapping[tuple[int, int], float]  # (x,y) -> Peierls phase on +hop from site


def sigma_x() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=np.complex128)


def sigma_y() -> np.ndarray:
    return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


def sigma_z() -> np.ndarray:
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def d_vector(kx, ky, m: float = 1.0, lam: float = 1.0) -> np.ndarray:
    """d(k) for H = d · σ. Broadcasting over kx, ky."""
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    dx = np.sin(kx)
    dy = lam * np.sin(ky)
    dz = m + np.cos(kx) + np.cos(ky)
    return np.stack([dx, dy, dz], axis=-1)


def Hk(kx: float, ky: float, m: float = 1.0, lam: float = 1.0) -> np.ndarray:
    d = d_vector(kx, ky, m=m, lam=lam)
    return d[0] * sigma_x() + d[1] * sigma_y() + d[2] * sigma_z()


def hop_x_block() -> np.ndarray:
    """+x hopping: (σz + i σx) / 2."""
    return 0.5 * (sigma_z() + 1j * sigma_x())


def hop_y_block(lam: float) -> np.ndarray:
    """+y hopping: (σz + i λ σy) / 2."""
    return 0.5 * (sigma_z() + 1j * float(lam) * sigma_y())


def onsite_block(m: float) -> np.ndarray:
    return float(m) * sigma_z()


def lambda_tanh(
    Nx: int,
    x0: float,
    w: float,
    lam0: float = 1.0,
    *,
    antiperiodic: bool = False,
) -> np.ndarray:
    """
    Wall profile λ(x).

    antiperiodic=False (cylinder): two zeros per circuit via sin(2π …).
    antiperiodic=True (Möbius): one zero; λ(x+Nx) = -λ(x) when extended.
    """
    x = np.arange(Nx, dtype=float)
    if w <= 0:
        raise ValueError("w must be positive")
    if antiperiodic:
        # sin(π(x-x0)/Nx) is antiperiodic under x -> x+Nx
        scale = Nx / (np.pi * w)
        return lam0 * np.tanh(np.sin(np.pi * (x - x0) / Nx) * scale)
    scale = Nx / (2.0 * np.pi * w)
    return lam0 * np.tanh(np.sin(2.0 * np.pi * (x - x0) / Nx) * scale)


def _add_block(
    rows: list[int],
    cols: list[int],
    data: list[complex],
    i_site: int,
    j_site: int,
    block: np.ndarray,
) -> None:
    for a in range(2):
        for b in range(2):
            val = block[a, b]
            if val == 0:
                continue
            rows.append(lattice.orb_index(i_site, a))
            cols.append(lattice.orb_index(j_site, b))
            data.append(complex(val))


def H_realspace(
    Nx: int,
    Ny: int,
    m: float = 1.0,
    lam: float = 1.0,
    bc_x: geometry.BcX = "cylinder",
    lam_x: Optional[Sequence[float]] = None,
    phi_x: Optional[PhaseMap] = None,
    phi_y: Optional[PhaseMap] = None,
) -> sparse.csr_matrix:
    """
    Real-space QWZ (2 orbitals/site).

    lam_x: per-column λ(x); if None, use constant `lam`.
    phi_x / phi_y: Peierls phases on +x / +y hops keyed by source (x, y).
    Built only through geometry.hop_x for the x-direction.
    """
    if lam_x is None:
        lam_of = np.full(Nx, float(lam), dtype=float)
    else:
        lam_of = np.asarray(lam_x, dtype=float)
        if lam_of.shape != (Nx,):
            raise ValueError("lam_x must have shape (Nx,)")

    N = lattice.dim(Nx, Ny)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    onsite = onsite_block(m)
    tx = hop_x_block()

    for x in range(Nx):
        for y in range(Ny):
            i = lattice.site_index(x, y, Ny)
            _add_block(rows, cols, data, i, i, onsite)

            nxt = geometry.hop_x(x, y, Nx, Ny, bc_x)
            if nxt is not None:
                x2, y2 = nxt
                j = lattice.site_index(x2, y2, Ny)
                phase = 0.0 if phi_x is None else float(phi_x.get((x, y), 0.0))
                blk = tx * np.exp(1j * phase)
                _add_block(rows, cols, data, i, j, blk)
                _add_block(rows, cols, data, j, i, blk.conj().T)

            y2 = geometry.y_neighbor(y, Ny, +1)
            if y2 is not None:
                j = lattice.site_index(x, y2, Ny)
                phase = 0.0 if phi_y is None else float(phi_y.get((x, y), 0.0))
                ty = hop_y_block(lam_of[x]) * np.exp(1j * phase)
                _add_block(rows, cols, data, i, j, ty)
                _add_block(rows, cols, data, j, i, ty.conj().T)

    return sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)


def H_ribbon(
    kx: float,
    Ny: int,
    m: float = 1.0,
    lam: float = 1.0,
) -> np.ndarray:
    """
    QWZ ribbon: Bloch in x (momentum kx), open in y.

    Hilbert space: Ny sites × 2 orbitals, indexed as orb_index(y, orb)
    with site = y (single column).
    """
    N = 2 * int(Ny)
    H = np.zeros((N, N), dtype=np.complex128)
    onsite = onsite_block(m)
    tx = hop_x_block()
    ty = hop_y_block(lam)
    phase = np.exp(1j * float(kx))
    # +x Bloch: t_x e^{ikx} + h.c. on each y
    hx = tx * phase
    hx_hc = hx.conj().T
    for y in range(Ny):
        i0 = 2 * y
        H[i0 : i0 + 2, i0 : i0 + 2] += onsite
        H[i0 : i0 + 2, i0 : i0 + 2] += hx + hx_hc
        if y < Ny - 1:
            j0 = 2 * (y + 1)
            H[i0 : i0 + 2, j0 : j0 + 2] += ty
            H[j0 : j0 + 2, i0 : i0 + 2] += ty.conj().T
    return H
