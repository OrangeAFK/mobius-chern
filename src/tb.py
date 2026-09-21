"""Single-orbital nearest-neighbor tight-binding on the square lattice."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from . import geometry, lattice


def dispersion(kx, ky, t: float = 1.0):
    """E(k) = -2t (cos kx + cos ky)."""
    return -2.0 * t * (np.cos(kx) + np.cos(ky))


def H_kspace(kx: float, ky: float, t: float = 1.0) -> complex:
    return complex(dispersion(kx, ky, t=t))


def H_realspace(
    Nx: int,
    Ny: int,
    t: float = 1.0,
    bc_x: str = "open",
) -> sparse.csr_matrix:
    """
    Real-space single-orbital TB Hamiltonian.

    bc_x: 'open' | 'cylinder' (periodic in x). y is always open.
    """
    if bc_x not in ("open", "cylinder"):
        raise ValueError("tb.H_realspace supports bc_x in {'open','cylinder'}")
    N = lattice.n_sites(Nx, Ny)
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []

    def add(i: int, j: int, val: complex) -> None:
        rows.append(i)
        cols.append(j)
        data.append(val)

    for x in range(Nx):
        for y in range(Ny):
            i = lattice.site_index(x, y, Ny)
            # +x hop
            nxt = geometry.hop_x(x, y, Nx, Ny, bc_x)
            if nxt is not None:
                x2, y2 = nxt
                j = lattice.site_index(x2, y2, Ny)
                add(i, j, -t)
                add(j, i, -t)
            # +y hop (open)
            y2 = geometry.y_neighbor(y, Ny, +1)
            if y2 is not None:
                j = lattice.site_index(x, y2, Ny)
                add(i, j, -t)
                add(j, i, -t)

    return sparse.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.complex128)
