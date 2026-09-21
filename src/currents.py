"""Bond currents from a density matrix."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse

from . import lattice


def bond_current_orbital(
    rho: np.ndarray,
    t_ji: np.ndarray,
    i_site: int,
    j_site: int,
) -> float:
    """
    Orbital-summed Hermitian bond current I_ij = 2 Im Tr_orb (t_ji ρ_ij).

    t_ji is the 2x2 hopping block from j -> i (i.e. H_{ij} block when
    orbitals are contiguous), and ρ_ij is the 2x2 block of the density matrix.
    """
    i0 = lattice.orb_index(i_site, 0)
    j0 = lattice.orb_index(j_site, 0)
    rho_ij = rho[i0 : i0 + 2, j0 : j0 + 2]
    # I = 2 Im Tr(t_ji ρ_ij); with H_ij = t_{i<-j}, usually t_ji = H_block(j,i)
    val = np.trace(t_ji @ rho_ij)
    return float(2.0 * np.imag(val))


def extract_hop_block(H, i_site: int, j_site: int) -> np.ndarray:
    """2x2 block H_{orb_i, orb_j} for sites i <- from matrix rows i, cols j."""
    if sparse.issparse(H):
        H = H.tocsr()
        block = np.zeros((2, 2), dtype=np.complex128)
        for a in range(2):
            for b in range(2):
                block[a, b] = H[lattice.orb_index(i_site, a), lattice.orb_index(j_site, b)]
        return block
    i0 = lattice.orb_index(i_site, 0)
    j0 = lattice.orb_index(j_site, 0)
    return np.asarray(H[i0 : i0 + 2, j0 : j0 + 2])


def bond_current(
    rho: np.ndarray,
    H,
    i_site: int,
    j_site: int,
) -> float:
    """I_ij = 2 Im Tr(t_ji ρ_ij) with t_ji = H_{j,i} block."""
    t_ji = extract_hop_block(H, j_site, i_site)
    return bond_current_orbital(rho, t_ji, i_site, j_site)


def bond_currents(
    rho: np.ndarray,
    H,
    bonds: Iterable[tuple[int, int]],
) -> list[tuple[int, int, float]]:
    """
    Compute currents for site bonds.

    bonds: iterable of (i_site, j_site).
    Returns list of (i_site, j_site, I).
    """
    out: list[tuple[int, int, float]] = []
    for i_site, j_site in bonds:
        I = bond_current(rho, H, i_site, j_site)
        out.append((i_site, j_site, I))
    return out


def directed_bonds_from_geometry(
    Nx: int,
    Ny: int,
    bc_x: str,
) -> list[tuple[int, int]]:
    """All +x and +y site bonds as (i_site, j_site)."""
    from . import geometry

    bonds: list[tuple[int, int]] = []
    for x, y, x2, y2 in geometry.iter_plus_x_bonds(Nx, Ny, bc_x):  # type: ignore[arg-type]
        i = lattice.site_index(x, y, Ny)
        j = lattice.site_index(x2, y2, Ny)
        bonds.append((i, j))
    for x, y, x2, y2 in geometry.iter_plus_y_bonds(Nx, Ny):
        i = lattice.site_index(x, y, Ny)
        j = lattice.site_index(x2, y2, Ny)
        bonds.append((i, j))
    return bonds
