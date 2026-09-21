"""Bianco–Resta local Chern marker in a per-site chart."""

from __future__ import annotations

import numpy as np

from . import geometry, lattice


def _position_ops(
    Nx: int,
    Ny: int,
    bc_x: geometry.BcX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonal position operators on the orbital Hilbert space.

    Cylinder / Möbius: X uses minimal-image coordinates on the ring
    (values in (-Nx/2, Nx/2]). Y is the open-strip row index.

    Möbius chart: place the coordinate cut opposite the geometric seam so
    the seam is interior. Crossing the seam in the covering space flips y
    (PLAN §2): sites with x > cut use Y = (Ny - 1 - y) when measuring in
    the chart continuous through the seam. Here the cut is at mid-column
    ``Nx // 2``, and the half containing the seam (x near 0 / Nx-1) keeps
    ordinary Y on one side and flipped Y on the other relative to that cut.
    """
    N = lattice.dim(Nx, Ny)
    X = np.zeros(N, dtype=np.float64)
    Y = np.zeros(N, dtype=np.float64)
    cut = Nx // 2
    for x in range(Nx):
        if bc_x in ("cylinder", "mobius"):
            xm = ((x - cut + Nx // 2) % Nx) - Nx // 2
        else:
            xm = float(x)
        for y in range(Ny):
            if bc_x == "mobius" and x >= cut:
                # second chart across the mid cut: y flipped so that
                # continuing through the geometric seam matches hop_x
                y_chart = float(Ny - 1 - y)
            else:
                y_chart = float(y)
            site = lattice.site_index(x, y, Ny)
            for orb in range(2):
                idx = lattice.orb_index(site, orb)
                X[idx] = float(xm)
                Y[idx] = y_chart
    return X, Y


def _comm(A_diag: np.ndarray, P: np.ndarray) -> np.ndarray:
    """[A, P] for diagonal A: A_i P_ij - P_ij A_j."""
    return A_diag[:, None] * P - P * A_diag[None, :]


def local_chern(
    P: np.ndarray,
    Nx: int,
    Ny: int,
    bc_x: geometry.BcX = "cylinder",
) -> np.ndarray:
    """
    C(r) = -2π Im ⟨r| P [[X,P],[Y,P]] |r⟩, orbital-traced per site.

    Returns array shape (Nx, Ny).
    """
    X, Y = _position_ops(Nx, Ny, bc_x)
    XP = _comm(X, P)
    YP = _comm(Y, P)
    # [[X,P],[Y,P]] = XP @ YP - YP @ XP
    M = XP @ YP - YP @ XP
    PM = P @ M

    C = np.zeros((Nx, Ny), dtype=np.float64)
    for x in range(Nx):
        for y in range(Ny):
            site = lattice.site_index(x, y, Ny)
            acc = 0.0
            for orb in range(2):
                i = lattice.orb_index(site, orb)
                acc += float(np.imag(PM[i, i]))
            C[x, y] = -2.0 * np.pi * acc
    return C


def average_Cx(C: np.ndarray) -> np.ndarray:
    """C(x) = ⟨C⟩_y."""
    return np.mean(C, axis=1)


def hall_vector(
    C: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    """h = C n̂ with normals shaped (Nx, Ny, 3). Returns same shape."""
    return C[..., None] * normals
