"""Peierls phases for hole flux and plaquette flux tubes."""

from __future__ import annotations

from typing import Dict, Tuple

from . import geometry

PhaseMap = Dict[Tuple[int, int], float]


def hole_phases(
    Nx: int,
    Ny: int,
    Phi: float,
    bc_x: geometry.BcX = "cylinder",
) -> PhaseMap:
    """
    Uniform Peierls twist on all +x hoppings: each +x bond gets phase Phi/Nx
    so that a loop around the hole accumulates Phi.
    """
    if Nx <= 0:
        raise ValueError("Nx must be positive")
    dphi = float(Phi) / float(Nx)
    phases: PhaseMap = {}
    for x, y, _, _ in geometry.iter_plus_x_bonds(Nx, Ny, bc_x):
        phases[(x, y)] = dphi
    return phases


def tube_phases(
    Nx: int,
    Ny: int,
    x_tube: int,
    y0: int,
    Phi: float,
    bc_x: geometry.BcX = "cylinder",
) -> PhaseMap:
    """
    Phase string for a flux tube through the plaquette whose lower-left
    corner is (x_tube, y0). Returns a map for ``phi_x`` in
    ``qwz.H_realspace`` (phases on +x hops).

    Puts phase ``Phi`` on +x bonds at column ``x_tube`` for ``y = 0 .. y0``.
    Intermediate plaquettes cancel (Phi - Phi = 0); only the target
    plaquette at y0 carries net flux Phi. The string runs to the open
    bottom edge and does not cross the Mobius x-seam.
    ``bc_x`` is accepted for API symmetry with ``hole_phases``.
    """
    del bc_x  # vertical cut of +x bonds; seam handled by hop_x elsewhere
    x_tube = int(x_tube) % Nx
    y0 = int(y0)
    if not (0 <= y0 < Ny - 1):
        raise ValueError("y0 must index a plaquette lower-left with y0 in [0, Ny-2]")
    phases: PhaseMap = {}
    for y in range(0, y0 + 1):
        phases[(x_tube, y)] = float(Phi)
    return phases
