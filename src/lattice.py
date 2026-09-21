"""Square-lattice indexing and Brillouin-zone grids."""

from __future__ import annotations

import numpy as np

N_ORB = 2
DEFAULT_NX = 80
DEFAULT_NY = 16


def site_index(x: int, y: int, Ny: int) -> int:
    return int(x) * int(Ny) + int(y)


def xy_of(i: int, Ny: int) -> tuple[int, int]:
    i = int(i)
    return i // Ny, i % Ny


def orb_index(site: int, orb: int, n_orb: int = N_ORB) -> int:
    return int(site) * int(n_orb) + int(orb)


def dim(Nx: int, Ny: int, n_orb: int = N_ORB) -> int:
    return int(Nx) * int(Ny) * int(n_orb)


def n_sites(Nx: int, Ny: int) -> int:
    return int(Nx) * int(Ny)


def bz_mesh(Nk: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform mesh on [-pi, pi) x [-pi, pi)."""
    k = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    return np.meshgrid(k, k, indexing="ij")


def high_symmetry_path(n_per_seg: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gamma -> X -> M -> Gamma path.

    Returns (kx, ky, s) where s is arc-length parameter along the path.
    """
    gamma = np.array([0.0, 0.0])
    X = np.array([np.pi, 0.0])
    M = np.array([np.pi, np.pi])
    segs = [(gamma, X), (X, M), (M, gamma)]
    kx_list: list[np.ndarray] = []
    ky_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    s0 = 0.0
    for a, b in segs:
        t = np.linspace(0.0, 1.0, n_per_seg, endpoint=False)
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        ds = np.linalg.norm(b - a)
        s = s0 + t * ds
        kx_list.append(pts[:, 0])
        ky_list.append(pts[:, 1])
        s_list.append(s)
        s0 += ds
    # include Gamma endpoint
    kx_list.append(np.array([0.0]))
    ky_list.append(np.array([0.0]))
    s_list.append(np.array([s0]))
    return (
        np.concatenate(kx_list),
        np.concatenate(ky_list),
        np.concatenate(s_list),
    )
