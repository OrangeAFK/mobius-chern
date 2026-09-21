"""Dense / sparse eigensolvers and occupied projectors."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def eigh(H) -> tuple[np.ndarray, np.ndarray]:
    """Full Hermitian diagonalization. Returns (evals, evecs) with evecs as columns."""
    if sparse.issparse(H):
        Hd = H.toarray()
    else:
        Hd = np.asarray(H)
    evals, evecs = np.linalg.eigh(Hd)
    return evals, evecs


def eigsh_near_zero(H, k: int = 20, sigma: float = 0.0, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Shift-invert sparse eigensolve around sigma (default 0)."""
    if not sparse.issparse(H):
        H = sparse.csr_matrix(H)
    k = min(int(k), H.shape[0] - 2)
    if k < 1:
        raise ValueError("need matrix dimension > 2 for eigsh_near_zero")
    evals, evecs = eigsh(H, k=k, sigma=sigma, which="LM", **kwargs)
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def n_occupied(dim: int, fill_half: bool = True) -> int:
    """Half-filling for two-band insulator: dim/2 occupied states."""
    if fill_half:
        return dim // 2
    raise ValueError("only half-filling supported")


def projector(evecs: np.ndarray, n_occ: int) -> np.ndarray:
    """P = V_occ V_occ† from the lowest n_occ eigenvectors (columns)."""
    V = evecs[:, :n_occ]
    return V @ V.conj().T


def occupied_density_matrix(
    evals: np.ndarray,
    evecs: np.ndarray,
    n_occ: int | None = None,
) -> np.ndarray:
    if n_occ is None:
        n_occ = n_occupied(evecs.shape[0])
    return projector(evecs, n_occ)
