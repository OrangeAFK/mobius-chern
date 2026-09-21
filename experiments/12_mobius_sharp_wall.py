"""Experiment 12 — Möbius gluing, sharp seam wall (prior art calibration)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import lattice, marker, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
LAM = 1.0
NX, NY = 40, 12
EDGE_TRIM = 2
SEAM_WIN = 3
GAP_CUT = 0.5
SEAM_FRAC = 0.45
ARC_MIN = 0.25

NOTE_MD = """# Möbius sharp seam (Experiment 12)

**Prior art — not a new claim.** This note calibrates against Beugeling, Quelle & Morais Smith
and Huang & Lee.

## Beugeling: QH needs a domain wall

On a non-orientable band a global Chern number (and a uniform perpendicular field /
Hall conductivity) cannot be defined smoothly: the Chern density is parity-odd and
orientation-dependent (Beugeling, Quelle & Morais Smith, Phys. Rev. B **89**, 235112
(2014); arXiv:1403.6998). Local quantum-Hall physics on a Möbius strip therefore
requires an orientation flip — a chirality **domain wall**.

## This calibration

Qi–Wu–Zhang with **uniform** `λ` plus Möbius gluing `(N_x-1, y) → (0, N_y-1-y)` makes
the geometric **seam** that wall. The in-gap seam doublet and the opposite-sign local
marker on the two arcs meeting at the seam are the expected Jackiw–Rebbi /
orientation-obstruction signature. Huang & Lee (Phys. Rev. B **84**, 193106 (2011);
arXiv:1107.1411) showed the related edge-current twist; Experiment 13 reproduces
those current patterns.

Do not read this as a first observation of Möbius Chern edge modes.
"""


def site_ldos(psi: np.ndarray) -> np.ndarray:
    dens = np.abs(psi).reshape(NX * NY, 2) ** 2
    return dens.sum(axis=1).reshape(NX, NY)


def seam_mask() -> np.ndarray:
    """Columns near the Möbius seam (x=0 and x=Nx-1)."""
    xs = np.arange(NX)
    d0 = np.minimum(xs, NX - xs)  # distance to seam on the ring
    return d0 <= SEAM_WIN


def midplane_seam_frac(rho: np.ndarray) -> float:
    mid = rho[:, EDGE_TRIM : NY - EDGE_TRIM]
    col = mid.sum(axis=1)
    tot = float(col.sum()) + 1e-30
    return float(col[seam_mask()].sum() / tot)


def find_seam_modes(evals: np.ndarray, evecs: np.ndarray) -> list[int]:
    order = np.argsort(np.abs(evals.real))
    picked: list[int] = []
    for n in order:
        if abs(evals.real[n]) >= GAP_CUT:
            break
        rho = site_ldos(evecs[:, n])
        if midplane_seam_frac(rho) >= SEAM_FRAC:
            picked.append(int(n))
        if len(picked) >= 2:
            break
    return picked


def arc_means(C: np.ndarray) -> tuple[float, float]:
    """
    Bulk-y means on the two arcs meeting at the seam.
    Left arc: small x (approaching seam from +x side of cut chart).
    Right arc: large x.
    """
    y0, y1 = EDGE_TRIM, NY - EDGE_TRIM
    # stay a few columns off the exact seam sites
    left = C[1 : 1 + EDGE_TRIM + 2, y0:y1]
    right = C[NX - (EDGE_TRIM + 3) : NX - 1, y0:y1]
    return float(np.mean(left)), float(np.mean(right))


def write_note() -> None:
    (RESULTS / "mobius_sharp.md").write_text(NOTE_MD, encoding="utf-8")


def plot_spectrum(evals_c: np.ndarray, evals_m: np.ndarray, seam_idx: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, ev, title in (
        (axes[0], evals_c, "Cylinder"),
        (axes[1], evals_m, "Mobius (uniform lam)"),
    ):
        idx = np.arange(len(ev))
        ax.plot(idx, ev.real, "k.", ms=2)
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.set_ylim(-1.2, 1.2)
        ax.set_xlabel("state index")
        ax.set_title(title)
    if seam_idx:
        axes[1].plot(
            seam_idx,
            evals_m.real[seam_idx],
            "o",
            ms=8,
            color="C3",
            label="seam modes",
        )
        axes[1].legend(fontsize=8)
    axes[0].set_ylabel("E")
    fig.suptitle(
        "Exp 12: spectrum vs cylinder (prior art; Beugeling / Huang-Lee)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "12_spectrum_vs_cylinder.png", dpi=150)
    plt.close(fig)


def plot_ldos(evecs: np.ndarray, seam_idx: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if seam_idx:
        rho = np.sum([site_ldos(evecs[:, n]) for n in seam_idx], axis=0)
    else:
        rho = np.zeros((NX, NY))
    im = ax.imshow(rho.T, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.axvline(0, color="w", ls="--", lw=0.9)
    ax.axvline(NX - 1, color="w", ls="--", lw=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        "LDOS of Mobius seam doublet (Huang-Lee / Beugeling calibration)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "12_ldos_seam.png", dpi=150)
    plt.close(fig)


def plot_marker(C: np.ndarray, mean_l: float, mean_r: float) -> None:
    Cx = np.mean(C[:, EDGE_TRIM : NY - EDGE_TRIM], axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    im = axes[0].imshow(C.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(r"Mobius chart $C(\mathbf{r})$")
    xs = np.arange(NX)
    axes[1].plot(xs, Cx, "C0-")
    axes[1].axhline(0.0, color="0.5", lw=0.5)
    axes[1].axvline(0.5, color="C3", ls="--", lw=0.8, label="seam")
    axes[1].text(2, mean_l, f"arc L={mean_l:.2f}", fontsize=8, color="C3")
    axes[1].text(NX - 12, mean_r, f"arc R={mean_r:.2f}", fontsize=8, color="C3")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$C(x)$")
    axes[1].set_title("Opposite sign on arcs at seam (noisy OK)")
    axes[1].legend(fontsize=8)
    fig.suptitle("Exp 12 marker (Beugeling: QH needs a domain wall)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "12_marker_arcs.png", dpi=150)
    plt.close(fig)


def main() -> None:
    write_note()

    H_c = qwz.H_realspace(NX, NY, m=M, lam=LAM, bc_x="cylinder")
    H_m = qwz.H_realspace(NX, NY, m=M, lam=LAM, bc_x="mobius")
    evals_c, _ = spectr.eigh(H_c)
    evals_m, evecs_m = spectr.eigh(H_m)

    seam_idx = find_seam_modes(evals_m, evecs_m)
    plot_spectrum(evals_c, evals_m, seam_idx)
    plot_ldos(evecs_m, seam_idx)

    n_occ = int(np.sum(evals_m.real < -1e-12))
    if n_occ < H_m.shape[0] // 4:
        n_occ = spectr.n_occupied(H_m.shape[0])
    P = spectr.projector(evecs_m, n_occ)
    C = marker.local_chern(P, NX, NY, bc_x="mobius")
    mean_l, mean_r = arc_means(C)
    plot_marker(C, mean_l, mean_r)

    pass_modes = len(seam_idx) == 2
    pass_flip = mean_l * mean_r < 0 and abs(mean_l) >= ARC_MIN and abs(mean_r) >= ARC_MIN

    summary = {
        "Nx": NX,
        "Ny": NY,
        "lam": LAM,
        "seam_indices": seam_idx,
        "seam_energies": [float(evals_m.real[i]) for i in seam_idx],
        "n_seam_modes": len(seam_idx),
        "arc_mean_left": mean_l,
        "arc_mean_right": mean_r,
        "n_occ": n_occ,
        "pass_seam_modes": pass_modes,
        "pass_marker_flip": pass_flip,
        "note": "results/mobius_sharp.md",
        "citations": [
            "Beugeling, Quelle, Morais Smith, PRB 89, 235112 (2014); arXiv:1403.6998",
            "Huang & Lee, PRB 84, 193106 (2011); arXiv:1107.1411",
        ],
    }
    (RESULTS / "12_mobius_sharp.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 12 - Mobius sharp wall (prior art)")
    print(f"  seam modes: {len(seam_idx)} E={[float(evals_m.real[i]) for i in seam_idx]}")
    print(f"  arc means: L={mean_l:.3f} R={mean_r:.3f}")
    assert pass_modes, f"expected 2 seam modes, got {len(seam_idx)}"
    assert pass_flip, f"marker arc flip failed: L={mean_l}, R={mean_r}"
    print("  PASS")


if __name__ == "__main__":
    main()
