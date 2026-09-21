"""Experiment 02 — Real-space diagonalization of square-lattice TB."""

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

from src import lattice, spectr, tb  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

T = 1.0
NX, NY = 24, 16
BAND = 4.0 * T
EPS = 1e-8


def psi_density(psi: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    return np.abs(psi).reshape(Nx, Ny) ** 2


def dominant_kx(psi: np.ndarray, Nx: int, Ny: int) -> tuple[int, float]:
    """FFT along x of mid-y strip; return (n, kx)."""
    dens = psi.reshape(Nx, Ny)
    # take complex amplitude along a bulk row
    y0 = Ny // 2
    fk = np.fft.fft(dens[:, y0])
    n = int(np.argmax(np.abs(fk)))
    if n > Nx // 2:
        n -= Nx
    kx = 2.0 * np.pi * n / Nx
    return n, float(kx)


def main() -> None:
    H_open = tb.H_realspace(NX, NY, t=T, bc_x="open")
    H_cyl = tb.H_realspace(NX, NY, t=T, bc_x="cylinder")
    ev_o, vec_o = spectr.eigh(H_open)
    ev_c, vec_c = spectr.eigh(H_cyl)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, ev, title in (
        (axes[0], ev_o, "open"),
        (axes[1], ev_c, "cylinder"),
    ):
        ax.plot(np.arange(len(ev)), ev.real, "|", color="C0", ms=6)
        ax.axhline(-BAND, color="0.4", ls="--", lw=0.8)
        ax.axhline(+BAND, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel("level index")
        ax.set_title(title)
    axes[0].set_ylabel(r"$E$")
    fig.suptitle("Real-space spectrum vs infinite-lattice band edges")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_spectrum.png", dpi=150)
    plt.close(fig)

    mid = len(ev_o) // 2
    # near lower band edge
    edge_idx = 1
    dens_mid = psi_density(vec_o[:, mid], NX, NY)
    dens_edge = psi_density(vec_o[:, edge_idx], NX, NY)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, dens, title in (
        (axes[0], dens_mid, f"mid-spectrum n={mid}"),
        (axes[1], dens_edge, f"near-edge n={edge_idx}"),
    ):
        im = ax.imshow(dens.T, origin="lower", aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
    fig.suptitle(r"Ordinary $|\psi|^2$ (open sample)")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_psi_density.png", dpi=150)
    plt.close(fig)

    n_qx, kx = dominant_kx(vec_c[:, mid], NX, NY)
    in_band_open = bool(np.all(ev_o.real >= -BAND - EPS) and np.all(ev_o.real <= BAND + EPS))
    in_band_cyl = bool(np.all(ev_c.real >= -BAND - EPS) and np.all(ev_c.real <= BAND + EPS))

    summary = {
        "Nx": NX,
        "Ny": NY,
        "t": T,
        "E_min_open": float(ev_o.real.min()),
        "E_max_open": float(ev_o.real.max()),
        "E_min_cylinder": float(ev_c.real.min()),
        "E_max_cylinder": float(ev_c.real.max()),
        "band_edges": [-BAND, BAND],
        "cylinder_mid_n": n_qx,
        "cylinder_mid_kx": kx,
        "pass_inside_band": in_band_open and in_band_cyl,
    }
    (RESULTS / "02_realspace.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 02 - real-space TB")
    print(f"  open E in [{ev_o.real.min():.4f},{ev_o.real.max():.4f}]")
    print(f"  cyl  E in [{ev_c.real.min():.4f},{ev_c.real.max():.4f}]")
    print(f"  cylinder mid-state dominant n={n_qx}, kx={kx:.4f} (~ 2*pi*n/Nx)")
    assert summary["pass_inside_band"], "eigenvalues outside infinite-lattice band"
    print("  PASS")


if __name__ == "__main__":
    main()
