"""Experiment 01 — Square-lattice tight-binding dispersion."""

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

from src import lattice, tb  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

T = 1.0
NK = 96


def main() -> None:
    kx_path, ky_path, s = lattice.high_symmetry_path(80)
    E_path = tb.dispersion(kx_path, ky_path, t=T)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(s, E_path, color="C0", lw=1.5)
    ax.axhline(0.0, color="0.5", lw=0.6)
    # mark high-symmetry points
    n = 80
    ticks_s = [s[0], s[n], s[2 * n], s[-1]]
    ax.set_xticks(ticks_s)
    ax.set_xticklabels([r"$\Gamma$", r"$X$", r"$M$", r"$\Gamma$"])
    ax.set_ylabel(r"$E(\mathbf{k})$")
    ax.set_title("Square-lattice NN tight-binding band path")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_band_path.png", dpi=150)
    plt.close(fig)

    kx, ky = lattice.bz_mesh(NK)
    E = tb.dispersion(kx, ky, t=T)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.pcolormesh(kx, ky, E, shading="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label=r"$E(\mathbf{k})$")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_aspect("equal")
    ax.set_title("Dispersion heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES / "01_dispersion_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(E.ravel(), bins=80, density=True, color="C0", alpha=0.85)
    ax.axvline(0.0, color="C3", ls="--", lw=1.0, label="van Hove")
    ax.set_xlabel(r"$E$")
    ax.set_ylabel("DOS (arb.)")
    ax.set_title("Density of states (k-grid)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "01_dos.png", dpi=150)
    plt.close(fig)

    E_G = float(tb.dispersion(0.0, 0.0, t=T))
    E_X = float(tb.dispersion(np.pi, 0.0, t=T))
    E_M = float(tb.dispersion(np.pi, np.pi, t=T))
    # DOS peak near E=0: count fraction of states in a window
    dos_peak_frac = float(np.mean(np.abs(E) < 0.15))

    summary = {
        "t": T,
        "E_Gamma": E_G,
        "E_X": E_X,
        "E_M": E_M,
        "band_edges": [float(E.min()), float(E.max())],
        "dos_near_zero_frac": dos_peak_frac,
        "pass_extrema": abs(E_G - (-4 * T)) < 1e-12 and abs(E_M - (4 * T)) < 1e-12,
        "pass_saddle_X": abs(E_X) < 1e-12,
    }
    (RESULTS / "01_dispersion.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 01 - square TB dispersion")
    print(f"  E(Gamma)={E_G:.6f}, E(X)={E_X:.6f}, E(M)={E_M:.6f}")
    print(f"  band [{E.min():.4f}, {E.max():.4f}]; DOS near 0 frac={dos_peak_frac:.3f}")
    assert summary["pass_extrema"], "extrema at Gamma/M failed"
    assert summary["pass_saddle_X"], "saddle at X failed"
    print("  PASS")


if __name__ == "__main__":
    main()
