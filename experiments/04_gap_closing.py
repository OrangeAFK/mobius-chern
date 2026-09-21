"""Experiment 04 — Gap closings and the mass parameter."""

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

from src import berry, lattice, qwz  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

LAM = 1.0
NK = 64
M_GRID = np.linspace(-3.0, 3.0, 121)
CRITICAL = (-2.0, 0.0, 2.0)


def min_gap(m: float, lam: float = LAM, Nk: int = NK) -> float:
    kx, ky = lattice.bz_mesh(Nk)
    d = qwz.d_vector(kx, ky, m=m, lam=lam)
    # E± = ±|d|, gap = 2 min |d|
    return float(2.0 * np.min(np.linalg.norm(d, axis=-1)))


def main() -> None:
    gaps = np.array([min_gap(m) for m in M_GRID])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(M_GRID, gaps, color="C0", lw=1.5)
    for mc in CRITICAL:
        ax.axvline(mc, color="C3", ls="--", lw=0.9)
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\min_{\mathbf{k}}|E_+-E_-|$")
    ax.set_title(rf"QWZ gap vs $m$ ($\lambda={LAM:g}$)")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_gap_vs_m.png", dpi=150)
    plt.close(fig)

    # d̂ just below / above m=0
    m_lo, m_hi = -0.2, 0.2
    kx, ky = lattice.bz_mesh(40)
    d_lo = berry.dhat(kx, ky, m=m_lo, lam=LAM)
    d_hi = berry.dhat(kx, ky, m=m_hi, lam=LAM)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, d, mval in ((axes[0], d_lo, m_lo), (axes[1], d_hi, m_hi)):
        im = ax.pcolormesh(kx, ky, d[..., 2], shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\hat{d}_z$")
        ax.set_aspect("equal")
        ax.set_title(rf"$m={mval:+.1f}$")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")
    fig.suptitle(r"$\hat{d}_z$ across the $m=0$ transition")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_dhat_across_transition.png", dpi=150)
    plt.close(fig)

    gaps_at_crit = {str(mc): min_gap(mc, Nk=96) for mc in CRITICAL}
    # also sample near critical on fine grid of M_GRID
    near = {}
    for mc in CRITICAL:
        idx = int(np.argmin(np.abs(M_GRID - mc)))
        near[str(mc)] = float(gaps[idx])

    dz_G_lo = float(berry.dhat(0.0, 0.0, m=m_lo, lam=LAM)[2])
    dz_G_hi = float(berry.dhat(0.0, 0.0, m=m_hi, lam=LAM)[2])

    pass_close = all(g < 0.05 for g in gaps_at_crit.values())
    summary = {
        "lam": LAM,
        "gaps_at_critical": gaps_at_crit,
        "gaps_on_grid_near_critical": near,
        "dhat_z_Gamma": {"m_lo": dz_G_lo, "m_hi": dz_G_hi},
        "pass_gap_closes": pass_close,
        "pass_texture_changes": dz_G_lo * dz_G_hi < 0 or abs(dz_G_lo - dz_G_hi) > 0.1,
    }
    (RESULTS / "04_gap.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 04 - gap closings")
    for mc, g in gaps_at_crit.items():
        print(f"  gap(m={mc})={g:.4e}")
    print(f"  dhat_z(Gamma): m={m_lo} -> {dz_G_lo:.3f}, m={m_hi} -> {dz_G_hi:.3f}")
    assert summary["pass_gap_closes"], "gap did not close at m=0,+/-2"
    print("  PASS")


if __name__ == "__main__":
    main()
