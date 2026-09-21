"""Experiment 03 — QWZ as a map k → d̂(k)."""

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

from src import berry, lattice  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
NK = 48


def plot_dhat(lam: float, out: Path) -> dict:
    kx, ky = lattice.bz_mesh(NK)
    d = berry.dhat(kx, ky, m=M, lam=lam)
    labels = [r"$\hat{d}_x$", r"$\hat{d}_y$", r"$\hat{d}_z$"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    for ax, i, lab in zip(axes.ravel()[:3], range(3), labels):
        im = ax.pcolormesh(kx, ky, d[..., i], shading="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(lab)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$k_x$")
        ax.set_ylabel(r"$k_y$")

    # RGB map of d̂ mapped to [0,1]
    rgb = 0.5 * (d + 1.0)
    rgb = np.clip(rgb, 0, 1)
    ax = axes[1, 1]
    ax.imshow(
        np.transpose(rgb, (1, 0, 2)),
        origin="lower",
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        aspect="equal",
    )
    # quiver subsample
    step = max(NK // 12, 1)
    ax.quiver(
        kx[::step, ::step],
        ky[::step, ::step],
        d[::step, ::step, 0],
        d[::step, ::step, 1],
        color="k",
        scale=15,
        width=0.003,
    )
    ax.set_title(r"RGB($\hat{d}$) + quiver $d_x,d_y$")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    fig.suptitle(rf"QWZ $\hat{{d}}(\mathbf{{k}})$, $m={M}$, $\lambda={lam:+g}$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return {
        "lam": lam,
        "d_x_mean": float(d[..., 0].mean()),
        "d_y_mean": float(d[..., 1].mean()),
        "d_z_mean": float(d[..., 2].mean()),
    }


def main() -> None:
    info_p = plot_dhat(+1.0, FIGURES / "03_dhat_lam_p.png")
    info_m = plot_dhat(-1.0, FIGURES / "03_dhat_lam_m.png")

    # orientation foreshadow: flipping λ flips d_y and reverses texture orientation
    kx, ky = lattice.bz_mesh(NK)
    dp = berry.dhat(kx, ky, m=M, lam=+1.0)
    dm = berry.dhat(kx, ky, m=M, lam=-1.0)
    # d_y(λ=-1) ≈ -d_y(λ=+1); d_x,z unchanged
    dy_flip = float(np.max(np.abs(dm[..., 1] + dp[..., 1])))
    dx_same = float(np.max(np.abs(dm[..., 0] - dp[..., 0])))

    summary = {
        "m": M,
        "plus": info_p,
        "minus": info_m,
        "max_abs_dy_plus_minus": dy_flip,
        "max_abs_dx_diff": dx_same,
        "pass_dy_reflection": dy_flip < 1e-12 and dx_same < 1e-12,
    }
    (RESULTS / "03_dhat.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 03 - QWZ d-hat textures")
    print(f"  d_y flip residual={dy_flip:.2e}, d_x same residual={dx_same:.2e}")
    assert summary["pass_dy_reflection"], "lam -> -lam should flip d_y only"
    print("  PASS (textures related by d_y reflection / orientation reversal)")


if __name__ == "__main__":
    main()
