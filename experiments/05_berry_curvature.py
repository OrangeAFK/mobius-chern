"""Experiment 05 — Berry curvature of the valence band."""

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

from src import berry  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
NK = 64


def raw_chern(F: np.ndarray, Nk: int) -> float:
    dk = 2.0 * np.pi / Nk
    return float(np.sum(F) * dk * dk / (2.0 * np.pi))


def plot_F(lam: float, out: Path) -> dict:
    kx, ky, F = berry.berry_curvature_mesh(m=M, lam=lam, Nk=NK)
    C_raw = raw_chern(F, NK)
    vmax = float(np.max(np.abs(F)))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.pcolormesh(kx, ky, F, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label=r"$\mathcal{F}(\mathbf{k})$")
    ax.set_aspect("equal")
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_title(rf"Berry curvature, $m={M}$, $\lambda={lam:+g}$  ($C_{{\mathrm{{raw}}}}={C_raw:.3f}$)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return {
        "lam": lam,
        "C_raw": C_raw,
        "F_min": float(F.min()),
        "F_max": float(F.max()),
        "F_abs_max": vmax,
    }


def main() -> None:
    info_p = plot_F(+1.0, FIGURES / "05_berry_lam_p.png")
    info_m = plot_F(-1.0, FIGURES / "05_berry_lam_m.png")

    flip = info_p["C_raw"] * info_m["C_raw"] < 0
    near_one = abs(abs(info_p["C_raw"]) - 1.0) < 0.15 and abs(abs(info_m["C_raw"]) - 1.0) < 0.15

    summary = {
        "m": M,
        "Nk": NK,
        "plus": info_p,
        "minus": info_m,
        "pass_sign_flip": flip,
        "pass_integral_near_one": near_one,
    }
    (RESULTS / "05_berry.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 05 - Berry curvature")
    print(f"  C_raw(lam=+1)={info_p['C_raw']:.4f}, C_raw(lam=-1)={info_m['C_raw']:.4f}")
    assert flip and near_one, "Berry integral / sign flip failed"
    print("  PASS")


if __name__ == "__main__":
    main()
