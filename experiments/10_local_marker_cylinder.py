"""Experiment 10 — Bianco–Resta local Chern marker on a cylinder."""

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

from src import marker, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
NX, NY = 24, 12
EDGE_TRIM = 2
BULK_TOL = 0.25


def bulk_mean(C: np.ndarray) -> float:
    """Exclude open edges in y and the X-chart branch cut near x=0, Nx-1."""
    return float(np.mean(C[EDGE_TRIM : NX - EDGE_TRIM, EDGE_TRIM : NY - EDGE_TRIM]))


def run_marker(lam: float) -> dict:
    H = qwz.H_realspace(NX, NY, m=M, lam=lam, bc_x="cylinder")
    evals, evecs = spectr.eigh(H)
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    P = spectr.projector(evecs, n_occ)
    C = marker.local_chern(P, NX, NY, bc_x="cylinder")
    Cx = marker.average_Cx(C)
    bm = bulk_mean(C)
    residual = abs(bm - float(np.sign(lam)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    im = axes[0].imshow(C.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title(rf"$C(\mathbf{{r}})$, $\lambda={lam:+g}$")

    axes[1].plot(np.arange(NX), Cx, color="C0")
    axes[1].axhline(float(np.sign(lam)), color="0.4", ls="--", lw=0.8)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$C(x)$")
    axes[1].set_title(f"y-average; bulk mean={bm:.3f}")
    fig.tight_layout()
    tag = "p" if lam > 0 else "m"
    fig.savefig(FIGURES / f"10_marker_lam_{tag}.png", dpi=150)
    plt.close(fig)

    edge_mean = float(np.mean(np.abs(C[:, :EDGE_TRIM]))) + float(
        np.mean(np.abs(C[:, NY - EDGE_TRIM :]))
    )
    return {
        "lam": lam,
        "bulk_mean": bm,
        "residual_vs_sgn": residual,
        "Cx": Cx.tolist(),
        "edge_abs_mean": edge_mean / 2.0,
        "C_min": float(C.min()),
        "C_max": float(C.max()),
    }


def main() -> None:
    info_p = run_marker(+1.0)
    info_m = run_marker(-1.0)

    pass_p = abs(info_p["bulk_mean"] - 1.0) < BULK_TOL
    pass_m = abs(info_m["bulk_mean"] - (-1.0)) < BULK_TOL

    summary = {
        "Nx": NX,
        "Ny": NY,
        "edge_trim": EDGE_TRIM,
        "plus": info_p,
        "minus": info_m,
        "pass_bulk_plus": pass_p,
        "pass_bulk_minus": pass_m,
    }
    (RESULTS / "10_marker.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 10 - local marker cylinder")
    print(f"  bulk C(lam=+1)={info_p['bulk_mean']:.3f} (residual {info_p['residual_vs_sgn']:.3f})")
    print(f"  bulk C(lam=-1)={info_m['bulk_mean']:.3f} (residual {info_m['residual_vs_sgn']:.3f})")
    assert pass_p and pass_m, "marker bulk plateau pass failed"
    print("  PASS")


if __name__ == "__main__":
    main()
