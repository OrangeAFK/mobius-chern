"""Experiment 06 — Chern number and the (m, λ) phase diagram."""

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

from src.berry import chern_number  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

NK = 48
M_VALS = np.linspace(-3.0, 3.0, 41)
LAM_VALS = np.array([-1.0, 1.0])


def main() -> None:
    C = np.zeros((len(LAM_VALS), len(M_VALS)), dtype=int)
    for i, lam in enumerate(LAM_VALS):
        for j, m in enumerate(M_VALS):
            C[i, j] = chern_number(m=float(m), lam=float(lam), Nk=NK)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(
        C,
        origin="lower",
        aspect="auto",
        extent=[M_VALS[0], M_VALS[-1], -0.5, len(LAM_VALS) - 0.5],
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )
    ax.set_yticks(range(len(LAM_VALS)))
    ax.set_yticklabels([rf"$\lambda={lam:+g}$" for lam in LAM_VALS])
    ax.set_xlabel(r"$m$")
    ax.set_title("QWZ Chern number phase diagram")
    fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], label=r"$C$")
    for mc in (-2, 0, 2):
        ax.axvline(mc, color="k", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "06_phase_diagram.png", dpi=150)
    plt.close(fig)

    labeled = {
        "(m,lam)=(1,+1)": chern_number(1.0, 1.0, Nk=NK),
        "(m,lam)=(1,-1)": chern_number(1.0, -1.0, Nk=NK),
        "(m,lam)=(3,+1)": chern_number(3.0, 1.0, Nk=NK),
        "(m,lam)=(-3,+1)": chern_number(-3.0, 1.0, Nk=NK),
        "(m,lam)=(0.5,+1)": chern_number(0.5, 1.0, Nk=NK),
        "(m,lam)=(0.5,-1)": chern_number(0.5, -1.0, Nk=NK),
    }

    # pass: |m|<2 lobe has C=sgn(lam); large |m| trivial
    lobe_ok = labeled["(m,lam)=(1,+1)"] == 1 and labeled["(m,lam)=(1,-1)"] == -1
    lobe_ok = lobe_ok and labeled["(m,lam)=(0.5,+1)"] == 1 and labeled["(m,lam)=(0.5,-1)"] == -1
    trivial_ok = labeled["(m,lam)=(3,+1)"] == 0 and labeled["(m,lam)=(-3,+1)"] == 0

    summary = {
        "Nk": NK,
        "m_grid": M_VALS.tolist(),
        "lam_grid": LAM_VALS.tolist(),
        "C_grid": C.tolist(),
        "labeled": labeled,
        "pass_lobe": lobe_ok,
        "pass_trivial": trivial_ok,
    }
    (RESULTS / "06_chern.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 06 - Chern phase diagram")
    for k, v in labeled.items():
        print(f"  C{k} = {v}")
    assert lobe_ok and trivial_ok, "phase diagram pass criteria failed"
    print("  PASS")


if __name__ == "__main__":
    main()
