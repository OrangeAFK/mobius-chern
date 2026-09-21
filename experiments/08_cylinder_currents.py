"""Experiment 08 — Occupied-state edge currents on a cylinder."""

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

from src import currents, lattice, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
NX, NY = 32, 16


def analyze_currents(lam: float) -> dict:
    H = qwz.H_realspace(NX, NY, m=M, lam=lam, bc_x="cylinder")
    evals, evecs = spectr.eigh(H)
    # Near-zero edge crossings: occupy E < -eps to avoid half-filling ambiguity.
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    rho = spectr.projector(evecs, n_occ)
    bonds = currents.directed_bonds_from_geometry(NX, NY, "cylinder")
    I_list = currents.bond_currents(rho, H, bonds)

    edge_I = []
    bulk_I = []
    xs, ys, us, vs, cs = [], [], [], [], []
    for i_site, j_site, I in I_list:
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > NX // 2:
            dx = dx - np.sign(dx) * NX
        mx = 0.5 * (x1 + x2)
        my = 0.5 * (y1 + y2)
        xs.append(mx)
        ys.append(my)
        us.append(dx * I)
        vs.append(dy * I)
        cs.append(I)
        if dy == 0 and ((x2 - x1) % NX == 1 or (x1 == NX - 1 and x2 == 0)):
            if y1 <= 1 or y1 >= NY - 2:
                edge_I.append(I)
            elif 3 <= y1 <= NY - 4:
                bulk_I.append(I)

    mean_edge = float(np.mean(edge_I)) if edge_I else 0.0
    mean_bulk = float(np.mean(np.abs(bulk_I))) if bulk_I else 0.0
    bot_x = []
    top_x = []
    for i_site, j_site, I in I_list:
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        if y1 == y2 and ((x2 - x1) % NX == 1 or (x1 == NX - 1 and x2 == 0)):
            if y1 == 0:
                bot_x.append(I)
            if y1 == NY - 1:
                top_x.append(I)
    mean_bot = float(np.mean(bot_x)) if bot_x else 0.0
    mean_top = float(np.mean(top_x)) if top_x else 0.0

    fig, ax = plt.subplots(figsize=(9, 3.5))
    # subsample arrows for clarity
    step = 2
    for x, y, u, v, c in zip(xs[::step], ys[::step], us[::step], vs[::step], cs[::step]):
        if abs(u) + abs(v) < 1e-8:
            continue
        ax.quiver(
            x,
            y,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=0.15,
            width=0.004,
            color="C0" if c >= 0 else "C3",
            alpha=0.85,
        )
    ax.set_xlim(-1, NX)
    ax.set_ylim(-1, NY)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"Occupied bond currents, $\lambda={lam:+g}$")
    fig.tight_layout()
    tag = "p" if lam > 0 else "m"
    fig.savefig(FIGURES / f"08_currents_lam_{tag}.png", dpi=150)
    plt.close(fig)

    return {
        "lam": lam,
        "n_occ": n_occ,
        "mean_bot_x": mean_bot,
        "mean_top_x": mean_top,
        "mean_edge_x": mean_edge,
        "mean_abs_bulk_x": mean_bulk,
        "n_bot": len(bot_x),
        "n_top": len(top_x),
    }


def main() -> None:
    info_p = analyze_currents(+1.0)
    info_m = analyze_currents(-1.0)

    reverse = info_p["mean_bot_x"] * info_m["mean_bot_x"] < 0
    counter_p = info_p["mean_bot_x"] * info_p["mean_top_x"] < 0
    counter_m = info_m["mean_bot_x"] * info_m["mean_top_x"] < 0
    bulk_small = info_p["mean_abs_bulk_x"] < 0.2 * abs(info_p["mean_bot_x"]) + 1e-9

    summary = {
        "Nx": NX,
        "Ny": NY,
        "plus": info_p,
        "minus": info_m,
        "pass_reverses_with_lam": reverse,
        "pass_bulk_small": bulk_small,
        "pass_counter_propagating": counter_p and counter_m,
    }
    (RESULTS / "08_currents.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 08 - cylinder currents")
    print(
        f"  lam=+1 bot={info_p['mean_bot_x']:.4e} top={info_p['mean_top_x']:.4e} "
        f"bulk|I|={info_p['mean_abs_bulk_x']:.4e}"
    )
    print(
        f"  lam=-1 bot={info_m['mean_bot_x']:.4e} top={info_m['mean_top_x']:.4e} "
        f"bulk|I|={info_m['mean_abs_bulk_x']:.4e}"
    )
    assert reverse and counter_p and counter_m and bulk_small, "current pass criteria failed"
    print("  PASS")


if __name__ == "__main__":
    main()
