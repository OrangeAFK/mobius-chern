"""Experiment 13 — Reproduce Huang-Lee currents (prior art; not a new claim)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import currents, geometry, lattice, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
LAM = 1.0
NX, NY = 40, 12
I_MIN_PLOT = 1e-4
EDGE_ECUT = 0.35
CITE_HL = "Huang & Lee, PRB 84, 193106 (2011); arXiv:1107.1411"
CITE_B = "Beugeling et al., PRB 89, 235112 (2014); arXiv:1403.6998"


def _bonds(bc_x: str):
    return currents.directed_bonds_from_geometry(NX, NY, bc_x)


def occupied_currents(bc_x: str) -> tuple[list[tuple[int, int, float]], int]:
    """Full valence projector (cylinder / Huang Fig. 1)."""
    H = qwz.H_realspace(NX, NY, m=M, lam=LAM, bc_x=bc_x)  # type: ignore[arg-type]
    evals, evecs = spectr.eigh(H)
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    rho = spectr.projector(evecs, n_occ)
    return currents.bond_currents(rho, H, _bonds(bc_x)), n_occ


def valence_edge_currents(bc_x: str) -> tuple[list[tuple[int, int, float]], int]:
    """
    Small-bias / in-gap occupied edge currents (PLAN Exp 13).

    On open and Mobius strips the full valence projector cancels bond currents
    numerically; the near-zero valence edge subspace carries the Huang-Lee pattern.
    """
    H = qwz.H_realspace(NX, NY, m=M, lam=LAM, bc_x=bc_x)  # type: ignore[arg-type]
    evals, evecs = spectr.eigh(H)
    sel = np.where((evals.real < -1e-12) & (np.abs(evals.real) < EDGE_ECUT))[0]
    if len(sel) < 2:
        sel = np.where(np.abs(evals.real) < EDGE_ECUT)[0]
        sel = sel[evals.real[sel] < 0]
    V = evecs[:, sel]
    rho = V @ V.conj().T
    return currents.bond_currents(rho, H, _bonds(bc_x)), int(len(sel))


def quiver_unrolled(
    I_list: list[tuple[int, int, float]],
    title: str,
    out: Path,
    *,
    step: int = 2,
    scale: float = 0.12,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.5))
    xs, ys, us, vs, cs = [], [], [], [], []
    for i_site, j_site, I in I_list:
        if abs(I) < I_MIN_PLOT:
            continue
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > NX // 2:
            dx -= int(np.sign(dx)) * NX
        # seam hop on Mobius: large |dy| — draw as short wrap marker
        if abs(dy) > NY // 2:
            dy = int(np.sign(dy)) * min(abs(dy), 2)
        xs.append(0.5 * (x1 + x2) if abs(x2 - x1) <= NX // 2 else x1)
        ys.append(0.5 * (y1 + y2))
        us.append(dx * I)
        vs.append(dy * I)
        cs.append(I)
    for x, y, u, v, c in zip(xs[::step], ys[::step], us[::step], vs[::step], cs[::step]):
        if abs(u) + abs(v) < 1e-10:
            continue
        ax.quiver(
            x,
            y,
            u,
            v,
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=0.004,
            color="C0" if c >= 0 else "C3",
            alpha=0.85,
        )
    ax.set_xlim(-1, NX)
    ax.set_ylim(-1, NY)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def edge_Ix_stats(I_list: list[tuple[int, int, float]]) -> dict:
    bot, top, bulk = [], [], []
    for i_site, j_site, I in I_list:
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        if y1 != y2:
            continue
        if not ((x2 - x1) % NX == 1 or (x1 == NX - 1 and x2 == 0)):
            continue
        if y1 == 0:
            bot.append(I)
        elif y1 == NY - 1:
            top.append(I)
        elif 3 <= y1 <= NY - 4:
            bulk.append(I)
    return {
        "mean_bot": float(np.mean(bot)) if bot else 0.0,
        "mean_top": float(np.mean(top)) if top else 0.0,
        "mean_abs_bulk": float(np.mean(np.abs(bulk))) if bulk else 0.0,
    }


def cut_Iy_stats(I_list: list[tuple[int, int, float]]) -> dict:
    """+y currents on left (x=0) and right (x=Nx-1) edges; Huang co-prop after y-flip."""
    left, right = [], []
    bulk = []
    for i_site, j_site, I in I_list:
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        if x1 != x2 or y2 != y1 + 1:
            continue
        if x1 == 0:
            left.append((y1, I))
        elif x1 == NX - 1:
            right.append((y1, I))
        elif 3 <= x1 <= NX - 4 and 3 <= y1 <= NY - 4:
            bulk.append(I)

    mean_L = float(np.mean([I for _, I in left])) if left else 0.0
    # Mobius identification y -> Ny-1-y flips the sense of +y on the right cut
    mean_R = float(np.mean([I for _, I in right])) if right else 0.0
    mean_R_flip = -mean_R
    return {
        "mean_Iy_left": mean_L,
        "mean_Iy_right": mean_R,
        "mean_Iy_right_yflip": mean_R_flip,
        "mean_abs_bulk_Iy": float(np.mean(np.abs(bulk))) if bulk else 0.0,
        "n_left": len(left),
        "n_right": len(right),
    }


def seam_current_stats(I_list: list[tuple[int, int, float]]) -> dict:
    """Currents near Mobius seam columns vs bulk interior."""
    seam, edge, bulk = [], [], []
    for i_site, j_site, I in I_list:
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        mx = 0.5 * (x1 + x2)
        # seam neighborhood: columns near 0 / Nx-1, or the wrapping bond
        near_seam = min(x1, x2, NX - 1 - x1, NX - 1 - x2) <= 1 or (
            (x1 == NX - 1 and x2 == 0) or (x2 == NX - 1 and x1 == 0)
        )
        on_open_edge = y1 <= 1 or y1 >= NY - 2 or y2 <= 1 or y2 >= NY - 2
        if near_seam:
            seam.append(abs(I))
        elif on_open_edge:
            edge.append(abs(I))
        elif 4 <= min(x1, x2) <= NX - 5 and 3 <= min(y1, y2) <= NY - 4:
            bulk.append(abs(I))
    return {
        "mean_abs_seam": float(np.mean(seam)) if seam else 0.0,
        "mean_abs_edge": float(np.mean(edge)) if edge else 0.0,
        "mean_abs_bulk": float(np.mean(bulk)) if bulk else 0.0,
        "n_seam": len(seam),
    }


def plot_3d_mobius(I_list: list[tuple[int, int, float]], out: Path) -> None:
    X, Y, Z, _n = geometry.embed_lattice_mobius(NX, NY)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    # faint lattice
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), c="0.7", s=4, depthshade=False)
    # current segments
    shown = 0
    for i_site, j_site, I in I_list:
        if abs(I) < 3e-4:
            continue
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        p0 = np.array([X[x1, y1], Y[x1, y1], Z[x1, y1]])
        p1 = np.array([X[x2, y2], Y[x2, y2], Z[x2, y2]])
        # skip huge visual jumps across embed discontinuity if any
        if np.linalg.norm(p1 - p0) > 1.2:
            continue
        color = "C0" if I >= 0 else "C3"
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=color,
            lw=0.8 + 40.0 * abs(I),
            alpha=0.85,
        )
        shown += 1
        if shown > 800:
            break
    ax.set_title(
        f"Sealed Mobius currents (Huang-Lee Fig. 3 analogue)\n{CITE_HL}"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_network_cartoon(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("equal")
    # two loops (counter-circulating) meeting at junction S
    # left loop
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(3.2 + 1.6 * np.cos(t), 3.0 + 1.8 * np.sin(t), "C0-", lw=2.0)
    # right loop
    ax.plot(6.8 + 1.6 * np.cos(t), 3.0 + 1.8 * np.sin(t), "C3-", lw=2.0)
    # junction box
    ax.add_patch(plt.Rectangle((4.6, 2.4), 0.8, 1.2, fill=True, facecolor="0.92", edgecolor="k", lw=1.2))
    ax.text(5.0, 3.0, r"$S$", ha="center", va="center", fontsize=14)
    # arrows: opposite circulation
    ax.annotate("", xy=(3.2, 4.7), xytext=(2.2, 4.2), arrowprops=dict(arrowstyle="->", color="C0", lw=1.5))
    ax.annotate("", xy=(3.2, 1.3), xytext=(4.2, 1.8), arrowprops=dict(arrowstyle="->", color="C0", lw=1.5))
    ax.annotate("", xy=(6.8, 4.7), xytext=(7.8, 4.2), arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))
    ax.annotate("", xy=(6.8, 1.3), xytext=(5.8, 1.8), arrowprops=dict(arrowstyle="->", color="C3", lw=1.5))
    # wall channels (two co-propagating into S)
    ax.annotate("", xy=(4.6, 3.4), xytext=(3.8, 3.8), arrowprops=dict(arrowstyle="->", color="0.2", lw=1.2))
    ax.annotate("", xy=(4.6, 2.6), xytext=(3.8, 2.2), arrowprops=dict(arrowstyle="->", color="0.2", lw=1.2))
    ax.annotate("", xy=(6.2, 3.8), xytext=(5.4, 3.4), arrowprops=dict(arrowstyle="->", color="0.2", lw=1.2))
    ax.annotate("", xy=(6.2, 2.2), xytext=(5.4, 2.6), arrowprops=dict(arrowstyle="->", color="0.2", lw=1.2))
    ax.text(3.2, 5.4, "loop A", ha="center", color="C0")
    ax.text(6.8, 5.4, "loop B (opposite)", ha="center", color="C3")
    ax.text(5.0, 0.6, "Two-channel chiral network (NOT a single racetrack)", ha="center", fontsize=10)
    ax.set_axis_off()
    ax.set_title(
        f"Exp 13 network cartoon (prior art)\n{CITE_HL}; {CITE_B}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    # --- 1. Cylinder (Huang Fig. 1) ---
    I_cyl, n_cyl = occupied_currents("cylinder")
    st_cyl = edge_Ix_stats(I_cyl)
    quiver_unrolled(
        I_cyl,
        f"Cylinder edge currents (Huang-Lee Fig. 1 analogue)\n{CITE_HL}",
        FIGURES / "13_currents_cylinder.png",
    )
    pass_cyl = st_cyl["mean_bot"] * st_cyl["mean_top"] < 0 and abs(st_cyl["mean_bot"]) > 1e-6

    # --- 2. Cut open (Huang Fig. 2) ---
    I_cut, n_cut = valence_edge_currents("open")
    st_cut = cut_Iy_stats(I_cut)
    quiver_unrolled(
        I_cut,
        f"Cut-open strip: co-propagating on cut (Huang-Lee Fig. 2 analogue)\n"
        f"{CITE_HL}; QH needs a wall: {CITE_B}",
        FIGURES / "13_currents_cut_open.png",
        scale=0.08,
    )
    # co-prop after y-flip: mean_L and mean_R_flip same sign; both >> bulk
    co = st_cut["mean_Iy_left"] * st_cut["mean_Iy_right_yflip"] > 0
    strong = (
        abs(st_cut["mean_Iy_left"]) > 3.0 * st_cut["mean_abs_bulk_Iy"] + 1e-12
        and abs(st_cut["mean_Iy_right"]) > 3.0 * st_cut["mean_abs_bulk_Iy"] + 1e-12
    )
    pass_cut = co and strong

    # --- 3. Sealed Mobius (Huang Fig. 3) ---
    I_mob, n_mob = valence_edge_currents("mobius")
    st_mob = seam_current_stats(I_mob)
    quiver_unrolled(
        I_mob,
        f"Sealed Mobius: twist persists (Huang-Lee Fig. 3 analogue)\n{CITE_HL}",
        FIGURES / "13_currents_sealed.png",
        scale=0.08,
    )
    plot_3d_mobius(I_mob, FIGURES / "13_currents_mobius_3d.png")
    pass_seal = st_mob["mean_abs_seam"] > 3.0 * st_mob["mean_abs_bulk"] + 1e-12

    plot_network_cartoon(FIGURES / "13_network_cartoon.png")

    summary = {
        "Nx": NX,
        "Ny": NY,
        "lam": LAM,
        "cylinder": {**st_cyl, "n_occ": n_cyl, "pass": pass_cyl},
        "cut_open": {**st_cut, "n_occ": n_cut, "pass_co_prop_yflip": co, "pass_strong": strong, "pass": pass_cut},
        "sealed_mobius": {**st_mob, "n_occ": n_mob, "pass": pass_seal},
        "pass_all": pass_cyl and pass_cut and pass_seal,
        "citations": [CITE_HL, CITE_B],
        "note": "Reproduction of Huang-Lee / Beugeling; not a novelty claim.",
    }
    (RESULTS / "13_huang_lee.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 13 - Huang-Lee currents (prior art)")
    print(
        f"  cylinder: bot={st_cyl['mean_bot']:.4e} top={st_cyl['mean_top']:.4e} "
        f"counter={pass_cyl}"
    )
    print(
        f"  cut: IyL={st_cut['mean_Iy_left']:.4e} IyR_flip={st_cut['mean_Iy_right_yflip']:.4e} "
        f"co={co} strong={strong}"
    )
    print(
        f"  sealed: |I|_seam={st_mob['mean_abs_seam']:.4e} "
        f"|I|_bulk={st_mob['mean_abs_bulk']:.4e}"
    )
    assert pass_cyl, "cylinder counter-propagating failed"
    assert pass_cut, "cut-open co-propagating (Huang Fig. 2) failed"
    assert pass_seal, "sealed Mobius seam currents (Huang Fig. 3) failed"
    print("  PASS")


if __name__ == "__main__":
    main()
