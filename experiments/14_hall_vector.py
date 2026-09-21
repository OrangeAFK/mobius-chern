"""Experiment 14 — Hall vector field h = C n-hat on the embedded Mobius strip."""

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

from src import geometry, marker, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
LAM0 = 1.0
NX, NY = 40, 12
X0 = NX / 4.0
WS = (0.5, 2.0, 5.0)
EDGE_TRIM = 2
PLATEAU_TOL = 0.35
PRIMARY_W = 2.0


def circular_dist(xs: np.ndarray, center: int, Nx: int) -> np.ndarray:
    return np.minimum((xs - center) % Nx, (center - xs) % Nx)


def wall_column(lam_x: np.ndarray) -> int:
    return int(np.argmin(np.abs(lam_x)))


def compute_C(w: float) -> tuple[np.ndarray, np.ndarray]:
    lam_x = qwz.lambda_tanh(NX, X0, w, lam0=LAM0, antiperiodic=True)
    H = qwz.H_realspace(NX, NY, m=M, bc_x="mobius", lam_x=lam_x)
    evals, evecs = spectr.eigh(H)
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    P = spectr.projector(evecs, n_occ)
    C = marker.local_chern(P, NX, NY, bc_x="mobius")
    return C, lam_x


def plateau_means(C: np.ndarray, lam_x: np.ndarray, w: float) -> dict:
    """
    Plateaus in the first chart half (unflipped Y), away from wall/seam.
    Expect Cx ~ sgn(lambda).
    """
    xs = np.arange(NX)
    wall = wall_column(lam_x)
    half_w = max(2, int(np.ceil(1.5 * w)))
    near_wall = circular_dist(xs, wall, NX) <= half_w
    first_half = (xs > EDGE_TRIM) & (xs < NX // 2 - 2)
    Cx = np.mean(C[:, EDGE_TRIM : NY - EDGE_TRIM], axis=1)
    safe = first_half & ~near_wall
    pos = safe & (lam_x > 0.5)
    neg = safe & (lam_x < -0.5)
    mean_pos = float(np.mean(Cx[pos])) if np.any(pos) else float("nan")
    mean_neg = float(np.mean(Cx[neg])) if np.any(neg) else float("nan")
    return {
        "wall_col": wall,
        "Cx": Cx.tolist(),
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "residual_pos": abs(mean_pos - 1.0) if mean_pos == mean_pos else 999.0,
        "residual_neg": abs(mean_neg - (-1.0)) if mean_neg == mean_neg else 999.0,
        "n_pos": int(np.sum(pos)),
        "n_neg": int(np.sum(neg)),
    }


def h_wall_vs_plateau(h: np.ndarray, C: np.ndarray, lam_x: np.ndarray, w: float) -> dict:
    """Soft band where |Cx| is small near the lambda zero; |h| suppressed vs plateaus."""
    xs = np.arange(NX)
    wall = wall_column(lam_x)
    Cx = np.mean(C[:, EDGE_TRIM : NY - EDGE_TRIM], axis=1)
    mag = np.linalg.norm(h, axis=-1)
    mag_mid = mag[:, EDGE_TRIM : NY - EDGE_TRIM]
    soft = (
        (np.abs(Cx) < 0.45)
        & (xs > EDGE_TRIM)
        & (xs < NX - EDGE_TRIM)
        & (circular_dist(xs, wall, NX) <= max(3, int(np.ceil(2 * w)) + 1))
    )
    first = (xs > EDGE_TRIM) & (xs < NX // 2 - 2)
    plat = first & (np.abs(lam_x) > 0.7) & (np.abs(Cx) > 0.6)
    near_m = float(mag_mid[soft, :].mean()) if np.any(soft) else 0.0
    plat_m = float(mag_mid[plat, :].mean()) if np.any(plat) else 1.0
    return {
        "mean_abs_h_wall": near_m,
        "mean_abs_h_plateau": plat_m,
        "soft_width": int(np.sum(soft)),
        "wall_suppressed": bool(np.any(soft) and near_m < 0.7 * plat_m),
    }


def seam_h_continuity(h: np.ndarray, C: np.ndarray) -> dict:
    """
    Continuity through the seam: |h| stays O(1) and matches on both inward
    columns. Scalar C shows the branch-cut jump; |h| does not.
    """
    x_l, x_r = 2, NX - 3
    y0, y1 = EDGE_TRIM, NY - EDGE_TRIM
    mag_l = np.linalg.norm(h[x_l, y0:y1], axis=-1)
    mag_r = np.linalg.norm(h[x_r, y0:y1], axis=-1)
    Cx = np.mean(C[:, y0:y1], axis=1)
    # magnitude continuity
    mag_rel = float(np.mean(np.abs(mag_l - mag_r)) / (0.5 * (mag_l.mean() + mag_r.mean()) + 1e-12))
    # scalar C jumps across seam ends (branch cut); |h| should not collapse
    c_jump = abs(float(Cx[x_l] - Cx[x_r]))
    h_ok = float(mag_l.mean()) > 0.4 and float(mag_r.mean()) > 0.4
    return {
        "mag_rel_diff": mag_rel,
        "C_jump_across_seam": c_jump,
        "mean_abs_h_L": float(mag_l.mean()),
        "mean_abs_h_R": float(mag_r.mean()),
        "continuous": bool(h_ok and mag_rel < 0.5),
    }


def plot_hall_vector_signed(X, Y, Z, h, C, w: float, out: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, color="0.88", alpha=0.3, linewidth=0)
    step_x, step_y = 2, 2
    for x in range(0, NX, step_x):
        for y in range(EDGE_TRIM, NY - EDGE_TRIM, step_y):
            vec = h[x, y]
            nrm = float(np.linalg.norm(vec))
            if nrm < 0.08:
                ax.scatter([X[x, y]], [Y[x, y]], [Z[x, y]], c="k", s=8, depthshade=False)
                continue
            color = "C0" if C[x, y] >= 0 else "C3"
            ax.quiver(
                X[x, y],
                Y[x, y],
                Z[x, y],
                vec[0],
                vec[1],
                vec[2],
                length=0.22,
                normalize=True,
                color=color,
                arrow_length_ratio=0.4,
                linewidth=0.7,
            )
    ax.set_title(
        rf"$\mathbf{{h}}=C\hat{{n}}$ on Mobius ($w={w:g}$)"
        "\nMain fig: chart-free Hall field (zeros = wall)"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_scalar_branch_cut(X, Y, Z, C, out: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    Cclip = np.clip(C, -1.5, 1.5)
    norm = plt.Normalize(vmin=-1.5, vmax=1.5)
    colors = plt.cm.RdBu_r(norm(Cclip))
    ax.plot_surface(
        X, Y, Z, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
    )
    ax.set_title(
        "WRONG: scalar C on the embedding (seam branch cut)\n"
        "Retire this; use h = C n-hat instead"
    )
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_Cx(all_Cx: dict[float, list], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(NX)
    for w, Cx in all_Cx.items():
        ax.plot(xs, Cx, label=rf"$w={w:g}$")
    ax.axhline(1.0, color="0.5", ls="--", lw=0.7)
    ax.axhline(-1.0, color="0.5", ls="--", lw=0.7)
    ax.axhline(0.0, color="0.4", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$C(x)$")
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(r"$C(x)=\langle C\rangle_y$ for antiperiodic $\tanh$ walls")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    X, Y, Z, normals = geometry.embed_lattice_mobius(NX, NY)
    by_w = {}
    all_Cx: dict[float, list] = {}
    C_for_wrong = None

    for w in WS:
        C, lam_x = compute_C(w)
        plat = plateau_means(C, lam_x, w)
        h = marker.hall_vector(C, normals)
        hw = h_wall_vs_plateau(h, C, lam_x, w)
        cont = seam_h_continuity(h, C)
        tag = f"{w:g}".replace(".", "p")
        plot_hall_vector_signed(X, Y, Z, h, C, w, FIGURES / f"14_hall_vector_w{tag}.png")
        all_Cx[w] = plat["Cx"]
        if abs(w - PRIMARY_W) < 1e-12:
            C_for_wrong = C
        by_w[str(w)] = {
            "wall_col": plat["wall_col"],
            "mean_pos": plat["mean_pos"],
            "mean_neg": plat["mean_neg"],
            "residual_pos": plat["residual_pos"],
            "residual_neg": plat["residual_neg"],
            "n_pos": plat["n_pos"],
            "n_neg": plat["n_neg"],
            "h_wall": hw,
            "seam_continuity": cont,
            "pass_plateau": bool(
                plat["n_pos"] > 0
                and plat["n_neg"] > 0
                and plat["residual_pos"] < PLATEAU_TOL
                and plat["residual_neg"] < PLATEAU_TOL
            ),
            "pass_h_vanishes": bool(hw["wall_suppressed"]),
            "pass_h_continuous": bool(cont["continuous"]),
        }

    plot_Cx(all_Cx, FIGURES / "14_Cx_vs_w.png")
    if C_for_wrong is not None:
        plot_scalar_branch_cut(X, Y, Z, C_for_wrong, FIGURES / "14_scalar_C_branch_cut.png")

    primary = by_w[str(PRIMARY_W)]
    pass_all = bool(
        primary["pass_plateau"]
        and primary["pass_h_vanishes"]
        and primary["pass_h_continuous"]
    )
    summary = {
        "Nx": NX,
        "Ny": NY,
        "x0": X0,
        "ws": list(WS),
        "primary_w": PRIMARY_W,
        "by_w": by_w,
        "pass": pass_all,
    }
    (RESULTS / "14_hall_vector.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 14 - Hall vector h = C n-hat")
    for w in WS:
        info = by_w[str(w)]
        print(
            f"  w={w:g}: C+={info['mean_pos']:.3f} C-={info['mean_neg']:.3f} "
            f"|h|_wall={info['h_wall']['mean_abs_h_wall']:.3f} "
            f"|h|_plat={info['h_wall']['mean_abs_h_plateau']:.3f} "
            f"soft_w={info['h_wall']['soft_width']} "
            f"mag_rel={info['seam_continuity']['mag_rel_diff']:.3f}"
        )
    assert primary["pass_plateau"], "plateau +/-1 failed"
    assert primary["pass_h_vanishes"], "h does not vanish on wall band"
    assert primary["pass_h_continuous"], "h not continuous through seam"
    print("  PASS")


if __name__ == "__main__":
    main()
