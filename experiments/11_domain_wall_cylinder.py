"""Experiment 11 — Mass / chirality domain wall on a cylinder."""

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

from src import currents, lattice, marker, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
LAM0 = 1.0
NX, NY = 40, 12
X0 = NX / 4.0
WS = (0.5, 2.0, 5.0)
PRIMARY_W = 2.0
EDGE_TRIM = 2
GAP_CUT = 0.5
WALL_MID_FRAC = 0.85
WALL_WIN = 3
CX_SIDE_MIN = 0.3


def site_ldos(psi: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """Orbital-summed |psi|^2 on sites, shape (Nx, Ny)."""
    dens = np.abs(psi).reshape(Nx * Ny, 2) ** 2
    site = dens.sum(axis=1)
    return site.reshape(Nx, Ny)


def wall_columns(lam_x: np.ndarray) -> tuple[int, int]:
    """Two columns nearest to lambda zeros (one per half-circuit)."""
    Nx = len(lam_x)
    half = Nx // 2
    i0 = int(np.argmin(np.abs(lam_x[:half])))
    i1 = half + int(np.argmin(np.abs(lam_x[half:])))
    return i0, i1


def circular_window(Nx: int, center: int, half_w: int) -> np.ndarray:
    xs = np.arange(Nx)
    d = np.minimum((xs - center) % Nx, (center - xs) % Nx)
    return d <= half_w


def midplane_wall_frac(rho: np.ndarray, w0: int, w1: int) -> float:
    """Fraction of mid-strip LDOS sitting near either wall."""
    mid = rho[:, EDGE_TRIM : NY - EDGE_TRIM]
    col = mid.sum(axis=1)
    tot = float(col.sum()) + 1e-30
    mask = circular_window(NX, w0, WALL_WIN) | circular_window(NX, w1, WALL_WIN)
    return float(col[mask].sum() / tot)


def orbital_mask_near_wall(wall_col: int) -> np.ndarray:
    """Boolean mask on orbital Hilbert space for columns near wall_col."""
    N = lattice.dim(NX, NY)
    mask = np.zeros(N, dtype=bool)
    xs = circular_window(NX, wall_col, WALL_WIN)
    for x in np.where(xs)[0]:
        for y in range(NY):
            site = lattice.site_index(int(x), y, NY)
            for orb in range(2):
                mask[lattice.orb_index(site, orb)] = True
    return mask


def classify_wall_modes(
    evals: np.ndarray,
    evecs: np.ndarray,
    lam_x: np.ndarray,
    w: float,
) -> dict:
    """
    Pick the four in-gap wall modes (hybridized across the two walls), then
    rotate the subspace to localize two modes on each wall.
    """
    del w  # width only used by caller for plots / JSON
    w0, w1 = wall_columns(lam_x)
    order = np.argsort(np.abs(evals.real))
    picked: list[int] = []
    for n in order:
        if abs(evals.real[n]) >= GAP_CUT:
            break
        rho = site_ldos(evecs[:, n], NX, NY)
        if midplane_wall_frac(rho, w0, w1) >= WALL_MID_FRAC:
            picked.append(int(n))
        if len(picked) >= 4:
            break

    modes: list[dict] = []
    if len(picked) == 4:
        V = evecs[:, picked]
        m0 = orbital_mask_near_wall(w0).astype(float)
        M0 = V.conj().T @ (m0[:, None] * V)
        M0 = 0.5 * (M0 + M0.conj().T)
        weights, U = np.linalg.eigh(M0)
        order_w = np.argsort(weights.real)[::-1]
        # largest two wall-0 weights -> wall 0; remaining -> wall 1
        assign = [(0, order_w[0]), (0, order_w[1]), (1, order_w[2]), (1, order_w[3])]
        for wall_id, ui in assign:
            coeff = U[:, ui]
            psi = V @ coeff
            # representative energy: expectation in original eigenbasis
            E = float(np.real(coeff.conj() @ np.diag(evals.real[picked]) @ coeff))
            modes.append(
                {
                    "index": int(picked[int(np.argmax(np.abs(coeff)))]),
                    "E": E,
                    "wall": wall_id,
                    "ldos": site_ldos(psi, NX, NY),
                    "psi": psi,
                }
            )

    return {
        "wall_cols": (int(w0), int(w1)),
        "picked": picked,
        "modes": modes,
        "n_modes": len(modes),
        "n_per_wall": (
            sum(1 for m in modes if m["wall"] == 0),
            sum(1 for m in modes if m["wall"] == 1),
        ),
    }


def wall_y_currents(H, psis: list[np.ndarray], wall_col: int) -> list[float]:
    """Mean +y bond current near wall_col for each wavefunction."""
    bonds: list[tuple[int, int]] = []
    for dx in (-1, 0, 1):
        x = (wall_col + dx) % NX
        for y in range(NY - 1):
            i = lattice.site_index(x, y, NY)
            j = lattice.site_index(x, y + 1, NY)
            bonds.append((i, j))

    means: list[float] = []
    for psi in psis:
        rho = np.outer(psi, psi.conj())
        I_list = currents.bond_currents(rho, H, bonds)
        means.append(float(np.mean([I for _, _, I in I_list])))
    return means


def cx_plateau_means(Cx: np.ndarray, lam_x: np.ndarray, w0: int, w1: int) -> dict:
    """
    Bulk Cx means on the two lambda plateaus (away from walls and X chart cut).
    """
    xs = np.arange(NX)
    # exclude chart cut columns and wall neighborhoods
    half_w = max(3, int(np.ceil(2.5)))
    near_wall = circular_window(NX, w0, half_w) | circular_window(NX, w1, half_w)
    chart = (xs < EDGE_TRIM) | (xs >= NX - EDGE_TRIM)
    bulk = ~(near_wall | chart)

    pos = bulk & (lam_x > 0.3)
    neg = bulk & (lam_x < -0.3)
    mean_pos = float(np.mean(Cx[pos])) if np.any(pos) else float("nan")
    mean_neg = float(np.mean(Cx[neg])) if np.any(neg) else float("nan")
    return {
        "mean_Cx_lam_pos": mean_pos,
        "mean_Cx_lam_neg": mean_neg,
        "n_pos": int(np.sum(pos)),
        "n_neg": int(np.sum(neg)),
    }


def plot_spectrum(evals: np.ndarray, picked: list[int], w: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.arange(len(evals))
    ax.plot(idx, evals.real, "k.", ms=2, label="spectrum")
    if picked:
        ax.plot(picked, evals.real[picked], "o", ms=7, color="C3", label="wall modes")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_xlabel("state index")
    ax.set_ylabel("E")
    ax.set_title(rf"Cylinder domain wall, $w={w:g}$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_ldos(modes: list[dict], w: float, wall_cols: tuple[int, int], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    if modes:
        rho = np.sum([m["ldos"] for m in modes], axis=0)
    else:
        rho = np.zeros((NX, NY))
    im = ax.imshow(rho.T, origin="lower", aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for xc in wall_cols:
        ax.axvline(xc, color="w", ls="--", lw=0.9, alpha=0.8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(rf"Wall-mode LDOS sum, $w={w:g}$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cx(lam_x: np.ndarray, Cx: np.ndarray, w: float, wall_cols: tuple[int, int], out: Path) -> None:
    xs = np.arange(NX)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(xs, lam_x, "C0-", label=r"$\lambda(x)$")
    ax.plot(xs, Cx, "C3-", label=r"$C(x)$")
    for xc in wall_cols:
        ax.axvline(xc, color="0.4", ls="--", lw=0.8)
    ax.axhline(0.0, color="0.5", lw=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\lambda$, $C$")
    ax.set_title(rf"Marker flip across walls, $w={w:g}$")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cartoon(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.set_aspect("equal")
    # strip
    ax.add_patch(plt.Rectangle((0.5, 0.8), 9, 2.4, fill=False, lw=1.5))
    # walls
    for x, label in ((3.0, r"$|\Delta C|=2$"), (7.5, r"$|\Delta C|=2$")):
        ax.plot([x, x], [0.8, 3.2], "C3-", lw=2.5)
        ax.annotate("", xy=(x, 3.0), xytext=(x, 1.0), arrowprops=dict(arrowstyle="->", color="C0", lw=1.5))
        ax.annotate("", xy=(x + 0.25, 3.0), xytext=(x + 0.25, 1.0), arrowprops=dict(arrowstyle="->", color="C0", lw=1.5))
        ax.text(x, 3.5, label, ha="center", fontsize=9)
    ax.text(1.5, 2.0, r"$C=+1$", fontsize=10)
    ax.text(4.8, 2.0, r"$C=-1$", fontsize=10)
    ax.text(8.5, 2.0, r"$C=+1$", fontsize=10)
    ax.set_axis_off()
    ax.set_title("Two walls, two co-propagating channels each")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def run_one(w: float) -> dict:
    lam_x = qwz.lambda_tanh(NX, X0, w, lam0=LAM0, antiperiodic=False)
    H = qwz.H_realspace(NX, NY, m=M, bc_x="cylinder", lam_x=lam_x)
    evals, evecs = spectr.eigh(H)

    info = classify_wall_modes(evals, evecs, lam_x, w)
    w0, w1 = info["wall_cols"]
    modes = info["modes"]

    # co-propagating: same-sign <Iy> for the two modes on each wall
    co_ok = True
    co_detail: dict = {}
    for wall_id, col in ((0, w0), (1, w1)):
        psis = [m["psi"] for m in modes if m["wall"] == wall_id]
        if len(psis) < 2:
            co_ok = False
            co_detail[f"wall{wall_id}"] = {"n": len(psis), "Iy": []}
            continue
        Iy = wall_y_currents(H, psis[:2], col)
        same = (Iy[0] * Iy[1] > 0) and (abs(Iy[0]) > 1e-10) and (abs(Iy[1]) > 1e-10)
        co_ok = co_ok and same
        co_detail[f"wall{wall_id}"] = {"n": len(psis), "Iy": Iy, "same_sign": same}

    # marker
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    P = spectr.projector(evecs, n_occ)
    C = marker.local_chern(P, NX, NY, bc_x="cylinder")
    # y-average excluding open edges for Cx used in flip check
    C_bulk_y = C[:, EDGE_TRIM : NY - EDGE_TRIM]
    Cx = np.mean(C_bulk_y, axis=1)
    plateaus = cx_plateau_means(Cx, lam_x, w0, w1)
    flip = (
        plateaus["mean_Cx_lam_pos"] * plateaus["mean_Cx_lam_neg"] < 0
        and abs(plateaus["mean_Cx_lam_pos"]) >= CX_SIDE_MIN
        and abs(plateaus["mean_Cx_lam_neg"]) >= CX_SIDE_MIN
    )

    tag = f"{w:g}".replace(".", "p")
    plot_spectrum(evals, info["picked"], w, FIGURES / f"11_spectrum_w{tag}.png")
    plot_ldos(modes, w, (w0, w1), FIGURES / f"11_ldos_w{tag}.png")
    plot_cx(lam_x, Cx, w, (w0, w1), FIGURES / f"11_Cx_w{tag}.png")

    n_modes = info["n_modes"]
    n0, n1 = info["n_per_wall"]
    pass_modes = n_modes == 4 and n0 == 2 and n1 == 2

    return {
        "w": w,
        "wall_cols": [w0, w1],
        "n_modes": n_modes,
        "n_per_wall": [n0, n1],
        "picked": info["picked"],
        "mode_energies": [float(evals.real[i]) for i in info["picked"]],
        "co_propagating": co_ok,
        "co_detail": co_detail,
        "plateaus": plateaus,
        "Cx_flip": flip,
        "n_occ": n_occ,
        "pass_modes": pass_modes,
        "pass_co": co_ok,
        "pass_flip": flip,
        "Cx": Cx.tolist(),
        "lam_x": lam_x.tolist(),
    }


def main() -> None:
    plot_cartoon(FIGURES / "11_cartoon_walls.png")
    results = [run_one(w) for w in WS]
    primary = next(r for r in results if abs(r["w"] - PRIMARY_W) < 1e-12)

    summary = {
        "Nx": NX,
        "Ny": NY,
        "x0": X0,
        "ws": list(WS),
        "primary_w": PRIMARY_W,
        "by_w": [
            {
                "w": r["w"],
                "wall_cols": r["wall_cols"],
                "n_modes": r["n_modes"],
                "n_per_wall": r["n_per_wall"],
                "mode_energies": r["mode_energies"],
                "co_propagating": r["co_propagating"],
                "co_detail": r["co_detail"],
                "plateaus": r["plateaus"],
                "Cx_flip": r["Cx_flip"],
                "pass_modes": r["pass_modes"],
                "pass_co": r["pass_co"],
                "pass_flip": r["pass_flip"],
            }
            for r in results
        ],
        "pass_modes": primary["pass_modes"],
        "pass_co": primary["pass_co"],
        "pass_flip": primary["pass_flip"],
    }
    (RESULTS / "11_domain_wall.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 11 - domain wall cylinder")
    for r in results:
        print(
            f"  w={r['w']:g}: modes={r['n_modes']} "
            f"(per wall {r['n_per_wall']}), co={r['co_propagating']}, "
            f"Cx_flip={r['Cx_flip']}"
        )
        print(
            f"    plateaus Cx(lam+)= {r['plateaus']['mean_Cx_lam_pos']:.3f}, "
            f"Cx(lam-)= {r['plateaus']['mean_Cx_lam_neg']:.3f}"
        )
    assert primary["pass_modes"], (
        f"expected 4 wall modes (2+2) at w={PRIMARY_W}, got "
        f"{primary['n_modes']} {primary['n_per_wall']}"
    )
    assert primary["pass_co"], f"co-propagating check failed at w={PRIMARY_W}"
    assert primary["pass_flip"], f"C(x) flip failed at w={PRIMARY_W}"
    print("  PASS")


if __name__ == "__main__":
    main()
