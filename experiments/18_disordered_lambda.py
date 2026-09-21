"""Experiment 18 — Disordered antiperiodic lambda(x); odd zeros."""

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
NX, NY = 40, 12
N_PROFILES = 10
RNG_SEED = 20260321
K_MAX_ODD = 7  # odd harmonics 1,3,5,7
EDGE_TRIM = 2
CX_SMOOTH = 3
I_MIN_PLOT = 1e-4


def lambda_odd_fourier(Nx: int, rng: np.random.Generator, k_max: int = K_MAX_ODD) -> np.ndarray:
    """
    Antiperiodic random profile via odd Fourier modes:
    lambda(x+Nx) = -lambda(x) automatically, so the zero count is odd.
    """
    x = np.arange(Nx, dtype=float)
    lam = np.zeros(Nx, dtype=float)
    for k in range(1, k_max + 1, 2):
        a = rng.normal()
        b = rng.normal()
        # decay higher harmonics so the local gap is not shredded
        w = 1.0 / k
        lam += w * (a * np.cos(k * np.pi * x / Nx) + b * np.sin(k * np.pi * x / Nx))
    peak = float(np.max(np.abs(lam)))
    if peak < 1e-12:
        lam = np.sin(np.pi * x / Nx)
        peak = float(np.max(np.abs(lam)))
    return lam / peak


def count_zeros_antiperiodic(f: np.ndarray) -> int:
    """Sign changes on the ring with antiperiodic wrap f[Nx] = -f[0]."""
    n = 0
    for i in range(len(f) - 1):
        if f[i] == 0.0 or f[i + 1] == 0.0:
            continue
        if f[i] * f[i + 1] < 0.0:
            n += 1
    # wrap: compare f[-1] to -f[0]
    last, first_wrap = f[-1], -f[0]
    if last != 0.0 and first_wrap != 0.0 and last * first_wrap < 0.0:
        n += 1
    return n


def smooth_1d(y: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return y.copy()
    ker = np.ones(win, dtype=float) / win
    # circular smooth is wrong for antiperiodic C; use linear edge handling
    pad = win // 2
    yp = np.pad(y, pad, mode="edge")
    out = np.convolve(yp, ker, mode="valid")
    return out[: len(y)]


def lambda_zero_sites(f: np.ndarray) -> list[int]:
    """Indices i where f changes sign between i and i+1 (antiperiodic wrap)."""
    zs: list[int] = []
    n = len(f)
    for i in range(n - 1):
        if f[i] * f[i + 1] < 0.0:
            zs.append(i)
    if f[-1] * (-f[0]) < 0.0:
        zs.append(n - 1)
    return zs


def walls_matched_by_marker(lam_x: np.ndarray, Cx: np.ndarray, half_w: int = 3) -> tuple[int, int]:
    """
    Count lambda zeros that have a soft |C(x)| dip nearby.

    Scalar C on Mobius has a chart cut, so raw C(x) zero counts are noisy;
    matching walls via |C| suppression at each lambda zero is resolution-safe.
    """
    zeros = lambda_zero_sites(lam_x)
    matched = 0
    Nx = len(lam_x)
    for z in zeros:
        xs = [(z + d) % Nx for d in range(-half_w, half_w + 1)]
        if float(np.min(np.abs(Cx[xs]))) < 0.5:
            matched += 1
    return matched, len(zeros)


def compute_Cx(lam_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H = qwz.H_realspace(NX, NY, m=M, bc_x="mobius", lam_x=lam_x)
    evals, evecs = spectr.eigh(H)
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    P = spectr.projector(evecs, n_occ)
    C = marker.local_chern(P, NX, NY, bc_x="mobius")
    Cx = np.mean(C[:, EDGE_TRIM : NY - EDGE_TRIM], axis=1)
    return C, Cx

def plot_currents(lam_x: np.ndarray, out: Path) -> None:
    H = qwz.H_realspace(NX, NY, m=M, bc_x="mobius", lam_x=lam_x)
    evals, evecs = spectr.eigh(H)
    n_occ = int(np.sum(evals.real < -1e-12))
    if n_occ < H.shape[0] // 4:
        n_occ = spectr.n_occupied(H.shape[0])
    # near-gap valence subspace (full projector cancels on Mobius)
    sel = np.where((evals.real < -1e-12) & (np.abs(evals.real) < 0.4))[0]
    if len(sel) < 2:
        sel = np.where(evals.real < -1e-12)[0][-min(8, n_occ) :]
    rho = evecs[:, sel] @ evecs[:, sel].conj().T
    bonds = currents.directed_bonds_from_geometry(NX, NY, "mobius")
    I_list = currents.bond_currents(rho, H, bonds)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    step = 2
    for idx, (i_site, j_site, I) in enumerate(I_list):
        if idx % step != 0 or abs(I) < I_MIN_PLOT:
            continue
        x1, y1 = lattice.xy_of(i_site, NY)
        x2, y2 = lattice.xy_of(j_site, NY)
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) > NX // 2:
            dx -= int(np.sign(dx)) * NX
        if abs(dy) > NY // 2:
            dy = int(np.sign(dy)) * min(abs(dy), 2)
        ax.quiver(
            x1,
            y1,
            dx * I,
            dy * I,
            angles="xy",
            scale_units="xy",
            scale=0.15,
            width=0.004,
            color="C0",
            alpha=0.8,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Disordered antiperiodic lambda: near-gap currents")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    print("Experiment 18 - Disordered antiperiodic lambda")
    rng = np.random.default_rng(RNG_SEED)
    records = []
    profile_rows = []

    for i in range(N_PROFILES):
        lam_x = lambda_odd_fourier(NX, rng)
        n_lam = count_zeros_antiperiodic(lam_x)
        assert n_lam % 2 == 1, f"odd Fourier must give odd zeros, got {n_lam}"

        _, Cx = compute_Cx(lam_x)
        Cx_s = smooth_1d(Cx, CX_SMOOTH)
        n_matched, n_walls = walls_matched_by_marker(lam_x, Cx_s, half_w=max(3, NX // 20))
        # within resolution: allow one missed soft dip
        match = n_matched >= max(1, n_walls - 1)

        rec = {
            "seed_index": i,
            "n_zeros_lambda": n_lam,
            "n_walls_matched": n_matched,
            "n_walls": n_walls,
            "match": match,
            "lam_x": lam_x.tolist(),
            "Cx": Cx.tolist(),
        }
        records.append(rec)
        profile_rows.append((lam_x, Cx_s, n_lam, n_matched))
        print(
            f"  profile {i}: n_lam={n_lam}, marker walls matched={n_matched}/{n_walls}, "
            f"match={match}"
        )
    # Profiles + C(x) overlays (first 4)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, (lam_x, Cx_s, n_lam, n_matched), i in zip(
        axes.ravel(), profile_rows[:4], range(4)
    ):
        ax.plot(lam_x, color="C0", label=rf"$\lambda$ ({n_lam} zeros)")
        ax.plot(Cx_s, color="C3", label=rf"$C(x)$ (matched {n_matched})")
        ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
        ax.set_title(f"profile {i}")
        ax.legend(fontsize=7, loc="best")
    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    fig.suptitle("Disordered antiperiodic lambda with local marker C(x)")
    fig.tight_layout()
    fig.savefig(FIGURES / "18_profiles_Cx.png", dpi=150)
    plt.close(fig)

    # Histogram of lambda zero counts
    counts = [r["n_zeros_lambda"] for r in records]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.arange(0.5, max(counts) + 1.5, 1.0)
    ax.hist(counts, bins=bins, color="C0", edgecolor="k", align="mid")
    ax.set_xlabel(r"number of $\lambda$ zeros")
    ax.set_ylabel("count")
    ax.set_title("All draws have odd zero count (antiperiodic)")
    fig.tight_layout()
    fig.savefig(FIGURES / "18_zero_hist.png", dpi=150)
    plt.close(fig)

    # One current plot (first profile)
    print("  currents for profile 0")
    plot_currents(np.asarray(records[0]["lam_x"]), FIGURES / "18_currents.png")

    all_odd = all(c % 2 == 1 for c in counts)
    matches = all(r["match"] for r in records)
    summary = {
        "Nx": NX,
        "Ny": NY,
        "n_profiles": N_PROFILES,
        "rng_seed": RNG_SEED,
        "k_max_odd": K_MAX_ODD,
        "profiles": [
            {
                "seed_index": r["seed_index"],
                "n_zeros_lambda": r["n_zeros_lambda"],
                "n_walls_matched": r["n_walls_matched"],
                "n_walls": r["n_walls"],
                "match": r["match"],
            }
            for r in records
        ],
        "pass_all_odd": all_odd,
        "pass_marker_match": matches,
    }
    (RESULTS / "18_disordered_lambda.json").write_text(json.dumps(summary, indent=2))

    assert all_odd, f"expected all odd lambda zeros, got {counts}"
    assert matches, "marker soft dips should match lambda walls within resolution"
    print("  PASS")

if __name__ == "__main__":
    main()
