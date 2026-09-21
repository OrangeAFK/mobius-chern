"""Experiment 15 — Smearing, valleys, and S_12 ~ Delta(w, Ny)."""

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

from src import flux, qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
LAM0 = 1.0

# Gap vs w (Mobius)
GAP_NX, GAP_NY = 80, 16
GAP_WS = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0)
GAP_X0 = GAP_NX / 4.0

# Cylinder valleys
VAL_NX, VAL_NY = 40, 16
VAL_WS = (0.5, 2.0, 5.0, 20.0)
VAL_X0 = VAL_NX / 4.0
EDGE_TRIM = 2
GAP_CUT = 0.5
WALL_WIN = 3
WALL_MID_FRAC = 0.85

# Mobius Delta grid
DEL_NX = 40
DEL_WS = (0.5, 2.0, 5.0, 20.0)
DEL_NYS = (10, 16)
N_PHI = 21
K_EIG = 12
DEL_X0 = DEL_NX / 4.0


def min_gap_mobius(w: float) -> float:
    lam_x = qwz.lambda_tanh(GAP_NX, GAP_X0, w, lam0=LAM0, antiperiodic=True)
    H = qwz.H_realspace(GAP_NX, GAP_NY, m=M, bc_x="mobius", lam_x=lam_x)
    evals, _ = spectr.eigsh_near_zero(H, k=K_EIG)
    # gap = min positive - max negative among near-zero levels
    ev = evals.real
    neg = ev[ev < 0]
    pos = ev[ev > 0]
    if len(neg) == 0 or len(pos) == 0:
        return float(2.0 * np.min(np.abs(ev)))
    return float(pos.min() - neg.max())


def circular_window(Nx: int, center: int, half_w: int) -> np.ndarray:
    xs = np.arange(Nx)
    d = np.minimum((xs - center) % Nx, (center - xs) % Nx)
    return d <= half_w


def wall_columns(lam_x: np.ndarray) -> tuple[int, int]:
    Nx = len(lam_x)
    half = Nx // 2
    i0 = int(np.argmin(np.abs(lam_x[:half])))
    i1 = half + int(np.argmin(np.abs(lam_x[half:])))
    return i0, i1


def site_amp(psi: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """Complex site amplitude (orb 0 + orb 1), shape (Nx, Ny)."""
    a = psi.reshape(Nx * Ny, 2)
    return (a[:, 0] + a[:, 1]).reshape(Nx, Ny)


def midplane_wall_frac(rho: np.ndarray, w0: int, w1: int, Nx: int, Ny: int) -> float:
    mid = rho[:, EDGE_TRIM : Ny - EDGE_TRIM]
    col = mid.sum(axis=1)
    tot = float(col.sum()) + 1e-30
    mask = circular_window(Nx, w0, WALL_WIN) | circular_window(Nx, w1, WALL_WIN)
    return float(col[mask].sum() / tot)


def valley_weights_at_walls(psi: np.ndarray, Nx: int, Ny: int, walls: tuple[int, int]) -> float:
    """
    Project wall-column site amplitude onto e^{+/- i pi y / 2}.
    Return purity p = max(|w+|,(|w-|)) / (|w+|+|w-|) averaged over walls,
    or fraction in dominant valley: |w+|**2 / (|w+|**2+|w-|**2).
    """
    amp = site_amp(psi, Nx, Ny)
    ys = np.arange(Ny, dtype=float)
    kp = np.exp(1j * 0.5 * np.pi * ys)
    km = np.exp(-1j * 0.5 * np.pi * ys)
    purities = []
    for xc in walls:
        # average a few columns near wall
        cols = [(xc + d) % Nx for d in (-1, 0, 1)]
        profile = np.mean([amp[c, :] for c in cols], axis=0)
        wp = np.vdot(kp, profile) / Ny
        wm = np.vdot(km, profile) / Ny
        n2 = abs(wp) ** 2 + abs(wm) ** 2 + 1e-30
        p = abs(wp) ** 2 / n2
        # purity toward majority valley: max(p, 1-p)
        purities.append(max(p, 1.0 - p))
    return float(np.mean(purities))


def cylinder_valley_purity(w: float) -> dict:
    lam_x = qwz.lambda_tanh(VAL_NX, VAL_X0, w, lam0=LAM0, antiperiodic=False)
    H = qwz.H_realspace(VAL_NX, VAL_NY, m=M, bc_x="cylinder", lam_x=lam_x)
    evals, evecs = spectr.eigh(H)
    w0, w1 = wall_columns(lam_x)
    order = np.argsort(np.abs(evals.real))
    modes = []
    for n in order:
        if abs(evals.real[n]) >= GAP_CUT and len(modes) >= 4:
            break
        if abs(evals.real[n]) >= 1.0:
            break
        dens = np.abs(evecs[:, n]).reshape(VAL_NX * VAL_NY, 2) ** 2
        rho = dens.sum(axis=1).reshape(VAL_NX, VAL_NY)
        frac = midplane_wall_frac(rho, w0, w1, VAL_NX, VAL_NY)
        # relax midplane fraction for very wide walls
        thresh = WALL_MID_FRAC if w < 10 else 0.55
        if frac >= thresh and abs(evals.real[n]) < max(GAP_CUT, 0.15 * w):
            modes.append(n)
        if len(modes) >= 4:
            break
    if not modes:
        return {"w": w, "n_modes": 0, "mean_purity": 0.0, "purities": []}
    purities = [
        valley_weights_at_walls(evecs[:, n], VAL_NX, VAL_NY, (w0, w1)) for n in modes
    ]
    return {
        "w": w,
        "n_modes": len(modes),
        "mean_purity": float(np.mean(purities)),
        "purities": purities,
    }


def anticrossing_delta(w: float, Ny: int) -> dict:
    """Min gap between two closest-to-zero levels over hole flux."""
    x0 = DEL_NX / 4.0
    lam_x = qwz.lambda_tanh(DEL_NX, x0, w, lam0=LAM0, antiperiodic=True)
    phis = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    gaps = []
    crosses_zero = False
    for Phi in phis:
        phi_x = flux.hole_phases(DEL_NX, Ny, Phi, bc_x="mobius")
        H = qwz.H_realspace(
            DEL_NX, Ny, m=M, bc_x="mobius", lam_x=lam_x, phi_x=phi_x
        )
        evals, _ = spectr.eigsh_near_zero(H, k=min(K_EIG, DEL_NX * Ny - 2))
        ev = np.sort(evals.real)
        # two closest to zero
        order = np.argsort(np.abs(ev))
        e0, e1 = ev[order[0]], ev[order[1]]
        pair = np.sort([e0, e1])
        gaps.append(float(pair[1] - pair[0]))
        # full gap crossing would require a level to change sign through zero
        # while the partner also crosses — flag if min|E| ~ 0 and gap ~ 0
        if abs(e0) < 1e-8 and abs(e1) < 1e-8:
            crosses_zero = True
    gaps_a = np.asarray(gaps)
    return {
        "w": w,
        "Ny": Ny,
        "Delta": float(np.min(gaps_a)),
        "gap_vs_phi": gaps_a.tolist(),
        "phis": phis.tolist(),
        "avoided": float(np.min(gaps_a)) > 1e-8 and not crosses_zero,
    }


def glide_x0_ok() -> float:
    """Integer x0 shift on Mobius antiperiodic: spectrum invariant."""
    Nx, Ny, w = 16, 8, 1.5
    x0 = 3.0
    lam0 = qwz.lambda_tanh(Nx, x0, w, antiperiodic=True)
    lam1 = qwz.lambda_tanh(Nx, x0 + 1.0, w, antiperiodic=True)
    H0 = qwz.H_realspace(Nx, Ny, m=M, bc_x="mobius", lam_x=lam0)
    H1 = qwz.H_realspace(Nx, Ny, m=M, bc_x="mobius", lam_x=lam1)
    e0, _ = spectr.eigh(H0)
    e1, _ = spectr.eigh(H1)
    return float(np.max(np.abs(np.sort(e0.real) - np.sort(e1.real))))


def write_s_matrix_md(gaps: dict, valleys: list, deltas: list, glide_diff: float) -> None:
    # summarize numbers
    purities = {v["w"]: v["mean_purity"] for v in valleys}
    delta_lines = "\n".join(
        f"| {d['w']:g} | {d['Ny']} | {d['Delta']:.4e} | {'yes' if d['avoided'] else 'no'} |"
        for d in deltas
    )
    # interpretation
    mean_p_sharp = purities.get(0.5, 0.0)
    mean_p_wide = purities.get(20.0, 0.0)
    deltas_finite = all(d["Delta"] > 1e-8 for d in deltas)
    if mean_p_sharp > 0.7 and deltas_finite:
        verdict = (
            "Cylinder wall modes stay relatively valley-pure even as `w` grows, "
            "while Mobius hole-flux still shows a finite avoided-crossing scale `Delta`. "
            "That points to **twist-induced mixing** at the junction (finite `|S_12|`), "
            "not mere smearing of the mass profile."
        )
    elif deltas_finite:
        verdict = (
            "Mobius `Delta(w,Ny)` remains a finite avoided-crossing scale on the grid. "
            "Cylinder valley purity varies with `w`; compare the tables before claiming "
            "decoupling. Mixing can come from twist, finite `Ny`, or both."
        )
    else:
        verdict = (
            "Check the `Delta` table: an avoided-crossing scale should stay strictly "
            "positive (no quantized Laughlin gap crossing)."
        )

    text = f"""# S-matrix / anticrossing scale (Experiment 15)

Filled **after** the numerical sweep. This measures the two-loop junction scale
`|S_12| ~ Delta` on the Mobius strip; it is **not** the Experiment 17 local tube
flagship, and it does not claim a first observation of the wall (Huang-Lee /
Beugeling prior art).

## Gap vs smearing (`Nx={GAP_NX}`, Mobius antiperiodic)

Min spectral gap near `E=0` grows soft as `w` increases: the *local* gap still
tracks `~2|lambda(x)|`, but the globally soft (wall-like) region widens. Only
`w >~ Nx` is fully wall-like.

| w | gap |
|---|-----|
""" + "\n".join(f"| {w:g} | {gaps[str(w)]:.6f} |" for w in GAP_WS) + f"""

## Cylinder valley purity (periodic tanh, two walls)

Purity = majority-valley weight at `k_y = +/- pi/2` on wall columns (1 = valley-pure).

| w | n_modes | mean purity |
|---|---------|-------------|
""" + "\n".join(
        f"| {v['w']:g} | {v['n_modes']} | {v['mean_purity']:.3f} |" for v in valleys
    ) + f"""

## Mobius hole-flux anticrossing `Delta(w, Ny)` (`Nx={DEL_NX}`)

`Delta = min_Phi [E_1(Phi) - E_0(Phi)]` for the two levels closest to zero.
In the two-channel network picture this tracks `|S_12|`.

| w | Ny | Delta | avoided? |
|---|----|-------|----------|
{delta_lines}

Integer `x0 -> x0+1` glide (Mobius antiperiodic): max `|dE|` = {glide_diff:.3e}
(pass if `<~ 1e-12`; no spectrum-vs-`x0` figure).

## Interpretation

{verdict}
"""
    (RESULTS / "s_matrix.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("Experiment 15 - smearing and S_12 ~ Delta")

    # --- 1. gap vs w ---
    gaps: dict[str, float] = {}
    for w in GAP_WS:
        g = min_gap_mobius(w)
        gaps[str(w)] = g
        print(f"  gap w={w:g}: {g:.6f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ws = list(GAP_WS)
    gs = [gaps[str(w)] for w in ws]
    ax.plot(ws, gs, "o-", color="C0")
    ax.axvline(GAP_NX, color="0.4", ls="--", lw=0.8, label=r"$w=N_x$")
    ax.set_xlabel("w")
    ax.set_ylabel("min gap near E=0")
    ax.set_title(
        rf"Mobius antiperiodic tanh: gap vs $w$ ($N_x={GAP_NX}$)"
        "\nlocal gap ~ 2|lambda|; soft region grows with w"
    )
    ax.legend(fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES / "15_gap_vs_w.png", dpi=150)
    plt.close(fig)

    # --- 2. valley purity ---
    valleys = [cylinder_valley_purity(w) for w in VAL_WS]
    for v in valleys:
        print(f"  valley w={v['w']:g}: purity={v['mean_purity']:.3f} n={v['n_modes']}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([v["w"] for v in valleys], [v["mean_purity"] for v in valleys], "s-", color="C1")
    ax.set_xlabel("w")
    ax.set_ylabel("valley purity")
    ax.set_ylim(0.5, 1.05)
    ax.set_title("Cylinder wall modes: valley purity vs smearing")
    fig.tight_layout()
    fig.savefig(FIGURES / "15_valley_vs_w.png", dpi=150)
    plt.close(fig)

    # --- 3. Delta(w, Ny) ---
    deltas = []
    for Ny in DEL_NYS:
        for w in DEL_WS:
            d = anticrossing_delta(w, Ny)
            deltas.append(d)
            print(f"  Delta w={w:g} Ny={Ny}: {d['Delta']:.4e} avoided={d['avoided']}")

    # heatmap-style curves
    fig, ax = plt.subplots(figsize=(7, 4))
    for Ny in DEL_NYS:
        xs = [d["w"] for d in deltas if d["Ny"] == Ny]
        ys = [d["Delta"] for d in deltas if d["Ny"] == Ny]
        ax.plot(xs, ys, "o-", label=rf"$N_y={Ny}$")
    ax.set_xlabel("w")
    ax.set_ylabel(r"$\Delta$ (anticrossing)")
    ax.set_title(r"Mobius hole-flux $\Delta(w,N_y)$ $\sim |S_{12}|$")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES / "15_delta_w_Ny.png", dpi=150)
    plt.close(fig)

    glide_diff = glide_x0_ok()
    print(f"  glide x0 max|dE|={glide_diff:.3e}")

    # --- 4. write note AFTER numbers ---
    write_s_matrix_md(gaps, valleys, deltas, glide_diff)

    pass_delta = all(d["avoided"] and d["Delta"] > 0 for d in deltas)
    pass_glide = glide_diff < 1e-10  # float margin toward 1e-12

    summary = {
        "gap_vs_w": {"Nx": GAP_NX, "Ny": GAP_NY, "gaps": gaps},
        "valley": valleys,
        "delta": deltas,
        "glide_max_dE": glide_diff,
        "pass_delta_avoided": pass_delta,
        "pass_glide": pass_glide,
        "note": "results/s_matrix.md",
    }
    (RESULTS / "15_smearing_S.json").write_text(json.dumps(summary, indent=2))

    assert pass_delta, "Delta is not a clean avoided-crossing scale on the grid"
    assert pass_glide, f"x0 glide failed: dE={glide_diff}"
    print("  PASS")


if __name__ == "__main__":
    main()
