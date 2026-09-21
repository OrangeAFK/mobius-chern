"""Experiment 17 — Local Laughlin (flux tube) on Mobius [flagship]."""

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

# Spectrum panels (eigsh): larger strip
SPEC_NX, SPEC_NY = 40, 16
SPEC_W = 2.0
SPEC_X0 = SPEC_NX / 4.0
N_PHI_SPEC = 21
K_EIG = 16

# Signed pump vs x_tube (full eigh for local charge): smaller for cost
LINE_NX, LINE_NY = 32, 12
LINE_W = 2.0
LINE_X0 = LINE_NX / 4.0
N_PHI_LINE = 13
X_STEP = 4
HALF_WIN = 2
JUMP_CUT = 0.25  # discard discontinuous dQ from E=0 occupation jumps

# Plateau samples across (w, Ny)
WS = (0.5, 2.0, 5.0)
NYS = (10, 16)
SAMPLE_NX = 32


def plateau_columns(lam_x: np.ndarray, w: float) -> tuple[int, int, int]:
    Nx = len(lam_x)
    wall = int(np.argmin(np.abs(lam_x)))
    xs = np.arange(Nx)
    half_w = max(3, int(np.ceil(2.0 * w)) + 1)
    dwall = np.minimum((xs - wall) % Nx, (wall - xs) % Nx)
    away = dwall > half_w
    pos = np.where(away & (lam_x > 0.5))[0]
    neg = np.where(away & (lam_x < -0.5))[0]
    if len(pos) == 0:
        pos = np.where(lam_x > 0.5)[0]
    if len(neg) == 0:
        neg = np.where(lam_x < -0.5)[0]
    x_pos = int(pos[len(pos) // 2])
    x_neg = int(neg[len(neg) // 2])
    return x_pos, wall, x_neg


def tube_spectrum(lam_x: np.ndarray, Ny: int, x_tube: int, y0: int, Nx: int) -> dict:
    """Near-zero eigsh spectrum vs Phi_tube (for plots)."""
    phis = np.linspace(0.0, 2.0 * np.pi, N_PHI_SPEC)
    spectrum = np.zeros((N_PHI_SPEC, K_EIG))
    for i, Phi in enumerate(phis):
        phi_x = flux.tube_phases(Nx, Ny, x_tube, y0, Phi, bc_x="mobius")
        H = qwz.H_realspace(Nx, Ny, m=M, bc_x="mobius", lam_x=lam_x, phi_x=phi_x)
        k = min(K_EIG, H.shape[0] - 2)
        evals, _ = spectr.eigsh_near_zero(H, k=k)
        near = np.sort(evals.real[np.argsort(np.abs(evals.real))[:K_EIG]])
        if len(near) < K_EIG:
            near = np.concatenate([near, np.zeros(K_EIG - len(near))])
        spectrum[i] = near
    return {"phis": phis, "spectrum": spectrum}


def local_charge_pump(
    lam_x: np.ndarray,
    Ny: int,
    x_tube: int,
    y0: int,
    Nx: int,
    n_phi: int = N_PHI_LINE,
) -> dict:
    """
    Signed local Laughlin response via charge near the tube column.

    Same idea as Exp 09 edge-charge pump: sum smooth dQ, zeroing jumps from
    E=0 occupation reshuffles (PH-symmetric spectrum makes raw crossing
    counts cancel). Direction tracks sgn C(x_tube) on the plateaus.
    """
    phis = np.linspace(0.0, 2.0 * np.pi, n_phi)
    Q = np.zeros(n_phi)
    for i, Phi in enumerate(phis):
        phi_x = flux.tube_phases(Nx, Ny, x_tube, y0, Phi, bc_x="mobius")
        H = qwz.H_realspace(Nx, Ny, m=M, bc_x="mobius", lam_x=lam_x, phi_x=phi_x)
        evals, evecs = spectr.eigh(H)
        n_occ = int(np.sum(evals.real < -1e-12))
        if n_occ < H.shape[0] // 4:
            n_occ = spectr.n_occupied(H.shape[0])
        dens = np.zeros((Nx, Ny))
        for n in range(n_occ):
            dens += (np.abs(evecs[:, n]) ** 2).reshape(Nx, Ny, 2).sum(axis=2)
        cols = [(x_tube + d) % Nx for d in range(-HALF_WIN, HALF_WIN + 1)]
        Q[i] = float(dens[cols, :].sum())

    dQ = np.diff(Q)
    dQ_smooth = dQ.copy()
    dQ_smooth[np.abs(dQ) > JUMP_CUT] = 0.0
    pumped = float(dQ_smooth.sum())
    signed = int(np.sign(pumped)) if abs(pumped) > 0.5 else 0
    return {
        "phis": phis,
        "Q_local": Q,
        "pumped": pumped,
        "signed": signed,
    }


def main() -> None:
    print("Experiment 17 - Local flux tube (flagship)")
    all_runs: dict = {}

    # --- Spectrum panels at three x_tube (eigsh) ---
    lam_spec = qwz.lambda_tanh(SPEC_NX, SPEC_X0, SPEC_W, lam0=LAM0, antiperiodic=True)
    x_pos, x_wall, x_neg = plateau_columns(lam_spec, SPEC_W)
    y0_spec = SPEC_NY // 2 - 1
    print(f"  spectra Nx={SPEC_NX} Ny={SPEC_NY}: x+={x_pos}, wall={x_wall}, x-={x_neg}")

    specs = {}
    for label, xt in (("pos", x_pos), ("wall", x_wall), ("neg", x_neg)):
        specs[label] = tube_spectrum(lam_spec, SPEC_NY, xt, y0_spec, SPEC_NX)
        print(f"    spectrum {label} x={xt} done")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    titles = {
        "pos": rf"$C=+1$ plateau ($x={x_pos}$)",
        "wall": rf"wall ($x={x_wall}$)",
        "neg": rf"$C=-1$ plateau ($x={x_neg}$)",
    }
    for ax, key in zip(axes, ("pos", "wall", "neg")):
        d = specs[key]
        for n in range(d["spectrum"].shape[1]):
            ax.plot(d["phis"], d["spectrum"][:, n], color="C0", lw=0.7)
        ax.axhline(0.0, color="0.4", ls="--", lw=0.8)
        ax.set_xlabel(r"$\Phi_{\mathrm{tube}}$")
        ax.set_title(titles[key])
    axes[0].set_ylabel(r"$E$")
    fig.suptitle(
        rf"Local flux tube spectra, Mobius $w={SPEC_W}$, $N_y={SPEC_NY}$ "
        r"(crossing direction from local charge pump below)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "17_tube_spectra.png", dpi=150)
    plt.close(fig)

    # --- Flagship: signed pump vs x_tube ---
    lam_line = qwz.lambda_tanh(LINE_NX, LINE_X0, LINE_W, lam0=LAM0, antiperiodic=True)
    y0_line = LINE_NY // 2 - 1
    wall_line = int(np.argmin(np.abs(lam_line)))
    xs = list(range(0, LINE_NX, X_STEP))
    signed_line = []
    pumped_line = []
    print(f"  signed vs x_tube (Nx={LINE_NX}, Ny={LINE_NY}, step={X_STEP})")
    for xt in xs:
        data = local_charge_pump(lam_line, LINE_NY, xt, y0_line, LINE_NX)
        signed_line.append(data["signed"])
        pumped_line.append(data["pumped"])
        print(f"    x={xt:2d}: pumped={data['pumped']:+.3f}, signed={data['signed']:+d}")

    xp, xw, xn = plateau_columns(lam_line, LINE_W)
    # nearest sampled columns to plateaus
    def nearest(target: int) -> int:
        return min(xs, key=lambda x: min((x - target) % LINE_NX, (target - x) % LINE_NX))

    xt_pos, xt_neg = nearest(xp), nearest(xn)
    i_pos, i_neg = xs.index(xt_pos), xs.index(xt_neg)
    s_pos, s_neg = signed_line[i_pos], signed_line[i_neg]
    q_pos, q_neg = pumped_line[i_pos], pumped_line[i_neg]
    print(f"  plateau samples: x+={xt_pos} signed={s_pos:+d} Q={q_pos:+.3f}; "
          f"x-={xt_neg} signed={s_neg:+d} Q={q_neg:+.3f}")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.step(xs, signed_line, where="mid", color="C0", lw=1.5, label="signed pump")
    ax.plot(xs, pumped_line, "o-", color="C1", ms=4, lw=1.0, label=r"$\Delta Q_{\mathrm{local}}$")
    ax.axhline(0.0, color="0.5", ls="--", lw=0.8)
    ax.axvline(wall_line, color="C3", ls=":", lw=1.2, label=f"wall x={wall_line}")
    ax.plot(np.arange(LINE_NX), np.sign(lam_line), color="0.6", lw=1.0, label=r"$\mathrm{sgn}\,\lambda$")
    ax.set_xlabel(r"$x_{\mathrm{tube}}$")
    ax.set_ylabel("signed / local charge")
    ax.set_title(
        rf"Flagship: tube pump flips across wall ($w={LINE_W}$, $N_y={LINE_NY}$)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.set_ylim(-2.5, 2.5)
    fig.tight_layout()
    fig.savefig(FIGURES / "17_signed_vs_x.png", dpi=150)
    plt.close(fig)

    all_runs["line"] = {
        "Nx": LINE_NX,
        "Ny": LINE_NY,
        "w": LINE_W,
        "xs": xs,
        "signed": signed_line,
        "pumped": pumped_line,
        "wall": wall_line,
        "x_pos": xt_pos,
        "x_neg": xt_neg,
    }

    # --- Report grid: plateau pumps vs (w, Ny) ---
    for w in WS:
        for Ny in NYS:
            key = f"w{w}_Ny{Ny}"
            print(f"  plateau sample {key}")
            Nx = SAMPLE_NX
            lam_x = qwz.lambda_tanh(Nx, Nx / 4.0, w, lam0=LAM0, antiperiodic=True)
            xp2, xw2, xn2 = plateau_columns(lam_x, w)
            y0 = Ny // 2 - 1
            pp = local_charge_pump(lam_x, Ny, xp2, y0, Nx, n_phi=N_PHI_LINE)
            pn = local_charge_pump(lam_x, Ny, xn2, y0, Nx, n_phi=N_PHI_LINE)
            pw = local_charge_pump(lam_x, Ny, xw2, y0, Nx, n_phi=N_PHI_LINE)
            all_runs[key] = {
                "x_pos": xp2,
                "x_wall": xw2,
                "x_neg": xn2,
                "pumped": {"pos": pp["pumped"], "wall": pw["pumped"], "neg": pn["pumped"]},
                "signed": {"pos": pp["signed"], "wall": pw["signed"], "neg": pn["signed"]},
            }
            print(
                f"    pos={pp['signed']:+d} ({pp['pumped']:+.2f}), "
                f"wall={pw['signed']:+d} ({pw['pumped']:+.2f}), "
                f"neg={pn['signed']:+d} ({pn['pumped']:+.2f})"
            )

    flip = s_pos * s_neg < 0
    magnitude = abs(q_pos) > 0.5 and abs(q_neg) > 0.5
    pass_ok = flip and magnitude and abs(s_pos) == 1 and abs(s_neg) == 1

    summary = {
        "spec": {"Nx": SPEC_NX, "Ny": SPEC_NY, "w": SPEC_W, "x_pos": x_pos, "x_wall": x_wall, "x_neg": x_neg},
        "runs": all_runs,
        "pass_flip": bool(flip),
        "pass_magnitude": bool(magnitude),
        "note": (
            "Signed crossings from local charge pump near tube "
            "(naive eigsh zero-crossing counts cancel under PH symmetry)."
        ),
    }
    (RESULTS / "17_laughlin_tube.json").write_text(json.dumps(summary, indent=2))

    assert pass_ok, (
        f"tube pump should flip across wall "
        f"(pos signed={s_pos} Q={q_pos:.2f}, neg signed={s_neg} Q={q_neg:.2f})"
    )
    print("  PASS")


if __name__ == "__main__":
    main()
