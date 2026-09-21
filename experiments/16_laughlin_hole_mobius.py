"""Experiment 16 — Global Laughlin (hole flux) fails on Mobius."""

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
NX = 40
N_PHI = 25
K_EIG = 16
X0 = NX / 4.0


def track_signed_crossings(spectrum: np.ndarray) -> dict:
    """
    Continuous nearest-neighbor matching of near-zero levels vs Phi.

    A signed crossing is a tracked branch that changes sign between steps.
    Anticrossing bounces (approach and retreat without a net sign change of a
    continuing branch) contribute 0 to the signed sum when both partners
    bounce; we report both the raw signed sum and an unpaired-crossing count.
    """
    n_phi, k = spectrum.shape
    if n_phi < 2:
        return {"signed": 0, "n_crossings": 0, "events": []}

    # Track k branches by greedy nearest matching.
    tracked = spectrum[0].copy()
    signed = 0
    events: list[dict] = []
    for i in range(1, n_phi):
        cur = spectrum[i].copy()
        used = np.zeros(k, dtype=bool)
        new_tracked = np.zeros(k)
        for j in range(k):
            dists = np.abs(cur - tracked[j])
            dists[used] = np.inf
            m = int(np.argmin(dists))
            used[m] = True
            new_tracked[j] = cur[m]
            prev = tracked[j]
            nxt = cur[m]
            if prev == 0.0 or nxt == 0.0:
                continue
            if prev * nxt < 0.0:
                # Crossing direction: negative -> positive is +1 (Laughlin sense).
                s = int(np.sign(nxt - prev))
                if s == 0:
                    s = 1 if nxt > 0 else -1
                signed += s
                events.append({"phi_idx": i, "branch": j, "sign": s, "E_prev": float(prev), "E_next": float(nxt)})
        tracked = new_tracked

    # Unpaired: net signed flow after canceling +/- pairs at the same step.
    by_step: dict[int, int] = {}
    for ev in events:
        by_step[ev["phi_idx"]] = by_step.get(ev["phi_idx"], 0) + ev["sign"]
    unpaired = int(sum(1 for v in by_step.values() if v != 0))
    return {
        "signed": int(signed),
        "n_crossings": len(events),
        "n_unpaired_steps": unpaired,
        "events": events,
    }


def hole_flux_sweep(
    *,
    bc_x: str,
    Ny: int,
    lam: float | None = None,
    lam_x: np.ndarray | None = None,
) -> dict:
    phis = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    spectrum = np.zeros((N_PHI, K_EIG))
    pair_gaps = []
    min_abs = []
    for i, Phi in enumerate(phis):
        phi_x = flux.hole_phases(NX, Ny, Phi, bc_x=bc_x)  # type: ignore[arg-type]
        H = qwz.H_realspace(
            NX,
            Ny,
            m=M,
            lam=1.0 if lam is None else lam,
            bc_x=bc_x,  # type: ignore[arg-type]
            lam_x=lam_x,
            phi_x=phi_x,
        )
        k = min(K_EIG, H.shape[0] - 2)
        evals, _ = spectr.eigsh_near_zero(H, k=k)
        ev = np.sort(evals.real)
        # pad / trim to K_EIG for storage
        order = np.argsort(np.abs(ev))[:K_EIG]
        near = np.sort(ev[order])
        if len(near) < K_EIG:
            pad = np.full(K_EIG - len(near), np.nan)
            near = np.concatenate([near, pad])
        spectrum[i] = near

        order2 = np.argsort(np.abs(ev))
        e0, e1 = ev[order2[0]], ev[order2[1]]
        pair = np.sort([e0, e1])
        pair_gaps.append(float(pair[1] - pair[0]))
        min_abs.append(float(np.min(np.abs(ev))))

    track = track_signed_crossings(np.nan_to_num(spectrum, nan=0.0))
    return {
        "phis": phis,
        "spectrum": spectrum,
        "pair_gaps": np.asarray(pair_gaps),
        "min_abs_E": np.asarray(min_abs),
        "Delta": float(np.min(pair_gaps)),
        "closest_approach": float(np.min(min_abs)),
        "track": track,
    }


def plot_panel(ax, data: dict, title: str) -> None:
    phis = data["phis"]
    spec = data["spectrum"]
    for n in range(spec.shape[1]):
        y = spec[:, n]
        if np.all(np.isnan(y)):
            continue
        ax.plot(phis, y, color="C0", lw=0.7)
    ax.axhline(0.0, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel(r"$\Phi$")
    ax.set_ylabel(r"$E$")
    ax.set_title(title)


def main() -> None:
    print("Experiment 16 - Mobius hole flux (global Laughlin fails)")

    # Cylinder contrast: quantized flow succeeds (same lattice size).
    cyl = hole_flux_sweep(bc_x="cylinder", Ny=16, lam=+1.0)
    print(
        f"  cylinder: signed={cyl['track']['signed']}, "
        f"crossings={cyl['track']['n_crossings']}, closest={cyl['closest_approach']:.4f}"
    )

    sharp = hole_flux_sweep(
        bc_x="mobius",
        Ny=16,
        lam_x=qwz.lambda_tanh(NX, X0, 0.5, lam0=LAM0, antiperiodic=True),
    )
    print(
        f"  mobius sharp: signed={sharp['track']['signed']}, "
        f"unpaired={sharp['track']['n_unpaired_steps']}, "
        f"Delta={sharp['Delta']:.4e}, closest={sharp['closest_approach']:.4f}"
    )

    smeared = hole_flux_sweep(
        bc_x="mobius",
        Ny=16,
        lam_x=qwz.lambda_tanh(NX, X0, 5.0, lam0=LAM0, antiperiodic=True),
    )
    print(
        f"  mobius smeared: signed={smeared['track']['signed']}, "
        f"unpaired={smeared['track']['n_unpaired_steps']}, "
        f"Delta={smeared['Delta']:.4e}, closest={smeared['closest_approach']:.4f}"
    )

    # Weaker approach panel (Ny=10 smeared)
    weak = hole_flux_sweep(
        bc_x="mobius",
        Ny=10,
        lam_x=qwz.lambda_tanh(NX, X0, 5.0, lam0=LAM0, antiperiodic=True),
    )
    print(
        f"  mobius Ny=10 smeared: unpaired={weak['track']['n_unpaired_steps']}, "
        f"closest={weak['closest_approach']:.4f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    plot_panel(axes[0], cyl, "Cylinder (Exp 09 style): crossings")
    plot_panel(axes[1], sharp, r"Mobius sharp $w=0.5$: anticross")
    plot_panel(axes[2], smeared, r"Mobius smeared $w=5$: anticross")
    fig.suptitle(
        "Hole flux: cylinder pumps; Mobius anticrosses "
        r"($\int C=0$, no quantized Hall). See Exp 15 for $\Delta(w,N_y)$."
    )
    fig.tight_layout()
    out_fig = FIGURES / "16_laughlin_hole_mobius.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_fig}")

    def pack(d: dict) -> dict:
        return {
            "signed": d["track"]["signed"],
            "n_crossings": d["track"]["n_crossings"],
            "n_unpaired_steps": d["track"]["n_unpaired_steps"],
            "Delta": d["Delta"],
            "closest_approach": d["closest_approach"],
            "min_abs_E_vs_phi": d["min_abs_E"].tolist(),
            "pair_gaps": d["pair_gaps"].tolist(),
        }

    # Pass: Mobius has no robust unpaired quantized crossing.
    # Cylinder should show nonzero spectral-flow activity for contrast.
    pass_mobius = (
        sharp["track"]["n_unpaired_steps"] == 0
        and smeared["track"]["n_unpaired_steps"] == 0
        and weak["track"]["n_unpaired_steps"] == 0
    )
    # Also require finite avoided-crossing scale (not a full through-gap Laughlin flow).
    pass_delta = sharp["Delta"] > 1e-10 and smeared["Delta"] > 1e-10

    summary = {
        "Nx": NX,
        "n_phi": N_PHI,
        "cylinder": pack(cyl),
        "mobius_sharp_Ny16_w0.5": pack(sharp),
        "mobius_smeared_Ny16_w5": pack(smeared),
        "mobius_smeared_Ny10_w5": pack(weak),
        "pass_no_unpaired_mobius": pass_mobius,
        "pass_finite_Delta": pass_delta,
        "caption": (
            "Mobius hole-flux spectrum anticrosses; net Hall response vanishes "
            "because C as a section integrates to zero. Anticrossing scale: Exp 15. "
            "Not a 'pump returns via the wall' reading."
        ),
    }
    (RESULTS / "16_laughlin_hole_mobius.json").write_text(json.dumps(summary, indent=2))

    assert pass_mobius and pass_delta, (
        f"Mobius hole flux should anticross without unpaired crossings "
        f"(unpaired sharp/smeared/weak="
        f"{sharp['track']['n_unpaired_steps']}/"
        f"{smeared['track']['n_unpaired_steps']}/"
        f"{weak['track']['n_unpaired_steps']}, "
        f"Delta={sharp['Delta']:.3e}/{smeared['Delta']:.3e})"
    )
    print("  PASS")


if __name__ == "__main__":
    main()
