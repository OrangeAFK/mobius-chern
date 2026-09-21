"""Experiment 09 — Hole-flux Laughlin pump on a cylinder."""

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
NX, NY = 24, 12
N_PHI = 25
K_EIG = 24
EDGE_ROWS = 2


def run_flux_sweep(lam: float) -> dict:
    phis = np.linspace(0.0, 2.0 * np.pi, N_PHI)
    spectrum = np.zeros((N_PHI, K_EIG))
    Q_bot = np.zeros(N_PHI)
    Q_top = np.zeros(N_PHI)

    for i, Phi in enumerate(phis):
        phi_x = flux.hole_phases(NX, NY, Phi, bc_x="cylinder")
        H = qwz.H_realspace(NX, NY, m=M, lam=lam, bc_x="cylinder", phi_x=phi_x)
        evals, evecs = spectr.eigh(H)
        # near-zero window for the spectrum plot
        order = np.argsort(np.abs(evals.real))[:K_EIG]
        spectrum[i] = np.sort(evals.real[order])

        n_occ = int(np.sum(evals.real < -1e-12))
        qb = 0.0
        qt = 0.0
        for n in range(n_occ):
            dens = np.abs(evecs[:, n]).reshape(NX, NY, 2) ** 2
            site = dens.sum(axis=2)
            qb += float(site[:, :EDGE_ROWS].sum())
            qt += float(site[:, NY - EDGE_ROWS :].sum())
        Q_bot[i] = qb
        Q_top[i] = qt

    # Signed pumped charge: sum of dQ excluding the final gauge-reset jump.
    dQ = np.diff(Q_bot)
    wrap = int(np.argmax(np.abs(dQ)))
    dQ_smooth = dQ.copy()
    dQ_smooth[wrap] = 0.0
    pumped = float(dQ_smooth.sum())
    signed = int(np.sign(pumped)) if abs(pumped) > 0.5 else 0

    return {
        "phis": phis,
        "spectrum": spectrum,
        "Q_bot": Q_bot,
        "Q_top": Q_top,
        "pumped_charge": pumped,
        "signed_crossings": signed,
    }


def plot_flow(data: dict, lam: float, out: Path) -> None:
    phis = data["phis"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for n in range(data["spectrum"].shape[1]):
        axes[0].plot(phis, data["spectrum"][:, n], color="C0", lw=0.7)
    axes[0].axhline(0.0, color="0.4", ls="--", lw=0.8)
    axes[0].set_xlabel(r"$\Phi$")
    axes[0].set_ylabel(r"$E$")
    axes[0].set_title("Near-zero spectrum")

    axes[1].plot(phis, data["Q_bot"], label="bottom edge", color="C0")
    axes[1].plot(phis, data["Q_top"], label="top edge", color="C3")
    axes[1].set_xlabel(r"$\Phi$")
    axes[1].set_ylabel("edge charge")
    axes[1].set_title(f"Laughlin pump: Delta Q_bot={data['pumped_charge']:+.2f}")
    axes[1].legend()
    fig.suptitle(rf"Cylinder hole flux, $\lambda={lam:+g}$")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    results = {}
    for lam in (+1.0, -1.0):
        data = run_flux_sweep(lam)
        tag = "p" if lam > 0 else "m"
        plot_flow(data, lam, FIGURES / f"09_laughlin_lam_{tag}.png")
        results[f"lam_{tag}"] = {
            "lam": lam,
            "pumped_charge": data["pumped_charge"],
            "signed_crossings": data["signed_crossings"],
            "Q_bot": data["Q_bot"].tolist(),
            "Q_top": data["Q_top"].tolist(),
        }
        print(
            f"  lam={lam:+g}: pumped_charge={data['pumped_charge']:+.3f}, "
            f"signed={data['signed_crossings']}"
        )

    c_p = results["lam_p"]["signed_crossings"]
    c_m = results["lam_m"]["signed_crossings"]
    q_p = results["lam_p"]["pumped_charge"]
    q_m = results["lam_m"]["pumped_charge"]

    one_each = abs(c_p) == 1 and abs(c_m) == 1
    opposite = c_p * c_m < 0
    # Also require |Q| ~ 1
    magnitude = abs(q_p) > 0.7 and abs(q_m) > 0.7

    summary = {
        "Nx": NX,
        "Ny": NY,
        "n_phi": N_PHI,
        "results": results,
        "pass_one_each": one_each,
        "pass_opposite": opposite,
        "pass_magnitude": magnitude,
    }
    (RESULTS / "09_laughlin.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 09 - Laughlin cylinder")
    assert one_each and opposite and magnitude, (
        f"spectral flow pass failed (c_p={c_p}, c_m={c_m}, Q={q_p:.2f}/{q_m:.2f})"
    )
    print("  PASS")


if __name__ == "__main__":
    main()
