"""Experiment 07 — Cylinder edge spectrum (QWZ ribbon)."""

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

from src import qwz, spectr  # noqa: E402

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

M = 1.0
NK = 80
EDGE_ROWS = 2


def edge_weights(evecs: np.ndarray, Ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Per eigenstate: weight on bottom / top EDGE_ROWS rows (orbital summed)."""
    # evecs shape (2*Ny, nband); site y occupies [2y:2y+2]
    nband = evecs.shape[1]
    bot = np.zeros(nband)
    top = np.zeros(nband)
    for n in range(nband):
        psi = evecs[:, n]
        dens = np.abs(psi).reshape(Ny, 2) ** 2
        site = dens.sum(axis=1)
        bot[n] = site[:EDGE_ROWS].sum()
        top[n] = site[-EDGE_ROWS:].sum()
    return bot, top


def ribbon_bands(Ny: int, lam: float) -> dict:
    kx = np.linspace(-np.pi, np.pi, NK, endpoint=False)
    # store all bands
    nb = 2 * Ny
    E = np.zeros((NK, nb))
    bot_w = np.zeros((NK, nb))
    top_w = np.zeros((NK, nb))
    for i, k in enumerate(kx):
        H = qwz.H_ribbon(k, Ny, m=M, lam=lam)
        evals, evecs = spectr.eigh(H)
        E[i] = evals.real
        b, t = edge_weights(evecs, Ny)
        bot_w[i] = b
        top_w[i] = t

    # hybridization: dual edge weight on near-zero states (larger at small Ny)
    hyb = 0.0
    for i in range(NK):
        for n in range(nb):
            if abs(E[i, n]) < 0.6:
                hyb = max(hyb, float(min(bot_w[i, n], top_w[i, n])))

    # min valence-conduction separation (often ~0 when edges cross)
    gap_k = E[:, nb // 2] - E[:, nb // 2 - 1]
    min_gap = float(np.min(gap_k))

    slope_bot = _edge_slope(kx, E, bot_w)
    slope_top = _edge_slope(kx, E, top_w)

    return {
        "kx": kx,
        "E": E,
        "bot_w": bot_w,
        "top_w": top_w,
        "min_gap": min_gap,
        "hybridization": hyb,
        "slope_bot": slope_bot,
        "slope_top": slope_top,
    }


def _edge_slope(kx: np.ndarray, E: np.ndarray, w: np.ndarray) -> float:
    """Estimate velocity of edge-localized branch near E=0."""
    pts_k = []
    pts_e = []
    for i in range(len(kx)):
        for n in range(E.shape[1]):
            if w[i, n] > 0.35 and abs(E[i, n]) < 0.8:
                pts_k.append(kx[i])
                pts_e.append(E[i, n])
    if len(pts_k) < 4:
        return 0.0
    pts_k = np.asarray(pts_k)
    pts_e = np.asarray(pts_e)
    # local finite difference along sorted kx of nearest-to-zero points per kx bin
    order = np.argsort(pts_k)
    pts_k, pts_e = pts_k[order], pts_e[order]
    # unique kx: take energy closest to 0
    uniq = {}
    for k, e in zip(pts_k, pts_e):
        if k not in uniq or abs(e) < abs(uniq[k]):
            uniq[k] = e
    ks = np.array(sorted(uniq))
    es = np.array([uniq[k] for k in ks])
    if len(ks) < 3:
        return 0.0
    # central slope from polyfit
    coef = np.polyfit(ks, es, 1)
    return float(coef[0])


def plot_bands(data: dict, Ny: int, lam: float, out: Path) -> None:
    kx, E, bot_w, top_w = data["kx"], data["E"], data["bot_w"], data["top_w"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in range(E.shape[1]):
        c = bot_w[:, n] - top_w[:, n]  # +1 bottom, -1 top
        sc = ax.scatter(
            kx,
            E[:, n],
            c=c,
            cmap="coolwarm",
            s=8,
            vmin=-1,
            vmax=1,
            linewidths=0,
        )
    fig.colorbar(sc, ax=ax, label="bottom weight - top weight")
    ax.axhline(0.0, color="0.5", lw=0.6)
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$E$")
    ax.set_title(rf"QWZ ribbon $N_y={Ny}$, $\lambda={lam:+g}$, min gap={data['min_gap']:.3f}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    results = {}
    for Ny in (10, 16):
        for lam in (+1.0, -1.0):
            key = f"Ny{Ny}_lam{int(lam):+d}"
            data = ribbon_bands(Ny, lam)
            tag = "p" if lam > 0 else "m"
            plot_bands(data, Ny, lam, FIGURES / f"07_edges_Ny{Ny}_lam_{tag}.png")
            results[key] = {
                "Ny": Ny,
                "lam": lam,
                "min_gap": data["min_gap"],
                "hybridization": data["hybridization"],
                "slope_bot": data["slope_bot"],
                "slope_top": data["slope_top"],
            }
            print(
                f"  Ny={Ny} lam={lam:+g}: hyb={data['hybridization']:.4f}, "
                f"slope_bot={data['slope_bot']:.3f}, slope_top={data['slope_top']:.3f}"
            )

    h10 = results["Ny10_lam+1"]["hybridization"]
    h16 = results["Ny16_lam+1"]["hybridization"]
    s_bot_p = results["Ny16_lam+1"]["slope_bot"]
    s_top_p = results["Ny16_lam+1"]["slope_top"]
    s_bot_m = results["Ny16_lam-1"]["slope_bot"]
    opposite = s_bot_p * s_top_p < 0
    reverse = s_bot_p * s_bot_m < 0
    # finite-width: edge slope magnitudes stay O(1); require opposite + reverse
    # (hybridization metric is optional diagnostics — edges are spatially segregated)
    summary = {
        "panels": results,
        "hyb_Ny10": h10,
        "hyb_Ny16": h16,
        "pass_opposite_edges": opposite,
        "pass_chirality_reverses": reverse,
    }
    (RESULTS / "07_edges.json").write_text(json.dumps(summary, indent=2))

    print("Experiment 07 - cylinder edge spectrum")
    print(f"  hybridization Ny=10 -> {h10:.4f}, Ny=16 -> {h16:.4f}")
    assert opposite and reverse, "edge spectrum pass failed"
    print("  PASS")


if __name__ == "__main__":
    main()
