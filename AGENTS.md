# Agent notes

- **`PLAN.md` is the source of truth** for physics, experiment order, outputs, and pass criteria. Read it before implementing.
- **Do not edit `PLAN.md` or `pdfs/`.**
- **One Part per agent session.** Finish that Part’s experiments and tests before starting another.
- If `pdfs/` is missing, use the arXiv copies: [arXiv:1107.1411](https://arxiv.org/abs/1107.1411) (Huang & Lee) and [arXiv:1403.6998](https://arxiv.org/abs/1403.6998) (Beugeling, Quelle & Morais Smith). Read them before coding Part 7 (Experiments 12–13).

## Implementation gotchas

- **Occupation on cylinders:** use states with `E < -1e-12`, not blind `dim//2` half-fill — near-zero edge modes break naive half-filling (currents, marker, flux sweeps).
- **Marker bulk average:** exclude open `y` edge rows **and** the `x` chart cut near `x ≈ 0, Nx−1` (minimal-image branch); otherwise bulk means can cancel to ~0 despite ±1 plateaus. See `experiments/10_local_marker_cylinder.py` and `tests/test_chern_cylinder.py`.
- **Cylinder Laughlin (Exp 09):** naive zero-crossing counts on `eigsh` levels are flaky (hybridization / anticrossings). Prefer signed **edge-charge pump** ΔQ vs Φ ≈ sgn(λ); keep spectrum plots, don’t rely on crossing counts alone.
- **Scripts:** match existing `experiments/` pattern (Agg backend, ASCII `print`s for Windows cp1252, `figures/` + `results/*.json`, assert PLAN pass criteria).
- **Extend `src/` only when needed;** don’t duplicate physics or pass criteria here — those stay in `PLAN.md`.
