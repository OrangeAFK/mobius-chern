# Chern insulator on a Möbius strip

Learning series: orientation obstruction, the local Chern marker as a Hall vector field, and local vs global Laughlin pumping.

## Series

| Part | Title | Experiments |
|------|-------|-------------|
| 1 | The quote, and a lattice you already know | 01–02 |
| 2 | Two-band models and the Bloch sphere | 03–04 |
| 3 | Berry curvature and the Chern number | 05–06 |
| 4 | Edges: bulk–boundary on a cylinder | 07–08 |
| 5 | Laughlin’s pump (where it works) | 09–10 |
| 6 | Domain walls and Jackiw–Rebbi | 11 |
| 7 | Onto a Möbius strip (prior art) | 12–13 |
| 8 | The Hall vector field | 14 |
| 9 | Can you smooth the inversion away? | 15 |
| 10 | Eigenvalues of the inverted strip | 16–17 |
| 11 | Odd zeros without a tanh | 18 |

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python experiments/NN_....py
```

Each experiment writes figures under `figures/` and a short summary under `results/`.

## Prior art

Domain-wall modes and Möbius edge-current patterns are prior art; this series reproduces them as calibration.

- L.-T. Huang and D.-H. Lee, Phys. Rev. B **84**, 193106 (2011); [arXiv:1107.1411](https://arxiv.org/abs/1107.1411); [doi:10.1103/PhysRevB.84.193106](https://doi.org/10.1103/PhysRevB.84.193106)
- W. Beugeling, A. Quelle, and C. Morais Smith, Phys. Rev. B **89**, 235112 (2014); [arXiv:1403.6998](https://arxiv.org/abs/1403.6998); [doi:10.1103/PhysRevB.89.235112](https://doi.org/10.1103/PhysRevB.89.235112)
