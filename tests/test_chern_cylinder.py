"""Bianco–Resta local Chern marker on a cylinder (convention lock)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import marker, qwz, spectr  # noqa: E402

NX, NY = 16, 10
EDGE_TRIM = 2


def bulk_mean(C: np.ndarray) -> float:
    return float(C[EDGE_TRIM : NX - EDGE_TRIM, EDGE_TRIM : NY - EDGE_TRIM].mean())


class TestChernCylinder(unittest.TestCase):
    def _bulk(self, lam: float) -> float:
        H = qwz.H_realspace(NX, NY, m=1.0, lam=lam, bc_x="cylinder")
        evals, evecs = spectr.eigh(H)
        n_occ = int((evals.real < -1e-12).sum())
        if n_occ < H.shape[0] // 4:
            n_occ = spectr.n_occupied(H.shape[0])
        P = spectr.projector(evecs, n_occ)
        C = marker.local_chern(P, NX, NY, bc_x="cylinder")
        return bulk_mean(C)

    def test_bulk_marker_positive_lambda(self):
        bm = self._bulk(+1.0)
        self.assertGreater(bm, 0.5, msg=f"bulk C={bm}")
        self.assertLess(abs(bm - 1.0), 0.5, msg=f"bulk C={bm}")

    def test_bulk_marker_negative_lambda(self):
        bm = self._bulk(-1.0)
        self.assertLess(bm, -0.5, msg=f"bulk C={bm}")
        self.assertLess(abs(bm - (-1.0)), 0.5, msg=f"bulk C={bm}")


if __name__ == "__main__":
    unittest.main()
