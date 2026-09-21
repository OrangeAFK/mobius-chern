"""Integer x0 shift of tanh lambda is a lattice glide (spectrum invariant)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import qwz, spectr  # noqa: E402

NX, NY = 16, 8
W = 1.5
X0 = 3.0
TOL = 1e-10


class TestGlideX0(unittest.TestCase):
    def test_lambda_is_cyclic_roll(self):
        lam0 = qwz.lambda_tanh(NX, X0, W, antiperiodic=False)
        lam1 = qwz.lambda_tanh(NX, X0 + 1.0, W, antiperiodic=False)
        np.testing.assert_allclose(lam1, np.roll(lam0, 1), atol=1e-14)

    def test_spectrum_invariant_under_integer_x0(self):
        lam0 = qwz.lambda_tanh(NX, X0, W, antiperiodic=False)
        lam1 = qwz.lambda_tanh(NX, X0 + 1.0, W, antiperiodic=False)
        H0 = qwz.H_realspace(NX, NY, m=1.0, bc_x="cylinder", lam_x=lam0)
        H1 = qwz.H_realspace(NX, NY, m=1.0, bc_x="cylinder", lam_x=lam1)
        e0, _ = spectr.eigh(H0)
        e1, _ = spectr.eigh(H1)
        diff = float(np.max(np.abs(np.sort(e0.real) - np.sort(e1.real))))
        self.assertLess(
            diff,
            TOL,
            msg=f"max |E(x0)-E(x0+1)| = {diff} (want < {TOL})",
        )


if __name__ == "__main__":
    unittest.main()
