"""Möbius gluing and Hermitian smoke tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import geometry, qwz  # noqa: E402


class TestGluing(unittest.TestCase):
    def test_mobius_hop_lands_on_reversed_y(self):
        Nx, Ny = 8, 6
        for y in range(Ny):
            dest = geometry.hop_x(Nx - 1, y, Nx, Ny, "mobius")
            self.assertEqual(dest, (0, Ny - 1 - y))

    def test_cylinder_hop_no_y_flip(self):
        Nx, Ny = 8, 6
        for y in range(Ny):
            dest = geometry.hop_x(Nx - 1, y, Nx, Ny, "cylinder")
            self.assertEqual(dest, (0, y))

    def test_mobius_hamiltonian_hermitian(self):
        H = qwz.H_realspace(6, 4, m=1.0, lam=1.0, bc_x="mobius")
        Hd = H.toarray()
        self.assertTrue(
            (abs(Hd - Hd.conj().T) < 1e-12).all(),
            "Möbius H is not Hermitian",
        )


if __name__ == "__main__":
    unittest.main()
