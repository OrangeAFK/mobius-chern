"""k-space Chern number smoke tests for default QWZ point."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.berry import chern_number  # noqa: E402


class TestChernKspace(unittest.TestCase):
    def test_chern_default_positive_lambda(self):
        C = chern_number(m=1.0, lam=1.0, Nk=32)
        self.assertEqual(C, 1)

    def test_chern_default_negative_lambda(self):
        C = chern_number(m=1.0, lam=-1.0, Nk=32)
        self.assertEqual(C, -1)


if __name__ == "__main__":
    unittest.main()
