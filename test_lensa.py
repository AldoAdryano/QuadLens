# test_lensa.py
import unittest

import numpy as np

from lensa import urutkan_quad, cocokkan, sisi, orientasi


class TestQuadGeometry(unittest.TestCase):
    def test_urutkan_rotasi_invarian(self):
        base = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        q0 = urutkan_quad(base)
        for shift in (1, 2, 3):
            rotated = base[shift:] + base[:shift]
            np.testing.assert_allclose(urutkan_quad(rotated), q0, atol=1e-4)

    def test_urutkan_mulai_kiri_atas(self):
        q = urutkan_quad([(100, 100), (0, 100), (100, 0), (0, 0)])
        self.assertEqual(list(q[0]), [0, 0])

    def test_cocokkan_identity(self):
        q = urutkan_quad([(0, 0), (10, 0), (10, 10), (0, 10)])
        np.testing.assert_allclose(cocokkan(q, q), q)

    def test_cocokkan_recovers_roll(self):
        q = urutkan_quad([(0, 0), (10, 0), (10, 10), (0, 10)])
        rolled = np.roll(q, 2, axis=0)
        np.testing.assert_allclose(cocokkan(rolled, q), q, atol=1e-4)

    def test_cocokkan_none_prev(self):
        q = urutkan_quad([(0, 0), (10, 0), (10, 10), (0, 10)])
        np.testing.assert_allclose(cocokkan(q, None), q)

    def test_sisi_and_orientasi_square(self):
        q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
        self.assertEqual(sisi(q), (10.0, 10.0, 10.0, 10.0))
        r, p, y = orientasi(q)
        self.assertAlmostEqual(r, 0.0, places=3)
        self.assertAlmostEqual(p, 0.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)
