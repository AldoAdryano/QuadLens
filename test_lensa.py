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

import cv2

from lensa import warp_efek, komposit, kurung_quad


def identitas(roi, t):
    out = roi.copy()
    out[:] = (0, 0, 255)
    return out


class TestWarp(unittest.TestCase):
    def frame(self):
        return np.full((240, 320, 3), 50, np.uint8)

    def quad(self):
        return np.array([[10, 10], [200, 10], [200, 150], [10, 150]], np.float32)

    def test_warp_shapes_and_meta(self):
        efek, balik, meta = warp_efek(self.frame(), self.quad(), identitas, 0.0)
        self.assertIsNotNone(efek)
        self.assertEqual(efek.shape[1], 190)   # lebar quad
        self.assertEqual(efek.shape[0], 140)
        bx1, by1, bx2, by2, q_lokal = meta
        self.assertGreaterEqual(bx1, 0)
        self.assertLessEqual(bx2, 320)
        self.assertEqual(balik.shape, (by2 - by1, bx2 - bx1, 3))

    def test_warp_rejects_tiny_quad(self):
        tiny = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32)
        efek, balik, meta = warp_efek(self.frame(), tiny, identitas, 0.0)
        self.assertIsNone(efek)

    def test_komposit_paints_red_in_quad(self):
        tampil = self.frame().copy()
        efek, balik, meta = warp_efek(tampil, self.quad(), identitas, 0.0)
        out = komposit(tampil, balik, meta)
        b, g, r = out[80, 100]
        self.assertEqual((int(b), int(g), int(r)), (0, 0, 255))

    def test_kurung_quad_draws(self):
        img = np.zeros((240, 320, 3), np.uint8)
        kurung_quad(img, self.quad(), (255, 255, 255))
        self.assertTrue(img.any())
