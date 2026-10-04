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

import lensa as L


class TestLenses(unittest.TestCase):
    def roi(self, h=64, w=96):
        rng = np.random.default_rng(1)
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    def test_all_lenses_preserve_shape_dtype(self):
        for name, fn in L.LENSA_LIST:
            out = fn(self.roi(), 0.0)
            self.assertEqual(out.shape, (64, 96, 3), name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_lens_names(self):
        names = [n for n, _ in L.LENSA_LIST]
        self.assertEqual(names, ["MONO", "KONTRAS", "FILM", "GARIS",
                                 "AMBANG", "DITHER", "NEGATIF"])

    def test_registry_disjoint_from_filters(self):
        from filters import FILTROS
        lensa_fns = {fn for _, fn in L.LENSA_LIST}
        self.assertEqual(lensa_fns & set(FILTROS), set())

    def test_grade_retro_various_sizes(self):
        for h, w in [(480, 640), (720, 1280), (33, 51)]:
            img = np.full((h, w, 3), 120, np.uint8)
            out = L.grade_retro(img)
            self.assertEqual(out.shape, (h, w, 3))

    def test_grade_retro_returns_uint8(self):
        img = np.full((100, 100, 3), 200, np.uint8)
        out = L.grade_retro(img)
        self.assertEqual(out.dtype, np.uint8)


from hand_tracking import Hand, Hands
from lensa import LensaMode, LENSA_LIST
from modes import is_L


def l_hand(px_thumb, px_index, w=320, h=240):
    pts = [types.SimpleNamespace(x=0.5, y=0.5)] * 21
    pts[4] = types.SimpleNamespace(x=px_thumb[0] / w, y=px_thumb[1] / h)
    pts[8] = types.SimpleNamespace(x=px_index[0] / w, y=px_index[1] / h)
    f = {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": False}
    return Hand(landmarks=pts, label="Left", fingers=f)


def l_pair():
    a = l_hand((30, 30), (140, 30))
    b = l_hand((180, 200), (290, 200))
    return Hands(left=a, right=b, all=[a, b])


class TestLensаModePlaceholder(unittest.TestCase):
    pass


class TestLensaMode(unittest.TestCase):
    def frame(self):
        return np.full((240, 320, 3), 80, np.uint8)

    def test_no_quad_without_L(self):
        m = LensaMode(saver=lambda *a: self.fail("tidak boleh simpan"))
        out, p = m.update(self.frame(), Hands(), 0.0, -1)
        self.assertIsNone(p)
        self.assertEqual(m.status, "-")

    def test_L_builds_quad_and_grades_frame(self):
        m = LensaMode(saver=lambda *a: None)
        out, _ = m.update(self.frame(), l_pair(), 0.0, -1)
        self.assertIsNotNone(m.quad)
        self.assertTrue(out.any())

    def test_space_cycles_lens(self):
        m = LensaMode(saver=lambda *a: None)
        first = m.lens_name
        m.update(self.frame(), l_pair(), 0.0, ord(" "))
        self.assertNotEqual(m.lens_name, first)
        self.assertEqual(len(LENSA_LIST), 7)

    def test_hold_3s_saves_photo_and_resets(self):
        saved = []
        m = LensaMode(saver=lambda img, tag: saved.append((img, tag)))
        f = self.frame()
        for i in range(30):
            t = i * 0.1
            m.update(f, l_pair(), t, -1)
            self.assertEqual(saved, [], f"tersimpan terlalu cepat di t={t}")
        m.update(f, l_pair(), 4.2, -1)
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0][1].startswith("LENS"))
        self.assertIsNone(m.quad)

    def test_genGGAM_when_quad_small(self):
        m = LensaMode(saver=lambda *a: None)
        a = l_hand((150, 120), (156, 120))
        b = l_hand((160, 120), (166, 120))
        hs = Hands(left=a, right=b, all=[a, b])
        m.update(self.frame(), hs, 0.0, -1)
        self.assertEqual(m.status, "GENGGAM")

    def test_on_enter_resets(self):
        m = LensaMode(saver=lambda *a: None)
        m.update(self.frame(), l_pair(), 0.0, -1)
        m.on_enter(9.0)
        self.assertIsNone(m.quad)
        self.assertEqual(m.status, "-")


def clench_pair():
    a = l_hand((150, 120), (156, 120))
    b = l_hand((160, 120), (166, 120))
    return Hands(left=a, right=b, all=[a, b])


def mid_pair():
    a = l_hand((120, 60), (160, 60))
    b = l_hand((140, 180), (180, 180))
    return Hands(left=a, right=b, all=[a, b])


def diag_quad(hs, w=320, h=240):
    sudut = []
    for hand in hs.all:
        sudut.append((int(hand.landmarks[4].x * w), int(hand.landmarks[4].y * h)))
        sudut.append((int(hand.landmarks[8].x * w), int(hand.landmarks[8].y * h)))
    q = urutkan_quad(sudut[:4])
    return float(np.mean([np.linalg.norm(q[2] - q[0]), np.linalg.norm(q[3] - q[1])]))


class TestGantiLensJepret(unittest.TestCase):
    def frame(self):
        return np.full((240, 320, 3), 80, np.uint8)

    def feed_to_genggam(self, m, hands, t0, langkah=15):
        t = t0
        for _ in range(langkah):
            m.update(self.frame(), hands, t, -1)
            t += 0.1
            if m.status == "GENGGAM":
                return t
        self.fail("quad tidak pernah mencapai GEGGAM")
        return t

    def test_fixture_zona_ambang(self):
        self.assertLess(diag_quad(clench_pair()), L.MIN_BUKA)
        self.assertGreaterEqual(diag_quad(mid_pair()), L.MIN_BUKA)
        self.assertLess(diag_quad(mid_pair()), L.BUKA_LAGI)
        self.assertGreater(diag_quad(l_pair()), L.BUKA_LAGI)

    def test_jepret_ganti_lens_sekali_saja(self):
        m = LensaMode(saver=lambda *a: None)
        m.update(self.frame(), l_pair(), 0.0, -1)
        awal = m.idx
        t = self.feed_to_genggam(m, clench_pair(), 1.0)
        self.assertEqual(m.idx, awal + 1, "jepret kecil harus ganti lens persis sekali")
        for _ in range(10):
            m.update(self.frame(), clench_pair(), t, -1)
            t += 0.1
        self.assertEqual(m.idx, awal + 1, "menahan jepret tidak boleh ganti berulang")

    def test_buka_lalu_jepret_lagi_boleh(self):
        m = LensaMode(saver=lambda *a: None)
        m.update(self.frame(), l_pair(), 0.0, -1)
        awal = m.idx
        t = self.feed_to_genggam(m, clench_pair(), 1.0)
        self.assertEqual(m.idx, awal + 1)
        for _ in range(4):
            m.update(self.frame(), l_pair(), t, -1)
            t += 0.1
        t = self.feed_to_genggam(m, clench_pair(), t)
        self.assertEqual(m.idx, awal + 2, "setelah membuka penuh, jepret berikutnya harus berlaku")

    def test_zona_tengah_tidak_mengaktifkan_lagi(self):
        m = LensaMode(saver=lambda *a: None)
        m.update(self.frame(), l_pair(), 0.0, -1)
        awal = m.idx
        t = self.feed_to_genggam(m, clench_pair(), 1.0)
        self.assertEqual(m.idx, awal + 1)
        for _ in range(6):
            m.update(self.frame(), mid_pair(), t, -1)
            t += 0.1
        for _ in range(12):
            m.update(self.frame(), clench_pair(), t, -1)
            t += 0.1
        self.assertEqual(m.idx, awal + 1,
                         "zona histeresis (>= MIN_BUKA, < BUKA_LAGI) tidak boleh re-arm")


import types
