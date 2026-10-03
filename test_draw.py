import unittest

import numpy as np

from draw import DrawMode
from hand_tracking import Hand, Hands


def hand_with(fingers, tip=(0.5, 0.5), label="Right"):
    pts = [types.SimpleNamespace(x=0.5, y=0.5)] * 21
    pts[8] = types.SimpleNamespace(x=tip[0], y=tip[1])
    return Hand(landmarks=pts, label=label, fingers=fingers)


F_POINT = {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}
F_FIST = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
F_PALM = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
F_PEACE = {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}


def hs(*hands):
    all_ = list(hands)
    return Hands(left=all_[0] if all_ else None, right=None, all=all_)


import types


class TestDrawMode(unittest.TestCase):
    def frame(self):
        return np.zeros((240, 320, 3), np.uint8)

    def pump(self, m, hands, key=-1, start=0.0, n=6, step=0.05):
        out = None
        for i in range(n):
            out, _ = m.update(self.frame(), hands, start + i * step, key if i == 0 else -1)
        return out

    def test_start_requires_two_point_frames(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=1)
        self.assertEqual(m.status, "SIAP")
        out = self.pump(m, hs(hand_with(F_POINT)), n=4, start=1.0)
        self.assertEqual(m.status, "GAMBAR")
        self.assertGreater(m.goresan, 0)
        self.assertTrue(out.any(), "kanvas harus menyala")

    def test_fist_clears(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=6)
        self.assertGreater(m.goresan, 0)
        self.pump(m, hs(hand_with(F_FIST)), n=3, start=2.0)
        self.assertEqual(m.status, "HAPUS")
        self.assertEqual(m.goresan, 0)
        self.assertFalse(m.kanvas.any())

    def test_peace_status(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_PEACE)), n=3, start=1.0)
        self.assertEqual(m.status, "PINDAH")

    def test_canvas_persists_across_on_enter(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=6)
        self.assertGreater(m.goresan, 0)
        m.on_enter(5.0)
        self.assertGreater(m.goresan, 0)
        self.assertTrue(m.kanvas.any())

    def test_c_key_clears(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=6)
        m.update(self.frame(), hs(), 9.0, ord("c"))
        self.assertFalse(m.kanvas.any())

    def test_no_hands_status_siap(self):
        m = DrawMode()
        m.update(self.frame(), Hands(), 0.0, -1)
        self.assertEqual(m.status, "SIAP")

    def test_jump_gap_rejected(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT, tip=(0.9, 0.9))), n=6)
        self.assertGreater(m.goresan, 0)
        g = m.goresan
        m.update(self.frame(), hs(hand_with(F_POINT, tip=(0.05, 0.05))), 2.0, -1)
        self.assertEqual(m.goresan, g, "lompatan >160px tidak boleh menggambar")
        m.update(self.frame(), hs(hand_with(F_POINT, tip=(0.05, 0.05))), 2.05, -1)
        self.assertGreater(m.goresan, g)

    def test_two_hand_tap_cycles_pen_color(self):
        m = DrawMode()
        jauh = hs(hand_with(F_POINT, tip=(0.1, 0.5)), hand_with(F_POINT, tip=(0.9, 0.5)))
        dekat = hs(hand_with(F_POINT, tip=(0.5, 0.5)), hand_with(F_POINT, tip=(0.5, 0.52)))
        awal = m.idx_pena
        m.update(self.frame(), jauh, 0.0, -1)
        self.assertEqual(m.idx_pena, awal, "jarak jauh tidak boleh ganti warna")
        m.update(self.frame(), dekat, 0.05, -1)
        self.assertEqual(m.idx_pena, awal + 1, "tap telunjuk dua tangan harus ganti warna")
        m.update(self.frame(), dekat, 0.10, -1)
        self.assertEqual(m.idx_pena, awal + 1, "tap bertahan tidak boleh ganti berulang")
        m.update(self.frame(), jauh, 0.15, -1)
        self.assertEqual(m.idx_pena, awal + 1)
        m.update(self.frame(), dekat, 0.20, -1)
        self.assertEqual(m.idx_pena, awal + 2, "tap kedua setelah pisah harus ganti lagi")

    def test_original_hand_stays_active_when_second_appears(self):
        m = DrawMode()
        kanan = hand_with(F_POINT, tip=(0.2, 0.5), label="Right")
        self.pump(m, hs(kanan), n=6)
        self.assertGreater(m.goresan, 0)
        kiri = hand_with(F_FIST, tip=(0.7, 0.5), label="Left")
        m.update(self.frame(), hs(kiri, kanan), 5.0, -1)
        self.assertNotEqual(m.status, "HAPUS", "kepal di tangan BARU tidak boleh menghapus")
        self.assertGreater(m.goresan, 0)
        self.assertTrue(m.kanvas.any())

    def test_active_hand_reselects_when_original_leaves(self):
        m = DrawMode()
        kanan = hand_with(F_POINT, tip=(0.2, 0.5), label="Right")
        kiri = hand_with(F_FIST, tip=(0.7, 0.5), label="Left")
        self.pump(m, hs(kanan), n=6)
        self.assertGreater(m.goresan, 0)
        m.update(self.frame(), hs(kiri), 6.0, -1)
        self.assertEqual(m.status, "HAPUS", "setelah tangan aktif hilang, tangan tersisa yang dipakai")


if __name__ == "__main__":
    unittest.main()
