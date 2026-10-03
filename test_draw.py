import unittest

import numpy as np

from draw import DrawMode
from hand_tracking import Hand, Hands


def hand_with(fingers, tip=(0.5, 0.5)):
    pts = [types.SimpleNamespace(x=0.5, y=0.5)] * 21
    pts[8] = types.SimpleNamespace(x=tip[0], y=tip[1])
    return Hand(landmarks=pts, label="Right", fingers=fingers)


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


if __name__ == "__main__":
    unittest.main()
