# test_modes.py
import unittest

from modes import (
    is_L, is_tunjuk, is_tunjuk_ketat, is_telapak, is_kepal, is_peace,
)


def F(thumb=False, index=False, middle=False, ring=False, pinky=False):
    return {"thumb": thumb, "index": index, "middle": middle,
            "ring": ring, "pinky": pinky}


class TestClassifiers(unittest.TestCase):
    def test_L(self):
        self.assertTrue(is_L(F(thumb=True, index=True)))
        self.assertFalse(is_L(F(thumb=True, index=True, middle=True)))

    def test_tunjuk_ignores_thumb(self):
        self.assertTrue(is_tunjuk(F(index=True)))
        self.assertTrue(is_tunjuk(F(thumb=True, index=True)))
        self.assertFalse(is_tunjuk(F(thumb=True, index=True, middle=True)))

    def test_tunjuk_ketat_requires_thumb_down(self):
        self.assertTrue(is_tunjuk_ketat(F(index=True)))
        self.assertFalse(is_tunjuk_ketat(F(thumb=True, index=True)))

    def test_telapak_kepal_peace(self):
        self.assertTrue(is_telapak(F(True, True, True, True, True)))
        self.assertTrue(is_kepal(F()))
        self.assertTrue(is_peace(F(index=True, middle=True)))
        self.assertFalse(is_peace(F(index=True, middle=True, ring=True)))


from modes import HoldTransition


class TestHoldTransition(unittest.TestCase):
    def test_triggers_once_after_hold(self):
        t = HoldTransition(hold_s=2.0, enter_grace_s=0.0)
        self.assertFalse(t.update(True, 0.0))
        self.assertFalse(t.update(True, 1.9))
        self.assertTrue(t.update(True, 2.0))
        self.assertFalse(t.update(True, 2.5))

    def test_reset_after_grace(self):
        t = HoldTransition(hold_s=2.0, reset_grace_s=0.3, enter_grace_s=0.0)
        t.update(True, 0.0)
        t.update(True, 1.0)
        self.assertFalse(t.update(False, 1.1))   # masih dalam grace
        self.assertFalse(t.update(False, 1.5))   # lewat grace -> reset
        self.assertFalse(t.update(True, 1.6))
        self.assertTrue(t.update(True, 3.7))     # hitung ulang penuh

    def test_enter_grace_blocks(self):
        t = HoldTransition(hold_s=1.0, enter_grace_s=0.5)
        t.arm(10.0)
        self.assertFalse(t.update(True, 10.2))
        self.assertFalse(t.update(True, 10.6))
        self.assertTrue(t.update(True, 11.6))

    def test_short_hold_no_trigger(self):
        t = HoldTransition(hold_s=2.0, enter_grace_s=0.0)
        t.update(True, 0.0)
        self.assertFalse(t.update(False, 1.0))
        self.assertFalse(t.update(False, 2.0))


from hand_tracking import Hand, Hands
from modes import TransitionController


def h(fingers):
    return Hand(landmarks=[], label="Left", fingers=fingers)


FIST = F()
PALM = F(True, True, True, True, True)
POINT = F(index=True)
EMPTY = Hands()


class TestTransitionController(unittest.TestCase):
    def setUp(self):
        self.c = TransitionController()

    def hands(self, *hs):
        all_ = list(hs)
        left = all_[0] if all_ else None
        right = all_[1] if len(all_) > 1 else None
        return Hands(left=left, right=right, all=all_)

    def test_filters_kepal_requires_both(self):
        one = self.hands(h(FIST))
        self.assertIsNone(self.c.check("FILTERS", one, 100.0))
        self.c.arm("FILTERS", 100.0)
        self.assertIsNone(self.c.check("FILTERS", one, 103.0))
        both = self.hands(h(FIST), h(FIST))
        self.assertIsNone(self.c.check("FILTERS", both, 103.0))
        self.assertEqual(self.c.check("FILTERS", both, 105.0), "LENSA")

    def test_filters_tunjuk_ketat_single_hand(self):
        self.c.arm("FILTERS", 0.0)
        hs = self.hands(h(POINT))
        self.assertIsNone(self.c.check("FILTERS", hs, 3.0))
        self.assertEqual(self.c.check("FILTERS", hs, 5.0), "GAMBAR")

    def test_lensa_cells(self):
        self.c.arm("LENSA", 0.0)
        self.assertIsNone(self.c.check("LENSA", self.hands(h(POINT)), 3.0))
        self.assertEqual(self.c.check("LENSA", self.hands(h(POINT)), 5.0), "GAMBAR")
        self.c.arm("LENSA", 10.0)
        self.assertIsNone(self.c.check("LENSA", self.hands(h(FIST)), 13.0))
        self.assertEqual(self.c.check("LENSA", self.hands(h(FIST)), 15.0), "FILTERS")

    def test_gambar_telapak_ke_sebelumnya(self):
        self.c.arm("GAMBAR", 0.0)
        self.assertIsNone(self.c.check("GAMBAR", self.hands(h(PALM)), 3.0))
        self.assertEqual(self.c.check("GAMBAR", self.hands(h(PALM)), 5.0), "__sebelumnya__")

    def test_empty_hands_none(self):
        self.c.arm("FILTERS", 0.0)
        self.assertIsNone(self.c.check("FILTERS", EMPTY, 5.0))

    def test_enter_grace_prevents_instant(self):
        self.c.arm("FILTERS", 0.0)
        self.assertIsNone(self.c.check("FILTERS", self.hands(h(POINT)), 0.4))


import numpy as np

from modes import FiltersMode


def hand_at(x, y, fingers=None):
    pts = [types.SimpleNamespace(x=x, y=y)] * 21
    pts[4] = types.SimpleNamespace(x=x + 0.02, y=y + 0.02)
    pts[8] = types.SimpleNamespace(x=x - 0.02, y=y - 0.02)
    return Hand(landmarks=pts, label="Left", fingers=fingers or F(True, True))


def hands_pair():
    return Hands(left=hand_at(0.3, 0.4), right=hand_at(0.7, 0.4),
                 all=[hand_at(0.3, 0.4), hand_at(0.7, 0.4)])


class TestFiltersMode(unittest.TestCase):
    def frame(self):
        return np.zeros((480, 640, 3), np.uint8)

    def test_portal_only_with_two_hands(self):
        m = FiltersMode()
        f = self.frame()
        out, p = m.update(f, Hands(), 0.0, -1)
        self.assertIsNone(p)
        self.assertFalse(out.any())
        out, _ = m.update(f, hands_pair(), 0.0, -1)
        self.assertTrue(out.any(), "portal harus menggambar frame")

    def test_space_advances_filter(self):
        m = FiltersMode()
        first = m.lens_name
        m.update(self.frame(), Hands(), 0.0, ord(" "))
        self.assertNotEqual(m.lens_name, first)

    def test_on_enter_resets_index(self):
        m = FiltersMode()
        m.update(self.frame(), Hands(), 0.0, ord(" "))
        m.on_enter(1.0)
        first = FiltersMode().lens_name
        self.assertEqual(m.lens_name, first)

    def test_status_and_lens_name(self):
        m = FiltersMode()
        self.assertEqual(m.status, "-")
        self.assertTrue(m.lens_name.startswith("filtro"))

    def test_single_hand_no_portal(self):
        m = FiltersMode()
        f = self.frame()
        one = Hands(left=hand_at(0.3, 0.4), all=[hand_at(0.3, 0.4)])
        out, _ = m.update(f, one, 0.0, -1)
        self.assertFalse(out.any())


import types
