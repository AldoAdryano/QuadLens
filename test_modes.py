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
