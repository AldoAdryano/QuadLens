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


if __name__ == "__main__":
    unittest.main()
