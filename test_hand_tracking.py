# test_hand_tracking.py
import types
import unittest

from hand_tracking import Hand, Hands, build_hands


def lm(x, y):
    return types.SimpleNamespace(x=x, y=y)


def fake_landmarks(points):
    return types.SimpleNamespace(landmark=[lm(x, y) for x, y in points])


def fake_handedness(label):
    return types.SimpleNamespace(
        classification=[types.SimpleNamespace(label=label)]
    )


def make_results(pairs):
    return types.SimpleNamespace(
        multi_hand_landmarks=[p[0] for p in pairs],
        multi_handedness=[p[1] for p in pairs],
    )


def extended_points():
    # 21 titik gaya mediapipe: jarak tip > mcp terhadap wrist
    pts = [(0.5, 0.9)] * 21
    pts[4] = (0.2, 0.2)   # thumb tip
    pts[2] = (0.4, 0.6)   # thumb mcp
    pts[8] = (0.3, 0.2)   # index tip
    pts[5] = (0.45, 0.6)  # index mcp
    for tip, mcp in [(12, 9), (16, 13), (20, 17)]:
        pts[tip] = (0.5, 0.95)
        pts[mcp] = (0.5, 0.85)
    return pts


class TestBuildHands(unittest.TestCase):
    def test_empty_results(self):
        r = make_results([])
        hands = build_hands(r, 640, 480)
        self.assertIsNone(hands.left)
        self.assertIsNone(hands.right)
        self.assertEqual(hands.all, [])

    def test_label_inverted_after_flip(self):
        # frame di-flip sebelum detect: raw label di-invert mengikuti loop lama
        r = make_results([(fake_landmarks(extended_points()), fake_handedness("Left"))])
        hands = build_hands(r, 640, 480)
        self.assertIsNone(hands.left)
        self.assertIsNotNone(hands.right)
        self.assertEqual(hands.all[0].label, "Right")

    def test_raw_right_becomes_left_slot(self):
        r = make_results([(fake_landmarks(extended_points()), fake_handedness("Right"))])
        hands = build_hands(r, 640, 480)
        self.assertIsNotNone(hands.left)
        self.assertIsNone(hands.right)

    def test_fingers_populated(self):
        r = make_results([(fake_landmarks(extended_points()), fake_handedness("Left"))])
        hands = build_hands(r, 640, 480)
        f = hands.all[0].fingers
        self.assertTrue(f["thumb"])
        self.assertTrue(f["index"])
        self.assertFalse(f["middle"])

    def test_missing_handedness_ignored(self):
        r = types.SimpleNamespace(
            multi_hand_landmarks=[fake_landmarks(extended_points())],
            multi_handedness=None,
        )
        self.assertEqual(build_hands(r, 640, 480).all, [])


if __name__ == "__main__":
    unittest.main()
