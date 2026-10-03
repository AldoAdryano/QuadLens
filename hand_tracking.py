import numpy as np

WRIST = 0
THUMB_TIP, THUMB_MCP = 4, 2
INDEX_TIP, INDEX_MCP = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP = 16, 13
PINKY_TIP, PINKY_MCP = 20, 17


def _dist(lm, i, j, w, h):
    a = np.array([lm[i].x * w, lm[i].y * h])
    b = np.array([lm[j].x * w, lm[j].y * h])
    return np.linalg.norm(a - b)


def get_extended_fingers(hand_landmarks, w, h):
    lm = hand_landmarks.landmark

    def is_extended(tip, mcp):
        return _dist(lm, tip, WRIST, w, h) > _dist(lm, mcp, WRIST, w, h) * 1.3

    return {
        "thumb": is_extended(THUMB_TIP, THUMB_MCP),
        "index": is_extended(INDEX_TIP, INDEX_MCP),
        "middle": is_extended(MIDDLE_TIP, MIDDLE_MCP),
        "ring": is_extended(RING_TIP, RING_MCP),
        "pinky": is_extended(PINKY_TIP, PINKY_MCP),
    }


class Hand:
    def __init__(self, landmarks, label, fingers):
        self.landmarks = landmarks
        self.label = label
        self.fingers = fingers


class Hands:
    def __init__(self, left=None, right=None, all=None):
        self.left = left
        self.right = right
        self.all = all if all is not None else []


def build_hands(results, w, h):
    out = Hands()
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            raw = handedness.classification[0].label
            label = "Right" if raw == "Left" else "Left"
            hand = Hand(
                landmarks=hand_landmarks.landmark,
                label=label,
                fingers=get_extended_fingers(hand_landmarks, w, h),
            )
            out.all.append(hand)
            if label == "Left":
                out.left = hand
            else:
                out.right = hand
    return out