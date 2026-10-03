import argparse

import cv2
import mediapipe as mp

from hand_tracking import INDEX_TIP, THUMB_TIP
from geometry import render_portal, portal_width, ClosingGestureDetector
from filters import FILTROS
from frame_source import FrameSource


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def main():
    parser = argparse.ArgumentParser(description="Filtros AR con portal de manos")
    parser.add_argument(
        "--source",
        default="0",
        help="Indice de camara (ej. 0) o URL de stream (ej. http://IP:8080/video)",
    )
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = FrameSource(parse_source(args.source))
    if not cap.isOpened():
        raise RuntimeError(
            "No se pudo abrir la camara. Revisa el indice de camara, la URL del "
            "stream o los permisos."
        )

    filtro_index = 0
    closing_detector = ClosingGestureDetector()

    while True:
        ok, frame = cap.read()
        if not ok:
            if not cap.alive:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                raw_label = handedness.classification[0].label
                label = "Right" if raw_label == "Left" else "Left"

                if label == "Left":
                    left_hand = hand_landmarks
                else:
                    right_hand = hand_landmarks

        if left_hand is not None and right_hand is not None:
            lm_left = left_hand.landmark
            lm_right = right_hand.landmark

            p1 = (lm_left[INDEX_TIP].x * w, lm_left[INDEX_TIP].y * h)
            p2 = (lm_left[THUMB_TIP].x * w, lm_left[THUMB_TIP].y * h)
            p3 = (lm_right[INDEX_TIP].x * w, lm_right[INDEX_TIP].y * h)
            p4 = (lm_right[THUMB_TIP].x * w, lm_right[THUMB_TIP].y * h)

            width = portal_width(p1, p2, p3, p4)

            if closing_detector.update(width, w):
                filtro_index = (filtro_index + 1) % len(FILTROS)

            frame = render_portal(frame, p1, p2, p3, p4, FILTROS[filtro_index])

        cv2.imshow(" ", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()