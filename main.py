import argparse

import cv2
import mediapipe as mp

from hand_tracking import build_hands
from frame_source import FrameSource

import time

from modes import FiltersMode, TransitionController
from lensa import LensaMode
from draw import DrawMode
import capture


def parse_source(value: str):
    return int(value) if value.isdigit() else value


def draw_hud(frame, mode_name, mode_obj, fps, source_label, now, debug, hands_n):
    h, w = frame.shape[:2]
    import cv2 as _cv2
    from lensa import teks, FONT, CYAN, PUTIH, ABU, AMBER
    if int(now * 2) % 2 == 0:
        _cv2.circle(frame, (28, 30), 8, (60, 60, 235), _cv2.FILLED)
    teks(frame, "REC", (45, 38), 0.7, 2)
    teks(frame, time.strftime("%d.%m.%Y %H:%M:%S"), (w - 320, 38), 0.6, 2)
    teks(frame, f"{mode_name} | {mode_obj.lens_name} | FPS {int(fps)} | FOTO {capture.photo_count()}",
         (22, h - 24), 0.6, 2)
    teks(frame, f"SUMBER: {source_label}", (w - 320, h - 24), 0.55, 2, AMBER)
    st = mode_obj.status
    if st and st != "-":
        n = getattr(mode_obj, "goresan", None)
        label = f"{st}   GORESAN {n}" if n is not None else st
        teks(frame, label, (22, h - 92 if debug else h - 58), 0.6, 2)
    if debug:
        teks(frame, f"TANGAN {hands_n}", (22, h - 92), 0.5, 2, AMBER)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Filtros AR con portal de manos")
    parser.add_argument(
        "--source",
        default="0",
        help="Indice de camara (ej. 0) o URL de stream (ej. http://IP:8080/video)",
    )
    args = parser.parse_args()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)

    cap = FrameSource(parse_source(args.source))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "No se pudo abrir la camara. Revisa el indice de camara, la URL del "
            "stream o los permisos."
        )

    flash = capture.Flash()
    MODES = {
        "FILTERS": FiltersMode(),
        "LENSA": LensaMode(flash=flash),
        "GAMBAR": DrawMode(),
    }
    URUTAN = ["FILTERS", "LENSA", "GAMBAR"]
    transisi = TransitionController()

    mode = "FILTERS"
    sebelumnya = "FILTERS"
    MODES[mode].on_enter(0.0)
    transisi.arm(mode, 0.0)

    debug = False
    prev_time = 0.0
    source_label = str(args.source)

    def ganti(target, now):
        nonlocal mode, sebelumnya
        if target == mode:
            return
        if target == "__sebelumnya__":
            target = sebelumnya
        sebelumnya = mode
        mode = target
        MODES[mode].on_enter(now)
        transisi.arm(mode, now)

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
        now = time.time()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        detected = build_hands(results, w, h)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("d"):
            debug = not debug

        target = transisi.check(mode, detected, now)
        if target:
            ganti(target, now)

        if key in (ord("f"), ord("l"), ord("g")):
            ganti({"f": "FILTERS", "l": "LENSA", "g": "GAMBAR"}[key], now)
        elif key == ord("m"):
            ganti(URUTAN[(URUTAN.index(mode) + 1) % len(URUTAN)], now)

        mode_key = key if key in (ord(" "), ord("p"), ord("c")) else -1
        tampil, _ = MODES[mode].update(frame, detected, now, mode_key)
        tampil = flash.apply(tampil, now)

        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now
        tampil = draw_hud(tampil, mode, MODES[mode], fps, source_label,
                          now, debug, len(detected.all))

        if key == ord("s"):
            if capture.save(tampil, "MANUAL"):
                flash.trigger(now)

        cv2.imshow("Filters", tampil)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
