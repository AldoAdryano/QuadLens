import os
import time

import cv2
import numpy as np

FOTO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "foto")

_count = 0


def photo_count():
    return _count


def reset_count():
    global _count
    _count = 0


def save(img, tag="LENS", folder=FOTO_DIR):
    global _count
    try:
        os.makedirs(folder, exist_ok=True)
        base = os.path.join(folder, time.strftime(f"{tag}_%Y%m%d_%H%M%S"))
        nama = base + ".png"
        n = 1
        while os.path.exists(nama):
            nama = f"{base}_{n}.png"
            n += 1
        if not cv2.imwrite(nama, img):
            return None
        _count += 1
        return nama
    except OSError as exc:
        print("Gagal menyimpan foto:", exc)
        return None


class Flash:
    def __init__(self, durasi=0.30):
        self.durasi = durasi
        self._sampai = 0.0

    def trigger(self, now):
        self._sampai = now + self.durasi

    def apply(self, frame, now):
        if now < self._sampai:
            a = (self._sampai - now) / self.durasi
            return cv2.addWeighted(frame, 1 - a, np.full_like(frame, 255), a, 0)
        return frame
