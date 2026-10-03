import math

import cv2
import numpy as np


def urutkan_quad(titik):
    p = np.array(titik, dtype=np.float32)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    return np.roll(p, -int(np.argmin(p.sum(axis=1))), axis=0)


def cocokkan(q_baru, q_lama):
    if q_lama is None:
        return q_baru
    terbaik, skor_min = 0, None
    for r in range(4):
        skor = float(np.sum((np.roll(q_baru, -r, axis=0) - q_lama) ** 2))
        if skor_min is None or skor < skor_min:
            skor_min, terbaik = skor, r
    return np.roll(q_baru, -terbaik, axis=0)


def sisi(q):
    return (np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[1]),
            np.linalg.norm(q[2] - q[3]), np.linalg.norm(q[3] - q[0]))


def orientasi(q):
    atas, kanan, bawah, kiri = sisi(q)
    v = q[1] - q[0]
    return (math.degrees(math.atan2(v[1], v[0])),
            math.degrees(math.atan2(bawah - atas, bawah + atas + 1e-6)) * 2,
            math.degrees(math.atan2(kanan - kiri, kanan + kiri + 1e-6)) * 2)
