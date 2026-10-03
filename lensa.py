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


def warp_efek(frame, q, fn, t):
    Hf, Wf = frame.shape[:2]
    atas, kanan, bawah, kiri = sisi(q)
    lw, lh = int(max(atas, bawah)), int(max(kiri, kanan))
    if lw < 16 or lh < 16:
        return None, None, None
    tujuan = np.float32([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]])
    rect = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(q, tujuan),
                               (lw, lh))
    efek = fn(rect, t)

    bx1 = max(0, int(np.floor(q[:, 0].min())))
    by1 = max(0, int(np.floor(q[:, 1].min())))
    bx2 = min(Wf, int(np.ceil(q[:, 0].max())) + 1)
    by2 = min(Hf, int(np.ceil(q[:, 1].max())) + 1)
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        return None, None, None

    q_lokal = q - np.float32([bx1, by1])
    balik = cv2.warpPerspective(efek, cv2.getPerspectiveTransform(tujuan, q_lokal),
                                (bx2 - bx1, by2 - by1))
    return efek, balik, (bx1, by1, bx2, by2, q_lokal)


def komposit(tampil, balik, meta):
    bx1, by1, bx2, by2, q_lokal = meta
    mask = np.zeros((by2 - by1, bx2 - bx1), np.uint8)
    cv2.fillConvexPoly(mask, q_lokal.astype(np.int32), 255)
    cv2.copyTo(balik, mask, tampil[by1:by2, bx1:bx2])
    return tampil


def titik_int(p):
    return (int(round(p[0])), int(round(p[1])))


def kurung_quad(img, q, warna, tebal=3):
    for i in range(4):
        p = q[i]
        for j in ((i - 1) % 4, (i + 1) % 4):
            v = q[j] - p
            L = float(np.linalg.norm(v))
            if L < 2:
                continue
            cv2.line(img, titik_int(p), titik_int(p + v / L * min(L * 0.28, 55)),
                     warna, tebal, cv2.LINE_AA)
