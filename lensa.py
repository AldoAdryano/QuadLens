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


def _lut_retro():
    x = np.arange(256, dtype=np.float32)
    b = np.clip(18 + x * 0.92, 0, 255).astype(np.uint8)
    g = np.clip(6 + x * 0.97, 0, 255).astype(np.uint8)
    r = np.clip(x * 1.06 - 4, 0, 255).astype(np.uint8)
    return cv2.merge([b, g, r]).reshape(1, 256, 3)


LUT_RETRO = _lut_retro()
_vignette_cache = {}


def _vignette(w, h):
    key = (w, h)
    if key not in _vignette_cache:
        kx = cv2.getGaussianKernel(w, w * 0.55)
        ky = cv2.getGaussianKernel(h, h * 0.55)
        m = ky @ kx.T
        m = 0.35 + 0.65 * (m / m.max())
        _vignette_cache[key] = np.clip(m * 255, 0, 255).astype(np.uint8)[:, :, None].repeat(3, axis=2)
    return _vignette_cache[key]


def grade_retro(img):
    out = cv2.LUT(img, LUT_RETRO)
    out = cv2.multiply(out, _vignette(img.shape[1], img.shape[0]), scale=1 / 255)
    out[::3] = (out[::3] * 0.78).astype(np.uint8)
    return out


BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32) * (255.0 / 64.0)

CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
LUT_KONTRAS = np.clip(((np.arange(256) / 255.0) ** 1.6) * 300 - 22,
                      0, 255).astype(np.uint8)
_grain = {}


def abu(roi):
    return cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)


def ke_bgr(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def m_mono(roi, t):
    return ke_bgr(CLAHE.apply(abu(roi)))


def m_kontras(roi, t):
    return ke_bgr(cv2.LUT(CLAHE.apply(abu(roi)), LUT_KONTRAS))


def m_film(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    if (h, w) not in _grain:
        rng = np.random.default_rng(7)
        _grain[(h, w)] = [rng.normal(0, 11, (h, w)).astype(np.int16)
                          for _ in range(6)]
    g = np.clip(g.astype(np.int16) + _grain[(h, w)][int(t * 18) % 6],
                0, 255).astype(np.uint8)
    bloom = cv2.GaussianBlur(cv2.threshold(g, 195, 255, cv2.THRESH_TOZERO)[1],
                             (0, 0), 6)
    return ke_bgr(cv2.addWeighted(g, 1.0, bloom, 0.35, 0))


def m_garis(roi, t):
    g = cv2.GaussianBlur(abu(roi), (0, 0), 1.2)
    e = cv2.dilate(cv2.Canny(g, 45, 130), np.ones((2, 2), np.uint8))
    return ke_bgr(cv2.add(cv2.multiply(g, 0.14, dtype=cv2.CV_8U), e))


def m_ambang(roi, t):
    g = cv2.GaussianBlur(CLAHE.apply(abu(roi)), (0, 0), 1.0)
    return ke_bgr(cv2.threshold(g, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])


def m_dither(roi, t):
    g = CLAHE.apply(abu(roi))
    h, w = g.shape
    ubin = np.tile(BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w]
    return ke_bgr(np.where(g.astype(np.float32) > ubin, 255, 0).astype(np.uint8))


def m_negatif(roi, t):
    return ke_bgr(cv2.bitwise_not(CLAHE.apply(abu(roi))))


LENSA_LIST = [("MONO", m_mono), ("KONTRAS", m_kontras), ("FILM", m_film),
              ("GARIS", m_garis), ("AMBANG", m_ambang), ("DITHER", m_dither),
              ("NEGATIF", m_negatif)]
