import cv2
import numpy as np

from lensa import grade_retro, teks, titik_int, FONT, CYAN, MAGENTA, AMBER, PUTIH, ABU
from modes import is_tunjuk, is_peace, is_kepal

HALUS_PENA = 0.5
LOMPAT_MAKS = 160
MULAI_BUTUH = 2
HENTI_BUTUH = 4
SENTUH_MAKS = 40
TANGAN_PISAH = 60
PENA = [PUTIH, CYAN, MAGENTA, AMBER]


class DrawMode:
    def __init__(self):
        self.kanvas = None
        self.idx_pena = 1
        self.menggambar = False
        self.beruntun_ya = 0
        self.beruntun_tidak = 0
        self.pena_akhir = None
        self.pena_halus = None
        self._goresan = 0
        self._status = "SIAP"
        self.sentuh = False
        self.tangan_aktif = None

    @property
    def lens_name(self):
        return "-"

    @property
    def status(self):
        return self._status

    @property
    def goresan(self):
        return self._goresan

    def on_enter(self, now):
        self.menggambar = False
        self.beruntun_ya = 0
        self.beruntun_tidak = 0
        self.pena_akhir = None
        self.pena_halus = None
        self._status = "SIAP"
        self.sentuh = False
        self.tangan_aktif = None

    def update(self, frame, hands, now, key):
        h, w = frame.shape[:2]
        if self.kanvas is None or self.kanvas.shape[:2] != (h, w):
            self.kanvas = np.zeros((h, w, 3), np.uint8)

        if key == ord("c"):
            self.kanvas[:] = 0
            self._goresan = 0
            self.menggambar = False
            self.pena_akhir = self.pena_halus = None
            self.beruntun_ya = self.beruntun_tidak = 0
        if key == ord("p"):
            self.idx_pena = (self.idx_pena + 1) % len(PENA)

        if len(hands.all) >= 2:
            a = hands.all[0].landmarks[8]
            b = hands.all[1].landmarks[8]
            dx = (a.x - b.x) * w
            dy = (a.y - b.y) * h
            wa = hands.all[0].landmarks[0]
            wb = hands.all[1].landmarks[0]
            wx = (wa.x - wb.x) * w
            wy = (wa.y - wb.y) * h
            dekat = (dx * dx + dy * dy) ** 0.5 < SENTUH_MAKS
            tangan_pisah = (wx * wx + wy * wy) ** 0.5 >= TANGAN_PISAH
            aktif = dekat and tangan_pisah
            if aktif and not self.sentuh:
                self.idx_pena = (self.idx_pena + 1) % len(PENA)
            self.sentuh = aktif
        else:
            self.sentuh = False

        tampil = cv2.multiply(grade_retro(frame), 0.5, dtype=cv2.CV_8U)
        self._status = "SIAP"

        if hands.all:
            cocok = [x for x in hands.all if x.label == self.tangan_aktif]
            hand = cocok[0] if cocok else hands.all[0]
            self.tangan_aktif = hand.label
        else:
            hand = None
            self.tangan_aktif = None
        f = hand.fingers if hand else None
        ujung = None

        if f is not None and is_kepal(f):
            self.kanvas[:] = 0
            self._goresan = 0
            self.menggambar = False
            self.pena_akhir = self.pena_halus = None
            self.beruntun_ya = self.beruntun_tidak = 0
            self._status = "HAPUS"
        elif f is not None and is_peace(f):
            self._status = "PINDAH"
        else:
            gambar_ok = f is not None and is_tunjuk(f)
            if hand is not None:
                ujung = (int(hand.landmarks[8].x * w), int(hand.landmarks[8].y * h))

            if gambar_ok:
                self.beruntun_ya += 1
                self.beruntun_tidak = 0
            else:
                self.beruntun_tidak += 1
                self.beruntun_ya = 0

            if not self.menggambar and self.beruntun_ya >= MULAI_BUTUH:
                self.menggambar = True
            elif self.menggambar and self.beruntun_tidak >= HENTI_BUTUH:
                self.menggambar = False
                self.pena_akhir = self.pena_halus = None

            if self.menggambar and gambar_ok and ujung is not None:
                p = np.array(ujung, dtype=np.float32)
                self.pena_halus = p if self.pena_halus is None else \
                    self.pena_halus + HALUS_PENA * (p - self.pena_halus)
                titik = titik_int(self.pena_halus)
                if self.pena_akhir is not None:
                    d = abs(titik[0] - self.pena_akhir[0]) + abs(titik[1] - self.pena_akhir[1])
                    if d < LOMPAT_MAKS:
                        cv2.line(self.kanvas, self.pena_akhir, titik,
                                 PENA[self.idx_pena], 6, cv2.LINE_AA)
                        self._goresan += 1
                self.pena_akhir = titik
                self._status = "GAMBAR"
                cv2.circle(tampil, titik, 12, PENA[self.idx_pena], 2, cv2.LINE_AA)
            elif ujung is not None:
                cv2.circle(tampil, ujung, 9, ABU, 1, cv2.LINE_AA)

        for x in hands.all:
            px = [(int(l.x * w), int(l.y * h)) for l in x.landmarks]
            for con in __import__("mediapipe").solutions.hands.HAND_CONNECTIONS:
                cv2.line(tampil, px[con[0]], px[con[1]], (80, 70, 90), 1, cv2.LINE_AA)
            for i in (4, 8, 12, 16, 20):
                cv2.circle(tampil, px[i], 4, (60, 60, 235), cv2.FILLED)

        kecil = cv2.GaussianBlur(cv2.resize(self.kanvas, (max(1, w // 3), max(1, h // 3))),
                                 (0, 0), 4)
        tampil = cv2.add(tampil, cv2.resize(kecil, (w, h)))
        tampil = cv2.add(tampil, self.kanvas)
        return tampil, None
