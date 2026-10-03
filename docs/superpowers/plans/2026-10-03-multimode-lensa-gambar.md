# Multi-Mode FILTERS + LENSA + GAMBAR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menyatukan fitur RETROLENS (quad-L + warp perspektif + 7 lensa retro + auto-foto + mode gambar) ke dalam aplikasi Filters sebagai satu aplikasi multi-mode tanpa menghilangkan fitur lama.

**Architecture:** Pipeline tunggal di `main.py` (FrameSource → flip → build_hands → mode.update → HUD → imshow) dengan state machine 3 mode (FILTERS/LENSA/GAMBAR). Tiap mode adalah objek ber-interface sama; transisi gestur hold 2 detik dikonsolidasi di `TransitionController`. Logika portal FILTERS lama dipindah utuh ke `modes.py`.

**Tech Stack:** Python 3.11 (venv `./venv`), OpenCV 4.10, mediapipe 0.10.14 (`mp.solutions.hands`), NumPy, unittest (stdlib — **tanpa pytest, tanpa dependensi baru**).

**Spec:** `docs/superpowers/specs/2026-10-03-multimode-lensa-gambar-design.md`

**Sumber kode porting:** `/home/aldo/Downloads/main.py` (RETROLENS) — fungsi yang disebut task di bawah disalin **verbatim** dari file itu kecuali ada daftar modifikasi eksplisit.

## Global Constraints

- Jalankan semua test: `cd ~/Filters && ./venv/bin/python -m unittest discover -p "test_*.py" -v`
- **28 test existing wajib tetap hijau** setiap task (`test_main`, `test_launcher`, `test_frame_source`)
- Tanpa dependensi baru; API tracking tetap `mp.solutions.hands` (tanpa model `hand_landmarker.task`)
- Jangan ubah: `filters.py`, `geometry.py`, `frame_source.py`, `launcher.py`, `filters` (executable), `test_main.py`, `test_launcher.py`, `test_frame_source.py`
- `main.py` wajib mempertahankan `parse_source`, `--source`, dan alur `FrameSource` (dites `test_main`)
- Path foto absolut: `~/Filters/foto/` (path absolut tetap, bukan relatif lokasi script)
- String UI memakai bahasa Indonesia mengikuti gaya RETROLENS (`GENGGAM`, `TARIK UNTUK MEMBUKA`, dst)
- Jangan menambah komentar kode
- Konvensi commit: `feat:` / `test:` / `fix:`

---

### Task 1: `build_hands` — hasil deteksi terpadu

**Files:**
- Modify: `hand_tracking.py` (tambah di akhir file)
- Test: `test_hand_tracking.py` (baru)

**Interfaces:**
- Consumes: hasil objek `mp.solutions.hands` (`results.multi_hand_landmarks`, `results.multi_handedness`), fungsi `get_extended_fingers(hand_landmarks, w, h)` yang sudah ada
- Produces (dipakai Task 5, 10, 11, 12):
  ```python
  class Hand:      # .landmarks (sequence objek .x/.y), .label ("Left"|"Right"), .fingers (dict)
  class Hands:     # .left: Hand|None, .right: Hand|None, .all: list[Hand]
  def build_hands(results, w: int, h: int) -> Hands
  ```
  Kunci dict `fingers`: `"thumb"`, `"index"`, `"middle"`, `"ring"`, `"pinky"`.

- [ ] **Step 1: Tulis test yang gagal**

```python
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
        # frame di-flip sebelum detect: raw "Left" -> label "Left" slot
        r = make_results([(fake_landmarks(extended_points()), fake_handedness("Left"))])
        hands = build_hands(r, 640, 480)
        self.assertIsNotNone(hands.left)
        self.assertIsNone(hands.right)
        self.assertEqual(hands.all[0].label, "Left")

    def test_raw_right_becomes_right_slot(self):
        r = make_results([(fake_landmarks(extended_points()), fake_handedness("Right"))])
        hands = build_hands(r, 640, 480)
        self.assertIsNotNone(hands.right)
        self.assertIsNone(hands.left)

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
```

- [ ] **Step 2: Jalankan — harus gagal**

Run: `./venv/bin/python -m unittest test_hand_tracking -v`
Expected: FAIL/ERROR — `ImportError: cannot import name 'build_hands'`

- [ ] **Step 3: Implementasi minimal**

Tambahkan ke `hand_tracking.py`:

```python
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
```

- [ ] **Step 4: Jalankan — harus hijau + full suite**

Run: `./venv/bin/python -m unittest test_hand_tracking -v` → PASS
Run: `./venv/bin/python -m unittest discover -p "test_*.py"` → 33+ OK

- [ ] **Step 5: Commit**

```bash
git add hand_tracking.py test_hand_tracking.py
git commit -m "feat: build_hands unified detection result"
```

---

### Task 2: Classifier gestur jari

**Files:**
- Create: `modes.py`
- Test: `test_modes.py` (baru)

**Interfaces:**
- Consumes: dict `fingers` (Task 1)
- Produces (dipakai Task 4, 10, 11):
  ```python
  def is_L(f) -> bool            # thumb+index saja
  def is_tunjuk(f) -> bool       # index saja, jempol DIABAIKAN (untuk pena)
  def is_tunjuk_ketat(f) -> bool # is_tunjuk dan jempol turun (untuk transisi)
  def is_telapak(f) -> bool      # semua jari
  def is_kepal(f) -> bool        # nihil
  def is_peace(f) -> bool        # index+tengah saja
  ```

- [ ] **Step 1: Test gagal**

```python
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
```

- [ ] **Step 2:** Run `./venv/bin/python -m unittest test_modes -v` → Expected: `ModuleNotFoundError: No module named 'modes'`

- [ ] **Step 3: Implementasi**

```python
# modes.py
def is_L(f):
    return f["thumb"] and f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]


def is_tunjuk(f):
    return f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]


def is_tunjuk_ketat(f):
    return is_tunjuk(f) and not f["thumb"]


def is_telapak(f):
    return all(f.values())


def is_kepal(f):
    return not any(f.values())


def is_peace(f):
    return f["index"] and f["middle"] and not f["ring"] and not f["pinky"]
```

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_modes -v` → PASS
- [ ] **Step 5:**
```bash
git add modes.py test_modes.py
git commit -m "feat: finger gesture classifiers"
```

---

### Task 3: `HoldTransition`

**Files:**
- Modify: `modes.py`
- Test: `test_modes.py`

**Interfaces:**
- Produces (dipakai Task 4):
  ```python
  class HoldTransition:
      def __init__(self, hold_s=2.0, reset_grace_s=0.3, enter_grace_s=0.5)
      def arm(self, now: float) -> None
      def update(self, active: bool, now: float) -> bool  # True sekali saat hold penuh
  ```

- [ ] **Step 1: Test gagal (append ke `test_modes.py`)**

```python
from modes import HoldTransition


class TestHoldTransition(unittest.TestCase):
    def test_triggers_once_after_hold(self):
        t = HoldTransition(hold_s=2.0, enter_grace_s=0.0)
        self.assertFalse(t.update(True, 0.0))
        self.assertFalse(t.update(True, 1.9))
        self.assertTrue(t.update(True, 2.0))
        self.assertFalse(t.update(True, 2.5))

    def test_reset_after_grace(self):
        t = HoldTransition(hold_s=2.0, reset_grace_s=0.3, enter_grace_s=0.0)
        t.update(True, 0.0)
        t.update(True, 1.0)
        self.assertFalse(t.update(False, 1.1))   # masih dalam grace
        self.assertFalse(t.update(False, 1.5))   # lewat grace -> reset
        self.assertFalse(t.update(True, 1.6))
        self.assertTrue(t.update(True, 3.7))     # hitung ulang penuh

    def test_enter_grace_blocks(self):
        t = HoldTransition(hold_s=1.0, enter_grace_s=0.5)
        t.arm(10.0)
        self.assertFalse(t.update(True, 10.2))
        self.assertFalse(t.update(True, 10.6))
        self.assertTrue(t.update(True, 11.6))

    def test_short_hold_no_trigger(self):
        t = HoldTransition(hold_s=2.0, enter_grace_s=0.0)
        t.update(True, 0.0)
        self.assertFalse(t.update(False, 1.0))
        self.assertFalse(t.update(False, 2.0))
```

- [ ] **Step 2:** Run → Expected: `ImportError: cannot import name 'HoldTransition'`

- [ ] **Step 3: Implementasi (append ke `modes.py`)**

```python
class HoldTransition:
    def __init__(self, hold_s=2.0, reset_grace_s=0.3, enter_grace_s=0.5):
        self.hold_s = hold_s
        self.reset_grace_s = reset_grace_s
        self.enter_grace_s = enter_grace_s
        self._arm_until = 0.0
        self._start = None
        self._last_active = None
        self._latched = False

    def arm(self, now):
        self._arm_until = now + self.enter_grace_s
        self._start = None
        self._last_active = None
        self._latched = False

    def update(self, active, now):
        if now < self._arm_until:
            return False
        if active:
            self._last_active = now
            if self._start is None:
                self._start = now
            if not self._latched and now - self._start >= self.hold_s:
                self._latched = True
                return True
            return False
        if self._start is not None and now - self._last_active > self.reset_grace_s:
            self._start = None
            self._latched = False
        return False
```

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_modes -v` → PASS
- [ ] **Step 5:**
```bash
git add modes.py test_modes.py
git commit -m "feat: HoldTransition gesture timing with hysteresis"
```

---

### Task 4: `TransitionController` — tabel transisi mode

**Files:**
- Modify: `modes.py`
- Test: `test_modes.py`

**Interfaces:**
- Consumes: `Hand`/`Hands` (Task 1), classifier (Task 2), `HoldTransition` (Task 3)
- Produces (dipakai Task 12):
  ```python
  class TransitionController:
      def __init__(self)
      def arm(self, mode: str, now: float) -> None
      def check(self, mode: str, hands: "Hands", now: float) -> str | None
  # check mengembalikan "LENSA"|"GAMBAR"|"FILTERS"|"__sebelumnya__"|None
  # Tabel (spec §4): FILTERS: kepal(2 tangan)->LENSA, tunjuk_ketat->GAMBAR
  #                  LENSA:   tunjuk_ketat->GAMBAR, kepal->FILTERS
  #                  GAMBAR:  telapak->__sebelumnya__
  ```

- [ ] **Step 1: Test gagal (append ke `test_modes.py`)**

```python
from hand_tracking import Hand, Hands
from modes import TransitionController


def h(fingers):
    return Hand(landmarks=[], label="Left", fingers=fingers)


FIST = F()
PALM = F(True, True, True, True, True)
POINT = F(index=True)
EMPTY = Hands()


class TestTransitionController(unittest.TestCase):
    def setUp(self):
        self.c = TransitionController()

    def hands(self, *hs):
        all_ = list(hs)
        left = all_[0] if all_ else None
        right = all_[1] if len(all_) > 1 else None
        return Hands(left=left, right=right, all=all_)

    def test_filters_kepal_requires_both(self):
        one = self.hands(h(FIST))
        self.assertIsNone(self.c.check("FILTERS", one, 100.0))
        self.c.arm("FILTERS", 100.0)
        self.assertIsNone(self.c.check("FILTERS", one, 103.0))
        both = self.hands(h(FIST), h(FIST))
        self.assertEqual(self.c.check("FILTERS", both, 103.0), "LENSA")

    def test_filters_tunjuk_ketat_single_hand(self):
        self.c.arm("FILTERS", 0.0)
        hs = self.hands(h(POINT))
        self.assertEqual(self.c.check("FILTERS", hs, 3.0), "GAMBAR")

    def test_lensa_cells(self):
        self.c.arm("LENSA", 0.0)
        self.assertEqual(self.c.check("LENSA", self.hands(h(POINT)), 3.0), "GAMBAR")
        self.c.arm("LENSA", 10.0)
        self.assertEqual(self.c.check("LENSA", self.hands(h(FIST)), 13.0), "FILTERS")

    def test_gambar_telapak_ke_sebelumnya(self):
        self.c.arm("GAMBAR", 0.0)
        self.assertEqual(self.c.check("GAMBAR", self.hands(h(PALM)), 3.0), "__sebelumnya__")

    def test_empty_hands_none(self):
        self.c.arm("FILTERS", 0.0)
        self.assertIsNone(self.c.check("FILTERS", EMPTY, 5.0))

    def test_enter_grace_prevents_instant(self):
        self.c.arm("FILTERS", 0.0)
        self.assertIsNone(self.c.check("FILTERS", self.hands(h(POINT)), 0.4))
```

- [ ] **Step 2:** Run → Expected: `ImportError: cannot import name 'TransitionController'`

- [ ] **Step 3: Implementasi (append ke `modes.py`)**

```python
class TransitionController:
    TABLE = {
        "FILTERS": [("kepal", True, "LENSA"), ("tunjuk_ketat", False, "GAMBAR")],
        "LENSA": [("tunjuk_ketat", False, "GAMBAR"), ("kepal", False, "FILTERS")],
        "GAMBAR": [("telapak", False, "__sebelumnya__")],
    }
    GESTURES = {
        "kepal": is_kepal,
        "tunjuk_ketat": is_tunjuk_ketat,
        "telapak": is_telapak,
    }

    def __init__(self):
        self._holds = {
            mode: [HoldTransition() for _ in rows]
            for mode, rows in self.TABLE.items()
        }

    def arm(self, mode, now):
        for hold in self._holds[mode]:
            hold.arm(now)

    def check(self, mode, hands, now):
        for (gestur, dua_tangan, target), hold in zip(
            self.TABLE[mode], self._holds[mode]
        ):
            if dua_tangan:
                ok = (
                    hands.left is not None
                    and hands.right is not None
                    and is_kepal(hands.left.fingers)
                    and is_kepal(hands.right.fingers)
                )
            else:
                ok = any(self.GESTURES[gestur](x.fingers) for x in hands.all)
            if hold.update(ok, now):
                return target
        return None
```

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_modes -v` → PASS
- [ ] **Step 5:**
```bash
git add modes.py test_modes.py
git commit -m "feat: TransitionController mode transition table"
```

---

### Task 5: `FiltersMode` — regresi logika portal lama

**Files:**
- Modify: `modes.py`
- Test: `test_modes.py`

**Interfaces:**
- Consumes: `render_portal`, `portal_width`, `ClosingGestureDetector` (`geometry.py`, tak diubah), `FILTROS` (`filters.py`, tak diubah), `build_hands` (Task 1)
- Produces (dipakai Task 12):
  ```python
  class FiltersMode:
      def __init__(self)
      def on_enter(self, now: float) -> None   # reset indeks filter + detector
      def update(self, frame, hands, now, key) -> (frame, None)
      @property lens_name -> str               # nama fungsi filter aktif, mis. "filtro_1"
      @property status -> str                  # "-"
  ```

- [ ] **Step 1: Test gagal (append ke `test_modes.py`)**

```python
import numpy as np

from modes import FiltersMode


def hand_at(x, y, fingers=None):
    pts = [types.SimpleNamespace(x=x, y=y)] * 21
    pts[4] = types.SimpleNamespace(x=x + 0.02, y=y + 0.02)
    pts[8] = types.SimpleNamespace(x=x - 0.02, y=y - 0.02)
    return Hand(landmarks=pts, label="Left", fingers=fingers or F(True, True))


def hands_pair():
    return Hands(left=hand_at(0.3, 0.4), right=hand_at(0.7, 0.4),
                 all=[hand_at(0.3, 0.4), hand_at(0.7, 0.4)])


class TestFiltersMode(unittest.TestCase):
    def frame(self):
        return np.zeros((480, 640, 3), np.uint8)

    def test_portal_only_with_two_hands(self):
        m = FiltersMode()
        f = self.frame()
        out, p = m.update(f, Hands(), 0.0, -1)
        self.assertIsNone(p)
        self.assertFalse(out.any())
        out, _ = m.update(f, hands_pair(), 0.0, -1)
        self.assertTrue(out.any(), "portal harus menggambar frame")

    def test_space_advances_filter(self):
        m = FiltersMode()
        first = m.lens_name
        m.update(self.frame(), Hands(), 0.0, ord(" "))
        self.assertNotEqual(m.lens_name, first)

    def test_on_enter_resets_index(self):
        m = FiltersMode()
        m.update(self.frame(), Hands(), 0.0, ord(" "))
        m.on_enter(1.0)
        first = FiltersMode().lens_name
        self.assertEqual(m.lens_name, first)

    def test_status_and_lens_name(self):
        m = FiltersMode()
        self.assertEqual(m.status, "-")
        self.assertTrue(m.lens_name.startswith("filtro"))

    def test_single_hand_no_portal(self):
        m = FiltersMode()
        f = self.frame()
        one = Hands(left=hand_at(0.3, 0.4), all=[hand_at(0.3, 0.4)])
        out, _ = m.update(f, one, 0.0, -1)
        self.assertFalse(out.any())


import types
```

- [ ] **Step 2:** Run → Expected: `ImportError: cannot import name 'FiltersMode'`

- [ ] **Step 3: Implementasi (append ke `modes.py`; tambahkan import di atas file)**

Tambahkan di bagian import `modes.py`:
```python
import numpy as np

from geometry import ClosingGestureDetector, portal_width, render_portal
from filters import FILTROS
```

```python
class FiltersMode:
    def __init__(self):
        self.index = 0
        self.closing = ClosingGestureDetector()

    def on_enter(self, now):
        self.index = 0
        self.closing = ClosingGestureDetector()

    @property
    def lens_name(self):
        return FILTROS[self.index].__name__

    @property
    def status(self):
        return "-"

    def update(self, frame, hands, now, key):
        if key == ord(" "):
            self.index = (self.index + 1) % len(FILTROS)

        if hands.left is not None and hands.right is not None:
            h, w = frame.shape[:2]
            lm_l = hands.left.landmarks
            lm_r = hands.right.landmarks
            p1 = (lm_l[8].x * w, lm_l[8].y * h)
            p2 = (lm_l[4].x * w, lm_l[4].y * h)
            p3 = (lm_r[8].x * w, lm_r[8].y * h)
            p4 = (lm_r[4].x * w, lm_r[4].y * h)

            width = portal_width(p1, p2, p3, p4)
            if self.closing.update(width, w):
                self.index = (self.index + 1) % len(FILTROS)
            frame = render_portal(frame, p1, p2, p3, p4, FILTROS[self.index])
        return frame, None
```

Catatan perilaku: papan `test_main` lama tidak menyentuh `main.py` — perilaku portal di sini identik dengan loop lama (urutan titik `p1,p2,p3,p4` = kiri telunjuk, kiri jempol, kanan telunjuk, kanan jempol).

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_modes -v` → PASS; full discover → OK
- [ ] **Step 5:**
```bash
git add modes.py test_modes.py
git commit -m "feat: FiltersMode port of legacy portal loop"
```

---

### Task 6: `capture.py` — simpan foto + flash

**Files:**
- Create: `capture.py`
- Test: `test_capture.py` (baru)

**Interfaces:**
- Produces (dipakai Task 10, 12):
  ```python
  FOTO_DIR: str                       # ~/Filters/foto/
  def save(img, tag="LENS", folder=FOTO_DIR) -> str | None   # unik, tidak menimpa; +1 counter
  def photo_count() -> int
  def reset_count() -> None
  class Flash:
      def __init__(self, durasi=0.30)
      def trigger(self, now: float) -> None
      def apply(self, frame, now: float) -> frame            # putih menyala lalu normal
  ```

- [ ] **Step 1: Test gagal**

```python
# test_capture.py
import os
import tempfile
import time
import unittest

import numpy as np

import capture


class TestSave(unittest.TestCase):
    def setUp(self):
        capture.reset_count()
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def test_creates_dir_and_returns_path(self):
        target = os.path.join(self.dir.name, "baru")
        img = np.zeros((10, 10, 3), np.uint8)
        path = capture.save(img, "LENS", folder=target)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(path.startswith(target))
        self.assertEqual(capture.photo_count(), 1)

    def test_unique_names_no_overwrite(self):
        img = np.zeros((10, 10, 3), np.uint8)
        p1 = capture.save(img, "LENS", folder=self.dir.name)
        p2 = capture.save(img, "LENS", folder=self.dir.name)
        self.assertNotEqual(p1, p2)
        self.assertEqual(capture.photo_count(), 2)

    def test_bad_folder_returns_none(self):
        img = np.zeros((10, 10, 3), np.uint8)
        path = capture.save(img, "LENS", folder="/proc/tidak/boleh/x")
        self.assertIsNone(path)
        self.assertEqual(capture.photo_count(), 0)


class TestFlash(unittest.TestCase):
    def test_flash_fades_to_normal(self):
        f = capture.Flash(durasi=0.30)
        frame = np.full((4, 4, 3), 100, np.uint8)
        f.trigger(0.0)
        mid = f.apply(frame.copy(), 0.15)
        self.assertGreater(mid.mean(), frame.mean())
        after = f.apply(frame.copy(), 0.5)
        self.assertEqual(after.mean(), frame.mean())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** Run `./venv/bin/python -m unittest test_capture -v` → Expected: `ModuleNotFoundError: No module named 'capture'`

- [ ] **Step 3: Implementasi**

```python
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
```

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_capture -v` → PASS
- [ ] **Step 5:**
```bash
git add capture.py test_capture.py
git commit -m "feat: photo capture with unique naming and flash"
```

---

### Task 7: `lensa.py` — util geometri quad

**Files:**
- Create: `lensa.py`
- Test: `test_lensa.py` (baru)

**Interfaces:**
- Produces (dipakai Task 8, 10):
  ```python
  def urutkan_quad(pts) -> np.ndarray  # (4,2) float32, urut searah jarum dari kiri-atas
  def cocokkan(q_baru, q_lama) -> np.ndarray  # roll terbaik vs quad lama
  def sisi(q) -> tuple                   # (atas, kanan, bawah, kiri)
  def orientasi(q) -> (roll, pitch, yaw)  # derajat
  ```

- [ ] **Step 1: Test gagal**

```python
# test_lensa.py
import unittest

import numpy as np

from lensa import urutkan_quad, cocokkan, sisi, orientasi


class TestQuadGeometry(unittest.TestCase):
    def test_urutkan_rotasi_invarian(self):
        base = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        q0 = urutkan_quad(base)
        for shift in (1, 2, 3):
            rotated = base[shift:] + base[:shift]
            np.testing.assert_allclose(urutkan_quad(rotated), q0, atol=1e-4)

    def test_urutkan_mulai_kiri_atas(self):
        q = urutkan_quad([(100, 100), (0, 100), (100, 0), (0, 0)])
        self.assertEqual(list(q[0]), [0, 0])

    def test_cocokkan_identity(self):
        q = urutkan_quad([(0, 0), (10, 0), (10, 10), (0, 10)])
        np.testing.assert_allclose(cocokkan(q, q), q)

    def test_cocokkan_recovers_roll(self):
        q = urutkan_quad([(0, 0), (10, 0), (10, 10), (0, 10)])
        rolled = np.roll(q, 2, axis=0)
        np.testing.assert_allclose(cocokkan(rolled, q), q, atol=1e-4)

    def test_cocokkan_none_prev(self):
        q = urutkan_quad([(0, 0), (10, 0), (10, 10), (0, 10)])
        np.testing.assert_allclose(cocokkan(q, None), q)

    def test_sisi_and_orientasi_square(self):
        q = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32)
        self.assertEqual(sisi(q), (10.0, 10.0, 10.0, 10.0))
        r, p, y = orientasi(q)
        self.assertAlmostEqual(r, 0.0, places=3)
        self.assertAlmostEqual(p, 0.0, places=3)
        self.assertAlmostEqual(y, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** Run `./venv/bin/python -m unittest test_lensa -v` → Expected: `ModuleNotFoundError: No module named 'lensa'`

- [ ] **Step 3: Implementasi — salin verbatim dari `/home/aldo/Downloads/main.py` fungsi `urutkan_quad`, `cocokkan`, `sisi`, `orientasi` (baris 141–169) ke `lensa.py`**

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_lensa -v` → PASS
- [ ] **Step 5:**
```bash
git add lensa.py test_lensa.py
git commit -m "feat: quad geometry utils ported from RETROLENS"
```

---

### Task 8: `lensa.py` — warp perspektif & komposit

**Files:**
- Modify: `lensa.py`
- Test: `test_lensa.py`

**Interfaces:**
- Consumes: `sisi` (Task 7)
- Produces (dipakai Task 9, 10):
  ```python
  def warp_efek(frame, q, fn, t) -> (efek, balik, meta) | (None, None, None)
  # fn(roi, t) -> roi; meta = (bx1, by1, bx2, by2, q_lokal)
  def komposit(tampil, balik, meta) -> frame
  def kurung_quad(img, q, warna, tebal=3) -> None  # menggambar in-place
  ```

- [ ] **Step 1: Test gagal (append ke `test_lensa.py`)**

```python
import cv2

from lensa import warp_efek, komposit, kurung_quad


def identitas(roi, t):
    out = roi.copy()
    out[:] = (0, 0, 255)
    return out


class TestWarp(unittest.TestCase):
    def frame(self):
        return np.full((240, 320, 3), 50, np.uint8)

    def quad(self):
        return np.array([[10, 10], [200, 10], [200, 150], [10, 150]], np.float32)

    def test_warp_shapes_and_meta(self):
        efek, balik, meta = warp_efek(self.frame(), self.quad(), identitas, 0.0)
        self.assertIsNotNone(efek)
        self.assertEqual(efek.shape[1], 190)   # lebar quad
        self.assertEqual(efek.shape[0], 140)
        bx1, by1, bx2, by2, q_lokal = meta
        self.assertGreaterEqual(bx1, 0)
        self.assertLessEqual(bx2, 320)
        self.assertEqual(balik.shape, (by2 - by1, bx2 - bx1, 3))

    def test_warp_rejects_tiny_quad(self):
        tiny = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32)
        efek, balik, meta = warp_efek(self.frame(), tiny, identitas, 0.0)
        self.assertIsNone(efek)

    def test_komposit_paints_red_in_quad(self):
        tampil = self.frame().copy()
        efek, balik, meta = warp_efek(tampil, self.quad(), identitas, 0.0)
        out = komposit(tampil, balik, meta)
        b, g, r = out[80, 100]
        self.assertEqual((int(b), int(g), int(r)), (0, 0, 255))

    def test_kurung_quad_draws(self):
        img = np.zeros((240, 320, 3), np.uint8)
        kurung_quad(img, self.quad(), (255, 255, 255))
        self.assertTrue(img.any())


```

- [ ] **Step 2:** Run → Expected: `ImportError: cannot import name 'warp_efek'`

- [ ] **Step 3: Implementasi — salin verbatim dari `/home/aldo/Downloads/main.py` fungsi `warp_efek` (baris 172–193), `komposit` (196–201), `kurung_quad` (216–226) ke `lensa.py`. Tambahkan `import cv2` dan `import numpy as np` di atas `lensa.py` (lengkap dengan `import math`).**

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_lensa -v` → PASS
- [ ] **Step 5:**
```bash
git add lensa.py test_lensa.py
git commit -m "feat: perspective warp and composite ported from RETROLENS"
```

---

### Task 9: `lensa.py` — 7 lensa + grade retro (variasi ukuran frame)

**Files:**
- Modify: `lensa.py`
- Test: `test_lensa.py`

**Interfaces:**
- Produces (dipakai Task 10):
  ```python
  def grade_retro(img) -> img          # LUT + vignette + scanline, ukuran bebas (cache per ukuran)
  def m_mono/m_kontras/m_film/m_garis/m_ambang/m_dither/m_negatif(roi, t) -> roi
  LENSA_LIST = [("MONO", fn), ..., ("NEGATIF", fn)]   # TERPISAH dari FILTROS
  ```

**Modifikasi wajib terhadap salinan RETROLENS (beda dengan sumber):**
1. `grade_retro` asli memakai `VIGNETTE` tetap 960×540 → ganti jadi cache per ukuran (lihat kode di bawah)
2. Sisanya (`_lut_retro`, `BAYER8`, `CLAHE`, `LUT_KONTRAS`, `_grain`, `m_*`, `abu`, `ke_bgr`) disalin **verbatim**

- [ ] **Step 1: Test gagal (append ke `test_lensa.py`)**

```python
import lensa as L


class TestLenses(unittest.TestCase):
    def roi(self, h=64, w=96):
        rng = np.random.default_rng(1)
        return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)

    def test_all_lenses_preserve_shape_dtype(self):
        for name, fn in L.LENSA_LIST:
            out = fn(self.roi(), 0.0)
            self.assertEqual(out.shape, (64, 96, 3), name)
            self.assertEqual(out.dtype, np.uint8, name)

    def test_lens_names(self):
        names = [n for n, _ in L.LENSA_LIST]
        self.assertEqual(names, ["MONO", "KONTRAS", "FILM", "GARIS",
                                 "AMBANG", "DITHER", "NEGATIF"])

    def test_registry_disjoint_from_filters(self):
        from filters import FILTROS
        lensa_fns = {fn for _, fn in L.LENSA_LIST}
        self.assertEqual(lensa_fns & set(FILTROS), set())

    def test_grade_retro_various_sizes(self):
        for h, w in [(480, 640), (720, 1280), (33, 51)]:
            img = np.full((h, w, 3), 120, np.uint8)
            out = L.grade_retro(img)
            self.assertEqual(out.shape, (h, w, 3))

    def test_grade_retro_returns_uint8(self):
        img = np.full((100, 100, 3), 200, np.uint8)
        out = L.grade_retro(img)
        self.assertEqual(out.dtype, np.uint8)


```

- [ ] **Step 2:** Run → Expected: `ImportError: cannot import name 'grade_retro'`

- [ ] **Step 3: Implementasi**

Salin verbatim dari `/home/aldo/Downloads/main.py`: `_lut_retro` + `LUT_RETRO` (42–50), `BAYER8` (71–76), `CLAHE` + `LUT_KONTRAS` + `_grain` (78–81), `abu`, `ke_bgr`, `m_mono`, `m_kontras`, `m_film`, `m_garis`, `m_ambang`, `m_dither`, `m_negatif` (84–134), lalu `LENSA_LIST` (137–139).

**Ganti** fungsi `_vignette`/`VIGNETTE`/`grade_retro` asli (53–68) dengan versi cache ukuran berikut (logika piksel sama: gaussian 0.55, min 0.35, scanline `out[::3] *= 0.78`):

```python
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
```

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_lensa -v` → PASS
- [ ] **Step 5:**
```bash
git add lensa.py test_lensa.py
git commit -m "feat: retro lenses and grade with size-independent vignette"
```

---

### Task 10: `LensaMode` — state machine quad + auto-foto

**Files:**
- Modify: `lensa.py`
- Test: `test_lensa.py`

**Interfaces:**
- Consumes: Task 7–9, `is_L` (Task 2), `capture.save`/`Flash` (Task 6), `Hand`/`Hands` (Task 1)
- Produces (dipakai Task 12):
  ```python
  class LensaMode:
      def __init__(self, saver=None, flash=None)  # saver(img, tag); default capture.save
      def on_enter(self, now)                      # reset quad/countdown
      def update(self, frame, hands, now, key) -> (frame, None)
      @property lens_name -> str   # "MONO".."NEGATIF"; spasi = ganti lensa
      @property status -> str      # "GENGGAM"|"ATUR"|"DIAM"|"-"
  ```

- [ ] **Step 1: Test gagal (append ke `test_lensa.py`)**

```python
from hand_tracking import Hand, Hands
from lensa import LensaMode, LENSA_LIST
from modes import is_L


def l_hand(px_thumb, px_index, w=320, h=240):
    pts = [types.SimpleNamespace(x=0.5, y=0.5)] * 21
    pts[4] = types.SimpleNamespace(x=px_thumb[0] / w, y=px_thumb[1] / h)
    pts[8] = types.SimpleNamespace(x=px_index[0] / w, y=px_index[1] / h)
    f = {"thumb": True, "index": True, "middle": False, "ring": False, "pinky": False}
    return Hand(landmarks=pts, label="Left", fingers=f)


def l_pair():
    a = l_hand((30, 30), (140, 30))
    b = l_hand((180, 200), (290, 200))
    return Hands(left=a, right=b, all=[a, b])


class TestLensаModePlaceholder(unittest.TestCase):
    pass


class TestLensaMode(unittest.TestCase):
    def frame(self):
        return np.full((240, 320, 3), 80, np.uint8)

    def test_no_quad_without_L(self):
        m = LensaMode(saver=lambda *a: self.fail("tidak boleh simpan"))
        out, p = m.update(self.frame(), Hands(), 0.0, -1)
        self.assertIsNone(p)
        self.assertEqual(m.status, "-")

    def test_L_builds_quad_and_grades_frame(self):
        m = LensaMode(saver=lambda *a: None)
        out, _ = m.update(self.frame(), l_pair(), 0.0, -1)
        self.assertIsNotNone(m.quad)
        self.assertTrue(out.any())

    def test_space_cycles_lens(self):
        m = LensaMode(saver=lambda *a: None)
        first = m.lens_name
        m.update(self.frame(), l_pair(), 0.0, ord(" "))
        self.assertNotEqual(m.lens_name, first)
        self.assertEqual(len(LENSA_LIST), 7)

    def test_hold_3s_saves_photo_and_resets(self):
        saved = []
        m = LensaMode(saver=lambda img, tag: saved.append((img, tag)))
        f = self.frame()
        for i in range(40):
            t = i * 0.1
            m.update(f, l_pair(), t, -1)
            self.assertEqual(saved, [], f"tersimpan terlalu cepat di t={t}")
        m.update(f, l_pair(), 4.2, -1)
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0][1].startswith("LENS"))
        self.assertIsNone(m.quad)

    def test_genGGAM_when_quad_small(self):
        m = LensaMode(saver=lambda *a: None)
        a = l_hand((150, 120), (156, 120))
        b = l_hand((160, 120), (166, 120))
        hs = Hands(left=a, right=b, all=[a, b])
        m.update(self.frame(), hs, 0.0, -1)
        self.assertEqual(m.status, "GENGGAM")

    def test_on_enter_resets(self):
        m = LensaMode(saver=lambda *a: None)
        m.update(self.frame(), l_pair(), 0.0, -1)
        m.on_enter(9.0)
        self.assertIsNone(m.quad)
        self.assertEqual(m.status, "-")


import types
```

- [ ] **Step 2:** Run → Expected: `ImportError: cannot import name 'LensaMode'`

- [ ] **Step 3: Implementasi (append ke `lensa.py`)**

```python
import time as _time_unused
import capture
from modes import is_L

HALUS = 0.40
MIN_BUKA = 120
GOYANG = 9
TAHAN_FOTO = 3.0


class LensaMode:
    def __init__(self, saver=None, flash=None):
        self.saver = saver if saver is not None else capture.save
        self.flash = flash
        self.quad = None
        self.hilang = 99
        self.mulai_diam = None
        self.idx = 0
        self._status = "-"

    def on_enter(self, now):
        self.quad = None
        self.hilang = 99
        self.mulai_diam = None
        self._status = "-"

    @property
    def lens_name(self):
        return LENSA_LIST[self.idx][0]

    @property
    def status(self):
        return self._status

    def update(self, frame, hands, now, key):
        if key == ord(" "):
            self.idx = (self.idx + 1) % len(LENSA_LIST)

        h, w = frame.shape[:2]
        nama_lensa, fn = LENSA_LIST[self.idx]
        sudut = []
        for hand in hands.all:
            if is_L(hand.fingers):
                sudut.append((int(hand.landmarks[4].x * w), int(hand.landmarks[4].y * h)))
                sudut.append((int(hand.landmarks[8].x * w), int(hand.landmarks[8].y * h)))

        mentah = cocokkan(urutkan_quad(sudut[:4]), self.quad) if len(sudut) >= 4 else None

        geser_maks = 0.0
        if mentah is not None:
            if self.quad is None:
                self.quad = mentah
            else:
                geser_maks = float(np.max(np.linalg.norm(mentah - self.quad, axis=1)))
                self.quad = self.quad + HALUS * (mentah - self.quad)
            self.hilang = 0
        else:
            self.hilang += 1

        tampil = grade_retro(frame)
        self._status = "-"
        aktif = self.hilang < 6 and self.quad is not None

        if aktif:
            q = self.quad.astype(np.float32)
            diag = (np.linalg.norm(q[2] - q[0]) + np.linalg.norm(q[3] - q[1])) / 2

            if diag < MIN_BUKA:
                self._status = "GENGGAM"
                self.mulai_diam = None
                c = titik_int(q.mean(axis=0))
                r = int(18 + 6 * math.sin(now * 6))
                cv2.circle(tampil, c, r, CYAN, 2, cv2.LINE_AA)
                cv2.circle(tampil, c, 3, CYAN, cv2.FILLED)
                teks(tampil, "TARIK UNTUK MEMBUKA", (c[0] - 128, c[1] - 40), 0.6, 2)
            else:
                efek, balik, meta = warp_efek(frame, q, fn, now)
                if efek is not None:
                    tampil = komposit(tampil, balik, meta)

                    if geser_maks > GOYANG or self.mulai_diam is None:
                        self.mulai_diam = now
                    sisa = TAHAN_FOTO - (now - self.mulai_diam)
                    self._status = "ATUR" if geser_maks > GOYANG else "DIAM"
                    warna = AMBER if sisa > 1 else MAGENTA

                    cv2.polylines(tampil, [q.astype(np.int32)], True, SAMAR, 1, cv2.LINE_AA)
                    kurung_quad(tampil, q, warna)
                    n = q[0] + (q[1] - q[0]) * 0.02
                    teks(tampil, f"{nama_lensa}  {efek.shape[1]}x{efek.shape[0]}",
                         (int(n[0]), max(18, int(n[1]) - 12)), 0.5, 1, warna)

                    maju = min(1.0, max(0.0, 1 - sisa / TAHAN_FOTO))
                    a, b = q[3], q[2]
                    cv2.line(tampil, titik_int(a), titik_int(b), SAMAR, 4, cv2.LINE_AA)
                    cv2.line(tampil, titik_int(a), titik_int(a + (b - a) * maju),
                             warna, 4, cv2.LINE_AA)

                    if sisa <= 0:
                        hasil = self.saver(efek, nama_lensa)
                        if hasil and self.flash is not None:
                            self.flash.trigger(now)
                        self.mulai_diam = None
                        self.hilang = 99
                        self.quad = None
                    elif sisa < TAHAN_FOTO - 0.25:
                        angka = str(int(math.ceil(sisa)))
                        sk = 2.4 + 0.4 * abs(math.sin(sisa * math.pi))
                        (tw, th), _ = cv2.getTextSize(angka, FONT, sk, 6)
                        c = titik_int(q.mean(axis=0))
                        teks(tampil, angka, (int(c[0] - tw / 2), int(c[1] + th / 2)),
                             sk, 6, warna)
        else:
            self.mulai_diam = None
            if self.hilang > 20:
                self.quad = None

        for hand in hands.all:
            px = [(int(l.x * w), int(l.y * h)) for l in hand.landmarks]
            ok = is_L(hand.fingers)
            for con in KONEKSI:
                cv2.line(tampil, px[con.start], px[con.end],
                         (150, 120, 90) if ok else (95, 80, 105), 1, cv2.LINE_AA)
            for i in (4, 8):
                cv2.circle(tampil, px[i], 7, CYAN if ok else ABU, 2 if ok else 1, cv2.LINE_AA)

        if not hands.all:
            teks(tampil, "BENTUK 'L' DUA TANGAN, SATUKAN, LALU TARIK",
                 (22, h - 58), 0.55, 2, ABU)

        return tampil, None
```

Definisi warna/font/tinta yang dibutuhkan — tambahkan di bagian atas `lensa.py` (salin verbatim dari RETROLENS baris 32–38, plus `titik_int`, `teks`, `KONEKSI`):

```python
import math

FONT = cv2.FONT_HERSHEY_DUPLEX
CYAN = (255, 235, 0)
MAGENTA = (200, 0, 255)
AMBER = (0, 180, 255)
PUTIH = (240, 245, 250)
ABU = (120, 110, 130)
SAMAR = (55, 50, 62)


def titik_int(p):
    return (int(round(p[0])), int(round(p[1])))


def teks(img, s, org, skala, tebal=2, warna=PUTIH, aberasi=True):
    x, y = org
    if aberasi:
        cv2.putText(img, s, (x - 2, y), FONT, skala, MAGENTA, tebal, cv2.LINE_AA)
        cv2.putText(img, s, (x + 2, y), FONT, skala, CYAN, tebal, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, skala, warna, tebal, cv2.LINE_AA)


import mediapipe as mp
KONEKSI = mp.solutions.hands.HAND_CONNECTIONS
```

Catatan: `test_hold_3s_saves_photo_and_resets` memakai pose L stabil — pastikan `geser_maks` = 0 pada pemanggilan pertama (`self.quad is None` → `geser_maks 0` → `mulai_diam = now`), sehingga countdown mulai dari frame pertama dan selesai di t=3,0 (assert loop hanya sampai t=3,9; pemanggilan t=4,2 memicu simpan).

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_lensa -v` → PASS
- [ ] **Step 5:**
```bash
git add lensa.py test_lensa.py
git commit -m "feat: LensaMode quad state machine with auto photo"
```

---

### Task 11: `DrawMode` — kanvas & pena gesture

**Files:**
- Create: `draw.py`
- Test: `test_draw.py` (baru)

**Interfaces:**
- Consumes: classifier (Task 2), `grade_retro` (Task 9), `Hand`/`Hands` (Task 1)
- Produces (dipakai Task 12):
  ```python
  class DrawMode:
      def __init__(self)
      def on_enter(self, now)                      # reset state pena; kanvas DIPERTAHANKAN
      def update(self, frame, hands, now, key) -> (frame, None)
      @property lens_name -> str   # "-"
      @property status -> str      # "SIAP"|"GAMBAR"|"HAPUS"|"PINDAH"
      @property goresan -> int
  ```
  Key mode: `c` = hapus kanvas, `p` = ganti warna pena.

- [ ] **Step 1: Test gagal**

```python
# test_draw.py
import unittest

import numpy as np

from draw import DrawMode
from hand_tracking import Hand, Hands


def hand_with(fingers, tip=(0.5, 0.5)):
    pts = [types.SimpleNamespace(x=0.5, y=0.5)] * 21
    pts[8] = types.SimpleNamespace(x=tip[0], y=tip[1])
    return Hand(landmarks=pts, label="Right", fingers=fingers)


F_POINT = {"thumb": False, "index": True, "middle": False, "ring": False, "pinky": False}
F_FIST = {"thumb": False, "index": False, "middle": False, "ring": False, "pinky": False}
F_PALM = {"thumb": True, "index": True, "middle": True, "ring": True, "pinky": True}
F_PEACE = {"thumb": False, "index": True, "middle": True, "ring": False, "pinky": False}


def hs(*hands):
    all_ = list(hands)
    return Hands(left=all_[0] if all_ else None, right=None, all=all_)


import types


class TestDrawMode(unittest.TestCase):
    def frame(self):
        return np.zeros((240, 320, 3), np.uint8)

    def pump(self, m, hands, key=-1, start=0.0, n=6, step=0.05):
        out = None
        for i in range(n):
            out, _ = m.update(self.frame(), hands, start + i * step, key if i == 0 else -1)
        return out

    def test_start_requires_two_point_frames(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=1)
        self.assertEqual(m.status, "SIAP")
        out = self.pump(m, hs(hand_with(F_POINT)), n=4, start=1.0)
        self.assertEqual(m.status, "GAMBAR")
        self.assertGreater(m.goresan, 0)
        self.assertTrue(out.any(), "kanvas harus menyala")

    def test_fist_clears(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=6)
        self.assertGreater(m.goresan, 0)
        self.pump(m, hs(hand_with(F_FIST)), n=3, start=2.0)
        self.assertEqual(m.status, "HAPUS")
        self.assertEqual(m.goresan, 0)
        self.assertFalse(m.kanvas.any())

    def test_palm_status(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_PALM)), n=3, start=1.0)
        self.assertEqual(m.status, "PINDAH")

    def test_canvas_persists_across_on_enter(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=6)
        self.assertGreater(m.goresan, 0)
        m.on_enter(5.0)
        self.assertGreater(m.goresan, 0)
        self.assertTrue(m.kanvas.any())

    def test_c_key_clears(self):
        m = DrawMode()
        self.pump(m, hs(hand_with(F_POINT)), n=6)
        m.update(self.frame(), hs(), 9.0, ord("c"))
        self.assertFalse(m.kanvas.any())

    def test_no_hands_status_siap(self):
        m = DrawMode()
        m.update(self.frame(), Hands(), 0.0, -1)
        self.assertEqual(m.status, "SIAP")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2:** Run `./venv/bin/python -m unittest test_draw -v` → Expected: `ModuleNotFoundError: No module named 'draw'`

- [ ] **Step 3: Implementasi**

```python
import cv2
import numpy as np

from lensa import grade_retro, teks, titik_int, FONT, CYAN, MAGENTA, AMBER, PUTIH, ABU
from modes import is_tunjuk, is_peace, is_kepal

HALUS_PENA = 0.5
LOMPAT_MAKS = 160
MULAI_BUTUH = 2
HENTI_BUTUH = 4
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

        tampil = cv2.multiply(grade_retro(frame), 0.5, dtype=cv2.CV_8U)
        self._status = "SIAP"

        hand = hands.all[0] if hands.all else None
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
                cv2.line(tampil, px[con.start], px[con.end], (80, 70, 90), 1, cv2.LINE_AA)
            for i in (4, 8, 12, 16, 20):
                cv2.circle(tampil, px[i], 4, (60, 60, 235), cv2.FILLED)

        kecil = cv2.GaussianBlur(cv2.resize(self.kanvas, (max(1, w // 3), max(1, h // 3))),
                                 (0, 0), 4)
        tampil = cv2.add(tampil, cv2.resize(kecil, (w, h)))
        tampil = cv2.add(tampil, self.kanvas)
        teks(tampil, f"{self._status}   GORESAN {self._goresan}", (22, h - 58), 0.6, 2, ABU)
        return tampil, None
```

- [ ] **Step 4:** Run `./venv/bin/python -m unittest test_draw -v` → PASS; full discover → OK
- [ ] **Step 5:**
```bash
git add draw.py test_draw.py
git commit -m "feat: DrawMode canvas and pen gestures"
```

---

### Task 12: `main.py` — state machine, keyboard, HUD

**Files:**
- Modify: `main.py` (ganti isi `main()`; pertahankan `parse_source`, `--source`, import `FrameSource`)
- Test: `test_main.py` **tidak diubah** (wajib tetap hijau) + smoke e2e

**Interfaces:**
- Consumes: semua Task 1–11
- Produces: aplikasi utuh

Aturan keyboard di `main()`:
- Global: `q` keluar, `d` debug, `s` simpan `tampil` (MANUAL + flash), `f`/`l`/`g` pindah ke FILTERS/LENSA/GAMBAR, `m` cycle mode berikutnya
- Sisanya diteruskan ke `mode.update(..., key)` (`spasi`, `p`, `c`)
- `pindah == "__sebelumnya__"` → `mode_sebelumnya`; lainnya → nama mode
- Setiap perpindahan: `MODES[baru].on_enter(now)` + `transisi.arm(baru, now)`

- [ ] **Step 1: Test regresi — jalankan suite yang ada**

Run: `./venv/bin/python -m unittest test_main test_launcher test_frame_source -v`
Expected: PASS (semua) — gagal berarti `parse_source`/alur lama rusak, perbaiki dulu sebelum lanjut.

- [ ] **Step 2: Tulis `main.py` baru**

Isi `main()` (pertahankan blok argparse + `parse_source` persis seperti sekarang):

```python
import time

from hand_tracking import build_hands
from modes import FiltersMode, TransitionController
from lensa import LensaMode
from draw import DrawMode
import capture


def draw_hud(frame, mode_name, mode_obj, fps, source_label, now, debug, hands_n):
    h, w = frame.shape[:2]
    import cv2 as _cv2
    from lensa import teks, FONT, CYAN, PUTIH, ABU, AMBER
    if int(now * 2) % 2 == 0:
        _cv2.circle(frame, (28, 30), 8, (60, 60, 235), _cv2.FILLED)
    teks(frame, "REC", (45, 38), 0.7, 2)
    teks(frame, time.strftime("%d.%m.%Y %H:%M:%M".replace("MM", "MM")), (w - 320, 38), 0.6, 2)
    teks(frame, f"{mode_name} | {mode_obj.lens_name} | FPS {int(fps)} | FOTO {capture.photo_count()}",
         (22, h - 24), 0.6, 2)
    teks(frame, f"SUMBER: {source_label}", (w - 320, h - 24), 0.55, 2, AMBER)
    st = mode_obj.status
    if st and st != "-":
        teks(frame, st, (22, h - 92 if debug else h - 58), 0.6, 2)
    if debug:
        teks(frame, f"TANGAN {hands_n}", (22, h - 92), 0.5, 2, AMBER)
    return frame
```

Perbaiki: `time.strftime("%d.%m.%Y %H:%M:%S")` (pakai format itu persis — baris di atas hanya contoh, jangan salin bug `MM`).

Kerangka `main()` setelah argparse (ganti mulai dari pembuatan `hands`):

```python
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)

    cap = FrameSource(parse_source(args.source))
    if not cap.isOpened():
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

        if key == ord("s"):
            if capture.save(tampil, "MANUAL"):
                flash.trigger(now)

        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now
        tampil = draw_hud(tampil, mode, MODES[mode], fps, source_label,
                          now, debug, len(detected.all))
        cv2.imshow("Filters", tampil)

    cap.release()
    cv2.destroyAllWindows()
```

- [ ] **Step 3: Jalankan seluruh suite**

Run: `./venv/bin/python -m unittest discover -p "test_*.py" -v`
Expected: semua OK (existing + baru)

- [ ] **Step 4: Smoke e2e (jendela tampil lalu keluar via timeout)**

```bash
timeout 8 ./venv/bin/python main.py --source 0 >/dev/null 2>&1; echo $?           # 124
timeout 8 ./venv/bin/python main.py --source http://127.0.0.1:18080/video >/dev/null 2>&1; echo $?   # 124
```
Expected: `124` untuk keduanya (artinya loop jalan tanpa exception).

- [ ] **Step 5: Commit**
```bash
git add main.py
git commit -m "feat: multi-mode state machine, keyboard, HUD in main"
```

---

### Task 13: Verifikasi akhir & checklist e2e manual

**Files:**
- None (verifikasi)

- [ ] **Step 1: Suite penuh**

Run: `./venv/bin/python -m unittest discover -p "test_*.py" -v`
Expected: semuanya OK (≥ 60 test; 28 lama + 30+ baru)

- [ ] **Step 2: Launcher tetap jalan**

```bash
filters --check
printf "\n" | timeout 8 filters >/dev/null 2>&1; echo $?
```
Expected: deteksi ketiga jalur tercetak; exit `124` (app terbuka via default).

- [ ] **Step 3: Checklist manual (butuh kamera + tangan)**

Minta user menjalankan `filters` lalu verifikasi:
- [ ] Mode FILTERS tampil default: portal 2 tangan + filter berubah saat tangan didekatkan
- [ ] `spasi` mengganti filter; `m`/`f`/`l`/`g` berpindah mode
- [ ] Hold **kepal 2 tangan 2 detik** → masuk LENSA; hold kepal di LENSA → kembali FILTERS
- [ ] Dua tangan bentuk **L** → quad muncul; diam 3 detik → foto tersimpan di `~/Filters/foto/` + flash
- [ ] Hold **tunjuk_ketat 2 detik** → masuk GAMBAR; telunjuk menggambar; kepal menghapus; `p` ganti warna; `c` bersih
- [ ] Hold **telapak di GAMBAR** → kembali ke mode sebelumnya
- [ ] Stream WiFi/USB tetap responsif (FrameSource anti-delay)

- [ ] **Step 4: Commit bila ada perbaikan dari temuan checklist**
```bash
git add -A
git commit -m "fix: checklist fixes"
```

---

## Catatan Eksekusi

- Task 1–11 urut (dependensi antar interface); Task 12 wajib terakhir sebelum Task 13.
- Jika satu test gagal setelah implementasi: kembalikan ke siklus TDD task itu (perbaiki implementasi, jangan lanjut task berikutnya).
- Jangan ubah file yang tertera di "tidak berubah" pada Global Constraints.
