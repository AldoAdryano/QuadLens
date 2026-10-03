# QuadLens

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-4f4f4f?logo=mediapipe&logoColor=white)](https://github.com/google/mediapipe)
[![Tests](https://img.shields.io/badge/tests-91-brightgreen)](https://github.com/AldoAdryano/QuadLens)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

A multi-mode webcam playground: AR filters behind a hand-framed portal, perspective-warped photo lenses, and air-painting with gestures — all in one live camera app.

## Modes

| Mode | What it does |
|------|--------------|
| **FILTERS** | An AR portal framed by the thumb + index of both hands. 8 visual filters render inside it. Bring your hands together (pinch) to cycle filters. |
| **LENSA** | Make an **L** with each hand to form a quad. The quad is perspective-warped live with one of 7 retro lenses: `MONO`, `KONTRAS`, `FILM`, `GARIS`, `AMBANG`, `DITHER`, `NEGATIF`. Hold the quad still for 3 seconds → auto-photo with flash. |
| **GAMBAR** | Paint in the air. Point with your index to draw, close a fist to wipe the canvas, tap both index fingertips together to cycle pen color (white, cyan, magenta, amber). |

Switch modes with gestures (hold 2 seconds) or with the keyboard.

## Quick start

```bash
git clone https://github.com/AldoAdryano/QuadLens.git
cd QuadLens
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

./filters                       # interactive launcher: laptop cam / phone WiFi / phone USB
# or jump straight in:
python main.py --source 0                              # laptop webcam
python main.py --source http://192.168.1.5:8080/video  # phone stream (IP Webcam app)
```

`./filters` remembers your last source and offers a `--check` probe for camera / WiFi / USB before you start.

## Gesture cheat sheet

| From | Gesture (hold ~2 s) | To |
|------|--------------------|----|
| FILTERS | both hands closed into fists | LENSA |
| FILTERS | strict point (index only, no thumb) | GAMBAR |
| LENSA | strict point (index only, no thumb) | GAMBAR |
| LENSA | closed fist | FILTERS |
| GAMBAR | open palm | previous mode |

In-mode gestures:

- **FILTERS** — pinch the portal closed to advance the filter.
- **LENSA** — form the quad with two L-hands; spread to open, hold still to shoot.
- **GAMBAR** — index draws, fist clears, peace sign shows `PINDAH`.

## Keyboard

| Key | Action |
|-----|--------|
| `f` / `l` / `g` | Switch to FILTERS / LENSA / GAMBAR |
| `m` | Next mode |
| `s` | Save current frame as a photo (flash) |
| `space` | Next filter (FILTERS) / next lens (LENSA) |
| `p` | Next pen color (GAMBAR) |
| `c` | Clear canvas (GAMBAR) |
| `d` | Toggle debug overlay |
| `q` | Quit |

## How it works

```
 laptop cam · phone WiFi (IP Webcam) · phone USB/adb
        │
        ▼
   FrameSource ──► mirror ──► MediaPipe Hands ──► finger states
        │                                              │
        │              ┌───────────────────────────────┤
        │              ▼               ▼               ▼
        │          FILTERS           LENSA          GAMBAR
        │      portal + 8 fx    quad warp +      air painting
        │      pinch to cycle    7 retro lenses   pen gestures
        │              │          hold 3 s = photo      │
        │              └───────────────┬────────────────┘
        ▼                              ▼
      HUD ◄── fps · source · mode · status      flash · save (s / auto)
        │
        ▼
   window (OpenCV HighGUI) · photo → ~/Filters/foto/
```

Each frame runs through exactly one mode module; the HUD is drawn last, so the on-screen text always matches the active mode. Gestures are debounced with a 2-second hold (`HoldTransition`) so a stray frame never switches modes.

## Photos

Photos are always saved to `~/Filters/foto/`:

- `LENS_*.png` — auto-photo after holding the quad still for 3 s
- `MANUAL_*.png` — frames captured with `s`

The folder is git-ignored; nothing personal is committed.

## Project structure

```
QuadLens/
├── main.py            capture loop, keyboard, mode switching, HUD
├── launcher.py        interactive source picker (cam / WiFi / USB), state
├── frame_source.py    webcam, HTTP/MJPEG, and USB frame sources
├── modes.py           gesture transitions + FILTERS portal mode
├── filters.py         the 8 portal filters
├── lensa.py           quad tracking, perspective warp, 7 retro lenses
├── draw.py            air-painting mode (pen, canvas, gestures)
├── hand_tracking.py   MediaPipe landmarks → finger states
├── geometry.py        portal geometry + pinch detection
├── capture.py         photo saving, flash, counters
├── filters            `./filters` launcher shim
├── test_*.py          unit tests
└── docs/              design spec + implementation plan
```

## Tests

```bash
python -m unittest discover -p "test_*.py"
```

91 tests covering keyboard routing, modes, drawing, lenses, capture, launcher, and frame sources.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `camera busy` / black feed | Close other apps using the webcam; test with `./filters --check` |
| WiFi stream won't start | Phone and laptop on the **same WiFi**; IP Webcam must be running; URL format `http://<phone-ip>:8080/video` — probe with `./filters --check` |
| Gestures don't switch modes | Hold the pose ~2 s steadily; improve lighting — MediaPipe needs a clearly visible hand |
| Quad won't open in LENSA | Both hands need a clear **L** (thumb + index spread); the quad must span at least ~120 px |
| Window doesn't appear | A display is required (X11/Wayland); the app is not headless |
| Install errors from MediaPipe | Keep the pinned `mediapipe==0.10.14` |
| Photos "missing" | They always land in `~/Filters/foto/` (check `LENS_*` / `MANUAL_*`) |

Press `d` for the debug overlay (landmark info) while diagnosing.

## Extending

- **New filter:** define a function in `filters.py` that takes and returns a BGR crop, then append it to `FILTROS` — the cycle adjusts automatically.
- **New lens:** add an entry to `LENSA_LIST` in `lensa.py`.

## Stack

Python 3.10+ (tested on 3.11) · OpenCV 4.10 · MediaPipe 0.10.14 · NumPy

## License

MIT
