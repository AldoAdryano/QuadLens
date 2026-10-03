# QuadLens

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

## Extending

- **New filter:** define a function in `filters.py` that takes and returns a BGR crop, then append it to `FILTROS` — the cycle adjusts automatically.
- **New lens:** add an entry to `LENSA_LIST` in `lensa.py`.

## Stack

Python 3.10 · OpenCV · MediaPipe 0.10.14 · NumPy

## License

MIT
