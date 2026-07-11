# GamePigeon 8 Ball Shot Helper

Small Python tool for detecting a GamePigeon 8 Ball table from a screenshot, finding balls/pockets, solving a simple two-line shot, and optionally drawing a click-through overlay on top of your screen.

## Requirements

- macOS for live screen capture via `screencapture`
- Python 3.10+
- Python packages:
  - `Pillow`
  - `numpy`
  - `PyQt6` for the live click-through overlay

Install packages:

```bash
python3 -m pip install Pillow numpy PyQt6
```

macOS may also require Screen Recording permission for your terminal app.

## Pipeline

1. `capture.py` takes a screenshot after a short countdown.
2. `find_screen.py` crops the full screenshot down to the pool table.
3. `parse_screen.py` finds pockets, walls, balls, and custom targets.
4. `find_shot.py` finds valid shots for stripes or solids.
5. `display_shot.py` can show the selected shot as a transparent click-through overlay.

## Run

Capture the screen, solve for stripes, and show the overlay:

```bash
python3 capture.py
```

Capture and solve for solids:

```bash
python3 capture.py --group solids
```

Run on an existing screenshot:

```bash
python3 main.py stripes tests/full_test.png -o tmp/shot.png --json-output tmp/shot.json
```

Show an overlay from saved JSON:

```bash
python3 display_shot.py tmp/shot.json
```

Debug overlay placement:

```bash
python3 display_shot.py tmp/shot.json --debug --no-click-through
```

## Config Files

- `scale.xml`: table scale and display scale
- `border.xml`: table wall lines
- `ball.xml`: ball radius
- `targets.xml`: extra shot targets
- `colors.xml`: screen-crop colors
