from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from main import run_pipeline


# Captures a screenshot to the requested output path.
def _capture_screen(output_path: str | Path) -> None:
    outputPath = Path(output_path)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["screencapture", "-x", str(outputPath)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            from PIL import ImageGrab
        except ImportError as exc:
            raise SystemExit(
                "Could not capture the screen. On macOS, grant Screen Recording "
                "permission or install Pillow with ImageGrab support."
            ) from exc

        image = ImageGrab.grab()
        image.save(outputPath)


# Captures the screen, solves the shot, and optionally shows the overlay.
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture the screen, solve the shot, and optionally display an overlay."
    )
    parser.add_argument(
        "--group",
        choices=("stripes", "solids"),
        default="stripes",
        help="Target ball group. Defaults to stripes so capture.py works with no arguments.",
    )
    parser.add_argument("--delay", type=int, default=5, help="Seconds before capture.")
    parser.add_argument(
        "--screenshot",
        default="tmp/capture.png",
        help="Path for the captured screenshot.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tmp/capture-shot-overlay.png",
        help="Path for the solved shot image.",
    )
    parser.add_argument(
        "--json-output",
        default="tmp/capture-shot.json",
        help="Path for the shot result JSON.",
    )
    parser.add_argument(
        "--crop-output",
        default="tmp/capture-crop.png",
        help="Path for the intermediate table crop.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not launch the click-through overlay after solving.",
    )
    parser.add_argument(
        "--line-width",
        type=float,
        default=1.0,
        help="Line width for the click-through overlay.",
    )
    parser.add_argument(
        "--max-bounces",
        type=int,
        default=None,
        help="Maximum object-ball rail bounces to consider. Defaults to find_shot.py.",
    )
    parser.add_argument(
        "--pick-most-bounces",
        action="store_true",
        help="Pick the first shot with the most bounces instead of the fewest.",
    )
    args = parser.parse_args()

    for remaining in range(args.delay, 0, -1):
        print(f"Capturing screen in {remaining}...")
        time.sleep(1)

    _capture_screen(args.screenshot)
    print(f"Captured screenshot: {args.screenshot}")

    result = run_pipeline(
        image=args.screenshot,
        group=args.group,
        output=args.output,
        json_output=args.json_output,
        crop_output=args.crop_output,
        max_bounces=args.max_bounces,
        pick_smallest_bounces=not args.pick_most_bounces,
    )
    print(json.dumps(result, indent=2))

    if not args.no_display:
        subprocess.run(
            [
                sys.executable,
                "display_shot.py",
                args.json_output,
                "--line-width",
                str(args.line_width),
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
