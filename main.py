from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.find_screen import COLORS_XML, crop_screen
from src.find_shot import (
    MAX_BOUNCES,
    PICK_SMALLEST_BOUNCES,
    draw_shot,
    find_valid_shots,
    pick_valid_shot,
)
from src.parse_screen import TARGETS_XML, parse_screen


def run_pipeline(
    image: str | Path,
    group: str,
    output: str | Path = "tmp/shot_overlay.png",
    json_output: str | Path | None = None,
    targets: str | Path = TARGETS_XML,
    colors: str | Path = COLORS_XML,
    crop_output: str | Path = "tmp/screen_crop.png",
    max_bounces: int | None = None,
    pick_smallest_bounces: bool | None = None,
) -> dict:
    screen_result = crop_screen(image, crop_output, colors)
    parsed_image = screen_result["image"]

    parse_result = parse_screen(parsed_image, targets_path=targets)
    effective_max_bounces = MAX_BOUNCES if max_bounces is None else max_bounces
    effective_pick_smallest_bounces = (
        PICK_SMALLEST_BOUNCES
        if pick_smallest_bounces is None
        else pick_smallest_bounces
    )
    shot_kwargs = {"max_bounces": effective_max_bounces}
    valid_shots = find_valid_shots(parse_result, group, **shot_kwargs)

    pick_kwargs = {"pick_smallest_bounces": effective_pick_smallest_bounces}
    selected_shot = pick_valid_shot(valid_shots, **pick_kwargs)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    draw_shot(parsed_image, selected_shot, output_path, len(valid_shots))

    result = {
        "image": str(image),
        "screen_crop": screen_result,
        "parsed_image": parsed_image,
        "output": str(output_path),
        "group": group,
        "max_bounces": effective_max_bounces,
        "pick_smallest_bounces": effective_pick_smallest_bounces,
        "valid_shot_count": len(valid_shots),
        "selected_shot": selected_shot,
        "parse_result": parse_result,
    }

    if json_output:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and draw a GamePigeon 8 Ball shot.")
    parser.add_argument(
        "group",
        choices=("stripes", "solids"),
        help="The player group to solve for.",
    )
    parser.add_argument(
        "image",
        nargs="?",
        default="tests/test.png",
        help="Input screenshot path. Defaults to tests/test.png.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="tmp/shot_overlay.png",
        help="Output path for the shot overlay image.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to save the parse and shot result JSON.",
    )
    parser.add_argument(
        "--targets",
        default=TARGETS_XML,
        help="Optional targets.xml path. Defaults to the project targets.xml.",
    )
    parser.add_argument(
        "--colors",
        default=COLORS_XML,
        help="Optional colors.xml path for full-screenshot cropping.",
    )
    parser.add_argument(
        "--crop-output",
        default="tmp/screen_crop.png",
        help="Output path for the intermediate table crop.",
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

    result = run_pipeline(
        image=args.image,
        group=args.group,
        output=args.output,
        json_output=args.json_output,
        targets=args.targets,
        colors=args.colors,
        crop_output=args.crop_output,
        max_bounces=args.max_bounces,
        pick_smallest_bounces=not args.pick_most_bounces,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
