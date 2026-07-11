from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageDraw


Line = dict[str, Any]
Point = dict[str, Any]
Ball = dict[str, Any]
BORDER_XML = Path(__file__).resolve().parent.parent / "border.xml"
SCALE_XML = Path(__file__).resolve().parent.parent / "scale.xml"
BALL_XML = Path(__file__).resolve().parent.parent / "ball.xml"
TARGETS_XML = Path(__file__).resolve().parent.parent / "targets.xml"


def _component_boxes(mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue

            queue: deque[tuple[int, int]] = deque([(x, y)])
            seen[y, x] = True
            xs: list[int] = []
            ys: list[int] = []

            while queue:
                cx, cy = queue.popleft()
                xs.append(cx)
                ys.append(cy)

                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if (
                            0 <= nx < width
                            and 0 <= ny < height
                            and not seen[ny, nx]
                            and mask[ny, nx]
                        ):
                            seen[ny, nx] = True
                            queue.append((nx, ny))

            area = len(xs)
            if area >= min_area:
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                components.append(
                    {
                        "area": area,
                        "center": {
                            "x": float(sum(xs) / area),
                            "y": float(sum(ys) / area),
                        },
                        "bbox": {
                            "x1": int(min_x),
                            "y1": int(min_y),
                            "x2": int(max_x),
                            "y2": int(max_y),
                        },
                    }
                )

    return components


def _find_holes(rgb: np.ndarray, min_area: int = 500) -> list[Point]:
    dark = (rgb[:, :, 0] < 25) & (rgb[:, :, 1] < 25) & (rgb[:, :, 2] < 25)
    components = _component_boxes(dark, min_area=min_area)
    components = sorted(components, key=lambda component: component["area"], reverse=True)

    holes = components[:6]
    if len(holes) < 6:
        raise ValueError(f"Expected 6 pockets, found {len(holes)}.")

    labels = [
        "top_left",
        "top_right",
        "middle_left",
        "middle_right",
        "bottom_left",
        "bottom_right",
    ]
    by_label: dict[str, Point] = {}

    sorted_by_y = sorted(holes, key=lambda hole: hole["center"]["y"])
    top = sorted(sorted_by_y[:2], key=lambda hole: hole["center"]["x"])
    middle = sorted(sorted_by_y[2:4], key=lambda hole: hole["center"]["x"])
    bottom = sorted(sorted_by_y[4:6], key=lambda hole: hole["center"]["x"])

    for label, hole in zip(labels, top + middle + bottom):
        by_label[label] = {"label": label, **hole}

    return [by_label[label] for label in labels]


def _parse_relative_point(value: str) -> tuple[float, float]:
    x_text, y_text = value.split(",", maxsplit=1)
    return float(x_text), float(y_text)


def _holes_by_label(holes: list[Point]) -> dict[str, Point]:
    return {hole["label"]: hole for hole in holes}


def _hole_center(hole: Point) -> tuple[float, float]:
    return float(hole["center"]["x"]), float(hole["center"]["y"])


def _scale_from_reference(
    scale_path: str | Path, holes: list[Point]
) -> dict[str, float]:
    by_label = _holes_by_label(holes)
    root = ElementTree.parse(scale_path).getroot()
    reference = root.find("reference")
    if reference is None:
        return {"x": 1.0, "y": 1.0}

    ref_holes = {
        node.attrib["label"]: _parse_relative_point(node.attrib["offset"])
        for node in reference.findall("hole")
    }

    required_labels = ("top_left", "top_right", "bottom_left")
    missing = [
        label
        for label in required_labels
        if label not in by_label or label not in ref_holes
    ]
    if missing:
        raise ValueError(
            "Scale reference must include detected and XML holes for: "
            + ", ".join(missing)
        )

    detected_top_left_x, detected_top_left_y = _hole_center(by_label["top_left"])
    detected_top_right_x, _ = _hole_center(by_label["top_right"])
    _, detected_bottom_left_y = _hole_center(by_label["bottom_left"])

    ref_top_left_x, ref_top_left_y = ref_holes["top_left"]
    ref_top_right_x, _ = ref_holes["top_right"]
    _, ref_bottom_left_y = ref_holes["bottom_left"]

    reference_width = ref_top_right_x - ref_top_left_x
    reference_height = ref_bottom_left_y - ref_top_left_y
    if reference_width == 0 or reference_height == 0:
        raise ValueError("Border reference width and height must be non-zero.")

    return {
        "x": (detected_top_right_x - detected_top_left_x) / reference_width,
        "y": (detected_bottom_left_y - detected_top_left_y) / reference_height,
    }


def _load_wall_lines(
    border_path: str | Path, holes: list[Point], scale: dict[str, float]
) -> list[Line]:
    by_label = _holes_by_label(holes)
    origin_x, origin_y = _hole_center(by_label["top_left"])

    root = ElementTree.parse(border_path).getroot()
    lines: list[Line] = []
    for node in root.findall("line"):
        start_dx, start_dy = _parse_relative_point(node.attrib["start"])
        end_dx, end_dy = _parse_relative_point(node.attrib["end"])
        lines.append(
            {
                "label": node.attrib["label"],
                "start": {
                    "x": int(round(origin_x + start_dx * scale["x"])),
                    "y": int(round(origin_y + start_dy * scale["y"])),
                },
                "end": {
                    "x": int(round(origin_x + end_dx * scale["x"])),
                    "y": int(round(origin_y + end_dy * scale["y"])),
                },
            }
        )

    if len(lines) != 6:
        raise ValueError(f"Expected 6 border lines in {border_path}, found {len(lines)}.")
    return lines


def _table_bounds_from_lines(lines: list[Line]) -> dict[str, int]:
    xs = [
        point["x"]
        for line in lines
        for point in (line["start"], line["end"])
    ]
    ys = [
        point["y"]
        for line in lines
        for point in (line["start"], line["end"])
    ]
    return {
        "left": min(xs),
        "right": max(xs),
        "top": min(ys),
        "bottom": max(ys),
    }


def _load_ball_radius(ball_path: str | Path, scale: dict[str, float]) -> int:
    root = ElementTree.parse(ball_path).getroot()
    cue_ball = root.find("cue_ball")
    if cue_ball is None or "radius" not in cue_ball.attrib:
        raise ValueError(f"Expected <cue_ball radius=\"...\" /> in {ball_path}.")

    base_radius = float(cue_ball.attrib["radius"])
    scaled_radius = base_radius * ((scale["x"] + scale["y"]) / 2)
    return max(2, int(round(scaled_radius)))


def _load_shot_targets(
    targets_path: str | Path,
    holes: list[Point],
    scale: dict[str, float],
) -> list[Point]:
    targets_path = Path(targets_path)
    if not targets_path.exists():
        return []

    by_label = _holes_by_label(holes)
    origin_x, origin_y = _hole_center(by_label["top_left"])
    root = ElementTree.parse(targets_path).getroot()

    targets: list[Point] = []
    for index, node in enumerate(root.findall("target")):
        dx, dy = _parse_relative_point(node.attrib["point"])
        label = node.attrib.get("label", f"target_{index + 1}")
        targets.append(
            {
                "label": label,
                "kind": "target",
                "center": {
                    "x": float(origin_x + dx * scale["x"]),
                    "y": float(origin_y + dy * scale["y"]),
                },
            }
        )
    return targets


def _table_felt_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r < 90) & (g > 75) & (b > 70) & (g > r + 20) & (np.abs(g - b) < 60)


def _white_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 175) & (g > 175) & (b > 175) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55)


def _black_mask(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, :, 0] < 55) & (rgb[:, :, 1] < 55) & (rgb[:, :, 2] < 55)


def _circle_offsets(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return np.argwhere((x * x + y * y) <= radius * radius) - radius


def _circle_ratio(mask: np.ndarray, radius: int) -> np.ndarray:
    height, width = mask.shape
    ratios = np.zeros((height, width), dtype=np.uint16)
    offsets = _circle_offsets(radius)
    for dy, dx in offsets:
        ratios[radius : height - radius, radius : width - radius] += mask[
            radius + dy : height - radius + dy,
            radius + dx : width - radius + dx,
        ]
    return ratios / len(offsets)


def _inside_search_mask(
    shape: tuple[int, int],
    bounds: dict[str, int],
    holes: list[Point],
    radius: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    top = max(radius, bounds["top"] + radius)
    bottom = min(height - radius, bounds["bottom"] - radius)
    left = max(radius, bounds["left"] + radius)
    right = min(width - radius, bounds["right"] - radius)
    mask[top : bottom + 1, left : right + 1] = True

    y_indices, x_indices = np.ogrid[:height, :width]
    for hole in holes:
        hx, hy = _hole_center(hole)
        pocket_radius = max(
            hole["bbox"]["x2"] - hole["bbox"]["x1"],
            hole["bbox"]["y2"] - hole["bbox"]["y1"],
        ) / 2
        blocked_radius = pocket_radius + radius
        mask &= (x_indices - hx) ** 2 + (y_indices - hy) ** 2 > blocked_radius**2

    return mask


def _count_white_patches(white_patch: np.ndarray) -> tuple[int, list[int]]:
    components = _component_boxes(white_patch, min_area=1)
    areas = sorted((component["area"] for component in components), reverse=True)
    significant_areas = [area for area in areas if area >= 5]
    return len(significant_areas), significant_areas


def _classify_ball(
    rgb: np.ndarray, center_x: int, center_y: int, radius: int
) -> tuple[str, dict[str, Any]]:
    patch = rgb[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ]
    offsets = _circle_offsets(radius)
    circle_mask = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    circle_mask[offsets[:, 0] + radius, offsets[:, 1] + radius] = True

    white = _white_mask(patch) & circle_mask
    black = _black_mask(patch) & circle_mask
    white_ratio = float(white[circle_mask].mean())
    black_ratio = float(black[circle_mask].mean())
    white_patch_count, white_patch_areas = _count_white_patches(white)

    metrics = {
        "white_ratio": white_ratio,
        "black_ratio": black_ratio,
        "white_patch_count": white_patch_count,
        "white_patch_areas": white_patch_areas,
    }

    if white_ratio >= 0.85:
        return "cue", metrics
    if black_ratio >= 0.25:
        return "eight", metrics
    if white_patch_count >= 3 or (white_ratio >= 0.36 and white_patch_count >= 2):
        return "stripe", metrics
    return "solid", metrics


def _find_balls(
    rgb: np.ndarray,
    bounds: dict[str, int],
    holes: list[Point],
    radius: int,
) -> list[Ball]:
    felt = _table_felt_mask(rgb)
    fill_ratio = _circle_ratio(~felt, radius)
    search_area = _inside_search_mask(felt.shape, bounds, holes, radius)
    candidates = (fill_ratio >= 0.68) & search_area

    components = _component_boxes(candidates, min_area=3)
    balls: list[Ball] = []
    for component in sorted(components, key=lambda item: item["center"]["y"]):
        center_x = int(round(component["center"]["x"]))
        center_y = int(round(component["center"]["y"]))
        ball_type, metrics = _classify_ball(rgb, center_x, center_y, radius)
        balls.append(
            {
                "type": ball_type,
                "center": {"x": center_x, "y": center_y},
                "radius": radius,
                "candidate_area": component["area"],
                "fill_ratio": float(fill_ratio[center_y, center_x]),
                **metrics,
            }
        )

    return balls


def _draw_overlay(
    image: Image.Image,
    lines: list[Line],
    holes: list[Point],
    balls: list[Ball],
    shot_targets: list[Point],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    for line in lines:
        start = (line["start"]["x"], line["start"]["y"])
        end = (line["end"]["x"], line["end"]["y"])
        draw.line([start, end], fill=(255, 235, 59), width=4)
        draw.ellipse(
            (start[0] - 3, start[1] - 3, start[0] + 3, start[1] + 3),
            fill=(255, 235, 59),
        )
        draw.ellipse(
            (end[0] - 3, end[1] - 3, end[0] + 3, end[1] + 3),
            fill=(255, 235, 59),
        )

    for hole in holes:
        cx = int(round(hole["center"]["x"]))
        cy = int(round(hole["center"]["y"]))
        radius = 12
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(255, 64, 129),
            width=4,
        )
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 64, 129))

    for target in shot_targets:
        cx = int(round(target["center"]["x"]))
        cy = int(round(target["center"]["y"]))
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(244, 67, 54))
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=(255, 255, 255), width=2)

    ball_colors = {
        "cue": (255, 255, 255),
        "eight": (0, 0, 0),
        "solid": (76, 175, 80),
        "stripe": (33, 150, 243),
    }
    text_colors = {
        "cue": (0, 0, 0),
        "eight": (255, 255, 255),
        "solid": (255, 255, 255),
        "stripe": (255, 255, 255),
    }
    for ball in balls:
        cx = ball["center"]["x"]
        cy = ball["center"]["y"]
        radius = ball["radius"] + 3
        color = ball_colors[ball["type"]]
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=color,
            width=3,
        )
        label = ball["type"][0].upper()
        draw.text((cx - 3, cy - 6), label, fill=text_colors[ball["type"]])

    return overlay


def parse_screen(
    image_path: str | Path,
    output_path: str | Path | None = None,
    border_path: str | Path = BORDER_XML,
    scale_path: str | Path = SCALE_XML,
    ball_path: str | Path = BALL_XML,
    targets_path: str | Path = TARGETS_XML,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)

    holes = _find_holes(rgb)
    scale = _scale_from_reference(scale_path, holes)
    lines = _load_wall_lines(border_path, holes, scale)
    bounds = _table_bounds_from_lines(lines)
    ball_radius = _load_ball_radius(ball_path, scale)
    balls = _find_balls(rgb, bounds, holes, ball_radius)
    shot_targets = _load_shot_targets(targets_path, holes, scale)

    result = {
        "image": str(image_path),
        "border": str(border_path),
        "scale": str(scale_path),
        "ball_config": str(ball_path),
        "targets_config": str(targets_path),
        "border_scale": scale,
        "ball_radius": ball_radius,
        "table_bounds": bounds,
        "wall_lines": lines,
        "holes": holes,
        "shot_targets": shot_targets,
        "balls": balls,
    }

    if output_path is not None:
        output = _draw_overlay(image, lines, holes, balls, shot_targets)
        output.save(output_path)
        result["overlay"] = str(output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find GamePigeon 8 Ball table wall segments and pockets."
    )
    parser.add_argument("image", help="Input screenshot path.")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional output path for an annotated overlay image.",
    )
    parser.add_argument(
        "-b",
        "--border",
        default=BORDER_XML,
        help="XML border template with lines relative to the top-left pocket center.",
    )
    parser.add_argument(
        "-s",
        "--scale",
        default=SCALE_XML,
        help="XML scale reference shared by border and ball templates.",
    )
    parser.add_argument(
        "--balls",
        default=BALL_XML,
        help="XML ball template containing the cue-ball radius.",
    )
    parser.add_argument(
        "--targets",
        default=TARGETS_XML,
        help="XML shot target template with points relative to the top-left pocket center.",
    )
    args = parser.parse_args()

    result = parse_screen(
        args.image,
        args.output,
        args.border,
        args.scale,
        args.balls,
        args.targets,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
