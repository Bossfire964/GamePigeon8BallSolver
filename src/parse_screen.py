# AI Generatted for now

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image, ImageDraw
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from templates.parse_template import (
    Ball,
    Bbox,
    Bounds,
    Hole,
    Line,
    Point,
    Scale,
    ScreenResult,
    Target,
)


BORDER_XML = Path(__file__).resolve().parent.parent / "configs" / "border.xml"
SCALE_XML = Path(__file__).resolve().parent.parent / "configs" / "scale.xml"
BALL_XML = Path(__file__).resolve().parent.parent / "configs" / "ball.xml"
TARGETS_XML = Path(__file__).resolve().parent.parent / "configs" / "targets.xml"


# Finds connected component bounding boxes in a mask.
def _component_boxes(mask: np.ndarray, min_area: int) -> list[Hole]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[Hole] = []

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
                minX = min(xs)
                maxX = max(xs)
                minY = min(ys)
                maxY = max(ys)
                components.append(
                    Hole(
                        label="",
                        area=area,
                        center=Point(x=float(sum(xs) / area), y=float(sum(ys) / area)),
                        bbox=Bbox(x1=int(minX), y1=int(minY), x2=int(maxX), y2=int(maxY)),
                    )
                )

    return components


# Detects and labels the six pockets on the table.
def _find_holes(rgb: np.ndarray, min_area: int = 500) -> list[Hole]:
    dark = (rgb[:, :, 0] < 25) & (rgb[:, :, 1] < 25) & (rgb[:, :, 2] < 25)
    components = _component_boxes(dark, min_area=min_area)
    components = sorted(components, key=lambda component: component.area, reverse=True)

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
    by_label: dict[str, Hole] = {}

    sortedByY = sorted(holes, key=lambda hole: hole.center.y)
    top = sorted(sortedByY[:2], key=lambda hole: hole.center.x)
    middle = sorted(sortedByY[2:4], key=lambda hole: hole.center.x)
    bottom = sorted(sortedByY[4:6], key=lambda hole: hole.center.x)

    for label, hole in zip(labels, top + middle + bottom):
        hole.label = label
        by_label[label] = hole

    return [by_label[label] for label in labels]


# Parses a relative x,y point string from XML.
def _parse_relative_point(value: str) -> tuple[float, float]:
    xText, yText = value.split(",", maxsplit=1)
    return float(xText), float(yText)


# Indexes holes by their label.
def _holes_by_label(holes: list[Hole]) -> dict[str, Hole]:
    return {hole.label: hole for hole in holes}


# Returns a hole center as an x,y tuple.
def _hole_center(hole: Hole) -> tuple[float, float]:
    return float(hole.center.x), float(hole.center.y)


# Calculates scale from detected holes and the XML reference.
def _scale_from_reference(
    scale_path: str | Path, holes: list[Hole]
) -> dict[str, float]:
    byLabel = _holes_by_label(holes)
    root = ElementTree.parse(scale_path).getroot()
    reference = root.find("reference")
    if reference is None:
        return {"x": 1.0, "y": 1.0}

    refHoles = {
        node.attrib["label"]: _parse_relative_point(node.attrib["offset"])
        for node in reference.findall("hole")
    }

    required_labels = ("top_left", "top_right", "bottom_left")
    missing = [
        label
        for label in required_labels
        if label not in byLabel or label not in refHoles
    ]
    if missing:
        raise ValueError(
            "Scale reference must include detected and XML holes for: "
            + ", ".join(missing)
        )

    detectedTopLeftX, detectedTopLeftY = _hole_center(byLabel["top_left"])
    detectedTopRightX, _ = _hole_center(byLabel["top_right"])
    _, detectedBottomLeftY = _hole_center(byLabel["bottom_left"])

    refTopLeftX, refTopLeftY = refHoles["top_left"]
    refTopRightX, _ = refHoles["top_right"]
    _, refBottomLeftY = refHoles["bottom_left"]

    referenceWidth = refTopRightX - refTopLeftX
    referenceHeight = refBottomLeftY - refTopLeftY
    if referenceWidth == 0 or referenceHeight == 0:
        raise ValueError("Border reference width and height must be non-zero.")

    return {
        Scale(
            x=float((detectedTopRightX - detectedTopLeftX) / referenceWidth),
            y=float((detectedBottomLeftY - detectedTopLeftY) / referenceHeight)
        )
    }


# Loads the wall line positions from the border XML.
def _load_wall_lines(
    border_path: str | Path, holes: list[Hole], scale: dict[str, float]
) -> list[Line]:
    byLabel = _holes_by_label(holes)
    originX, originY = _hole_center(byLabel["top_left"])

    root = ElementTree.parse(border_path).getroot()
    lines: list[Line] = []
    for node in root.findall("line"):
        startDx, startDy = _parse_relative_point(node.attrib["start"])
        endDx, endDy = _parse_relative_point(node.attrib["end"])
        lines.append(
            Line(
                label=node.attrib["label"],
                start=Point(
                    x=int(round(originX + startDx * scale["x"])),
                    y=int(round(originY + startDy * scale["y"])),
                ),
                end=Point(
                    x=int(round(originX + endDx * scale["x"])),
                    y=int(round(originY + endDy * scale["y"])),
                ),
            )
        )

    return lines


# Calculates the overall table bounds from the wall lines.
def _table_bounds_from_lines(lines: list[Line]) -> dict[str, int]:
    xs = [
        point.x
        for line in lines
        for point in (line.start, line.end)
    ]
    ys = [
        point.y
        for line in lines
        for point in (line.start, line.end)
    ]
    return {
        Bounds(
            left=min(xs),
            right=max(xs),
            top=min(ys),
            bottom=max(ys)
        )
    }


# Loads the scaled cue-ball radius from XML.
def _load_ball_radius(ball_path: str | Path, scale: dict[str, float]) -> int:
    root = ElementTree.parse(ball_path).getroot()
    cueBall = root.find("cue_ball")
    if cueBall is None or "radius" not in cueBall.attrib:
        raise ValueError(f"Expected <cue_ball radius=\"...\" /> in {ball_path}.")

    baseRadius = float(cueBall.attrib["radius"])
    scaledRadius = baseRadius * ((scale["x"] + scale["y"]) / 2)
    return max(2, int(round(scaledRadius)))


# Loads extra shot targets from the XML config.
def _load_shot_targets(
    targets_path: str | Path,
    holes: list[Hole],
    scale: dict[str, float],
) -> list[Target]:
    targetsPath = Path(targets_path)
    if not targetsPath.exists():
        return []

    byLabel = _holes_by_label(holes)
    originX, originY = _hole_center(byLabel["top_left"])
    root = ElementTree.parse(targetsPath).getroot()

    targets: list[Target] = []
    for index, node in enumerate(root.findall("target")):
        dx, dy = _parse_relative_point(node.attrib["point"])
        label = node.attrib.get("label", f"target_{index + 1}")
        targets.append(
            Target(
                label=label,
                kind="target",
                center=Point(
                    x=float(originX + dx * scale["x"]),
                    y=float(originY + dy * scale["y"]),
                ),
            )
        )
    return targets


# Builds a mask for the playable table felt.
def _table_felt_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r < 90) & (g > 75) & (b > 70) & (g > r + 20) & (np.abs(g - b) < 60)


# Builds a mask for bright white pixels.
def _white_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 175) & (g > 175) & (b > 175) & (np.abs(r - g) < 55) & (np.abs(g - b) < 55)


# Builds a mask for near-black pixels.
def _black_mask(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, :, 0] < 55) & (rgb[:, :, 1] < 55) & (rgb[:, :, 2] < 55)


# Returns relative pixel offsets for a filled circle.
def _circle_offsets(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return np.argwhere((x * x + y * y) <= radius * radius) - radius


# Measures how much of each circle area is filled by a mask.
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


# Finds a wall line by label.
def _line_by_label(lines: list[Line], label: str) -> Line | None:
    for line in lines:
        if line.label == label:
            return line
    return None


# Limits ball search to the table interior away from pockets.
def _inside_search_mask(
    shape: tuple[int, int],
    bounds: dict[str, int],
    lines: list[Line],
    holes: list[Hole],
    radius: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    topLine = _line_by_label(lines, "top")
    bottomLine = _line_by_label(lines, "bottom")
    leftLine = _line_by_label(lines, "left_upper") or _line_by_label(lines, "left_lower")
    rightLine = _line_by_label(lines, "right_upper") or _line_by_label(lines, "right_lower")

    topEdge = bounds["top"]
    if topLine is not None:
        topEdge = max(topLine.start.y, topLine.end.y)

    bottomEdge = bounds["bottom"]
    if bottomLine is not None:
        bottomEdge = min(bottomLine.start.y, bottomLine.end.y)

    leftEdge = bounds["left"]
    if leftLine is not None:
        leftEdge = max(leftLine.start.x, leftLine.end.x)

    rightEdge = bounds["right"]
    if rightLine is not None:
        rightEdge = min(rightLine.start.x, rightLine.end.x)

    top = max(radius, topEdge + radius)
    bottom = min(height - radius, bottomEdge - radius)
    left = max(radius, leftEdge + radius)
    right = min(width - radius, rightEdge - radius)
    mask[top : bottom + 1, left : right + 1] = True

    yIndices, xIndices = np.ogrid[:height, :width]
    for hole in holes:
        hx, hy = _hole_center(hole)
        pocketRadius = max(
            hole.bbox.x2 - hole.bbox.x1,
            hole.bbox.y2 - hole.bbox.y1,
        ) / 2
        blockedRadius = pocketRadius + radius
        mask &= (xIndices - hx) ** 2 + (yIndices - hy) ** 2 > blockedRadius**2

    return mask


# Counts significant disconnected white patches inside a ball.
def _count_white_patches(white_patch: np.ndarray) -> tuple[int, list[int]]:
    components = _component_boxes(white_patch, min_area=1)
    areas = sorted((component.area for component in components), reverse=True)
    significantAreas = [area for area in areas if area >= 5]
    return len(significantAreas), significantAreas


# Classifies a detected ball by its pixel makeup.
def _classify_ball(
    rgb: np.ndarray, center_x: int, center_y: int, radius: int
) -> tuple[str, dict[str, Any]]:
    patch = rgb[
        center_y - radius : center_y + radius + 1,
        center_x - radius : center_x + radius + 1,
    ]
    offsets = _circle_offsets(radius)
    circleMask = np.zeros((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    circleMask[offsets[:, 0] + radius, offsets[:, 1] + radius] = True

    white = _white_mask(patch) & circleMask
    black = _black_mask(patch) & circleMask
    whiteRatio = float(white[circleMask].mean())
    blackRatio = float(black[circleMask].mean())
    whitePatchCount, whitePatchAreas = _count_white_patches(white)

    metrics = {
        "white_ratio": whiteRatio,
        "black_ratio": blackRatio,
        "white_patch_count": whitePatchCount,
        "white_patch_areas": whitePatchAreas,
    }

    if whiteRatio >= 0.85:
        return "cue", metrics
    if blackRatio >= 0.25:
        return "eight", metrics
    if whitePatchCount >= 3 or (whiteRatio >= 0.36 and whitePatchCount >= 2):
        return "stripe", metrics
    return "solid", metrics


# Detects and classifies the balls on the table.
def _find_balls(
    rgb: np.ndarray,
    bounds: dict[str, int],
    lines: list[Line],
    holes: list[Hole],
    radius: int,
) -> list[Ball]:
    felt = _table_felt_mask(rgb)
    fillRatio = _circle_ratio(~felt, radius)
    searchArea = _inside_search_mask(felt.shape, bounds, lines, holes, radius)
    candidates = (fillRatio >= 0.68) & searchArea

    components = _component_boxes(candidates, min_area=3)
    balls: list[Ball] = []
    for component in sorted(components, key=lambda item: item.center.y):
        centerX = int(round(component.center.x))
        centerY = int(round(component.center.y))
        ballType, metrics = _classify_ball(rgb, centerX, centerY, radius)
        balls.append(
            Ball(
                type=ballType,
                center=Point(x=centerX, y=centerY),
                radius=radius,
                candidate_area=component.area,
                fill_ratio=float(fillRatio[centerY, centerX]),
                white_ratio=float(metrics["white_ratio"]),
                black_ratio=float(metrics["black_ratio"]),
                white_patch_count=int(metrics["white_patch_count"]),
                white_patch_areas=[int(area) for area in metrics["white_patch_areas"]],
            )
        )

    return balls


# Draws an annotated overlay for parsed table geometry.
def _draw_overlay(
    image: Image.Image,
    lines: list[Line],
    holes: list[Hole],
    balls: list[Ball],
    shot_targets: list[Target],
) -> Image.Image:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)

    for line in lines:
        start = (line.start.x, line.start.y)
        end = (line.end.x, line.end.y)
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
        cx = int(round(hole.center.x))
        cy = int(round(hole.center.y))
        radius = 12
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=(255, 64, 129),
            width=4,
        )
        draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 64, 129))

    for target in shot_targets:
        cx = int(round(target.center.x))
        cy = int(round(target.center.y))
        draw.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=(244, 67, 54))
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=(255, 255, 255), width=2)

    ballColors = {
        "cue": (255, 255, 255),
        "eight": (0, 0, 0),
        "solid": (76, 175, 80),
        "stripe": (33, 150, 243),
    }
    textColors = {
        "cue": (0, 0, 0),
        "eight": (255, 255, 255),
        "solid": (255, 255, 255),
        "stripe": (255, 255, 255),
    }
    for ball in balls:
        cx = ball.center.x
        cy = ball.center.y
        radius = ball.radius + 3
        color = ballColors[ball.type]
        draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            outline=color,
            width=3,
        )
        label = ball.type[0].upper()
        draw.text((cx - 3, cy - 6), label, fill=textColors[ball.type])

    return overlay


# Parses a cropped table screenshot into geometry and balls.
def parse_screen(
    image_path: str | Path,
    output_path: str | Path | None = None,
    border_path: str | Path = BORDER_XML,
    scale_path: str | Path = SCALE_XML,
    ball_path: str | Path = BALL_XML,
    targets_path: str | Path = TARGETS_XML,
) -> ScreenResult:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)

    holes = _find_holes(rgb)
    scale = _scale_from_reference(scale_path, holes)
    lines = _load_wall_lines(border_path, holes, scale)
    bounds = _table_bounds_from_lines(lines)
    ballRadius = _load_ball_radius(ball_path, scale)
    balls = _find_balls(rgb, bounds, lines, holes, ballRadius)
    shotTargets = _load_shot_targets(targets_path, holes, scale)
    overlayPath: str | None = None

    if output_path is not None:
        output = _draw_overlay(image, lines, holes, balls, shotTargets)
        output.save(output_path)
        overlayPath = str(output_path)

    result = ScreenResult(
        image=str(image_path), #path to cropped image
        border=str(border_path), # path to border xml
        scale=str(scale_path), # path to scale xml
        ball_config=str(ball_path), # path to ball xml
        targets_config=str(targets_path), # path to targets xml
        border_scale=scale, # scale factors for x and y
        ball_radius=ballRadius, # scaled cue-ball radius
        table_bounds=bounds, # table bounds from wall lines
        wall_lines=lines, # wall lines from border xml
        holes=holes, # coordinates of the holes
        shot_targets=shotTargets, # coordinates of additional targets
        balls=balls, # coordinates and information of balls
        overlay=overlayPath, #not used???
    )
    return result


# Parses CLI arguments and prints the table parse result.
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
    print(json.dumps(result.getAllElements(), indent=2))


if __name__ == "__main__":
    main()
