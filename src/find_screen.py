# AI Generated for now

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np
from PIL import Image


COLORS_XML = Path(__file__).resolve().parent.parent / "configs" / "colors.xml"


# Parses an XML rgb attribute into integer channels.
def _parse_rgb(value: str) -> tuple[int, int, int]:
    r, g, b = value.split(",", maxsplit=2)
    return int(r), int(g), int(b)


# Loads the color detection config from XML.
def _load_colors(colors_path: str | Path) -> dict[str, dict[str, Any]]:
    root = ElementTree.parse(colors_path).getroot()
    colors: dict[str, dict[str, Any]] = {}
    for node in root.findall("color"):
        config: dict[str, Any] = dict(node.attrib)
        config["rgb"] = _parse_rgb(config["rgb"])
        for key, value in list(config.items()):
            if key not in {"name", "rgb"}:
                config[key] = int(value)
        colors[config["name"]] = config
    return colors


# Finds connected component bounding boxes in a mask.
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
                            and mask[ny, nx]
                            and not seen[ny, nx]
                        ):
                            seen[ny, nx] = True
                            queue.append((nx, ny))

            if len(xs) >= min_area:
                components.append(
                    {
                        "area": len(xs),
                        "bbox": {
                            "x1": min(xs),
                            "y1": min(ys),
                            "x2": max(xs),
                            "y2": max(ys),
                        },
                        "center": {
                            "x": sum(xs) / len(xs),
                            "y": sum(ys) / len(ys),
                        },
                    }
                )

    return components


# Builds a bounding box covering all selected components.
def _union_bbox(components: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "x1": min(component["bbox"]["x1"] for component in components),
        "y1": min(component["bbox"]["y1"] for component in components),
        "x2": max(component["bbox"]["x2"] for component in components),
        "y2": max(component["bbox"]["y2"] for component in components),
    }


# Returns the inclusive width of a bounding box.
def _bbox_width(bbox: dict[str, int]) -> int:
    return bbox["x2"] - bbox["x1"] + 1


# Returns the inclusive height of a bounding box.
def _bbox_height(bbox: dict[str, int]) -> int:
    return bbox["y2"] - bbox["y1"] + 1


# Builds a mask for the gray table background.
def _gray_mask(rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    baseR, baseG, baseB = config["rgb"]
    tolerance = config["tolerance"]
    neutralTolerance = config["neutral_tolerance"]

    nearBase = (
        (np.abs(r - baseR) <= tolerance)
        & (np.abs(g - baseG) <= tolerance)
        & (np.abs(b - baseB) <= tolerance)
    )
    neutral = (
        (np.abs(r - g) <= neutralTolerance)
        & (np.abs(g - b) <= neutralTolerance)
        & (r >= config["min"])
        & (r <= config["max"])
    )
    return nearBase & neutral


# Builds a mask for the brown table rails.
def _brown_mask(rgb: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    baseR, baseG, baseB = config["rgb"]
    tolerance = config["tolerance"]

    nearBase = (
        (np.abs(r - baseR) <= tolerance)
        & (np.abs(g - baseG) <= tolerance)
        & (np.abs(b - baseB) <= tolerance)
    )
    brownShape = (
        (r >= config["red_min"])
        & (r <= config["red_max"])
        & (g >= config["green_min"])
        & (g <= config["green_max"])
        & (b >= config["blue_min"])
        & (b <= config["blue_max"])
        & (r > g + config["red_over_green"])
        & (np.abs(g - b) <= config["green_blue_tolerance"])
    )
    return nearBase & brownShape


# Measures horizontal overlap between two bounding boxes.
def _overlap_ratio(a: dict[str, int], b: dict[str, int]) -> float:
    overlap = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]) + 1)
    return overlap / max(1, min(_bbox_width(a), _bbox_width(b)))


# Finds the broad gray background region around the table.
def _find_gray_crop(rgb: np.ndarray, config: dict[str, Any]) -> dict[str, int] | None:
    height, width, _ = rgb.shape
    minArea = max(500, int(height * width * 0.002))
    components = _component_boxes(_gray_mask(rgb, config), min_area=minArea)
    if not components:
        return None

    largest = max(components, key=lambda component: component["area"])
    largestBbox = largest["bbox"]
    selected = [
        component
        for component in components
        if _overlap_ratio(component["bbox"], largestBbox) >= 0.65
    ]
    return _union_bbox(selected)


# Finds the table crop by locating brown rails inside the gray region.
def _find_brown_table_crop(
    rgb: np.ndarray, gray_bbox: dict[str, int], config: dict[str, Any]
) -> dict[str, int] | None:
    crop = rgb[
        gray_bbox["y1"] : gray_bbox["y2"] + 1,
        gray_bbox["x1"] : gray_bbox["x2"] + 1,
    ]
    height, width, _ = crop.shape
    minArea = max(100, int(height * width * 0.004))
    components = _component_boxes(_brown_mask(crop, config), min_area=minArea)
    if len(components) < 4:
        return None

    railBbox = _union_bbox(components)
    return {
        "x1": gray_bbox["x1"] + railBbox["x1"],
        "y1": gray_bbox["y1"] + railBbox["y1"],
        "x2": gray_bbox["x1"] + railBbox["x2"],
        "y2": gray_bbox["y1"] + railBbox["y2"],
    }


# Crops a full screenshot down to the detected table area.
def crop_screen(
    image_path: str | Path,
    output_path: str | Path | None = None,
    colors_path: str | Path = COLORS_XML,
) -> dict[str, Any]:
    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    rgb = np.array(image)
    colors = _load_colors(colors_path)

    grayBbox = _find_gray_crop(rgb, colors["table_background_gray"])
    if grayBbox is None:
        return {
            "source": str(image_path),
            "image": str(image_path),
            "cropped": False,
            "reason": "gray background not found",
        }

    tableBbox = _find_brown_table_crop(rgb, grayBbox, colors["table_rail_brown"])
    if tableBbox is None:
        return {
            "source": str(image_path),
            "image": str(image_path),
            "cropped": False,
            "gray_bbox": grayBbox,
            "reason": "brown table rails not found",
        }

    if output_path is None:
        output_path = image_path
    outputPath = Path(output_path)
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    croppedImage = image.crop(
        (tableBbox["x1"], tableBbox["y1"], tableBbox["x2"] + 1, tableBbox["y2"] + 1)
    )
    croppedImage.save(outputPath)

    return {
        "source": str(image_path),
        "image": str(outputPath),
        "cropped": True,
        "gray_bbox": grayBbox,
        "table_bbox": tableBbox,
        "size": {"width": croppedImage.width, "height": croppedImage.height},
    }


# Parses CLI arguments and prints the crop result.
def main() -> None:
    parser = argparse.ArgumentParser(description="Crop a full screenshot down to the pool table.")
    parser.add_argument("image", help="Input screenshot path.")
    parser.add_argument(
        "-o",
        "--output",
        default="tmp/screen_crop.png",
        help="Output path for the cropped table image.",
    )
    parser.add_argument(
        "--colors",
        default=COLORS_XML,
        help="XML color configuration path.",
    )
    args = parser.parse_args()

    result = crop_screen(args.image, args.output, args.colors)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
