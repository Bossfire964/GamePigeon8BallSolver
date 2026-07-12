from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Point:
    x: float
    y: float


@dataclass
class Bbox:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class Bounds:
    left: int
    right: int
    top: int
    bottom: int


@dataclass
class Scale:
    x: float
    y: float


@dataclass
class Line:
    label: str
    start: Point
    end: Point


@dataclass
class Hole:
    label: str
    area: int
    center: Point
    bbox: Bbox


@dataclass
class Target:
    label: str
    kind: str
    center: Point


@dataclass
class Ball:
    type: str
    center: Point
    radius: int
    candidate_area: int
    fill_ratio: float
    white_ratio: float
    black_ratio: float
    white_patch_count: int
    white_patch_areas: list[int]


@dataclass
class ScreenResult:
    image: str
    border: str
    scale: str
    ball_config: str
    targets_config: str
    border_scale: Scale
    ball_radius: int
    table_bounds: Bounds
    wall_lines: list[Line]
    holes: list[Hole]
    shot_targets: list[Target]
    balls: list[Ball]
    overlay: str | None = None

    def getAllElements(self) -> dict[str, Any]:
        return asdict(self)
