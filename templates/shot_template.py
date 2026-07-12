from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from templates.parse_template import Ball, Hole, Line, Target


@dataclass
class Shot:
    group: str
    target_ball: Ball
    target: Hole | Target
    target_hole: Hole | None
    angle_delta: float
    bounces: int
    cue_line: Line
    object_lines: list[Line]
    object_line: Line

    def getAllElements(self) -> dict[str, Any]:
        return asdict(self)
