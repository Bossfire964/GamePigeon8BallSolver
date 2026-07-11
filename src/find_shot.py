from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


Point = dict[str, float]
Shot = dict[str, Any]

ANGLE_SWEEP_DEGREES = 180
ANGLE_STEP_DEGREES = 1
POCKET_ALIGNMENT_DEGREES = 1.0
BALL_CLEARANCE_SCALE = 1.0
BALL_CLEARANCE_PADDING_PIXELS = 1.0
WALL_EPSILON = 1e-6
MAX_BOUNCES = 1
PICK_SMALLEST_BOUNCES = True
RAY_LENGTH = 5000.0


def _point(item: dict[str, Any]) -> Point:
    center = item["center"]
    return {"x": float(center["x"]), "y": float(center["y"])}


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _angle(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))


def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


def _direction(angle: float) -> Point:
    radians = math.radians(angle)
    return {"x": math.cos(radians), "y": math.sin(radians)}


def _add_scaled(point: Point, direction: Point, scale: float) -> Point:
    return {
        "x": point["x"] + direction["x"] * scale,
        "y": point["y"] + direction["y"] * scale,
    }


def _wall_points(wall: dict[str, Any]) -> tuple[Point, Point]:
    return (
        {"x": float(wall["start"]["x"]), "y": float(wall["start"]["y"])},
        {"x": float(wall["end"]["x"]), "y": float(wall["end"]["y"])},
    )


def _sample_segment(start: Point, end: Point, step: float) -> list[Point]:
    distance = _distance(start, end)
    if distance == 0:
        return [start]

    sample_count = max(1, int(math.ceil(distance / step)))
    return [
        {
            "x": start["x"] + (end["x"] - start["x"]) * index / sample_count,
            "y": start["y"] + (end["y"] - start["y"]) * index / sample_count,
        }
        for index in range(sample_count + 1)
    ]


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    segment_x = end["x"] - start["x"]
    segment_y = end["y"] - start["y"]
    segment_length_squared = segment_x * segment_x + segment_y * segment_y
    if segment_length_squared == 0:
        return _distance(point, start)

    point_x = point["x"] - start["x"]
    point_y = point["y"] - start["y"]
    projection = (point_x * segment_x + point_y * segment_y) / segment_length_squared
    projection = max(0.0, min(1.0, projection))
    closest = {
        "x": start["x"] + segment_x * projection,
        "y": start["y"] + segment_y * projection,
    }
    return _distance(point, closest)


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b["y"] - a["y"]) * (c["x"] - b["x"]) - (
        b["x"] - a["x"]
    ) * (c["y"] - b["y"])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a["x"], c["x"]) - WALL_EPSILON <= b["x"] <= max(a["x"], c["x"]) + WALL_EPSILON
        and min(a["y"], c["y"]) - WALL_EPSILON <= b["y"] <= max(a["y"], c["y"]) + WALL_EPSILON
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    if abs(o1) <= WALL_EPSILON and _on_segment(a, c, b):
        return True
    if abs(o2) <= WALL_EPSILON and _on_segment(a, d, b):
        return True
    if abs(o3) <= WALL_EPSILON and _on_segment(c, a, d):
        return True
    if abs(o4) <= WALL_EPSILON and _on_segment(c, b, d):
        return True
    return False


def _cross(a: Point, b: Point) -> float:
    return a["x"] * b["y"] - a["y"] * b["x"]


def _ray_segment_intersection(
    origin: Point, direction: Point, wall_start: Point, wall_end: Point
) -> tuple[float, Point] | None:
    segment = {
        "x": wall_end["x"] - wall_start["x"],
        "y": wall_end["y"] - wall_start["y"],
    }
    denominator = _cross(direction, segment)
    if abs(denominator) <= WALL_EPSILON:
        return None

    delta = {"x": wall_start["x"] - origin["x"], "y": wall_start["y"] - origin["y"]}
    ray_t = _cross(delta, segment) / denominator
    wall_t = _cross(delta, direction) / denominator
    if ray_t <= WALL_EPSILON or wall_t < -WALL_EPSILON or wall_t > 1 + WALL_EPSILON:
        return None

    point = _add_scaled(origin, direction, ray_t)
    return ray_t, point


def _first_wall_hit(
    origin: Point, direction: Point, wall_lines: list[dict[str, Any]]
) -> dict[str, Any] | None:
    hits: list[dict[str, Any]] = []
    for wall in wall_lines:
        wall_start, wall_end = _wall_points(wall)
        intersection = _ray_segment_intersection(origin, direction, wall_start, wall_end)
        if intersection is None:
            continue
        distance, point = intersection
        hits.append({"distance": distance, "point": point, "wall": wall})

    if not hits:
        return None
    return min(hits, key=lambda hit: hit["distance"])


def _target_on_ray(
    origin: Point,
    direction: Point,
    target: Point,
    max_distance: float,
) -> tuple[float, float] | None:
    target_vector = {"x": target["x"] - origin["x"], "y": target["y"] - origin["y"]}
    along = target_vector["x"] * direction["x"] + target_vector["y"] * direction["y"]
    if along <= WALL_EPSILON or along > max_distance + WALL_EPSILON:
        return None

    closest = _add_scaled(origin, direction, along)
    miss_distance = _distance(closest, target)
    return along, miss_distance


def _reflect_direction(direction: Point, wall: dict[str, Any]) -> Point:
    wall_start, wall_end = _wall_points(wall)
    wall_vector = {
        "x": wall_end["x"] - wall_start["x"],
        "y": wall_end["y"] - wall_start["y"],
    }
    wall_length = math.hypot(wall_vector["x"], wall_vector["y"])
    if wall_length == 0:
        return {"x": -direction["x"], "y": -direction["y"]}

    unit_wall = {"x": wall_vector["x"] / wall_length, "y": wall_vector["y"] / wall_length}
    projection = direction["x"] * unit_wall["x"] + direction["y"] * unit_wall["y"]
    reflected = {
        "x": 2 * projection * unit_wall["x"] - direction["x"],
        "y": 2 * projection * unit_wall["y"] - direction["y"],
    }
    reflected_length = math.hypot(reflected["x"], reflected["y"])
    if reflected_length == 0:
        return {"x": -direction["x"], "y": -direction["y"]}
    return {
        "x": reflected["x"] / reflected_length,
        "y": reflected["y"] / reflected_length,
    }


def _line_clear_of_walls(start: Point, end: Point, wall_lines: list[dict[str, Any]]) -> bool:
    for wall in wall_lines:
        wall_start = {
            "x": float(wall["start"]["x"]),
            "y": float(wall["start"]["y"]),
        }
        wall_end = {
            "x": float(wall["end"]["x"]),
            "y": float(wall["end"]["y"]),
        }
        if _segments_intersect(start, end, wall_start, wall_end):
            return False
    return True


def _path_clear_of_balls(
    start: Point,
    end: Point,
    balls: list[dict[str, Any]],
    ignored_indices: set[int],
    moving_radius: float,
    configured_radius: float | None = None,
) -> bool:
    moving_clearance = max(float(moving_radius), float(configured_radius or 0.0))
    for index, ball in enumerate(balls):
        if index in ignored_indices:
            continue

        center = _point(ball)
        obstacle_radius = max(float(ball["radius"]), float(configured_radius or 0.0))
        clearance = (
            (moving_clearance + obstacle_radius) * BALL_CLEARANCE_SCALE
            + BALL_CLEARANCE_PADDING_PIXELS
        )
        if _point_segment_distance(center, start, end) <= clearance:
            return False
    return True


def _candidate_targets(parse_result: dict[str, Any], group: str) -> list[tuple[int, dict[str, Any]]]:
    target_type = "stripe" if group == "stripes" else "solid"
    return [
        (index, ball)
        for index, ball in enumerate(parse_result["balls"])
        if ball["type"] == target_type
    ]


def _shot_targets(parse_result: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for hole in parse_result["holes"]:
        targets.append({"kind": "hole", **hole})
    targets.extend(parse_result.get("shot_targets", []))
    return targets


def _trace_object_path(
    start: Point,
    outgoing_angle: float,
    parse_result: dict[str, Any],
    balls: list[dict[str, Any]],
    target_index: int,
    moving_radius: float,
    configured_radius: float,
    max_bounces: int,
) -> list[dict[str, Any]]:
    valid_paths: list[dict[str, Any]] = []
    path_start = start
    direction = _direction(outgoing_angle)
    object_lines: list[dict[str, Point]] = []

    for bounce_count in range(max_bounces + 1):
        wall_hit = _first_wall_hit(path_start, direction, parse_result["wall_lines"])
        wall_distance = wall_hit["distance"] if wall_hit else RAY_LENGTH

        reachable_targets: list[dict[str, Any]] = []
        for shot_target in _shot_targets(parse_result):
            shot_target_center = _point(shot_target)
            target_hit = _target_on_ray(
                path_start, direction, shot_target_center, wall_distance
            )
            if target_hit is None:
                continue
            target_distance, miss_distance = target_hit
            if miss_distance > max(moving_radius, 1.0):
                continue
            if not _path_clear_of_balls(
                path_start,
                shot_target_center,
                balls,
                ignored_indices={target_index},
                moving_radius=moving_radius,
                configured_radius=configured_radius,
            ):
                continue
            reachable_targets.append(
                {
                    "distance": target_distance,
                    "target": shot_target,
                    "target_center": shot_target_center,
                }
            )

        if reachable_targets:
            best_target = min(reachable_targets, key=lambda item: item["distance"])
            valid_paths.append(
                {
                    "target": best_target["target"],
                    "target_center": best_target["target_center"],
                    "object_lines": object_lines
                    + [{"start": path_start, "end": best_target["target_center"]}],
                    "bounces": bounce_count,
                }
            )

        if bounce_count >= max_bounces or wall_hit is None:
            break

        wall_point = wall_hit["point"]
        if not _path_clear_of_balls(
            path_start,
            wall_point,
            balls,
            ignored_indices={target_index},
            moving_radius=moving_radius,
            configured_radius=configured_radius,
        ):
            break

        object_lines.append({"start": path_start, "end": wall_point})
        direction = _reflect_direction(direction, wall_hit["wall"])
        path_start = _add_scaled(wall_point, direction, max(1.0, moving_radius * 0.25))

    return valid_paths


def find_valid_shots(
    parse_result: dict[str, Any],
    group: str,
    max_bounces: int = MAX_BOUNCES,
) -> list[Shot]:
    balls = parse_result["balls"]
    cue_candidates = [
        (index, ball) for index, ball in enumerate(balls) if ball["type"] == "cue"
    ]
    if not cue_candidates:
        raise ValueError("Could not solve shot because no cue ball was detected.")

    cue_index, cue_ball = cue_candidates[0]
    cue_center = _point(cue_ball)
    configured_radius = float(parse_result.get("ball_radius", cue_ball["radius"]))
    half_sweep = ANGLE_SWEEP_DEGREES / 2
    angle_samples = int(ANGLE_SWEEP_DEGREES / ANGLE_STEP_DEGREES) + 1
    valid_shots: list[Shot] = []

    for target_index, target_ball in _candidate_targets(parse_result, group):
        target_center = _point(target_ball)
        base_angle = _angle(cue_center, target_center)

        for sample_index in range(angle_samples):
            delta = -half_sweep + sample_index * ANGLE_STEP_DEGREES
            outgoing_angle = base_angle + delta

            first_line_clear = _path_clear_of_balls(
                cue_center,
                target_center,
                balls,
                ignored_indices={cue_index, target_index},
                moving_radius=float(cue_ball["radius"]),
                configured_radius=configured_radius,
            ) and _line_clear_of_walls(
                cue_center, target_center, parse_result["wall_lines"]
            )
            if not first_line_clear:
                continue

            object_paths = _trace_object_path(
                target_center,
                outgoing_angle,
                parse_result,
                balls,
                target_index,
                float(target_ball["radius"]),
                configured_radius,
                max_bounces,
            )
            for object_path in object_paths:
                valid_shots.append(
                    {
                        "group": group,
                        "target_ball": target_ball,
                        "target": object_path["target"],
                        "target_hole": object_path["target"]
                        if object_path["target"]["kind"] == "hole"
                        else None,
                        "angle_delta": delta,
                        "bounces": object_path["bounces"],
                        "cue_line": {
                            "start": cue_center,
                            "end": target_center,
                        },
                        "object_lines": object_path["object_lines"],
                        "object_line": {
                            "start": target_center,
                            "end": object_path["target_center"],
                        },
                    }
                )

    return valid_shots


def pick_valid_shot(
    valid_shots: list[Shot],
    pick_smallest_bounces: bool = PICK_SMALLEST_BOUNCES,
) -> Shot | None:
    if not valid_shots:
        return None
    if pick_smallest_bounces:
        min_bounces = min(shot.get("bounces", 0) for shot in valid_shots)
        tied_shots = [
            shot for shot in valid_shots if shot.get("bounces", 0) == min_bounces
        ]
        return random.choice(tied_shots)

    max_bounces = max(shot.get("bounces", 0) for shot in valid_shots)
    tied_shots = [
        shot for shot in valid_shots if shot.get("bounces", 0) == max_bounces
    ]
    return random.choice(tied_shots)


def pickValidShot(
    valid_shots: list[Shot],
    pick_smallest_bounces: bool = PICK_SMALLEST_BOUNCES,
) -> Shot | None:
    return pick_valid_shot(valid_shots, pick_smallest_bounces)


def draw_shot(
    image_path: str | Path,
    shot: Shot | None,
    output_path: str | Path,
    valid_shot_count: int = 0,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    if shot is None:
        draw.text((16, 16), "No valid shot found", fill=(255, 64, 129))
        image.save(output_path)
        return

    cue_start = shot["cue_line"]["start"]
    cue_end = shot["cue_line"]["end"]
    object_lines = shot.get("object_lines")
    if object_lines is None:
        object_lines = [shot["object_line"]]

    draw.line(
        [(cue_start["x"], cue_start["y"]), (cue_end["x"], cue_end["y"])],
        fill=(255, 235, 59),
        width=4,
    )
    for index, object_line in enumerate(object_lines):
        object_start = object_line["start"]
        object_end = object_line["end"]
        color = (255, 64, 129) if index == len(object_lines) - 1 else (255, 128, 0)
        draw.line(
            [(object_start["x"], object_start["y"]), (object_end["x"], object_end["y"])],
            fill=color,
            width=4,
        )

    for point, color in (
        (cue_start, (255, 255, 255)),
        (cue_end, (255, 235, 59)),
        (object_lines[-1]["end"], (255, 64, 129)),
    ):
        x = point["x"]
        y = point["y"]
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    label = f"{valid_shot_count} valid shot"
    if valid_shot_count != 1:
        label += "s"
    label += f", {shot.get('bounces', 0)} bounce"
    if shot.get("bounces", 0) != 1:
        label += "s"
    draw.text((16, 16), label, fill=(255, 255, 255))
    image.save(output_path)
