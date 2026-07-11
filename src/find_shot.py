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


# Returns a ball or target center as a point.
def _point(item: dict[str, Any]) -> Point:
    center = item["center"]
    return {"x": float(center["x"]), "y": float(center["y"])}


# Measures the distance between two points.
def _distance(a: Point, b: Point) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


# Returns the angle from one point to another in degrees.
def _angle(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b["y"] - a["y"], b["x"] - a["x"]))


# Normalizes the signed difference between two angles.
def _angle_diff(a: float, b: float) -> float:
    return (a - b + 180) % 360 - 180


# Converts an angle into a unit direction vector.
def _direction(angle: float) -> Point:
    radians = math.radians(angle)
    return {"x": math.cos(radians), "y": math.sin(radians)}


# Offsets a point by a scaled direction vector.
def _add_scaled(point: Point, direction: Point, scale: float) -> Point:
    return {
        "x": point["x"] + direction["x"] * scale,
        "y": point["y"] + direction["y"] * scale,
    }


# Returns the start and end points of a wall segment.
def _wall_points(wall: dict[str, Any]) -> tuple[Point, Point]:
    return (
        {"x": float(wall["start"]["x"]), "y": float(wall["start"]["y"])},
        {"x": float(wall["end"]["x"]), "y": float(wall["end"]["y"])},
    )


# Samples evenly spaced points along a segment.
def _sample_segment(start: Point, end: Point, step: float) -> list[Point]:
    distance = _distance(start, end)
    if distance == 0:
        return [start]

    sampleCount = max(1, int(math.ceil(distance / step)))
    return [
        {
            "x": start["x"] + (end["x"] - start["x"]) * index / sampleCount,
            "y": start["y"] + (end["y"] - start["y"]) * index / sampleCount,
        }
        for index in range(sampleCount + 1)
    ]


# Measures the minimum distance from a point to a segment.
def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    segmentX = end["x"] - start["x"]
    segmentY = end["y"] - start["y"]
    segmentLengthSquared = segmentX * segmentX + segmentY * segmentY
    if segmentLengthSquared == 0:
        return _distance(point, start)

    pointX = point["x"] - start["x"]
    pointY = point["y"] - start["y"]
    projection = (pointX * segmentX + pointY * segmentY) / segmentLengthSquared
    projection = max(0.0, min(1.0, projection))
    closest = {
        "x": start["x"] + segmentX * projection,
        "y": start["y"] + segmentY * projection,
    }
    return _distance(point, closest)


# Computes the orientation value for three points.
def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b["y"] - a["y"]) * (c["x"] - b["x"]) - (
        b["x"] - a["x"]
    ) * (c["y"] - b["y"])


# Checks whether a point lies on a segment.
def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return (
        min(a["x"], c["x"]) - WALL_EPSILON <= b["x"] <= max(a["x"], c["x"]) + WALL_EPSILON
        and min(a["y"], c["y"]) - WALL_EPSILON <= b["y"] <= max(a["y"], c["y"]) + WALL_EPSILON
    )


# Tests whether two segments intersect.
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


# Returns the 2D cross product of two vectors.
def _cross(a: Point, b: Point) -> float:
    return a["x"] * b["y"] - a["y"] * b["x"]


# Finds the first intersection between a ray and a wall segment.
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
    rayT = _cross(delta, segment) / denominator
    wallT = _cross(delta, direction) / denominator
    if rayT <= WALL_EPSILON or wallT < -WALL_EPSILON or wallT > 1 + WALL_EPSILON:
        return None

    point = _add_scaled(origin, direction, rayT)
    return rayT, point


# Finds the nearest wall hit from a ray cast.
def _first_wall_hit(
    origin: Point, direction: Point, wall_lines: list[dict[str, Any]]
) -> dict[str, Any] | None:
    hits: list[dict[str, Any]] = []
    for wall in wall_lines:
        wallStart, wallEnd = _wall_points(wall)
        intersection = _ray_segment_intersection(origin, direction, wallStart, wallEnd)
        if intersection is None:
            continue
        distance, point = intersection
        hits.append({"distance": distance, "point": point, "wall": wall})

    if not hits:
        return None
    return min(hits, key=lambda hit: hit["distance"])


# Checks whether a target lies close enough to a ray path.
def _target_on_ray(
    origin: Point,
    direction: Point,
    target: Point,
    max_distance: float,
) -> tuple[float, float] | None:
    targetVector = {"x": target["x"] - origin["x"], "y": target["y"] - origin["y"]}
    along = targetVector["x"] * direction["x"] + targetVector["y"] * direction["y"]
    if along <= WALL_EPSILON or along > max_distance + WALL_EPSILON:
        return None

    closest = _add_scaled(origin, direction, along)
    missDistance = _distance(closest, target)
    return along, missDistance


# Reflects a direction vector across a wall segment.
def _reflect_direction(direction: Point, wall: dict[str, Any]) -> Point:
    wallStart, wallEnd = _wall_points(wall)
    wallVector = {
        "x": wallEnd["x"] - wallStart["x"],
        "y": wallEnd["y"] - wallStart["y"],
    }
    wallLength = math.hypot(wallVector["x"], wallVector["y"])
    if wallLength == 0:
        return {"x": -direction["x"], "y": -direction["y"]}

    unitWall = {"x": wallVector["x"] / wallLength, "y": wallVector["y"] / wallLength}
    projection = direction["x"] * unitWall["x"] + direction["y"] * unitWall["y"]
    reflected = {
        "x": 2 * projection * unitWall["x"] - direction["x"],
        "y": 2 * projection * unitWall["y"] - direction["y"],
    }
    reflectedLength = math.hypot(reflected["x"], reflected["y"])
    if reflectedLength == 0:
        return {"x": -direction["x"], "y": -direction["y"]}
    return {
        "x": reflected["x"] / reflectedLength,
        "y": reflected["y"] / reflectedLength,
    }


# Checks whether a segment crosses any wall segment.
def _line_clear_of_walls(start: Point, end: Point, wall_lines: list[dict[str, Any]]) -> bool:
    for wall in wall_lines:
        wallStart = {
            "x": float(wall["start"]["x"]),
            "y": float(wall["start"]["y"]),
        }
        wallEnd = {
            "x": float(wall["end"]["x"]),
            "y": float(wall["end"]["y"]),
        }
        if _segments_intersect(start, end, wallStart, wallEnd):
            return False
    return True


# Checks whether a movement path is clear of blocking balls.
def _path_clear_of_balls(
    start: Point,
    end: Point,
    balls: list[dict[str, Any]],
    ignored_indices: set[int],
    moving_radius: float,
    configured_radius: float | None = None,
) -> bool:
    movingClearance = max(float(moving_radius), float(configured_radius or 0.0))
    for index, ball in enumerate(balls):
        if index in ignored_indices:
            continue

        center = _point(ball)
        obstacleRadius = max(float(ball["radius"]), float(configured_radius or 0.0))
        clearance = (
            (movingClearance + obstacleRadius) * BALL_CLEARANCE_SCALE
            + BALL_CLEARANCE_PADDING_PIXELS
        )
        if _point_segment_distance(center, start, end) <= clearance:
            return False
    return True


# Returns candidate balls for the requested group.
def _candidate_targets(parse_result: dict[str, Any], group: str) -> list[tuple[int, dict[str, Any]]]:
    targetType = "stripe" if group == "stripes" else "solid"
    return [
        (index, ball)
        for index, ball in enumerate(parse_result["balls"])
        if ball["type"] == targetType
    ]


# Collects all valid pocket and custom shot targets.
def _shot_targets(parse_result: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for hole in parse_result["holes"]:
        targets.append({"kind": "hole", **hole})
    targets.extend(parse_result.get("shot_targets", []))
    return targets


# Traces valid object-ball paths including bank shots.
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
    validPaths: list[dict[str, Any]] = []
    pathStart = start
    direction = _direction(outgoing_angle)
    objectLines: list[dict[str, Point]] = []

    for bounceCount in range(max_bounces + 1):
        wallHit = _first_wall_hit(pathStart, direction, parse_result["wall_lines"])
        wallDistance = wallHit["distance"] if wallHit else RAY_LENGTH

        reachableTargets: list[dict[str, Any]] = []
        for shotTarget in _shot_targets(parse_result):
            shotTargetCenter = _point(shotTarget)
            targetHit = _target_on_ray(
                pathStart, direction, shotTargetCenter, wallDistance
            )
            if targetHit is None:
                continue
            targetDistance, missDistance = targetHit
            if missDistance > max(moving_radius, 1.0):
                continue
            if not _path_clear_of_balls(
                pathStart,
                shotTargetCenter,
                balls,
                ignored_indices={target_index},
                moving_radius=moving_radius,
                configured_radius=configured_radius,
            ):
                continue
            reachableTargets.append(
                {
                    "distance": targetDistance,
                    "target": shotTarget,
                    "target_center": shotTargetCenter,
                }
            )

        if reachableTargets:
            bestTarget = min(reachableTargets, key=lambda item: item["distance"])
            validPaths.append(
                {
                    "target": bestTarget["target"],
                    "target_center": bestTarget["target_center"],
                    "object_lines": objectLines
                    + [{"start": pathStart, "end": bestTarget["target_center"]}],
                    "bounces": bounceCount,
                }
            )

        if bounceCount >= max_bounces or wallHit is None:
            break

        wallPoint = wallHit["point"]
        if not _path_clear_of_balls(
            pathStart,
            wallPoint,
            balls,
            ignored_indices={target_index},
            moving_radius=moving_radius,
            configured_radius=configured_radius,
        ):
            break

        objectLines.append({"start": pathStart, "end": wallPoint})
        direction = _reflect_direction(direction, wallHit["wall"])
        pathStart = _add_scaled(wallPoint, direction, max(1.0, moving_radius * 0.25))

    return validPaths


# Finds all valid cue-ball shots for the requested group.
def find_valid_shots(
    parse_result: dict[str, Any],
    group: str,
    max_bounces: int = MAX_BOUNCES,
) -> list[Shot]:
    balls = parse_result["balls"]
    cueCandidates = [
        (index, ball) for index, ball in enumerate(balls) if ball["type"] == "cue"
    ]
    if not cueCandidates:
        raise ValueError("Could not solve shot because no cue ball was detected.")

    cueIndex, cueBall = cueCandidates[0]
    cueCenter = _point(cueBall)
    configuredRadius = float(parse_result.get("ball_radius", cueBall["radius"]))
    halfSweep = ANGLE_SWEEP_DEGREES / 2
    angleSamples = int(ANGLE_SWEEP_DEGREES / ANGLE_STEP_DEGREES) + 1
    validShots: list[Shot] = []

    for targetIndex, targetBall in _candidate_targets(parse_result, group):
        targetCenter = _point(targetBall)
        baseAngle = _angle(cueCenter, targetCenter)

        for sampleIndex in range(angleSamples):
            delta = -halfSweep + sampleIndex * ANGLE_STEP_DEGREES
            outgoingAngle = baseAngle + delta

            firstLineClear = _path_clear_of_balls(
                cueCenter,
                targetCenter,
                balls,
                ignored_indices={cueIndex, targetIndex},
                moving_radius=float(cueBall["radius"]),
                configured_radius=configuredRadius,
            ) and _line_clear_of_walls(
                cueCenter, targetCenter, parse_result["wall_lines"]
            )
            if not firstLineClear:
                continue

            objectPaths = _trace_object_path(
                targetCenter,
                outgoingAngle,
                parse_result,
                balls,
                targetIndex,
                float(targetBall["radius"]),
                configuredRadius,
                max_bounces,
            )
            for objectPath in objectPaths:
                validShots.append(
                    {
                        "group": group,
                        "target_ball": targetBall,
                        "target": objectPath["target"],
                        "target_hole": objectPath["target"]
                        if objectPath["target"]["kind"] == "hole"
                        else None,
                        "angle_delta": delta,
                        "bounces": objectPath["bounces"],
                        "cue_line": {
                            "start": cueCenter,
                            "end": targetCenter,
                        },
                        "object_lines": objectPath["object_lines"],
                        "object_line": {
                            "start": targetCenter,
                            "end": objectPath["target_center"],
                        },
                    }
                )

    return validShots


# Picks one valid shot using the configured bounce preference.
def pick_valid_shot(
    valid_shots: list[Shot],
    pick_smallest_bounces: bool = PICK_SMALLEST_BOUNCES,
) -> Shot | None:
    if not valid_shots:
        return None
    if pick_smallest_bounces:
        minBounces = min(shot.get("bounces", 0) for shot in valid_shots)
        tiedShots = [
            shot for shot in valid_shots if shot.get("bounces", 0) == minBounces
        ]
        return random.choice(tiedShots)

    maxBounces = max(shot.get("bounces", 0) for shot in valid_shots)
    tiedShots = [
        shot for shot in valid_shots if shot.get("bounces", 0) == maxBounces
    ]
    return random.choice(tiedShots)


# Preserves the legacy camelCase shot-picker entry point.
def pickValidShot(
    valid_shots: list[Shot],
    pick_smallest_bounces: bool = PICK_SMALLEST_BOUNCES,
) -> Shot | None:
    return pick_valid_shot(valid_shots, pick_smallest_bounces)


# Draws the chosen shot path on top of the input image.
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

    cueStart = shot["cue_line"]["start"]
    cueEnd = shot["cue_line"]["end"]
    objectLines = shot.get("object_lines")
    if objectLines is None:
        objectLines = [shot["object_line"]]

    draw.line(
        [(cueStart["x"], cueStart["y"]), (cueEnd["x"], cueEnd["y"])],
        fill=(255, 235, 59),
        width=4,
    )
    for index, objectLine in enumerate(objectLines):
        objectStart = objectLine["start"]
        objectEnd = objectLine["end"]
        color = (255, 64, 129) if index == len(objectLines) - 1 else (255, 128, 0)
        draw.line(
            [(objectStart["x"], objectStart["y"]), (objectEnd["x"], objectEnd["y"])],
            fill=color,
            width=4,
        )

    for point, color in (
        (cueStart, (255, 255, 255)),
        (cueEnd, (255, 235, 59)),
        (objectLines[-1]["end"], (255, 64, 129)),
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
