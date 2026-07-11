from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from xml.etree import ElementTree


SCALE_XML = Path(__file__).resolve().parent / "scale.xml"


# Loads the saved shot result JSON file.
def _load_result(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# Calculates the overlay window geometry from the shot result.
def _overlay_geometry(result: dict) -> tuple[float, float, float, float]:
    crop = result.get("screen_crop", {})
    tableBbox = crop.get("table_bbox")
    if tableBbox:
        return (
            float(tableBbox["x1"]),
            float(tableBbox["y1"]),
            float(tableBbox["x2"] - tableBbox["x1"] + 1),
            float(tableBbox["y2"] - tableBbox["y1"] + 1),
        )

    bounds = result.get("parse_result", {}).get("table_bounds", {})
    if bounds:
        return (
            0.0,
            0.0,
            float(bounds["right"] - bounds["left"] + 1),
            float(bounds["bottom"] - bounds["top"] + 1),
        )
    return 0.0, 0.0, 468.0, 830.0


# Reads the screenshot-to-screen scale from the XML config.
def _scale_from_xml(scale_path: str | Path = SCALE_XML) -> float | None:
    try:
        root = ElementTree.parse(scale_path).getroot()
    except (FileNotFoundError, ElementTree.ParseError):
        return None

    display = root.find("display")
    if display is None:
        return None

    value = display.attrib.get("screenshot_to_screen_scale")
    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None


# Builds the Qt overlay window class for a solved shot.
class ShotOverlay:
    def __init__(
        self,
        result: dict,
        line_width: float = 1.0,
        screen_scale: float = 1.0,
        debug: bool = False,
        click_through: bool = True,
    ) -> None:
        try:
            from PyQt6.QtCore import Qt
            from PyQt6.QtGui import QColor, QPainter, QPen
            from PyQt6.QtWidgets import QMainWindow
        except ImportError as exc:
            raise SystemExit(
                "PyQt6 is required for the click-through overlay. Install it with: "
                "python3 -m pip install PyQt6"
            ) from exc

        class _Window(QMainWindow):
            def __init__(self, shot_result: dict) -> None:
                super().__init__()
                self.result = shot_result
                self.line_width = max(float(line_width), 0.1)
                self.screen_scale = max(screen_scale, 0.01)
                self.debug = debug
                x, y, width, height = _overlay_geometry(shot_result)
                self.setWindowTitle("8 Ball Shot Overlay")
                self.setGeometry(
                    int(round(x / self.screen_scale)),
                    int(round(y / self.screen_scale)),
                    int(round(width / self.screen_scale)),
                    int(round(height / self.screen_scale)),
                )

                flags = (
                    Qt.WindowType.Window
                    | Qt.WindowType.FramelessWindowHint
                    | Qt.WindowType.WindowStaysOnTopHint
                    | Qt.WindowType.WindowDoesNotAcceptFocus
                )
                if click_through:
                    flags |= Qt.WindowType.WindowTransparentForInput
                self.setWindowFlags(flags)
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
                self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

            # Keeps the overlay window above the game window.
            def keep_on_top(self) -> None:
                self.raise_()
                self.repaint()

            # Draws the cue and object-ball path lines.
            def paintEvent(self, event) -> None:  # noqa: N802
                shot = self.result.get("selected_shot")
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)

                if self.debug:
                    painter.fillRect(self.rect(), QColor(255, 0, 85, 35))
                    painter.setPen(QPen(QColor(255, 0, 85, 220), 2))
                    painter.drawRect(self.rect().adjusted(1, 1, -2, -2))

                if not shot:
                    if self.debug:
                        painter.setPen(QColor(255, 255, 255, 255))
                        painter.drawText(16, 28, "No selected shot in JSON")
                    return

                cueLine = shot["cue_line"]
                objectLines = shot.get("object_lines")
                if objectLines is None:
                    objectLines = [shot["object_line"]]

                cuePen = QPen(QColor(255, 235, 59, 245))
                objectPen = QPen(QColor(255, 0, 85, 245))
                bouncePen = QPen(QColor(255, 128, 0, 245))
                for pen in (cuePen, objectPen, bouncePen):
                    pen.setWidthF(self.line_width)

                painter.setPen(cuePen)
                painter.drawLine(
                    int(round(cueLine["start"]["x"] / self.screen_scale)),
                    int(round(cueLine["start"]["y"] / self.screen_scale)),
                    int(round(cueLine["end"]["x"] / self.screen_scale)),
                    int(round(cueLine["end"]["y"] / self.screen_scale)),
                )

                for index, objectLine in enumerate(objectLines):
                    painter.setPen(objectPen if index == len(objectLines) - 1 else bouncePen)
                    painter.drawLine(
                        int(round(objectLine["start"]["x"] / self.screen_scale)),
                        int(round(objectLine["start"]["y"] / self.screen_scale)),
                        int(round(objectLine["end"]["x"] / self.screen_scale)),
                        int(round(objectLine["end"]["y"] / self.screen_scale)),
                    )

        self.window_class = _Window


# Opens the click-through overlay window for a saved shot result.
def display_shot(
    json_path: str | Path,
    line_width: float = 1.0,
    scale: float | None = None,
    scale_path: str | Path = SCALE_XML,
    debug: bool = False,
    click_through: bool = True,
) -> None:
    try:
        from PyQt6.QtCore import QTimer, Qt
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise SystemExit(
            "PyQt6 is required for the click-through overlay. Install it with: "
            "python3 -m pip install PyQt6"
        ) from exc

    result = _load_result(json_path)
    app = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_DontShowIconsInMenus, True)
    if scale is None:
        scale = _scale_from_xml(scale_path)
    if scale is None:
        screen = app.primaryScreen()
        scale = float(screen.devicePixelRatio()) if screen is not None else 1.0

    overlayFactory = ShotOverlay(
        result,
        line_width=line_width,
        screen_scale=scale,
        debug=debug,
        click_through=click_through,
    )
    overlay = overlayFactory.window_class(result)
    overlay.show()
    overlay.keep_on_top()
    keepOnTopTimer = QTimer()
    keepOnTopTimer.timeout.connect(overlay.keep_on_top)
    keepOnTopTimer.start(2000)
    overlay.repaint()
    print(
        "Overlay shown:",
        {
            "geometry": {
                "x": overlay.x(),
                "y": overlay.y(),
                "width": overlay.width(),
                "height": overlay.height(),
            },
            "screen_scale": scale,
            "scale_source": str(scale_path),
            "has_selected_shot": result.get("selected_shot") is not None,
            "click_through": click_through,
        },
    )
    sys.exit(app.exec())


# Parses CLI arguments and launches the shot overlay.
def main() -> None:
    parser = argparse.ArgumentParser(description="Display a click-through shot overlay.")
    parser.add_argument("json", help="Shot result JSON produced by main.py or capture.py.")
    parser.add_argument("--line-width", type=float, default=1.0, help="Overlay line width.")
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Screenshot-pixels to screen-points scale. Overrides scale.xml.",
    )
    parser.add_argument(
        "--scale-file",
        default=SCALE_XML,
        help="XML scale file containing display screenshot_to_screen_scale.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Draw a translucent debug box around the overlay window.",
    )
    parser.add_argument(
        "--no-click-through",
        action="store_true",
        help="Temporarily allow the overlay window to receive clicks while debugging.",
    )
    args = parser.parse_args()

    display_shot(
        args.json,
        line_width=args.line_width,
        scale=args.scale,
        scale_path=args.scale_file,
        debug=args.debug,
        click_through=not args.no_click_through,
    )


if __name__ == "__main__":
    main()
