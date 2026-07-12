"""Pure frame annotator: (frame, per-person PPE assignment) -> annotated frame.

Person boxes + labels are drawn with `supervision`. PPE status icons (one per
required item, green if worn / red if missing) are alpha-blended on top,
stacked vertically beside each person box, clipped so they never draw
outside the frame.
"""

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from app import config

SAFE_COLOR = sv.Color(r=34, g=197, b=94)
UNSAFE_COLOR = sv.Color(r=239, g=68, b=68)

_SAFE_BOX = sv.BoxAnnotator(color=SAFE_COLOR, thickness=2, color_lookup=sv.ColorLookup.INDEX)
_UNSAFE_BOX = sv.BoxAnnotator(color=UNSAFE_COLOR, thickness=2, color_lookup=sv.ColorLookup.INDEX)
_SAFE_LABEL = sv.LabelAnnotator(
    color=SAFE_COLOR, text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX
)
_UNSAFE_LABEL = sv.LabelAnnotator(
    color=UNSAFE_COLOR, text_scale=0.5, text_thickness=1, color_lookup=sv.ColorLookup.INDEX
)

ICON_SIZE = 24
ICON_GAP = 3


def _load_icons() -> dict:
    icons = {}
    icons_dir = Path(config.ICONS_DIR)
    for item in config.REQUIRED_PPE:
        for status in ("green", "red"):
            path = icons_dir / f"{item}_{status}.png"
            if path.exists():
                icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if icon is not None:
                    icon = cv2.resize(icon, (ICON_SIZE, ICON_SIZE), interpolation=cv2.INTER_AREA)
                icons[(item, status)] = icon
    return icons


_ICONS = _load_icons()


def _alpha_blend(frame: np.ndarray, icon: np.ndarray, x: int, y: int) -> None:
    """Blend an RGBA icon onto frame at (x, y), clipped to frame bounds."""
    fh, fw = frame.shape[:2]
    ih, iw = icon.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + iw), min(fh, y + ih)
    if x1 >= x2 or y1 >= y2:
        return  # fully outside frame

    icon_x1, icon_y1 = x1 - x, y1 - y
    icon_crop = icon[icon_y1 : icon_y1 + (y2 - y1), icon_x1 : icon_x1 + (x2 - x1)]

    if icon_crop.shape[2] == 4:
        alpha = icon_crop[:, :, 3:4].astype(np.float32) / 255.0
        rgb = icon_crop[:, :, :3].astype(np.float32)
    else:
        alpha = np.ones((icon_crop.shape[0], icon_crop.shape[1], 1), dtype=np.float32)
        rgb = icon_crop.astype(np.float32)

    roi = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (rgb * alpha + roi * (1 - alpha)).astype(np.uint8)


def annotate(frame: np.ndarray, assignments: dict) -> np.ndarray:
    """assignments: {track_id: {"box", "conf", "ppe", "missing"}} from association.py"""
    if not assignments:
        return frame

    annotated = frame.copy()

    for track_id, data in assignments.items():
        box = np.array([data["box"]], dtype=np.float32)
        confidence = np.array([data["conf"]], dtype=np.float32)
        tracker_id = np.array([track_id])
        detection = sv.Detections(xyxy=box, confidence=confidence, tracker_id=tracker_id)

        safe = len(data["missing"]) == 0
        label = f"ID {track_id} - {'SAFE' if safe else 'UNSAFE'}"

        box_annotator = _SAFE_BOX if safe else _UNSAFE_BOX
        label_annotator = _SAFE_LABEL if safe else _UNSAFE_LABEL
        annotated = box_annotator.annotate(annotated, detection)
        annotated = label_annotator.annotate(annotated, detection, labels=[label])

        x1, y1, x2, _ = data["box"]
        icon_x = x2 + ICON_GAP
        icon_y = y1
        for item in config.REQUIRED_PPE:
            status = "green" if item in data["ppe"] else "red"
            icon = _ICONS.get((item, status))
            if icon is not None:
                _alpha_blend(annotated, icon, icon_x, icon_y)
            icon_y += ICON_SIZE + ICON_GAP

    return annotated
