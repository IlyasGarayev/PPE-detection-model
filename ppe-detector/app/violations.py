"""Snapshot dedup + saving for unsafe persons.

One violation session should be created per video (see main.py) so that
track-ID dedup doesn't leak across unrelated uploads.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

from app import config


class ViolationLogger:
    def __init__(self, violations_dir: str = config.VIOLATIONS_DIR):
        self.dir = Path(violations_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "log.jsonl"
        self._logged_ids: set[int] = set()

    def maybe_log(self, track_id: int, frame, person_box, missing_list: list) -> None:
        """Log a snapshot + jsonl entry the first time this track ID is unsafe."""
        if not missing_list:
            return  # safe, nothing to log
        if track_id in self._logged_ids:
            return  # already logged once

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        image_path = self.dir / f"{timestamp}_id{track_id}.jpg"
        cv2.imwrite(str(image_path), frame)

        entry = {
            "timestamp": timestamp,
            "track_id": track_id,
            "missing": missing_list,
            "image": image_path.name,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

        self._logged_ids.add(track_id)
