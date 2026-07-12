"""Model loading (device-safe) and a thin per-frame track() wrapper."""

from pathlib import Path

import torch
from ultralytics import YOLO

from app import config


class Detection:
    __slots__ = ("xyxy", "class_name", "confidence", "track_id")

    def __init__(self, xyxy, class_name, confidence, track_id):
        self.xyxy = xyxy  # (x1, y1, x2, y2) ints
        self.class_name = class_name  # lowercased str
        self.confidence = confidence  # float
        self.track_id = track_id  # int or None


class Detector:
    def __init__(self, model_path: str = config.MODEL_PATH):
        weights = Path(model_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                "Train a model and copy 'best.pt' into the 'models/' directory "
                "before starting the app (see README.md)."
            )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(str(weights))
        self.model.to(self.device)
        self.names = self.model.names  # {id: name}

    def reset_tracker(self) -> None:
        """Drop tracker state so a new video starts track IDs from scratch.

        `model.track(..., persist=True)` reuses tracker state across calls so
        IDs stay stable within one video's frames. Clearing the predictor
        forces re-initialization on the next call, so a freshly uploaded
        video doesn't inherit IDs (or momentum) from a previous one.
        """
        self.model.predictor = None

    def track(self, frame) -> list[Detection]:
        """Run detection + tracking on a single BGR frame."""
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)

        for i in range(len(boxes)):
            class_id = int(cls[i])
            class_name = str(self.names.get(class_id, class_id)).lower()
            track_id = int(ids[i]) if ids[i] is not None else None
            x1, y1, x2, y2 = xyxy[i]
            detections.append(
                Detection(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    class_name=class_name,
                    confidence=float(conf[i]),
                    track_id=track_id,
                )
            )

        return detections
