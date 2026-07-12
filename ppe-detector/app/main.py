"""FastAPI app: upload endpoint + WebSocket streaming of annotated frames."""

import base64
import tempfile
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import association, config
from app.annotator import annotate
from app.detector import Detector
from app.violations import ViolationLogger

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="PPE Compliance Detector")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

Path(config.VIOLATIONS_DIR).mkdir(parents=True, exist_ok=True)

# Fail fast and loud if weights are missing — no silent fallback to a
# generic (COCO) model. This runs at import time, i.e. when uvicorn starts.
detector = Detector()

# video_id -> saved file path, populated by /upload, consumed by the
# WebSocket route. Single-user/local app: an in-memory dict is enough.
_uploaded_videos: dict[str, Path] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_VIDEO_EXTENSIONS)}",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {config.MAX_UPLOAD_MB} MB.",
        )

    video_id = uuid.uuid4().hex
    tmp_dir = Path(tempfile.gettempdir()) / "ppe-detector-uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / f"{video_id}{ext}"
    dest.write_bytes(contents)

    _uploaded_videos[video_id] = dest
    return {"video_id": video_id}


@app.websocket("/ws/{video_id}")
async def ws_stream(websocket: WebSocket, video_id: str):
    await websocket.accept()

    video_path = _uploaded_videos.get(video_id)
    if video_path is None or not video_path.exists():
        await websocket.send_json({"error": f"Unknown or expired video_id '{video_id}'."})
        await websocket.close()
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send_json({"error": "Could not open uploaded video."})
        await websocket.close()
        return

    # Fresh tracker + fresh dedup set so this video doesn't inherit state
    # (track IDs, already-logged violations) from a previous upload.
    detector.reset_tracker()
    violation_logger = ViolationLogger()
    unique_ids: set[int] = set()
    client_connected = True

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = detector.track(frame)
            persons = [d for d in detections if d.class_name in config.PERSON_CLASSES]
            equipment = [d for d in detections if d.class_name in config.PPE_CLASSES]

            assignments = association.assign_ppe(persons, equipment)

            for track_id, data in assignments.items():
                unique_ids.add(track_id)
                violation_logger.maybe_log(track_id, frame, data["box"], data["missing"])

            annotated = annotate(frame, assignments)

            detected_count = len(assignments)
            safe_count = sum(1 for data in assignments.values() if not data["missing"])
            compliance_pct = round((safe_count / detected_count) * 100) if detected_count else 0

            ok_jpeg, buffer = cv2.imencode(".jpg", annotated)
            if not ok_jpeg:
                continue
            b64_frame = base64.b64encode(buffer).decode("utf-8")

            try:
                await websocket.send_json(
                    {
                        "frame": b64_frame,
                        "counts": {
                            "detected": detected_count,
                            "safe": safe_count,
                            "compliance": compliance_pct,
                            "total_unique": len(unique_ids),
                        },
                    }
                )
            except (WebSocketDisconnect, RuntimeError):
                client_connected = False
                break

        if client_connected:
            try:
                await websocket.send_json({"done": True})
            except (WebSocketDisconnect, RuntimeError):
                client_connected = False
    except WebSocketDisconnect:
        client_connected = False
    finally:
        cap.release()
        _uploaded_videos.pop(video_id, None)
        video_path.unlink(missing_ok=True)
        if client_connected:
            try:
                await websocket.close()
            except RuntimeError:
                pass
