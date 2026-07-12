# Build Prompt — Real-Time PPE Compliance Detection App (FastAPI + YOLO11)

You are a **senior computer-vision / backend engineer**. Build a complete, runnable project from scratch based on the specification below. Follow it exactly. Where the spec gives a rationale ("WHY"), respect it — do not "optimize" it away, because those choices are deliberate.

---

## 1. Goal

A web app for **workplace safety (PPE) compliance monitoring**. A user uploads a video; the app detects people and their protective equipment, tracks each person across frames, decides who is compliant ("safe"), streams the annotated video live to the browser, and shows live counters. Unsafe people are logged once with a snapshot.

This is a **local learning/testing project** — no authentication, no database, no multi-user concurrency. One upload processed at a time is fine.

---

## 2. Locked tech stack (do not substitute)

- **Backend:** FastAPI, served with `uvicorn`. No Docker.
- **Detection:** Ultralytics **YOLO11** (custom-trained weights).
- **Tracking:** Ultralytics built-in **ByteTrack** via `model.track(..., persist=True)`. Do **not** write a custom tracker.
- **Annotation:** `supervision` (Roboflow) for boxes/labels, plus a small custom overlay for PPE status icons.
- **Video I/O:** OpenCV (`opencv-python`).
- **Streaming:** **WebSocket**, sending base64-encoded JPEG frames + a JSON counter payload, one message per processed frame.
- **Frontend:** single `templates/index.html` using **Tailwind via CDN** + vanilla JS. UI language: **English**.

---

## 3. The dataset & model (critical context)

The model is trained on this Roboflow dataset (YOLOv11 export):
`ppe-mask-glove-mergev2` — **8 classes**: `head`, `helmet`, `mask`, `face`, `hand`, `vest`, `glove`, `person`.

Class meanings:
- `person` — a human (this is our own class; **do not rely on COCO** for person detection).
- `helmet`, `mask`, `vest`, `glove` — PPE items being worn.
- `head` — a bare head (no helmet), `face` — a bare face (no mask), `hand` — a bare hand (no glove). These are the "negative"/violation-signal classes.

**IMPORTANT — do not hardcode class indices.** Class order in `data.yaml` may differ from this list. Always resolve names from the loaded model at runtime (`model.names`, a dict of `{id: name}`) and match by **string name**, lowercased. Categorize each detection by looking up its class name in the config sets below.

The user trains in Google Colab with:
```python
model.train(data=".../data.yaml", epochs=50, imgsz=640,
            project="ppe", plots=True, save=True, batch=128)
```
Trained weights land at `ppe/train/weights/best.pt` and the user will copy them to `models/best.pt` in this project.

---

## 4. Locked behavioral decisions

| Decision | Value | WHY |
|---|---|---|
| Required PPE for "safe" | `helmet` AND `mask` AND `vest` (glove NOT required) | User's rule. Must be a configurable list in `config.py`. |
| Equipment→person association | **Containment**, threshold **0.5** | An equipment box is tiny vs a person box, so IoU is near-zero even for a perfect match. Containment = `intersection_area / equipment_box_area` answers "what fraction of the equipment sits inside this person?" — correct for small-in-large matching. |
| Per-frame processing | Process **every** frame (no frame skipping) | User's choice. |
| Streaming mode | Real-time: read → process → push frame over WebSocket as it's ready | Live counters. |
| Output video | **Not saved** — live display only, discarded after | User's choice. |
| Device | Use GPU if available, else CPU: `"cuda" if torch.cuda.is_available() else "cpu"` | User's choice; must not crash on CPU-only machines. |
| Missing weights | If `models/best.pt` is absent → **fail with a clear error** at startup / on first use. Do NOT silently fall back to a COCO model. | User wants an explicit error, not misleading person-only detection. |
| Violation logging | Save **one** snapshot per unique track ID, the first time that person is classified unsafe | Saving every frame would produce thousands of near-duplicate files. Dedup by track ID. |
| UI language | English | User's choice. |
| PPE status icons | Claude Code generates them (see §7) | User has none. |

---

## 5. Project structure

Create exactly this layout:

```
ppe-detector/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI app: routes, upload, WebSocket, processing loop
│   ├── config.py          # all tunables: paths, class sets, thresholds, required PPE
│   ├── detector.py        # model loading (device-safe) + per-frame track() wrapper
│   ├── association.py     # containment math + per-person PPE assignment
│   ├── compliance.py      # safe/unsafe rule from a person's PPE set
│   ├── annotator.py       # supervision boxes/labels + PPE status icon overlay
│   └── violations.py      # snapshot dedup + saving (id + timestamp + missing PPE)
├── models/
│   └── .gitkeep           # user drops best.pt here
├── static/
│   └── icons/             # generated PNG icons (green/red helmet/mask/vest)
├── templates/
│   └── index.html         # Tailwind dashboard + WebSocket client
├── violations/            # runtime output (gitignored); create at startup if missing
├── scripts/
│   └── generate_icons.py  # one-off script to create the status icons
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 6. Module-by-module requirements

### `app/config.py`
Centralize everything tunable. Include at least:
- `MODEL_PATH = "models/best.pt"`
- `PERSON_CLASSES = {"person"}`
- `PPE_CLASSES = {"helmet", "mask", "vest", "glove"}`
- `REQUIRED_PPE = ["helmet", "mask", "vest"]` — changing this list changes the compliance rule
- `CONTAINMENT_THRESHOLD = 0.5`
- `CONF_THRESHOLD = 0.25`, `IOU_THRESHOLD = 0.45`
- `TRACKER = "bytetrack.yaml"`
- `VIOLATIONS_DIR = "violations"`
- `MAX_UPLOAD_MB` (pick a sane default, e.g. 300) and allowed extensions `{".mp4", ".avi", ".mov", ".mkv"}`

### `app/detector.py`
- Load YOLO11 once. Pick device with `torch.cuda.is_available()`.
- On load, if `MODEL_PATH` doesn't exist → raise a clear, user-facing exception (message tells them to place `best.pt` in `models/`).
- Expose the model's `names` dict.
- Provide a function that, given a BGR frame, calls `model.track(frame, persist=True, tracker=..., conf=..., iou=..., verbose=False)` and returns a normalized list of detections: each with `xyxy` (ints), `class_name` (lowercased str), `confidence` (float), and `track_id` (int or None — persons will have IDs; equipment may or may not).

### `app/association.py`
- `containment(equip_box, person_box) -> float`: intersection area divided by the **equipment** box area (guard against divide-by-zero).
- `assign_ppe(persons, equipment) -> dict`: for each person (keyed by `track_id`), build the set of PPE class names whose containment with that person > `CONTAINMENT_THRESHOLD`. If an equipment item matches multiple persons, assign it to the person with the highest containment. Return `{track_id: {"box": ..., "conf": ..., "ppe": set(), "missing": [...]}}`.

### `app/compliance.py`
- `is_safe(ppe_set) -> bool`: True iff every item in `REQUIRED_PPE` is present.
- `missing_ppe(ppe_set) -> list`: which required items are absent (used for labels + violation logs).

### `app/annotator.py`
- Use `supervision` for the person bounding boxes + a label (e.g. `ID {id} · SAFE` in green / `ID {id} · UNSAFE` in red).
- Additionally overlay small **PPE status icons** next to each person box: one icon per required PPE type, **green if worn, red if missing**, stacked vertically. Load the pre-generated PNG icons (RGBA with alpha) from `static/icons/` and alpha-blend them onto the frame. Handle boxes near frame edges (clip the overlay region so it never writes out of bounds).
- Keep this pure: input frame + assignment dict → annotated frame.

### `app/violations.py`
- Maintain an in-memory set of track IDs already logged.
- `maybe_log(track_id, frame, person_box, missing_list)`: if this ID is unsafe and not yet logged, save a JPEG crop (or full annotated frame — your choice, but be consistent) to `violations/` named with timestamp + id, and append a line to a `violations/log.jsonl` with `{timestamp, track_id, missing}`. Then mark the ID logged so it never repeats.

### `app/main.py`
- FastAPI app. Routes:
  - `GET /` → render `index.html`.
  - `POST /upload` → validate extension + size, save the file to a temp location, return an id/handle.
  - `WebSocket /ws/{video_id}` → open the saved video with OpenCV, loop over **every** frame:
    1. detect + track,
    2. split into persons vs equipment by class category,
    3. associate PPE (containment),
    4. compute per-person safe/unsafe + missing,
    5. update violation log,
    6. annotate the frame,
    7. accumulate the set of all unique person track IDs seen so far,
    8. JPEG-encode + base64 the frame,
    9. send a JSON message: `{ "frame": "<base64>", "counts": { "detected": <int, current frame>, "safe": <int, current frame>, "compliance": <0-100 %>, "total_unique": <int, cumulative> } }`.
  - Release the capture and close the socket cleanly at end of video. Handle client disconnects without crashing.
- Compliance % = `safe / detected * 100` for the current frame (0 when no persons detected).

### `templates/index.html`
- Tailwind (CDN). Clean, modern dashboard:
  - An upload control (choose file → POST to `/upload`, then open the WebSocket).
  - A main video panel that shows the streamed frames (draw base64 JPEG into an `<img>` or `<canvas>`).
  - Four counter cards: **Detected**, **Safe**, **Compliance %**, **Total Unique People**.
  - A status/progress indicator (processing / done).
- Vanilla JS only. No frameworks. Reconnect/cleanup gracefully when the stream ends.

### `scripts/generate_icons.py`
- Generate 6 PNG icons (helmet, mask, vest — each in green and red), RGBA with transparent background, ~64–100px, into `static/icons/`. Use PIL/Pillow (simple pictograms or clear symbols are fine). This runs once during setup.

### `requirements.txt`
Pin nothing exotic; include at least: `fastapi`, `uvicorn[standard]`, `ultralytics`, `supervision`, `opencv-python`, `python-multipart`, `pillow`, `numpy`. (`torch` comes via `ultralytics`.)

### `README.md`
- Setup: create venv, `pip install -r requirements.txt`, run `python scripts/generate_icons.py`, place `best.pt` in `models/`, run `uvicorn app.main:app --reload`, open `http://localhost:8000`.
- Note the Colab training command and where `best.pt` comes from (`ppe/train/weights/best.pt`).
- Explain the containment-vs-IoU rationale in one short paragraph.

---

## 7. Acceptance criteria (the app is done when all hold)

1. `uvicorn app.main:app` starts without error **when** `models/best.pt` exists; and gives a **clear, explicit error** when it doesn't (no silent COCO fallback).
2. Runs on a CPU-only machine (no CUDA) without crashing; uses GPU automatically when present.
3. Uploading a valid video streams annotated frames live in the browser, every frame processed, with person boxes labeled SAFE/UNSAFE.
4. Each person shows PPE status icons (green worn / red missing) for helmet, mask, vest; icons never render outside frame bounds.
5. A person is SAFE **iff** they have helmet + mask + vest (verify by editing `REQUIRED_PPE` and seeing behavior change).
6. The four counters update live and are correct: `detected` and `safe` are per-frame, `compliance` is their ratio as %, `total_unique` is cumulative distinct track IDs.
7. Each unsafe person produces **exactly one** snapshot + one `log.jsonl` entry (with track id, timestamp, missing PPE), no matter how many frames they appear in.
8. Class handling works regardless of `data.yaml` class order (names resolved from `model.names`, matched by string).
9. Code is modular per §5; no custom tracker; `supervision` used for annotation; all thresholds live in `config.py`.

---

## 8. Guardrails

- Do not hardcode class IDs. Do not add auth/database/multi-user. Do not save the output video. Do not fall back to a non-custom model when weights are missing. Do not skip frames. Keep modules focused and readable — this is a learning project, clarity beats cleverness.
- Before finishing, do a self-review against §7 and fix any gap.