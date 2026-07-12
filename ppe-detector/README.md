# PPE Compliance Detector

Real-time workplace PPE (personal protective equipment) compliance monitoring.
Upload a video; the app detects people and their PPE, tracks each person
across frames (ByteTrack), decides who is compliant, and streams the
annotated video live to the browser with live counters. Unsafe people are
logged once with a snapshot.

This is a local learning/testing project: no auth, no database, one upload
processed at a time.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_icons.py   # generates static/icons/*.png

# Place your trained weights here:
cp /path/to/best.pt models/best.pt

uvicorn app.main:app --reload
```

Open http://localhost:8000.

If `models/best.pt` is missing, `uvicorn app.main:app` will fail to start
with a clear error telling you where to put the weights — it will not
silently fall back to a generic COCO model.

## Training the model

The model is trained on a Roboflow-exported YOLOv11 dataset
(`ppe-mask-glove-mergev2`), 8 classes that encode worn/not-worn directly in
the class name: `head_whelmet`, `head_nohelmet`, `face_wmask`,
`face_nomask`, `hand_wglove`, `hand_noglove`, `vest`, `person`. Train in
Google Colab (or locally) with:

```python
model.train(data=".../data.yaml", epochs=50, imgsz=640,
            project="ppe", plots=True, save=True, batch=128)
```

Trained weights land at `ppe/train/weights/best.pt` — copy that file to
`models/best.pt` in this project.

Class order in `data.yaml` can vary between exports, so the app never
hardcodes class indices: it resolves class names from the loaded model
(`model.names`) at runtime and matches by lowercased string name.
`app/config.py`'s `PPE_CLASS_TO_ITEM` maps each "worn" class
(`head_whelmet`, `face_wmask`, `hand_wglove`, `vest`) to its normalized PPE
item name (`helmet`, `mask`, `glove`, `vest`); the "not worn" classes are
ignored, since the absence of a "worn" detection near a person already
means that item is missing — no need to double up on the negative signal.

## Why containment instead of IoU for PPE↔person matching

An equipment box (helmet, mask, glove...) is tiny compared to a person box,
so IoU (intersection over union) is near-zero even for a perfect match —
IoU penalizes the size mismatch. Containment instead measures
`intersection_area / equipment_box_area`: "what fraction of the equipment
sits inside this person's box?" That's the right question when matching a
small box against a large one, and it's what `app/association.py` uses,
with a threshold of 0.5.

## Project layout

- `app/config.py` — all tunables (paths, class sets, thresholds, required PPE).
- `app/detector.py` — model loading (GPU if available, else CPU) + tracking wrapper.
- `app/association.py` — containment math + per-person PPE assignment.
- `app/compliance.py` — safe/unsafe rule (edit `REQUIRED_PPE` in config.py to change it).
- `app/annotator.py` — box/label drawing (`supervision`) + PPE status icon overlay.
- `app/violations.py` — one snapshot + log line per unique unsafe track ID.
- `app/main.py` — FastAPI routes: upload, WebSocket streaming loop.
- `templates/index.html` — Tailwind (CDN) dashboard + WebSocket client.
- `scripts/generate_icons.py` — one-off script that generates the PPE status icons.
