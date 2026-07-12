# Debug Prompt — PPE App runs but detects/draws nothing

You are a **senior computer-vision engineer** debugging the PPE detection app you built earlier (FastAPI + YOLO11 + ByteTrack + supervision). There is a bug. **Do not guess and patch randomly** — follow the diagnostic protocol below in order, add temporary logging, report what each step reveals, then fix the root cause.

## Symptom (exact)

The app starts, a video is uploaded, and it streams to the browser **start to finish**, but the output looks **identical to the input**: no bounding boxes, no SAFE/UNSAFE labels, no PPE icons, and the counters (Detected / Safe / Compliance % / Total Unique) never change from 0. The video just plays through unmodified.

This means frames ARE flowing, but the detection→association→annotation chain is broken somewhere. Find exactly where.

## Two most likely root causes (check these first)

1. **Wrong or empty weights** — `models/best.pt` is missing/corrupt/not the trained model, so no detections are produced. (Recall the spec said the app must hard-error when weights are missing. If it's streaming with no error, either weights loaded but are wrong, or that error-handling was never implemented.)
2. **Class-name mismatch** — the trained model's class names in `model.names` do not match the strings in `PERSON_CLASSES` / `PPE_CLASSES` in `config.py`, so every detection is filtered out, the persons list is empty, and the raw frame is streamed unchanged.

## Diagnostic protocol — run in this order, log findings, don't skip

### Step 1 — Does the model produce ANY detections at all?
- Print `model.names` at startup. It **must** be the custom PPE classes (something like `head, helmet, mask, face, hand, vest, glove, person`). If it prints the 80 COCO classes (`person, bicycle, car, ...`) → the wrong model is loaded. Stop and fix the weights path.
- Print the absolute path of the loaded weights and its file size in bytes. Confirm it's the real trained `best.pt` (not 0 bytes, not `yolo11n.pt`).
- Run inference on **one** representative frame with a low threshold: `model.predict(frame, conf=0.1, verbose=True)`. Print the number of raw detections and each detection's class name + confidence.
  - **Zero detections** → go to Step 2.
  - **>0 detections** → detection works; the loss is downstream → go to Step 3.

### Step 2 — Model detects nothing
- Confirm `frame` passed to the model is a valid BGR `numpy.ndarray` (print `frame.shape`, `frame.dtype`) and not `None`/empty. A common bug: `cap.read()` returns `(ret, frame)` and `ret` is ignored, or frames are read but not passed through.
- Confirm the device load didn't silently fall back or fail.
- Sanity-check the weights themselves: was `best.pt` produced by a converged run? Load it and check it detects on a known training/validation image. If it detects on a still image but not on video frames → the frame you feed the model differs (color space, size, or it's actually blank) — inspect by saving one frame to disk and viewing it.

### Step 3 — Detections exist, but the persons/equipment lists come out empty (MOST LIKELY BUG)
- Print, side by side: the exact strings in `model.names.values()` **and** the exact strings in `PERSON_CLASSES` and `PPE_CLASSES` from `config.py`.
- The code splits detections into persons vs equipment by matching each detection's class name against those sets. Verify:
  - the match is truly **case-insensitive** (both sides `.lower()`), AND
  - the names actually correspond. If the trained model names a class `Person`/`worker`/`Hardhat`/`Safety Vest` etc. instead of the assumed `person`/`helmet`/`vest`, the config sets are simply wrong for this model.
- **Fix:** update `config.py` so `PERSON_CLASSES` / `PPE_CLASSES` / `REQUIRED_PPE` use the exact names from `model.names` (lowercased). Do not hardcode indices. If unsure of the real names, print them and adapt.
- If the persons list is empty, there is nothing to annotate and nothing to count → raw frame is streamed. This matches the symptom exactly.

### Step 4 — Persons exist, but the frame is still unchanged (wrong frame streamed)
- Verify the frame that gets JPEG-encoded and sent over the WebSocket is the **annotated** frame returned by the annotator, not the original `frame`/`image` variable.
- `supervision` annotators return a **new** image — they must be reassigned, e.g. `annotated = box_annotator.annotate(frame.copy(), detections)` and then `annotated` (not `frame`) is what you encode and send. A classic bug is annotating but then encoding the original.

### Step 5 — Everything draws, but counters stay at 0
- Confirm the per-frame JSON `counts` payload is actually sent with each frame, and the frontend JS parses it and writes to the DOM elements. Print the payload server-side and log it in the browser console client-side.

## What to do
1. Add temporary debug logging at each stage: raw detection count, `model.names`, persons count, equipment count, safe count — printed every ~30 frames (not every frame).
2. Work Step 1 → 5 in order. **Report what each step reveals** before fixing, so the root cause is identified, not masked.
3. Fix the root cause. Then remove the noisy debug logging (keep a single concise startup log of `model.names` and device).
4. Re-verify against the original acceptance criteria: boxes + labels appear, PPE icons show green/red correctly, all four counters update live, one violation snapshot per unsafe track ID.

## Guardrails
- Do not "fix" by lowering the confidence threshold to near-zero to force boxes — that hides the real bug.
- Do not switch to a COCO model to make something appear — the whole point is the custom PPE model.
- Do not hardcode class indices; always resolve names from `model.names`.
- Identify the actual root cause and state it plainly in your summary.