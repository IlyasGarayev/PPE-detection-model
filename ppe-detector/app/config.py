"""Central place for every tunable value used across the app."""

# Model
MODEL_PATH = "models/best.pt"

# Class categories — resolved by string name at runtime, never by index.
PERSON_CLASSES = {"person"}

# This model encodes worn/not-worn directly in the class name (e.g.
# "head_whelmet" vs "head_nohelmet") instead of separate equipment +
# bare-body-part classes. Only the "worn" classes are treated as equipment
# for association — the "not worn" classes carry no extra information for
# compliance (absence of a "worn" detection already means missing) so they
# are ignored rather than mapped.
PPE_CLASS_TO_ITEM = {
    "head_whelmet": "helmet",
    "face_wmask": "mask",
    "hand_wglove": "glove",
    "vest": "vest",
}
PPE_CLASSES = set(PPE_CLASS_TO_ITEM)

# Compliance rule: a person is "safe" iff they wear every item in this list.
# Change this list to change the compliance rule.
REQUIRED_PPE = ["helmet", "mask", "vest"]

# Equipment -> person association (see association.py for the containment rationale)
CONTAINMENT_THRESHOLD = 0.5

# Detection / tracking
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TRACKER = "bytetrack.yaml"

# Violation logging
VIOLATIONS_DIR = "violations"

# Upload constraints
MAX_UPLOAD_MB = 300
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Static assets
ICONS_DIR = "static/icons"
