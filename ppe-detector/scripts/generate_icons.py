"""One-off script: generate PPE status icons (green = worn, red = missing).

Run once during setup:
    python scripts/generate_icons.py
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "static" / "icons"

SIZE = 80
GREEN = (34, 197, 94, 255)
RED = (239, 68, 68, 255)
WHITE = (255, 255, 255, 255)


def _canvas():
    return Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))


def draw_helmet(color):
    img = _canvas()
    d = ImageDraw.Draw(img)
    # dome
    d.pieslice([12, 14, 68, 70], 180, 360, fill=color)
    d.rectangle([12, 42, 68, 52], fill=color)
    # brim
    d.rectangle([8, 50, 72, 58], fill=color, outline=None)
    # vent knob
    d.ellipse([36, 10, 44, 18], fill=color)
    return img


def draw_mask(color):
    img = _canvas()
    d = ImageDraw.Draw(img)
    # mask body
    d.rounded_rectangle([14, 28, 66, 58], radius=14, fill=color)
    # pleats
    for y in (36, 44, 52):
        d.line([18, y, 62, y], fill=WHITE, width=2)
    # ear loops
    d.arc([2, 14, 26, 60], start=260, end=100, fill=color, width=4)
    d.arc([54, 14, 78, 60], start=80, end=280, fill=color, width=4)
    return img


def draw_vest(color):
    img = _canvas()
    d = ImageDraw.Draw(img)
    # torso
    d.polygon(
        [(22, 12), (58, 12), (66, 70), (52, 70), (52, 34), (28, 34), (28, 70), (14, 70)],
        fill=color,
    )
    # collar notch
    d.polygon([(34, 12), (46, 12), (40, 24)], fill=(0, 0, 0, 0))
    # reflective stripe
    d.line([16, 46, 64, 46], fill=WHITE, width=5)
    return img


ICONS = {
    "helmet": draw_helmet,
    "mask": draw_mask,
    "vest": draw_vest,
}


def main():
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    for name, draw_fn in ICONS.items():
        for status, color in (("green", GREEN), ("red", RED)):
            img = draw_fn(color)
            out_path = ICONS_DIR / f"{name}_{status}.png"
            img.save(out_path)
            print(f"wrote {out_path}")


if __name__ == "__main__":
    sys.exit(main())
