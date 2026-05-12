#!/usr/bin/env python3
"""
Image Augmentation Script
==========================
Generates 100 variants per image (5x5 position grid × 4 zoom levels)
on a 384×384 white canvas. Same transforms applied identically to all images.

Usage:
    python3 generate_variants.py

Input:  ~/Desktop/VCC-2026/Data-Collection-Utilities/artificial collect/L1-Images/
Output: ~/Desktop/VCC-2026/Data-Collection-Utilities/artificial collect/L1-Output/<original_name>/<variant>.jpg
"""

import os
from pathlib import Path
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────────────────────

CANVAS_SIZE   = 384          # output canvas is always 384×384
GRID_COLS     = 5            # 5 columns
GRID_ROWS     = 5            # 5 rows  →  25 positions
ZOOM_LEVELS   = [0.40, 0.55, 0.70, 0.85]   # image occupies this fraction of canvas
CANVAS_BG     = (255, 255, 255)             # white background

# Root folder (absolute path)
ROOT = Path.home() / "Desktop/VCC-2026/Data-Collection-Utilities/artificial collect"
INPUT_DIR  = ROOT / "L1-Images"
OUTPUT_DIR = ROOT / "L1-Output"

# ─── BUILD TRANSFORM TABLE ────────────────────────────────────────────────────
# 25 positions × 4 zooms = 100 transforms, fixed/deterministic.
# Position is the top-left corner (x, y) of the pasted image on the canvas.
# The image is fully visible at all times (clamped to canvas bounds).

def build_transforms():
    transforms = []
    variant_id = 1
    for zoom in ZOOM_LEVELS:
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                transforms.append({
                    "id":   variant_id,
                    "zoom": zoom,
                    "col":  col,
                    "row":  row,
                })
                variant_id += 1
    return transforms   # 100 entries


def apply_transform(src_img: Image.Image, zoom: float, col: int, row: int) -> Image.Image:
    """
    Place src_img on a white 384×384 canvas.

    - Image is scaled proportionally so its longest side = zoom × CANVAS_SIZE.
    - Position is chosen from a 5×5 grid covering the valid placement range
      (i.e. the canvas area where the entire scaled image still fits).
    """
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), CANVAS_BG)

    # --- scale image (proportional, longest side → zoom * CANVAS_SIZE) ---
    max_side = int(zoom * CANVAS_SIZE)
    src_w, src_h = src_img.size
    scale = min(max_side / src_w, max_side / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    scaled = src_img.resize((new_w, new_h), Image.LANCZOS)

    # --- determine valid placement range ---
    # top-left x can range [0 .. CANVAS_SIZE - new_w]
    # top-left y can range [0 .. CANVAS_SIZE - new_h]
    x_range = CANVAS_SIZE - new_w   # could be 0 if image fills full width
    y_range = CANVAS_SIZE - new_h

    # divide range into GRID_COLS / GRID_ROWS steps (evenly spaced)
    if GRID_COLS > 1:
        x = int(col * x_range / (GRID_COLS - 1))
    else:
        x = x_range // 2

    if GRID_ROWS > 1:
        y = int(row * y_range / (GRID_ROWS - 1))
    else:
        y = y_range // 2

    canvas.paste(scaled, (x, y))
    return canvas


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_DIR.exists():
        print(f"❌  Input directory not found:\n    {INPUT_DIR}")
        return

    image_files = sorted([
        f for f in INPUT_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg")
    ])

    if not image_files:
        print(f"❌  No JPG images found in:\n    {INPUT_DIR}")
        return

    print(f"✅  Found {len(image_files)} images in L1-Images/")
    transforms = build_transforms()
    print(f"✅  {len(transforms)} transforms (5×5 positions × {len(ZOOM_LEVELS)} zoom levels)")
    print(f"📁  Output → {OUTPUT_DIR}\n")

    for img_path in image_files:
        # Output folder named after the original file (without extension)
        out_folder = OUTPUT_DIR / img_path.stem
        out_folder.mkdir(parents=True, exist_ok=True)

        src = Image.open(img_path).convert("RGB")

        for t in transforms:
            canvas = apply_transform(src, t["zoom"], t["col"], t["row"])

            # Filename: <stem>_v001_z40_r0_c0.jpg
            zoom_pct = int(t["zoom"] * 100)
            out_name = (
                f"{img_path.stem}"
                f"_v{t['id']:03d}"
                f"_z{zoom_pct}"
                f"_r{t['row']}_c{t['col']}"
                f".jpg"
            )
            canvas.save(out_folder / out_name, "JPEG", quality=95)

        print(f"  ✓  {img_path.name}  →  {len(transforms)} variants saved")

    total = len(image_files) * len(transforms)
    print(f"\n🎉  Done! {total} total images written to L1-Output/")
    print(f"\nFolder structure:")
    print(f"  L1-Output/")
    print(f"  ├── <image_name>/")
    print(f"  │   ├── <image_name>_v001_z40_r0_c0.jpg   ← zoom 40%, row 0, col 0 (top-left)")
    print(f"  │   ├── <image_name>_v002_z40_r0_c1.jpg")
    print(f"  │   ├── ... (100 variants per image)")
    print(f"  └── ...")


if __name__ == "__main__":
    main()