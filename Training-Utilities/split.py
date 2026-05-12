"""
Dataset Splitter — 70% Train / 20% Val / 10% Test
===================================================
Expected structure:
    <script location>/
    └── Dataset/
        ├── class_01/
        │   ├── img1.jpg
        │   └── ...
        ├── class_02/
        └── ... (25 folders total)

Output structure:
    <script location>/
    └── Dataset_split/
        ├── train/
        │   ├── class_01/
        │   └── ...
        ├── val/
        │   ├── class_01/
        │   └── ...
        └── test/
            ├── class_01/
            └── ...

Files are COPIED (not moved), so the original Dataset folder is preserved.
"""

import os
import json
import random
import shutil
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR        = Path("Dataset")           # input folder (relative to this script)
OUTPUT_DIR         = Path("Dataset_split")     # output folder
CLASS_NAMES_FILE   = Path("class_names.json")  # expected class names
SPLITS             = {"train": 0.70, "val": 0.20, "test": 0.10}
RANDOM_SEED        = 42                        # set to None for a different split each run
# ──────────────────────────────────────────────────────────────────────────────


def load_expected_classes(class_names_file: Path) -> list[str]:
    """Load the list of valid class names from a JSON file."""
    if not class_names_file.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_names_file.resolve()}\n"
            f"  Make sure '{class_names_file}' is in the same folder as this script."
        )
    with class_names_file.open() as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"'{class_names_file}' must contain a JSON array of strings.")
    return names


def validate_class_dirs(class_dirs: list[Path], expected_classes: list[str]) -> None:
    """
    Check that the folders in Dataset match class_names.json exactly.
    Prints a clear error and raises SystemExit if anything is off.
    """
    found_names    = {d.name for d in class_dirs}
    expected_names = set(expected_classes)

    missing  = expected_names - found_names   # in JSON but not on disk
    extra    = found_names - expected_names   # on disk but not in JSON
    ok       = found_names == expected_names

    if ok:
        print(f"  ✓ All {len(expected_classes)} class folders validated against class_names.json\n")
        return

    print("\n" + "!"*55)
    print("  ERROR: Dataset folder does not match class_names.json")
    print("  Your Dataset folder was likely created with a typo.")
    print("!"*55)

    if extra:
        print(f"\n  Unrecognized folders in Dataset ({len(extra)}):")
        for name in sorted(extra):
            suggestion = _closest_match(name, expected_names)
            hint = f"  ← did you mean '{suggestion}'?" if suggestion else ""
            print(f"    • '{name}'{hint}")

    if missing:
        print(f"\n  Folders expected by class_names.json but not found on disk ({len(missing)}):")
        for name in sorted(missing):
            print(f"    • '{name}'")

    print("\n  Please fix the folder names in your Dataset directory and re-run.")
    print("!"*55 + "\n")
    raise SystemExit(1)


def _closest_match(name: str, candidates: set[str], threshold: float = 0.7) -> str | None:
    """Return the closest string match using a simple character-overlap ratio."""
    best_ratio, best_match = 0.0, None
    name_lower = name.lower()
    for candidate in candidates:
        cand_lower = candidate.lower()
        # Jaccard similarity on character bigrams
        a = set(zip(name_lower, name_lower[1:]))
        b = set(zip(cand_lower, cand_lower[1:]))
        if not a and not b:
            ratio = 1.0 if name_lower == cand_lower else 0.0
        elif not a or not b:
            ratio = 0.0
        else:
            ratio = len(a & b) / len(a | b)
        if ratio > best_ratio:
            best_ratio, best_match = ratio, candidate
    return best_match if best_ratio >= threshold else None


def split_dataset(
    dataset_dir:      Path = DATASET_DIR,
    output_dir:       Path = OUTPUT_DIR,
    class_names_file: Path = CLASS_NAMES_FILE,
    splits:           dict = SPLITS,
    seed:             int  = RANDOM_SEED,
) -> None:
    # Validate ratios
    total = sum(splits.values())
    if not abs(total - 1.0) < 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir.resolve()}")

    class_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError(f"No sub-folders found inside '{dataset_dir}'")

    # ── Validate class names before doing anything else ──────────────────────
    expected_classes = load_expected_classes(class_names_file)
    validate_class_dirs(class_dirs, expected_classes)
    # ─────────────────────────────────────────────────────────────────────────

    print(f"\n{'='*55}")
    print(f"  Dataset Splitter  —  {'/'.join(str(int(v*100)) for v in splits.values())} split")
    print(f"{'='*55}")
    print(f"  Source  : {dataset_dir.resolve()}")
    print(f"  Output  : {output_dir.resolve()}")
    print(f"  Classes : {len(class_dirs)}")
    print(f"  Seed    : {seed}")
    print(f"{'='*55}\n")

    if seed is not None:
        random.seed(seed)

    split_names  = list(splits.keys())
    split_ratios = list(splits.values())

    grand_totals = {s: 0 for s in split_names}

    for class_dir in class_dirs:
        class_name = class_dir.name

        # Gather all files (non-recursive; add rglob("*") for nested files)
        files = [f for f in class_dir.iterdir() if f.is_file()]
        if not files:
            print(f"  [SKIP] '{class_name}' — no files found")
            continue

        random.shuffle(files)
        n = len(files)

        # Calculate split indices
        n_train = round(n * split_ratios[0])
        n_val   = round(n * split_ratios[1])

        buckets = {
            split_names[0]: files[:n_train],
            split_names[1]: files[n_train : n_train + n_val],
            split_names[2]: files[n_train + n_val :],   # remainder → test
        }

        print(f"  {class_name:<35}  total={n:>4}  "
              + "  ".join(f"{s}={len(b):>3}" for s, b in buckets.items()))

        for split_name, split_files in buckets.items():
            dest_dir = output_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_files:
                shutil.copy2(src, dest_dir / src.name)
            grand_totals[split_name] += len(split_files)

    print(f"\n{'='*55}")
    total_files = sum(grand_totals.values())
    print(f"  Done!  {total_files} files copied across {len(class_dirs)} classes")
    for s, count in grand_totals.items():
        pct = count / total_files * 100 if total_files else 0
        print(f"    {s:<6}: {count:>5} files  ({pct:.1f}%)")
    print(f"{'='*55}\n")
    print(f"  Output saved to: {output_dir.resolve()}\n")


if __name__ == "__main__":
    split_dataset()
