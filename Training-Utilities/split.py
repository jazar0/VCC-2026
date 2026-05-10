"""
Dataset Splitter
----------------
Splits files from a "Dataset" folder into train, test, and validation sets
with a 70 / 20 / 10 ratio, preserving the full directory structure.
 
Output layout:
    split_output/
    ├── train/   (70 %)
    ├── test/    (20 %)
    └── val/     (10 %)
 
Usage:
    python split_dataset.py                        # uses ./Dataset and ./split_output
    python split_dataset.py --src /path/to/data    # custom source
    python split_dataset.py --src data --out out   # custom source + destination
    python split_dataset.py --seed 99              # reproducible shuffle
"""
 
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
 
SPLITS = {"train": 0.70, "test": 0.20, "val": 0.10}
 
 
# ── Helpers ────────────────────────────────────────────────────────────────────
 
def collect_files(src: Path) -> dict[Path, list[Path]]:
    """
    Walk *src* and group files by their parent directory (relative to src).
    Returns {relative_dir: [absolute_file, ...], ...}.
    Directories that contain no files (only sub-dirs) are skipped.
    """
    groups: dict[Path, list[Path]] = defaultdict(list)
    for file in sorted(src.rglob("*")):
        if file.is_file():
            groups[file.parent.relative_to(src)].append(file)
    return dict(groups)
 
 
def split_list(items: list, ratios: dict[str, float], seed: int) -> dict[str, list]:
    """
    Shuffle *items* (deterministically) then cut according to *ratios*.
    Leftover files (due to rounding) go into the first split.
    """
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
 
    n = len(shuffled)
    result: dict[str, list] = {}
    used = 0
    names = list(ratios)
 
    for i, name in enumerate(names):
        if i == len(names) - 1:          # last bucket gets the remainder
            count = n - used
        else:
            count = round(n * ratios[name])
        result[name] = shuffled[used : used + count]
        used += count
 
    return result
 
 
def copy_files(files: list[Path], src_root: Path, dst_root: Path, rel_dir: Path) -> int:
    """Copy *files* into dst_root / rel_dir, creating dirs only when needed."""
    target_dir = dst_root / rel_dir
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, target_dir / f.name)
    return len(files)
 
 
# ── Main ───────────────────────────────────────────────────────────────────────
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Split a dataset into train / test / val.")
    parser.add_argument("--src",  default="Dataset",      help="Source dataset folder (default: ./Dataset)")
    parser.add_argument("--out",  default="split_output", help="Output folder          (default: ./split_output)")
    parser.add_argument("--seed", type=int, default=42,   help="Random seed            (default: 42)")
    args = parser.parse_args()
 
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
 
    if not src.exists():
        raise SystemExit(f"[ERROR] Source folder not found: {src}")
    if not src.is_dir():
        raise SystemExit(f"[ERROR] Source path is not a directory: {src}")
 
    print(f"Source      : {src}")
    print(f"Destination : {out}")
    print(f"Split ratio : train={SPLITS['train']:.0%}  test={SPLITS['test']:.0%}  val={SPLITS['val']:.0%}")
    print(f"Random seed : {args.seed}\n")
 
    # Gather files grouped by sub-directory
    groups = collect_files(src)
 
    if not groups:
        raise SystemExit("[ERROR] No files found in the source directory.")
 
    # Counters for summary
    totals: dict[str, int] = {name: 0 for name in SPLITS}
    warned_dirs: list[str] = []
 
    for rel_dir, files in sorted(groups.items()):
        if len(files) < len(SPLITS):
            warned_dirs.append(
                f"  {rel_dir}  ({len(files)} file{'s' if len(files) != 1 else ''}) "
                f"— too few for a clean split; all go to 'train'."
            )
            # Put everything in train rather than leaving splits empty
            dst = out / "train" / rel_dir
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
            totals["train"] += len(files)
            continue
 
        assigned = split_list(files, SPLITS, seed=args.seed)
        for split_name, split_files in assigned.items():
            n = copy_files(split_files, src, out / split_name, rel_dir)
            totals[split_name] += n
 
    # ── Summary ────────────────────────────────────────────────────────────────
    grand_total = sum(totals.values())
    print("=" * 50)
    print("Split complete!")
    print(f"  {'Split':<8}  {'Files':>6}  {'Actual %':>9}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*9}")
    for name, count in totals.items():
        pct = count / grand_total * 100 if grand_total else 0
        print(f"  {name:<8}  {count:>6}  {pct:>8.1f}%")
    print(f"  {'TOTAL':<8}  {grand_total:>6}")
    print("=" * 50)
 
    if warned_dirs:
        print("\n[WARNINGS] The following directories had too few files for a balanced split:")
        for w in warned_dirs:
            print(w)
 
    print(f"\nOutput written to: {out}")
 
 
if __name__ == "__main__":
    main()