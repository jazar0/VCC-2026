import os
import random
import argparse

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

def delete_images_from_classes(root_dir, count=400, dry_run=False):
    """
    For each subfolder (class) in root_dir, randomly delete `count` images.

    Args:
        root_dir: Path to the root directory containing class folders.
        count:    Number of images to delete per class folder.
        dry_run:  If True, only print what would be deleted without actually deleting.
    """
    class_folders = [
        entry.path
        for entry in os.scandir(root_dir)
        if entry.is_dir()
    ]

    if not class_folders:
        print(f"No subfolders found in '{root_dir}'.")
        return

    print(f"Found {len(class_folders)} class folder(s) in '{root_dir}'.")
    if dry_run:
        print("** DRY RUN — no files will actually be deleted **\n")

    total_deleted = 0

    for folder in sorted(class_folders):
        class_name = os.path.basename(folder)

        images = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        ]

        if len(images) < count:
            print(
                f"[SKIP] '{class_name}': only {len(images)} image(s) found "
                f"(need {count}). Skipping to avoid deleting the entire class."
            )
            continue

        to_delete = random.sample(images, count)

        deleted = 0
        for path in to_delete:
            if dry_run:
                print(f"  [dry-run] Would delete: {path}")
            else:
                os.remove(path)
            deleted += 1

        total_deleted += deleted
        action = "Would delete" if dry_run else "Deleted"
        print(f"[OK] '{class_name}': {action} {deleted} image(s). "
              f"Remaining: {len(images) - (deleted if not dry_run else 0)}")

    print(f"\nDone. Total images {'that would be' if dry_run else ''} deleted: {total_deleted}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly delete a fixed number of images from each class folder."
    )
    parser.add_argument(
        "root_dir",
        help="Path to the root directory containing class subfolders."
    )
    parser.add_argument(
        "--count", type=int, default=400,
        help="Number of images to delete per class (default: 400)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview deletions without actually removing any files."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (optional)."
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    delete_images_from_classes(args.root_dir, count=args.count, dry_run=args.dry_run)
