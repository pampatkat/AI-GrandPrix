"""Validate the gate images and their YOLO pose labels before training."""

import argparse
from collections import Counter
from pathlib import Path


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VALUES_PER_GATE = 5 + (8 * 3)


def validate_split(name, root):
    if not root.is_dir():
        return [], Counter(), [f"{name}: folder does not exist: {root}"]

    images = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    stats = Counter(images=len(images))
    errors = []

    for image in images:
        label = image.with_suffix(".txt")
        if not label.exists():
            # Missing label files are valid YOLO background/negative images.
            stats["background_images"] += 1
            continue

        stats["labeled_images"] += 1
        for line_number, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            stats["gates"] += 1
            fields = line.split()
            location = f"{label}:{line_number}"
            if len(fields) != VALUES_PER_GATE:
                errors.append(
                    f"{location}: expected {VALUES_PER_GATE} values, found {len(fields)}"
                )
                continue
            try:
                values = [float(value) for value in fields]
            except ValueError:
                errors.append(f"{location}: contains a non-numeric value")
                continue

            if values[0] != 0:
                errors.append(f"{location}: class must be 0, found {values[0]:g}")
            if any(value < 0 or value > 1 for value in values[1:5]):
                errors.append(f"{location}: bounding box values must be in [0, 1]")
            if values[3] <= 0 or values[4] <= 0:
                errors.append(f"{location}: bounding box width and height must be positive")

            for keypoint in range(8):
                offset = 5 + (keypoint * 3)
                x, y, visibility = values[offset:offset + 3]
                if x < 0 or x > 1 or y < 0 or y > 1:
                    errors.append(f"{location}: keypoint {keypoint} coordinates must be in [0, 1]")
                if visibility not in (0, 1, 2):
                    errors.append(f"{location}: keypoint {keypoint} visibility must be 0, 1, or 2")

    return images, stats, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path(r"E:\AIGP_3379\TrainingImages"))
    parser.add_argument("--val", type=Path,
                        default=Path(r"E:\AIGP_3379\ValidationImages\ValidationImages"))
    args = parser.parse_args()

    all_errors = []
    for name, path in (("train", args.train), ("val", args.val)):
        _, stats, errors = validate_split(name, path)
        all_errors.extend(errors)
        print(
            f"{name}: {stats['images']} images, {stats['labeled_images']} labeled, "
            f"{stats['background_images']} background, {stats['gates']} gates"
        )

    if all_errors:
        print(f"\nFound {len(all_errors)} label errors:")
        for error in all_errors[:100]:
            print(f"- {error}")
        if len(all_errors) > 100:
            print(f"- ...and {len(all_errors) - 100} more")
        raise SystemExit(1)
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
