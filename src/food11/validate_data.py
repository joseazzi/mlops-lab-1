import argparse
import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.food11.data import CLASSES

SPLITS = ("training", "validation", "evaluation")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_SIZE = (128, 128)


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )


def format_details(items: list[str]) -> str:
    if not items:
        return ""

    visible = items[:5]
    result = "; ".join(visible)

    if len(items) > 5:
        result += f"; and {len(items) - 5} more"

    return result


def print_result(
    name: str,
    passed: bool,
    details: str = "",
) -> None:
    status = "PASS" if passed else "FAIL"
    message = f"{status}: {name}"

    if details:
        message += f" — {details}"

    print(message)


def check_class_folders(
    dataset_root: Path,
) -> tuple[bool, str]:
    expected = set(CLASSES)
    failures = []

    for split in SPLITS:
        split_path = dataset_root / split

        if not split_path.is_dir():
            failures.append(f"{split}: folder is missing")
            continue

        actual = {
            path.name
            for path in split_path.iterdir()
            if path.is_dir()
        }

        missing = sorted(expected - actual)
        extra = sorted(actual - expected)

        if missing:
            failures.append(
                f"{split}: missing {missing}"
            )

        if extra:
            failures.append(
                f"{split}: extra {extra}"
            )

    return not failures, format_details(failures)


def check_images(
    dataset_root: Path,
) -> tuple[bool, str]:
    failures = []

    for split in SPLITS:
        for class_name in CLASSES:
            class_path = (
                dataset_root / split / class_name
            )

            if not class_path.is_dir():
                continue

            for image_path in image_paths(class_path):
                try:
                    with Image.open(image_path) as image:
                        image.load()

                        if image.mode != "RGB":
                            failures.append(
                                f"{image_path}: mode "
                                f"{image.mode}"
                            )

                        if image.size != EXPECTED_SIZE:
                            failures.append(
                                f"{image_path}: size "
                                f"{image.size}"
                            )
                except (
                    OSError,
                    UnidentifiedImageError,
                ) as error:
                    failures.append(
                        f"{image_path}: {error}"
                    )

    return not failures, format_details(failures)


def count_images(
    dataset_root: Path,
) -> dict[str, dict[str, int]]:
    counts = {}

    for split in SPLITS:
        counts[split] = {}

        for class_name in CLASSES:
            class_path = (
                dataset_root / split / class_name
            )

            if class_path.is_dir():
                count = len(image_paths(class_path))
            else:
                count = 0

            counts[split][class_name] = count

    return counts


def check_minimum_count(
    counts: dict[str, dict[str, int]],
    minimum: int,
) -> tuple[bool, str]:
    failures = []

    for split in SPLITS:
        for class_name in CLASSES:
            count = counts[split][class_name]

            if count < minimum:
                failures.append(
                    f"{split}/{class_name}: "
                    f"{count} < {minimum}"
                )

    return not failures, format_details(failures)


def check_cross_split_duplicates(
    dataset_root: Path,
) -> tuple[bool, str]:
    seen = {}
    failures = []

    for split in SPLITS:
        split_path = dataset_root / split

        if not split_path.is_dir():
            continue

        for image_path in image_paths(split_path):
            digest = hashlib.md5(
                image_path.read_bytes(),
                usedforsecurity=False,
            ).hexdigest()

            previous = seen.get(digest)

            if previous is not None:
                previous_split, previous_path = previous

                if previous_split != split:
                    failures.append(
                        f"{previous_path} and "
                        f"{image_path}"
                    )
            else:
                seen[digest] = (split, image_path)

    return not failures, format_details(failures)


def check_class_distribution(
    counts: dict[str, dict[str, int]],
    tau: float,
) -> tuple[bool, str]:
    training_total = sum(
        counts["training"].values()
    )
    validation_total = sum(
        counts["validation"].values()
    )

    if training_total == 0 or validation_total == 0:
        return False, "training or validation is empty"

    failures = []

    for class_name in CLASSES:
        training_share = (
            counts["training"][class_name]
            / training_total
        )
        validation_share = (
            counts["validation"][class_name]
            / validation_total
        )
        difference = abs(
            training_share - validation_share
        )

        if difference > tau:
            failures.append(
                f"{class_name}: difference "
                f"{difference:.4f} > {tau:.4f}"
            )

    return not failures, format_details(failures)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
    )
    args = parser.parse_args()

    counts = count_images(args.dataset)

    checks = [
        (
            "exactly 11 class folders per split",
            *check_class_folders(args.dataset),
        ),
        (
            "images are readable, RGB and 128x128",
            *check_images(args.dataset),
        ),
        (
            "minimum images per class",
            *check_minimum_count(
                counts,
                args.min_per_class,
            ),
        ),
        (
            "no duplicates across splits",
            *check_cross_split_duplicates(
                args.dataset
            ),
        ),
        (
            "class distributions are within tau",
            *check_class_distribution(
                counts,
                args.tau,
            ),
        ),
    ]

    for name, passed, details in checks:
        print_result(name, passed, details)

    return 0 if all(
        passed
        for _, passed, _ in checks
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
