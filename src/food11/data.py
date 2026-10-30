import shutil
from pathlib import Path

from PIL import Image

RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

CLASSES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

CATEGORIES = dict(enumerate(CLASSES))
SPLITS = ["training", "evaluation", "validation"]
IMAGE_SIZE = 128
MINI_LIMIT = 100


def category_from_filename(filename: str) -> str:
    """Return the Food-11 category contained in an image filename."""
    try:
        category_id = int(Path(filename).stem.split("_")[0])
        return CATEGORIES[category_id]
    except (ValueError, KeyError, IndexError) as error:
        raise ValueError(f"Invalid Food-11 filename: {filename}") from error


def resize_image(
    img: Image.Image,
    size: int = IMAGE_SIZE,
) -> Image.Image:
    """Convert an image to RGB and resize it to size x size."""
    return img.convert("RGB").resize(
        (size, size),
        Image.Resampling.LANCZOS,
    )


def prepare_data():
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    if MINI_DIR.exists():
        shutil.rmtree(MINI_DIR)

    for split in SPLITS:
        source_folder = RAW_DIR / split
        mini_counts = {
            category_id: 0
            for category_id in CATEGORIES
        }

        for image_path in sorted(source_folder.iterdir()):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in [
                ".jpg",
                ".jpeg",
                ".png",
            ]:
                continue

            try:
                category_name = category_from_filename(
                    image_path.name
                )
                category_id = CLASSES.index(category_name)
            except ValueError:
                print(f"Skipping {image_path.name}")
                continue

            processed_category = (
                PROCESSED_DIR / split / category_name
            )
            processed_category.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (
                processed_category / image_path.name
            )

            with Image.open(image_path) as image:
                resized = resize_image(image)
                resized.save(destination)

            if mini_counts[category_id] < MINI_LIMIT:
                mini_category = (
                    MINI_DIR / split / category_name
                )
                mini_category.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    destination,
                    mini_category / image_path.name,
                )
                mini_counts[category_id] += 1

        print(f"Finished {split}")


if __name__ == "__main__":
    prepare_data()
    print("Dataset preparation complete.")