from pathlib import Path
from PIL import Image
import shutil


RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

CATEGORIES = {
    0: "Bread",
    1: "Dairy product",
    2: "Dessert",
    3: "Egg",
    4: "Fried food",
    5: "Meat",
    6: "Noodles-Pasta",
    7: "Rice",
    8: "Seafood",
    9: "Soup",
    10: "Vegetable-Fruit",
}

SPLITS = ["training", "evaluation", "validation"]

IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100


def prepare_data():
    # Remove old processed folders if the script is run again
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

            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            # Example:
            # 0_123.jpg -> 0 -> Bread
            try:
                category_id = int(
                    image_path.stem.split("_")[0]
                )
            except ValueError:
                print(f"Skipping {image_path.name}")
                continue

            if category_id not in CATEGORIES:
                print(f"Unknown category: {image_path.name}")
                continue

            category_name = CATEGORIES[category_id]

            # Folder for the full processed dataset
            processed_category = (
                PROCESSED_DIR
                / split
                / category_name
            )

            processed_category.mkdir(
                parents=True,
                exist_ok=True
            )

            destination = (
                processed_category
                / image_path.name
            )

            # Open and resize the image to 128x128
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = image.resize(
                    IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )
                image.save(destination)

            # Add at most 100 images per category
            # to the mini dataset
            if mini_counts[category_id] < MINI_LIMIT:

                mini_category = (
                    MINI_DIR
                    / split
                    / category_name
                )

                mini_category.mkdir(
                    parents=True,
                    exist_ok=True
                )

                shutil.copy2(
                    destination,
                    mini_category / image_path.name
                )

                mini_counts[category_id] += 1

        print(f"Finished {split}")


if __name__ == "__main__":
    prepare_data()
    print("Dataset preparation complete.")