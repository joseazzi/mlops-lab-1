import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def make_loader(folder, batch_size, shuffle=False):
    dataset = datasets.ImageFolder(folder, transform=TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataset, loader


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images)
            loss = criterion(predictions, labels)
            total_loss += loss.item() * len(labels)
            correct += (predictions.argmax(dim=1) == labels).sum().item()
            count += len(labels)

    return total_loss / count, correct / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mini", "processed"], default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(42)
    folder_name = (
        "food11_processed_mini" if args.dataset == "mini"
        else "food11_processed"
    )
    data_root = PROJECT_ROOT / "data" / folder_name

    train_data, train_loader = make_loader(
        data_root / "training", args.batch_size, shuffle=True
    )
    val_data, val_loader = make_loader(
        data_root / "validation", args.batch_size
    )
    test_data, test_loader = make_loader(
        data_root / "evaluation", args.batch_size
    )

    if not (train_data.classes == val_data.classes == test_data.classes):
        raise ValueError("The class folders differ between data splits.")

    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("food11")

    with mlflow.start_run() as run:
        mlflow.log_params({
            "dataset": args.dataset,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "model": "resnet18",
            "trained_layers": "fc",
        })

        print("Loading pretrained ResNet18...")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            # Keep the pretrained backbone's batch normalization fixed.
            model.eval()
            total_loss = 0.0
            count = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(labels)
                count += len(labels)

            train_loss = total_loss / count
            val_loss, val_accuracy = evaluate(model, val_loader, criterion)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            print(
                f"Epoch {epoch}/{args.epochs}: "
                f"train_loss={train_loss:.3f}, "
                f"val_loss={val_loss:.3f}, "
                f"val_accuracy={val_accuracy:.3f}",
                flush=True,
            )

        _, test_accuracy = evaluate(model, test_loader, criterion)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.pytorch.log_model(
            model, name="model", serialization_format="pickle"
        )
        print(f"Test accuracy: {test_accuracy:.3f}")
        print(f"MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
