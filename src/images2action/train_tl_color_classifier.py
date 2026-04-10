"""
Train the traffic-light color classifier (ResNet18) on the tl_color dataset.

Expects data/datasets/tl_color/train and data/datasets/tl_color/val with class folders
(red, yellow, green). Saves best model and confusion matrix to artifacts.

Run from project root or use scripts/run_train_tl_color_classifier.py.
"""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from . import config as pkg_config


def get_default_paths():
    root = pkg_config.project_root()
    return {
        "data_root": os.path.join(root, "data", "datasets", "tl_color"),
        "model_dir": os.path.join(root, "artifacts", "models"),
        "figures_dir": os.path.join(root, "artifacts", "figures"),
    }


def train(
    data_root=None,
    model_dir=None,
    figures_dir=None,
    batch_size=32,
    epochs=5,
    lr=1e-4,
    img_size=224,
    seed=42,
):
    paths = get_default_paths()
    data_root = data_root or paths["data_root"]
    model_dir = model_dir or paths["model_dir"]
    figures_dir = figures_dir or paths["figures_dir"]

    pkg_config.ensure_dir(model_dir)
    pkg_config.ensure_dir(figures_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_root, "train"),
        transform=train_transform,
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_root, "val"),
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    class_names = train_dataset.classes
    print("Classes:", class_names)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    all_preds, all_labels = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_acc = val_correct / val_total

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {running_loss:.4f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_path = os.path.join(model_dir, f"tl_color_best_{stamp}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }, best_path)
            print(f"Saved best model to {best_path}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues",
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Traffic Light Color Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "tl_color_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")
    print("Training complete.")
    return best_path


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 traffic-light color classifier on tl_color dataset."
    )
    parser.add_argument("--data-root", default=None, help="data/datasets/tl_color root")
    parser.add_argument("--model-dir", default=None, help="Where to save best .pt")
    parser.add_argument("--figures-dir", default=None, help="Where to save confusion matrix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_root=args.data_root,
        model_dir=args.model_dir,
        figures_dir=args.figures_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        img_size=args.img_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
