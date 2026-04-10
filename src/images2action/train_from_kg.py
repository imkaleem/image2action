"""
Train a ResNet18 action classifier directly from the KG.

This mirrors the logic from `notebooks/tutorial.ipynb` but runs as a
script using a YAML config (see `config/tutorial.yaml`).
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Tuple

import rdflib
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from .nb_config import load_experiment_config


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


class KGImageDataset(Dataset):
    def __init__(self, data: List[Tuple[str, int]], transform=None, root: str = PROJECT_ROOT):
        self.data = data
        self.transform = transform
        self.root = root

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        path, label = self.data[idx]

        if not os.path.isabs(path):
            full_path = os.path.join(self.root, path)
        else:
            full_path = path

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Image not found: {full_path}")

        image = Image.open(full_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataset(cfg):
    schema_path = cfg["data"]["schema_path"]
    bdd_100k_val_path = cfg["data"]["bdd_ttl"]
    coco_traffic_val_path = cfg["data"]["coco_ttl"]

    if not os.path.isabs(schema_path):
        schema_path = os.path.join(PROJECT_ROOT, schema_path)
    if not os.path.isabs(bdd_100k_val_path):
        bdd_100k_val_path = os.path.join(PROJECT_ROOT, bdd_100k_val_path)
    if not os.path.isabs(coco_traffic_val_path):
        coco_traffic_val_path = os.path.join(PROJECT_ROOT, coco_traffic_val_path)

    print(f"Schema.ttl exists? {os.path.exists(schema_path)}")
    print(f"bdd_100k_val.ttl exists? {os.path.exists(bdd_100k_val_path)}")
    print(f"coco_traffic_val.ttl exists? {os.path.exists(coco_traffic_val_path)}")

    print("Loading RDF graphs...")
    g = rdflib.Graph()
    g.parse(schema_path, format="turtle")
    g.parse(bdd_100k_val_path, format="turtle")
    g.parse(coco_traffic_val_path, format="turtle")
    print(f"Graph loaded with {len(g)} triples")

    query = cfg["query"]["text"]
    results = g.query(query)

    data = []
    for row in results:
        path = str(row.path)
        action = str(row.action)
        data.append((path, action))

    print(f"Total samples from KG: {len(data)}")

    unique_actions = sorted(list(set([a for _, a in data])))
    label_map = {uri: idx for idx, uri in enumerate(unique_actions)}
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\nLabel Mapping:")
    for k, v in label_map.items():
        print(f"{k.split('/')[-1]} → {v}")

    dataset = [(path, label_map[action]) for path, action in data]
    return dataset, label_map, idx_to_label, unique_actions


def plot_confusion(model, loader, device, idx_to_label, unique_actions, save_path=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print(unique_actions)
    print(idx_to_label)

    # Plotting can be fragile across environments
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, 
                    annot=True, 
                    fmt="d", 
                    xticklabels=[idx_to_label[i].split('/')[-1] for i in range(len(unique_actions))],
                    yticklabels=[idx_to_label[i].split('/')[-1] for i in range(len(unique_actions))],
                    cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches="tight")
    except Exception as e:
        print(f"Warning: failed to plot/save confusion matrix ({e}). Skipping.")
    finally:
        plt.close()


def main(config_path: str = "config/tutorial.yaml"):
    # Ensure project root on sys.path when run directly
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    cfg = load_experiment_config(config_path)

    dataset, label_map, idx_to_label, unique_actions = build_dataset(cfg)

    train_data, val_data = train_test_split(
        dataset,
        test_size=cfg["training"]["val_split"],
        random_state=cfg["training"]["seed"],
    )

    # Use PILToTensor + ConvertImageDtype to avoid numpy dtype issues
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = KGImageDataset(train_data, transform, root=PROJECT_ROOT)
    val_dataset = KGImageDataset(val_data, transform, root=PROJECT_ROOT)

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(unique_actions))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    print(f"Using device: {device}")
    print(f"Run name: {cfg['output']['run_name']}")

    EPOCHS = cfg["training"]["num_epochs"]

    os.makedirs(cfg["output"]["model_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["metrics_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["figures_dir"], exist_ok=True)

    run_name = cfg["output"]["run_name"]

    def evaluate(model, loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training Loss: {running_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")

    # Save final model and metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(cfg["output"]["model_dir"], f"{run_name}_{timestamp}.pt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_map": label_map,
            "idx_to_label": idx_to_label,
            "config": cfg,
        },
        model_path,
    )

    metrics = {
        "epochs": EPOCHS,
        "val_accuracy": float(val_acc),
        "train_loss_last_epoch": float(running_loss),
    }

    metrics_path = os.path.join(
        cfg["output"]["metrics_dir"],
        f"{run_name}_{timestamp}_metrics.json",
    )
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")

    # Confusion matrix
    conf_mat_path = os.path.join(
        cfg["output"]["figures_dir"],
        f"{run_name}_{timestamp}_confusion_matrix.png",
    )
    plot_confusion(
        model,
        val_loader,
        device,
        idx_to_label,
        unique_actions,
        save_path=conf_mat_path,
    )
    print(f"Saved confusion matrix to {conf_mat_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a ResNet18 action classifier from the KG using a YAML config.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/tutorial.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)

