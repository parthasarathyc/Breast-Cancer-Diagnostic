"""
Training Pipeline for Agent 2 – Tumor Classification (ResNet18)
Includes Data Augmentation, Class Imbalance Weighting, and Macro-F1 Checkpointing.
"""

import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score

from .model import BreastCancerClassifier


def get_dataloaders(data_dir: str = "data/classification", batch_size: int = 16):
    """Creates train, val, and test DataLoaders with appropriate augmentations."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")

    train_dataset = datasets.ImageFolder(train_path, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_path, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(test_path, transform=eval_transforms)

    # Pin class ordering: {'benign': 0, 'malignant': 1, 'normal': 2} or custom
    class_to_idx = train_dataset.class_to_idx
    print(f"[Agent 2] Dataset Class Mapping: {class_to_idx}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_to_idx


def compute_class_weights(train_dataset, device):
    """Compute balanced inverse class weights to address data imbalance."""
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_classifier(
    data_dir: str = "data/classification",
    architecture: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = None,
    output_dir: str = "models/classification"
):
    """Trains ResNet on the BUSI ultrasound classification dataset."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print("==================================================")
    print("   Breast CARE-MA: Agent 2 Classification Training")
    print("==================================================")
    print(f"Data Dir:       {data_dir}")
    print(f"Architecture:   {architecture}")
    print(f"Epochs:         {epochs}")
    print(f"Batch Size:     {batch_size}")
    print(f"Learning Rate:  {lr}")
    print(f"Compute Device: {device}")
    print("==================================================")

    train_loader, val_loader, test_loader, class_to_idx = get_dataloaders(data_dir, batch_size)

    # Class weights for loss
    class_weights = compute_class_weights(train_loader.dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = BreastCancerClassifier(architecture=architecture, num_classes=len(class_to_idx), pretrained=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dest_path = Path(output_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    # Save class mapping
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    with open(dest_path / "class_mapping.json", "w") as f:
        json.dump(idx_to_class, f, indent=2)

    best_val_f1 = 0.0
    best_model_path = dest_path / f"{architecture}_tumor_classifier.pth"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(labels.cpu().numpy())

        scheduler.step()
        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} Macro-F1: {val_f1:.3f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "architecture": architecture
            }, best_model_path)
            print(f"  -> Checkpoint saved with Macro-F1: {val_f1:.4f}")

    print(f"\n[Agent 2] Training complete. Best model saved to: {best_model_path}")
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Agent 2 ResNet Classifier")
    parser.add_argument("--data", type=str, default="data/classification", help="Data directory")
    parser.add_argument("--arch", type=str, default="resnet18", help="Architecture (resnet18/resnet50)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 'cuda')")

    args = parser.parse_args()
    train_classifier(
        data_dir=args.data,
        architecture=args.arch,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        device=args.device
    )
