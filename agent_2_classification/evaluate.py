"""
Evaluation Pipeline for Agent 2 – Tumor Classification (ResNet)
Computes Test Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix.
"""

import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .model import BreastCancerClassifier


def evaluate_classifier(
    model_path: str = "models/classification/resnet18_tumor_classifier.pth",
    data_dir: str = "data/classification",
    architecture: str = "resnet18",
    batch_size: int = 16,
    device: str = "cpu"
):
    """Evaluates classifier on the test split."""
    device = torch.device(device)
    test_path = os.path.join(data_dir, "test")

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(test_path, transform=eval_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names = test_dataset.classes
    print("==================================================")
    print("   Breast CARE-MA: Agent 2 Classifier Evaluation  ")
    print("==================================================")
    print(f"Model Path:  {model_path}")
    print(f"Test Split:  {test_path} ({len(test_dataset)} images)")
    print(f"Classes:     {class_names}")
    print(f"Device:      {device}")
    print("==================================================")

    model = BreastCancerClassifier(architecture=architecture, num_classes=len(class_names), pretrained=False)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"[Agent 2] Loaded weights from: {model_path}")
    else:
        print("[Agent 2] Warning: Checkpoint not found. Evaluating with baseline weights.")

    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, digits=4)

    print("\n--- Test Split Classification Metrics ---")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1:  {macro_f1:.4f}")
    print("\nDetailed Per-Class Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("------------------------------------------\n")

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
        "report": report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Agent 2 Tumor Classifier")
    parser.add_argument("--model", type=str, default="models/classification/resnet18_tumor_classifier.pth")
    parser.add_argument("--data", type=str, default="data/classification")
    parser.add_argument("--arch", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    evaluate_classifier(
        model_path=args.model,
        data_dir=args.data,
        architecture=args.arch,
        device=args.device
    )
