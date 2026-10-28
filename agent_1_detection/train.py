"""
Training Pipeline for Agent 1 – Tumor Detection (YOLOv8)
Fine-tunes YOLOv8 on the prepared BUSI ultrasound dataset.
"""

import argparse
import os
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO


def train_detector(
    data_yaml: str = "data/yolo/data.yaml",
    weights: str = "yolov8n.pt",
    epochs: int = 15,
    imgsz: int = 640,
    batch: int = 8,
    device: str = None,
    output_model_dir: str = "models/detection"
):
    """
    Train YOLOv8 on the BUSI detection dataset.
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(
            f"Configuration file not found: {data_yaml}. Run data/prepare_data.py first."
        )

    # Determine compute device
    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    print("==================================================")
    print("   Breast CARE-MA: Agent 1 Tumor Detection Training")
    print("==================================================")
    print(f"Dataset Config: {data_yaml}")
    print(f"Base Weights:   {weights}")
    print(f"Epochs:         {epochs}")
    print(f"Image Size:     {imgsz}")
    print(f"Batch Size:     {batch}")
    print(f"Compute Device: {device}")
    print("==================================================")

    model = YOLO(weights)

    # Launch training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name="tumor_detector",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Copy best weights to project models directory
    dest_dir = Path(output_model_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    best_weights = Path("runs/detect/tumor_detector/weights/best.pt")

    if best_weights.exists():
        target_path = dest_dir / "yolov8_tumor_detection.pt"
        shutil.copy2(best_weights, target_path)
        print(f"\n[Agent 1] Best model weights saved to: {target_path}")
    else:
        print("\n[Agent 1] Note: best.pt was not generated (training may have been abbreviated).")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Agent 1 YOLOv8 Tumor Detector")
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Initial weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or '0')")

    args = parser.parse_args()
    train_detector(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
