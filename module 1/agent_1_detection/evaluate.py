"""
Evaluation Script for Agent 1 – Tumor Detection (YOLOv8)
Computes Precision, Recall, mAP50, and mAP50-95 on the held-out test split.
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO


def evaluate_detector(
    model_path: str = "models/detection/yolov8_tumor_detection.pt",
    data_yaml: str = "data/yolo/data.yaml",
    split: str = "test",
    device: str = "cpu"
):
    """
    Evaluates detector on the held-out test split.
    """
    if not os.path.exists(model_path):
        if os.path.exists("runs/detect/tumor_detector/weights/best.pt"):
            model_path = "runs/detect/tumor_detector/weights/best.pt"
        else:
            print(f"[Warning] Weights not found at {model_path}. Using base yolov8n.pt for pipeline validation.")
            model_path = "yolov8n.pt"

    print("==================================================")
    print("   Breast CARE-MA: Agent 1 Detector Evaluation   ")
    print("==================================================")
    print(f"Model Path:  {model_path}")
    print(f"Data Config: {data_yaml}")
    print(f"Split:       {split}")
    print(f"Device:      {device}")
    print("==================================================")

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        device=device,
        verbose=True
    )

    p = metrics.box.mp
    r = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    print("\n--- Evaluation Results on Test Split ---")
    print(f"Mean Precision (P):   {p:.4f}")
    print(f"Mean Recall (R):      {r:.4f}")
    print(f"mAP @ 0.50:           {map50:.4f}")
    print(f"mAP @ 0.50-0.95:      {map50_95:.4f}")
    print("----------------------------------------\n")

    return {
        "precision": float(p),
        "recall": float(r),
        "map50": float(map50),
        "map50_95": float(map50_95)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Agent 1 YOLOv8 Tumor Detector")
    parser.add_argument("--model", type=str, default="models/detection/yolov8_tumor_detection.pt")
    parser.add_argument("--data", type=str, default="data/yolo/data.yaml")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    evaluate_detector(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        device=args.device
    )
