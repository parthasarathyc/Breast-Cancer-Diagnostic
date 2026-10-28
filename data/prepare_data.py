"""
Dataset Preparation Pipeline for Breast CARE-MA
Converts BUSI (Breast Ultrasound Images Dataset) segmentation masks into
YOLOv8 bounding-box annotations and prepares stratified Train/Val/Test splits
for both Agent 1 (Tumor Detection) and Agent 2 (Tumor Classification).
"""

import os
import glob
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split


def find_dataset_path(custom_path: str = None) -> str:
    """Locate the BUSI dataset directory."""
    candidate_paths = [
        custom_path,
        os.path.expanduser(r"~\Downloads\archive\Dataset_BUSI_with_GT"),
        os.path.expanduser(r"~\Downloads\Dataset_BUSI_with_GT"),
        r"c:\Users\Nivek Nachimuthu\Downloads\archive\Dataset_BUSI_with_GT",
        os.path.join(os.getcwd(), "data", "raw", "Dataset_BUSI_with_GT"),
    ]
    for p in candidate_paths:
        if p and os.path.exists(p) and os.path.isdir(p):
            subdirs = os.listdir(p)
            if all(c in subdirs for c in ["benign", "malignant", "normal"]):
                return p
    raise FileNotFoundError(
        "Could not find Dataset_BUSI_with_GT. Checked paths:\n"
        + "\n".join(str(p) for p in candidate_paths if p)
    )


def extract_bboxes_from_masks(mask_paths: list, img_w: int, img_h: int, min_area: int = 30) -> list:
    """
    Extract bounding boxes from one or more binary segmentation masks.
    Returns list of normalized YOLO format bounding boxes: [x_center, y_center, width, height]
    """
    bboxes = []
    for m_path in mask_paths:
        mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if mask is None or np.sum(mask > 0) == 0:
            continue
        
        # Threshold to ensure binary
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            # Normalize coordinates [0.0, 1.0]
            x_center = (x + w / 2.0) / img_w
            y_center = (y + h / 2.0) / img_h
            norm_w = w / float(img_w)
            norm_h = h / float(img_h)
            
            # Clip to valid [0, 1] range
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))
            
            bboxes.append((x_center, y_center, norm_w, norm_h))
    return bboxes


def prepare_datasets(
    source_dir: str = None,
    output_base: str = "data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
):
    """
    Prepares both:
    1. YOLOv8 dataset structure (data/yolo/images/{train,val,test} & data/yolo/labels/{train,val,test})
    2. Classification dataset structure (data/classification/{train,val,test}/{normal,benign,malignant})
    Ensures ZERO data leakage by using identical patient image splits.
    """
    dataset_path = find_dataset_path(source_dir)
    print(f"Found BUSI Dataset at: {dataset_path}")

    # Collect all image records
    records = []
    categories = ["normal", "benign", "malignant"]
    class_to_id = {"normal": 0, "benign": 1, "malignant": 2}

    for cat in categories:
        cat_dir = os.path.join(dataset_path, cat)
        files = os.listdir(cat_dir)
        raw_imgs = [f for f in files if "_mask" not in f and f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for img_name in raw_imgs:
            base, ext = os.path.splitext(img_name)
            img_path = os.path.join(cat_dir, img_name)
            mask_pattern = os.path.join(cat_dir, f"{base}_mask*{ext}")
            masks = glob.glob(mask_pattern)

            records.append({
                "filename": img_name,
                "category": cat,
                "class_id": class_to_id[cat],
                "img_path": img_path,
                "mask_paths": masks
            })

    print(f"Total collected images: {len(records)}")
    for cat in categories:
        cnt = sum(1 for r in records if r["category"] == cat)
        print(f"  - {cat}: {cnt} images")

    # Stratified split: Train (70%), Val (15%), Test (15%)
    train_val_records, test_records = train_test_split(
        records,
        test_size=test_ratio,
        stratify=[r["category"] for r in records],
        random_state=random_seed
    )
    
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=val_rel_ratio,
        stratify=[r["category"] for r in train_val_records],
        random_state=random_seed
    )

    splits = {
        "train": train_records,
        "val": val_records,
        "test": test_records
    }

    print("\nDataset Split Summary (Zero Leakage):")
    for split_name, recs in splits.items():
        cat_counts = {c: sum(1 for r in recs if r["category"] == c) for c in categories}
        print(f"  {split_name.upper()}: {len(recs)} total -> {cat_counts}")

    # Create directories
    yolo_dir = Path(output_base) / "yolo"
    cls_dir = Path(output_base) / "classification"

    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for cat in categories:
            (cls_dir / split / cat).mkdir(parents=True, exist_ok=True)

    # Process images and annotations
    print("\nGenerating YOLO detection annotations and classification splits...")
    total_bboxes = 0

    for split_name, recs in splits.items():
        for r in recs:
            img = cv2.imread(r["img_path"])
            if img is None:
                continue
            h, w = img.shape[:2]
            base, ext = os.path.splitext(r["filename"])

            # 1. Classification dataset: copy image to appropriate class folder
            cls_dest = cls_dir / split_name / r["category"] / r["filename"]
            shutil.copy2(r["img_path"], cls_dest)

            # 2. YOLO dataset: copy image and create bounding box label file
            yolo_img_dest = yolo_dir / "images" / split_name / r["filename"]
            shutil.copy2(r["img_path"], yolo_img_dest)

            yolo_lbl_dest = yolo_dir / "labels" / split_name / f"{base}.txt"
            bboxes = extract_bboxes_from_masks(r["mask_paths"], img_w=w, img_h=h)

            with open(yolo_lbl_dest, "w") as f:
                # Class 0: tumor (single class tumor localization detector)
                for bbox in bboxes:
                    total_bboxes += 1
                    f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

    # Generate data.yaml for YOLOv8
    abs_yolo_dir = yolo_dir.resolve()
    yolo_config = {
        "path": str(abs_yolo_dir).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 1,
        "names": ["tumor"]
    }
    yaml_path = yolo_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yolo_config, f, default_flow_style=False)

    print(f"\nYOLO data.yaml written to: {yaml_path}")
    print(f"Total Bounding Boxes extracted: {total_bboxes}")
    print(f"YOLO dataset ready at: {yolo_dir}")
    print(f"Classification dataset ready at: {cls_dir}")

    return {
        "yolo_yaml": str(yaml_path),
        "total_images": len(records),
        "total_bboxes": total_bboxes,
        "splits": {k: len(v) for k, v in splits.items()}
    }


if __name__ == "__main__":
    prepare_datasets()
