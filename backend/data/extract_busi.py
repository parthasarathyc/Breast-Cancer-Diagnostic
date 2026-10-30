import os
import io
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

def extract_busi_dataset(output_dir="backend/data/busi_processed"):
    print("[*] Fetching SamIndia/BUSI parquet from Hugging Face cache...")
    parquet_path = hf_hub_download(
        repo_id="SamIndia/BUSI",
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset"
    )
    
    print(f"[*] Reading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"[*] Dataset shape: {df.shape}")
    print(f"[*] Class distribution:\n{df['class_label'].value_counts()}")
    
    # Create directories
    for split in ["train", "val", "test"]:
        for category in ["benign", "malignant", "normal"]:
            os.makedirs(os.path.join(output_dir, split, "images", category), exist_ok=True)
            os.makedirs(os.path.join(output_dir, split, "masks", category), exist_ok=True)
            
    os.makedirs(os.path.join("backend/data", "samples"), exist_ok=True)

    # Train (70%), Val (15%), Test (15%) stratified split
    train_df, test_val_df = train_test_split(
        df, test_size=0.30, stratify=df["class_label"], random_state=42
    )
    val_df, test_df = train_test_split(
        test_val_df, test_size=0.50, stratify=test_val_df["class_label"], random_state=42
    )

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    manifest_rows = []
    sample_saved = {"benign": False, "malignant": False, "normal": False}

    for split_name, split_data in splits.items():
        print(f"[*] Processing {split_name} split ({len(split_data)} records)...")
        for idx, row in split_data.iterrows():
            label = row["class_label"]
            img_id = str(row.get("image_id", idx))
            file_stem = f"{label}_{img_id}"

            # Save image
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_filename = f"{file_stem}.png"
            img_path = os.path.join(output_dir, split_name, "images", label, img_filename)
            img.save(img_path)

            # Save mask if available
            has_mask = False
            mask_filename = None
            if row["mask"] is not None and "bytes" in row["mask"] and row["mask"]["bytes"] is not None:
                mask_bytes = row["mask"]["bytes"]
                mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
                mask_filename = f"{file_stem}_mask.png"
                mask_path = os.path.join(output_dir, split_name, "masks", label, mask_filename)
                mask.save(mask_path)
                has_mask = True

            # Save representative samples for quick frontend/testing demo
            if not sample_saved[label]:
                sample_img_path = os.path.join("backend/data/samples", f"sample_{label}.png")
                img.save(sample_img_path)
                sample_saved[label] = True

            manifest_rows.append({
                "image_id": img_id,
                "split": split_name,
                "class_label": label,
                "image_path": img_path,
                "mask_path": mask_path if has_mask else None,
                "has_mask": has_mask
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_csv = os.path.join(output_dir, "metadata_manifest.csv")
    manifest_df.to_csv(manifest_csv, index=False)
    print(f"[OK] BUSI dataset successfully extracted and split. Saved manifest to {manifest_csv}")

if __name__ == "__main__":
    extract_busi_dataset()
