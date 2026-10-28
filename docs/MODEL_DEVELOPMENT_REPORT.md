# Breast CARE-MA: Agent 1 (Tumor Detection) & Agent 2 (Tumor Classification)
## Technical Architecture, Preprocessing Pipeline, and Model Development Report

**Project:** Breast CARE-MA – Explainable AI Breast Cancer Diagnostic Decision Support System  
**Team Member Responsibility:** Atchaya M (Detection & Classification AI Models)  
**Date:** October 2026  
**Status:** Implemented, Tested, & Integration-Ready  
**Medical Disclaimer:** *This system is an academic research prototype developed for educational and decision-support exploration. It is NOT certified as a medical device and must never substitute for professional clinical diagnosis.*

---

## 1. Project Overview & Existing Architecture Alignment

Based on the **Breast CARE-MA Project Plan and Architecture Document** and the **Overall Concept Report**, the project employs a 6-agent explainable AI architecture coordinated via FastAPI and workflow orchestration (n8n/CrewAI):

- **Agent 1 – Tumor Detection (YOLOv8):** Localizes tumor regions and lesion boundaries, returning normalized and pixel bounding boxes. Handles normal images without false alarms.
- **Agent 2 – Tumor Classification (ResNet18 / CNN):** Predicts class probabilities across `normal`, `benign`, and `malignant`.
- **Agent 3 – XAI / Grad-CAM (Kalaiarasi):** Hooks into the feature extraction layers of Agent 2's convolutional backbone to generate explainability heatmaps.
- **Agent 4 – Evidence Retrieval (Hariharasudhan):** Retrieves clinical reference evidence.
- **Agent 5 – Risk Assessment (Hariharasudhan):** Evaluates overall case risk.
- **Agent 6 – Report Generation (Hariharasudhan):** Synthesizes multi-agent outputs into a clinical summary.
- **Backend & Orchestrator (Partha):** FastAPI backend and multi-agent coordination.

---

## 2. Dataset Analysis & Preprocessing (BUSI Dataset)

### 2.1 Dataset Structure
The **Breast Ultrasound Images (BUSI)** dataset consists of 780 ultrasound images categorized into three classes:
- **Benign:** 437 images (454 mask files)
- **Malignant:** 210 images (211 mask files)
- **Normal:** 133 images (133 mask files)

### 2.2 Key Findings from Data Inspection
1. **Multi-Focal Lesions:** 16 benign cases and 1 malignant case contain multiple mask files (e.g., `benign (100)_mask.png` and `benign (100)_mask_1.png`), indicating multiple lesions in a single ultrasound scan.
2. **Normal Ultrasound Ground Truth:** All 133 normal masks contain zero positive pixels (all black). For YOLOv8 detection, these are correctly converted to empty label files (`.txt`), which serve as negative background samples to prevent false-positive detections on healthy tissue.
3. **Variable Dimensions:** Images range from 323x473 to 690x574 pixels in 3-channel RGB/grayscale.

### 2.3 Train / Validation / Test Stratified Split (Zero Leakage)
A stratified 70% / 15% / 15% train/val/test split was established with random seed `42` to guarantee zero data leakage:
- **Training Set (70%):** 545 images (`normal`: 93, `benign`: 305, `malignant`: 147)
- **Validation Set (15%):** 118 images (`normal`: 20, `benign`: 66, `malignant`: 32)
- **Test Set (15%):** 117 images (`normal`: 20, `benign`: 66, `malignant`: 31)

### 2.4 Annotation Conversion (Segmentation Mask -> YOLO Bounding Box)
- Binary masks were thresholded at 127.
- External contours were extracted via `cv2.findContours`.
- Contours smaller than 30 pixels were filtered to suppress noise artifacts.
- Bounding boxes `[x, y, w, h]` were converted to normalized YOLO format:
  `[class_id, x_center/W, y_center/H, box_width/W, box_height/H]`
- Extracted a total of **665 bounding boxes** across the dataset.

---

## 3. Agent 1: Tumor Detection Implementation

- **Architecture:** Ultralytics YOLOv8 (Nano baseline `yolov8n.pt` for high-speed inference, expandable to `yolov8s` / `yolov8m`).
- **Input:** Breast ultrasound image (path, bytes, PIL Image, or numpy array).
- **Output Schema:**
  ```json
  {
    "agent": "tumor_detection",
    "status": "success",
    "has_tumor": true,
    "tumors_detected": 1,
    "detections": [
      {
        "box_xyxy": [120, 85, 340, 310],
        "normalized_xywh": [0.45, 0.42, 0.40, 0.45],
        "confidence": 0.89,
        "label": "tumor",
        "area_pixels": 49500
      }
    ],
    "image_shape": [471, 562, 3],
    "inference_time_ms": 32.5,
    "model_version": "yolov8-tumor-detector-v1.0",
    "disclaimer": "Academic research prototype. Not certified for clinical diagnosis.",
    "message": "Detected 1 potential tumor region(s)."
  }
  ```
- **Normal Image Handling:** When no tumor is detected, returns `has_tumor: false`, `tumors_detected: 0`, `detections: []`, and descriptive message `"No tumor regions detected (compatible with normal ultrasound)."`.

---

## 4. Agent 2: Tumor Classification Implementation

- **Architecture:** PyTorch ResNet18 with ImageNet transfer learning and a custom 3-class classification head (`Linear(512, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 3)`).
- **Grad-CAM Hook Support:** The `extract_features` method exposes `layer4` convolutional feature maps for direct integration with Agent 3 (Explainability).
- **Class Imbalance Mitigation:** Weighted Cross-Entropy Loss with inverse class frequency weights:
  `weight_c = total_samples / (n_classes * count_c)`
- **Data Augmentation (Train only):** RandomHorizontalFlip, RandomRotation(15°), ColorJitter, and ImageNet normalization.
- **Output Schema:**
  ```json
  {
    "agent": "tumor_classification",
    "status": "success",
    "predicted_class": "benign",
    "confidence": 0.8521,
    "class_probabilities": {
      "normal": 0.0452,
      "benign": 0.8521,
      "malignant": 0.1027
    },
    "inference_time_ms": 28.4,
    "model_version": "resnet18-busi-v1.0",
    "disclaimer": "Academic research prototype. Not certified for clinical diagnosis.",
    "message": "Image classified as 'benign' with 85.2% confidence."
  }
  ```

---

## 5. Model Evaluation Results

### 5.1 Agent 2 – ResNet18 Classification Results (Held-Out Test Set: 117 Images)
Evaluated strictly on the held-out test split (zero data leakage):

- **Overall Accuracy:** 82.91%
- **Macro Average F1-Score:** 0.8194
- **Weighted Average F1-Score:** 0.8300

#### Per-Class Performance Breakdown:
| Class | Precision | Recall | F1-Score | Support (Cases) |
|---|---|---|---|---|
| **Benign** | 94.34% | 75.76% | 0.8403 | 66 |
| **Malignant** | 75.00% | 96.77% | 0.8451 | 31 |
| **Normal** | 70.83% | 85.00% | 0.7727 | 20 |
| **Macro Average** | **80.06%** | **85.84%** | **0.8194** | **117** |

#### Confusion Matrix:
| | Pred: Benign | Pred: Malignant | Pred: Normal |
|---|---|---|---|
| **Actual Benign (66)** | 50 | 9 | 7 |
| **Actual Malignant (31)** | 1 | 30 | 0 |
| **Actual Normal (20)** | 2 | 1 | 17 |

*Clinical Note:* The model demonstrates **96.77% sensitivity (recall) for malignant lesions** (30 of 31 malignant ultrasound scans identified), which is critical in cancer screening to minimize false negatives.

### 5.2 Agent 1 – YOLOv8 Tumor Localization Pipeline
- **Dataset Conversion:** Successfully extracted 665 ground-truth bounding boxes from segmentation masks.
- **Normal Samples:** 133 normal scans mapped to empty label files for negative background training.
- **Inference Latency:** Average 30–45 ms on CPU with YOLOv8n.
- **Status:** Full training pipeline (`agent_1_detection/train.py`), evaluation script (`agent_1_detection/evaluate.py`), and inference engine tested and operational. Ready for high-epoch training on GPU.

---

## 6. API Endpoints & Orchestration Integration

The service is implemented in FastAPI (`api/app.py`) with cached singleton model loading on application startup:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check and agent status |
| `POST` | `/api/v1/agents/detection` | Agent 1: Localizes tumor bounding boxes |
| `POST` | `/api/v1/agents/classification` | Agent 2: Predicts Normal / Benign / Malignant |
| `POST` | `/api/v1/pipeline/analyze` | Multi-Agent Pipeline: Combines Agent 1 & Agent 2 for n8n/orchestrator |

---

## 6. How to Run & Verify

### Install Dependencies
```powershell
pip install -r requirements.txt
```

### Run Automated Unit & Integration Tests
```powershell
python -m pytest -v
```

### Launch FastAPI Diagnostic Server
```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI documentation is available at `http://localhost:8000/docs`.

### Run Model Training Pipelines
- **Agent 1 (YOLOv8 Detection):**
  ```powershell
  python -m agent_1_detection.train --epochs 20 --batch 8 --imgsz 640
  ```
- **Agent 2 (ResNet18 Classification):**
  ```powershell
  python -m agent_2_classification.train --epochs 15 --batch 16 --lr 1e-4
  ```

### Run Held-Out Test Split Evaluations
- **Agent 1 Detection Evaluation:**
  ```powershell
  python -m agent_1_detection.evaluate
  ```
- **Agent 2 Classification Evaluation:**
  ```powershell
  python -m agent_2_classification.evaluate
  ```
