# Breast CARE-MA: Breast Cancer Diagnostic Decision Support System

An explainable multi-agent artificial intelligence decision support system for breast ultrasound diagnostics.

> [!IMPORTANT]
> **Academic Prototype Notice:** This system is an academic research prototype developed as part of a final-year B.Tech IT project. It is strictly for academic and educational evaluation and is not certified for clinical diagnostic use.

---

## Assigned Responsibilities (Atchaya M)
- **Agent 1 – Tumor Detection:** YOLOv8-based tumor localization, lesion bounding boxes, normalized coordinates, and handling of normal cases.
- **Agent 2 – Tumor Classification:** PyTorch ResNet18 transfer learning model classifying ultrasound into `normal`, `benign`, and `malignant`, with class probabilities and explainability hooks.

---

## Project Structure
```
Breast-Cancer-Diagnostic/
├── agent_1_detection/               # AGENT 1: TUMOR DETECTION
│   ├── __init__.py
│   ├── detector.py                  # Modular TumorDetectionAgent
│   ├── schemas.py                   # Pydantic data schemas
│   ├── train.py                     # YOLOv8 training pipeline
│   └── evaluate.py                  # Precision, Recall, mAP evaluation
│
├── agent_2_classification/          # AGENT 2: TUMOR CLASSIFICATION
│   ├── __init__.py
│   ├── classifier.py                # Modular TumorClassificationAgent
│   ├── model.py                     # ResNet18 architecture + Grad-CAM hooks
│   ├── schemas.py                   # Pydantic data schemas
│   ├── train.py                     # Training with class-weight balancing
│   └── evaluate.py                  # Accuracy, F1, and Confusion Matrix
│
├── api/                             # FASTAPI BACKEND SERVICE
│   ├── __init__.py
│   ├── app.py                       # Modular REST endpoints
│   └── dependencies.py              # Singleton model registry & cache
│
├── data/                            # DATASET PIPELINE
│   └── prepare_data.py              # Mask-to-bbox conversion & stratified splits
│
├── docs/                            # ARCHITECTURE & SPECIFICATIONS
│   ├── Breast_CARE_MA_Project_Plan_Architecture.pdf
│   └── MODEL_DEVELOPMENT_REPORT.md
│
├── tests/                           # AUTOMATED TESTS
│   ├── test_agent_1_detection.py
│   ├── test_agent_2_classification.py
│   └── test_api.py
│
├── requirements.txt                 # Project dependencies
└── README.md
```

---

## Quickstart

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Prepare Dataset Splits
Generates stratified 70% Train, 15% Val, 15% Test splits and converts BUSI segmentation masks into YOLO bounding boxes:
```powershell
python data/prepare_data.py
```

### 3. Run Automated Tests
```powershell
python -m pytest -v
```

### 4. Launch FastAPI Diagnostic Backend
```powershell
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation will be available at: `http://localhost:8000/docs`

---

## API Endpoints

- `GET /health` – System status and agent readiness
- `POST /api/v1/agents/detection` – Agent 1 Tumor Detection
- `POST /api/v1/agents/classification` – Agent 2 Tumor Classification
- `POST /api/v1/pipeline/analyze` – End-to-end multi-agent diagnostic analysis

---

## Git Workflow
- Active development branch: `achu`
- Integration branch: `develop`
- Production/Stable: `main`
