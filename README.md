# Breast CARE-MA: Explainable Multi-Agent AI Framework for Multimodal Breast Cancer Diagnostic Decision Support

**Institution:** Department of Information Technology, Nehru Institute of Engineering and Technology (Autonomous), Coimbatore  
**Guide:** Dr. N.K. SAKTHIVEL, Head of Department, IT  
**Team Members:**  
- Partha Sarathy C (721423205042) - Architecture, Native Backend Orchestration, Integration
- Atchaya M (721423205006) - Detection and Classification Models
- Kalaiarasi C (721423205024) - Frontend and Explainability (XAI) UI
- Hariharasudhan S (721423205019) - Clinical Evidence RAG, Risk Assessment & Report Generation

---

## 1. Project Overview & Key Architectural Innovation

### Core Problem Addressed
Traditional Computer-Aided Diagnosis (CADx) systems are monolithic "black-boxes" that output raw, uncalibrated probabilities without clinical justification, without verifiable anatomical features, and without contextualizing patient risk factors.

### Our Solution
**Breast CARE-MA** deconstructs breast cancer screening into **6 specialized, collaborative AI agents**. 
Crucially, **n8n has been completely replaced with a native, high-performance Python FastAPI asynchronous orchestrator**. This provides:
1. **Zero HTTP Latency Overhead:** Agents run directly within the backend memory space.
2. **Direct In-Memory Tensor Passing:** Preprocessed PyTorch tensors and CV2 image matrices are passed directly across agents without Base64 network bottlenecks.
3. **GPU VRAM Safety:** PyTorch CUDA device context is managed centrally with automated memory cleanup.
4. **Reliable Clinical Decision Reports:** Automated generation of ACR BI-RADS-compliant diagnostic PDF reports with ReportLab.

---

## 2. The 6-Agent Architecture

| Agent | Responsibility | Core Technology | Output |
| :--- | :--- | :--- | :--- |
| **Agent 1: Breast Tumor Detection** | Identifies whether a focal lesion is present; computes bounding boxes and binary segmentation masks. | OpenCV adaptive morphological filtering & segmentation | Bounding boxes, lesion area %, contour polygon, mask overlay |
| **Agent 2: Tumor Classification** | Classifies scan into `normal`, `benign`, or `malignant` with temperature scaling calibration. | Pre-trained EfficientNet-B0 / ConvNeXt via `timm` + Platt Calibration | Calibrated probability distribution (e.g., Benign: 88%, Malignant: 10%, Normal: 2%) |
| **Agent 3: Feature Analysis & XAI** | Generates visual attention heatmaps and extracts quantitative BI-RADS morphological descriptors. | Grad-CAM++ & contour biometric analysis | Heatmap overlay, Shape (oval/irregular), Margin (circumscribed/spiculated), Aspect ratio, Acoustic shadowing |
| **Agent 4: Evidence Retrieval (RAG)** | Semantically queries the American College of Radiology (ACR) BI-RADS 5th Edition Atlas. | Structured Knowledge Base & Vector Semantic Matcher | Top-3 clinical guidelines, risk definitions, and clinical citations |
| **Agent 5: Risk Assessment** | Fuses AI probabilities with patient clinical factors (age, family history, breast density, acoustic features). | Multimodal Bayesian Decision Matrix | ACR BI-RADS category (BI-RADS 1 to 5), calibrated malignancy risk score |
| **Agent 6: Report Generation** | Synthesizes an executive diagnostic summary and generates an official clinical PDF report. | Clinical Narrative Generator + ReportLab Engine | Structured JSON narrative, clinical follow-up actions, and downloadable PDF |

---

## 3. Dataset: `SamIndia/BUSI` (Breast Ultrasound Images)

The system is configured to train and evaluate on the standard **BUSI** benchmark from Hugging Face (`https://huggingface.co/datasets/SamIndia/BUSI`):
- **Benign:** 437 ultrasound scans + ground truth masks
- **Malignant:** 210 ultrasound scans + ground truth masks
- **Normal:** 133 ultrasound scans
- **Data Splits:** 70% Train (546 records), 15% Validation (117 records), 15% Test (117 records) stratified by class.

---

## 4. Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.14 / 3.11)
- Node.js 18+ and npm

### Step 1: Install Dependencies
```bash
# Backend dependencies
pip install -r backend/requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

### Step 2: (Optional) Re-Extract or Re-Train Models on BUSI Dataset
All 780 BUSI images, masks, and splits have already been extracted into `backend/data/busi_processed/`. If you want to run the dataset extractor or retrain the classifier:
```bash
# Dataset extractor
python backend/data/extract_busi.py

# Model training script
python backend/training/train_classifier.py
```

### Step 3: Run the Integration Test
```bash
$env:PYTHONPATH="backend"
python backend/test_pipeline.py
```

### Step 4: Launch the Full Application

**Option A (Using one-click scripts):**
- Double click `run_backend.bat`
- Double click `run_frontend.bat`

**Option B (From terminal):**
```bash
# Terminal 1 - Start FastAPI Backend (Port 8000)
$env:PYTHONPATH="backend"
python backend/app/main.py

# Terminal 2 - Start React Frontend (Port 5173)
cd frontend
npm run dev
```

Open your browser at: **`http://localhost:5173`**
- Click on any of the **1-Click BUSI Samples** (`Fibroadenoma (Benign)`, `Carcinoma (Malignant)`, `Normal Scan (Clear)`) or upload a custom ultrasound image.
- Review the live agent execution steps.
- Inspect the Grad-CAM++ heatmap, morphological descriptors, BI-RADS score, and click **Download PDF** to export the official diagnostic report!

---

## 5. Directory Structure
```
Breast-Cancer/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes.py            # FastAPI REST endpoints
│   │   │   └── schemas.py           # Pydantic schema models
│   │   ├── agents/
│   │   │   ├── agent1_detection.py  # Lesion detection & segmentation
│   │   │   ├── agent2_classifier.py # Calibrated classifier
│   │   │   ├── agent3_xai.py        # Grad-CAM++ & morphology analysis
│   │   │   ├── agent4_evidence.py   # Clinical knowledge base RAG
│   │   │   ├── agent5_risk.py       # Multimodal BI-RADS scoring
│   │   │   └── agent6_report.py     # Summary & PDF report generation
│   │   ├── knowledge_base/
│   │   │   └── birads_kb.json       # ACR BI-RADS Atlas 5th Edition data
│   │   ├── orchestrator/
│   │   │   └── pipeline.py          # Native Async Multi-Agent Orchestrator
│   │   └── main.py                  # FastAPI Application Entrypoint
│   ├── data/
│   │   ├── busi_processed/          # Stratified BUSI dataset (train/val/test)
│   │   ├── samples/                 # Demo test images
│   │   └── extract_busi.py          # Hugging Face BUSI fetcher & split script
│   ├── models/                      # Saved PyTorch model checkpoints
│   ├── reports/                     # Generated diagnostic PDF reports
│   ├── training/
│   │   └── train_classifier.py      # Classifier training script
│   ├── test_pipeline.py             # Multi-Agent pipeline test suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  # Main Decision Support UI Dashboard
│   │   ├── main.jsx
│   │   └── index.css                # Tailwind CSS v4 styling
│   ├── package.json
│   └── vite.config.js               # Vite config with backend proxy
├── run_backend.bat
├── run_frontend.bat
└── README.md
```
