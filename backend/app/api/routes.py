import os
import json
import uuid
import io
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from app.api.schemas import ClinicalDataInput, FullAnalysisResponse
from app.orchestrator.pipeline import NativeAgentOrchestrator

router = APIRouter(prefix="/api/v1")
orchestrator = NativeAgentOrchestrator()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Breast CARE-MA Multi-Agent Decision Support API",
        "orchestrator": "Native FastAPI Asynchronous Engine",
        "agents": [
            "Agent 1: Breast Tumor Detection (Morphological Segmentation)",
            "Agent 2: Tumor Classification (Calibrated Transfer Learning)",
            "Agent 3: Image Feature & XAI (Grad-CAM++ & BI-RADS Descriptors)",
            "Agent 4: Clinical Evidence Retrieval (ACR Atlas Knowledge Base)",
            "Agent 5: Risk Assessment (Multimodal Bayesian BI-RADS Scoring)",
            "Agent 6: Explainable Report Generation (Clinical Summary & PDF)"
        ]
    }

@router.post("/analyze", response_model=FullAnalysisResponse)
async def analyze_ultrasound(
    file: UploadFile = File(...),
    patient_age: int = Form(45),
    family_history: bool = Form(False),
    breast_density: str = Form("B - Scattered fibroglandular"),
    symptoms: str = Form("Routine screening / asymptomatic"),
    menopause_status: str = Form("premenopausal")
):
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    case_id = f"BC-{uuid.uuid4().hex[:8].upper()}"
    clinical_data = ClinicalDataInput(
        patient_age=patient_age,
        family_history=family_history,
        breast_density=breast_density,
        symptoms=symptoms,
        menopause_status=menopause_status
    )

    result = await orchestrator.run_pipeline(pil_image, clinical_data, case_id=case_id)
    return result

@router.post("/analyze/sample/{sample_type}", response_model=FullAnalysisResponse)
async def analyze_sample(
    sample_type: str,
    patient_age: int = 52,
    family_history: bool = False,
    breast_density: str = "C - Heterogeneously dense"
):
    """
    Run the 6-agent pipeline on pre-extracted BUSI samples for fast testing/demonstration.
    sample_type: 'benign', 'malignant', or 'normal'
    """
    sample_path = os.path.join("backend/data/samples", f"sample_{sample_type.lower()}.png")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail=f"Sample '{sample_type}' not found at {sample_path}")

    pil_image = Image.open(sample_path).convert("RGB")
    case_id = f"BUSI-{sample_type.upper()}-{uuid.uuid4().hex[:4].upper()}"

    clinical_data = ClinicalDataInput(
        patient_age=patient_age,
        family_history=family_history or (sample_type.lower() == "malignant"),
        breast_density=breast_density,
        symptoms="Palpable nodule" if sample_type.lower() != "normal" else "Routine screening"
    )

    result = await orchestrator.run_pipeline(pil_image, clinical_data, case_id=case_id)
    return result

@router.get("/reports/{case_id}/pdf")
async def download_pdf_report(case_id: str):
    pdf_filename = f"Report_{case_id}.pdf"
    pdf_path = os.path.join("backend/reports", pdf_filename)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF report not found for this case.")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"BreastCARE-MA_Report_{case_id}.pdf"
    )
