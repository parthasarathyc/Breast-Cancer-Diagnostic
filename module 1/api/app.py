"""
Breast CARE-MA Backend API
FastAPI service exposing modular endpoints for Agent 1 (Tumor Detection)
and Agent 2 (Tumor Classification), plus integrated multi-agent pipeline.
"""

import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_1_detection.schemas import TumorDetectionResponse
from agent_2_classification.schemas import TumorClassificationResponse
from .dependencies import ModelRegistry


# Maximum allowed upload size (25 MB)
MAX_FILE_SIZE = 25 * 1024 * 1024
ALLOWED_MIME_TYPES = ["image/png", "image/jpeg", "image/jpg", "image/bmp", "image/webp"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on server startup to avoid per-request latency."""
    ModelRegistry.initialize()
    yield


app = FastAPI(
    title="Breast CARE-MA AI Diagnostic Service",
    description="Multi-Agent Explainable AI Decision Support System: Agent 1 (Detection) & Agent 2 (Classification)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration (React + Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def validate_image_file(file: UploadFile) -> bytes:
    """Validates that uploaded file is a supported image and within size limits."""
    if file.content_type and file.content_type.lower() not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes)."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    return content


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Breast CARE-MA Diagnostic API"
    version: str = "1.0.0"
    agents_ready: list = ["agent_1_detection", "agent_2_classification"]
    disclaimer: str = "Academic research prototype. Not certified for clinical diagnosis."


class PipelineAnalysisResponse(BaseModel):
    analysis_id: str
    timestamp: str
    status: str
    agent_1_detection: TumorDetectionResponse
    agent_2_classification: TumorClassificationResponse
    summary: dict
    disclaimer: str = "Academic research prototype. Not certified for clinical diagnosis."


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """System health check and agent status."""
    return HealthResponse()


@app.post(
    "/api/v1/agents/detection",
    response_model=TumorDetectionResponse,
    tags=["Agent 1: Tumor Detection"]
)
async def detect_tumors(
    file: UploadFile = File(..., description="Breast ultrasound image (PNG/JPEG)"),
    conf_threshold: float = 0.25,
    detector=Depends(ModelRegistry.get_detection_agent)
):
    """
    Agent 1: Detects and localizes tumor bounding boxes in breast ultrasound images.
    Returns normalized coordinates, pixel coordinates, and confidence scores.
    """
    image_bytes = await validate_image_file(file)
    response = detector.predict(image_bytes, conf_threshold=conf_threshold)
    if response.status == "error":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=response.message)
    return response


@app.post(
    "/api/v1/agents/classification",
    response_model=TumorClassificationResponse,
    tags=["Agent 2: Tumor Classification"]
)
async def classify_tumor(
    file: UploadFile = File(..., description="Breast ultrasound image (PNG/JPEG)"),
    classifier=Depends(ModelRegistry.get_classification_agent)
):
    """
    Agent 2: Classifies ultrasound images into 'normal', 'benign', or 'malignant'.
    Returns predicted class, top confidence, and full probability distribution.
    """
    image_bytes = await validate_image_file(file)
    response = classifier.predict(image_bytes)
    if response.status == "error":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=response.message)
    return response


@app.post(
    "/api/v1/pipeline/analyze",
    response_model=PipelineAnalysisResponse,
    tags=["Integrated Pipeline"]
)
async def run_pipeline_analysis(
    file: UploadFile = File(..., description="Breast ultrasound image (PNG/JPEG)"),
    detector=Depends(ModelRegistry.get_detection_agent),
    classifier=Depends(ModelRegistry.get_classification_agent)
):
    """
    Multi-Agent Integrated Pipeline:
    Coordinates Agent 1 (Detection) and Agent 2 (Classification) in a unified workflow.
    Designed for consumption by n8n orchestrator and React frontend.
    """
    image_bytes = await validate_image_file(file)
    analysis_id = f"busi-{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    # Step 1: Agent 1 Tumor Detection
    det_res = detector.predict(image_bytes)

    # Step 2: Agent 2 Tumor Classification
    cls_res = classifier.predict(image_bytes)

    # Combined summary
    summary = {
        "analysis_id": analysis_id,
        "has_tumor": det_res.has_tumor,
        "tumors_detected": det_res.tumors_detected,
        "predicted_class": cls_res.predicted_class,
        "classification_confidence": cls_res.confidence,
        "requires_clinician_review": True
    }

    return PipelineAnalysisResponse(
        analysis_id=analysis_id,
        timestamp=now_iso,
        status="success",
        agent_1_detection=det_res,
        agent_2_classification=cls_res,
        summary=summary
    )
