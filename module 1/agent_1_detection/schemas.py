"""
Schemas and Data Contracts for Agent 1 – Tumor Detection
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Detected tumor bounding box details."""
    box_xyxy: List[int] = Field(..., description="[x_min, y_min, x_max, y_max] in pixel coordinates")
    normalized_xywh: List[float] = Field(..., description="[x_center, y_center, width, height] normalized (0.0 to 1.0)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    label: str = Field(default="tumor", description="Detected class label")
    area_pixels: int = Field(..., description="Bounding box area in pixels")


class TumorDetectionResponse(BaseModel):
    """Standardized response schema for Agent 1."""
    agent: str = Field(default="tumor_detection", description="Agent identifier")
    status: str = Field(..., description="'success' or 'error'")
    has_tumor: bool = Field(..., description="True if one or more tumor regions detected")
    tumors_detected: int = Field(..., description="Number of tumor regions localized")
    detections: List[BoundingBox] = Field(default_factory=list, description="List of localized tumor boxes")
    image_shape: Optional[List[int]] = Field(None, description="[Height, Width, Channels] of analyzed image")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    model_version: str = Field(default="yolov8n-busi-v1.0", description="Model architecture/version")
    disclaimer: str = Field(
        default="Academic research prototype. Not certified for clinical diagnosis.",
        description="Medical disclaimer"
    )
    message: Optional[str] = Field(None, description="Descriptive status or error message")
