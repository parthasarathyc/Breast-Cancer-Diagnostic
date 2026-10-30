"""
Schemas and Data Contracts for Agent 2 – Tumor Classification
"""

from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class ClassProbabilities(BaseModel):
    """Normalized probability distribution across all three diagnostic classes."""
    normal: float = Field(..., ge=0.0, le=1.0, description="Probability of normal tissue")
    benign: float = Field(..., ge=0.0, le=1.0, description="Probability of benign lesion")
    malignant: float = Field(..., ge=0.0, le=1.0, description="Probability of malignant lesion")


class TumorClassificationResponse(BaseModel):
    """Standardized response schema for Agent 2 matching project orchestrator contracts."""
    agent: str = Field(default="tumor_classification", description="Agent identifier")
    status: str = Field(..., description="'success' or 'error'")
    predicted_class: str = Field(..., description="'normal', 'benign', or 'malignant'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of top predicted class")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for each class")
    model_version: str = Field(default="resnet18-busi-v1.0", description="Model architecture/version")
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    disclaimer: str = Field(
        default="Academic research prototype. Not certified for clinical diagnosis.",
        description="Medical disclaimer distinguishing prediction from clinical diagnosis"
    )
    message: Optional[str] = Field(None, description="Descriptive status or error message")
