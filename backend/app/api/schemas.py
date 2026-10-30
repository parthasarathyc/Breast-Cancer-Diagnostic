from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ClinicalDataInput(BaseModel):
    patient_age: int = Field(default=45, ge=18, le=100, description="Patient age in years")
    family_history: bool = Field(default=False, description="First-degree family history of breast cancer")
    breast_density: str = Field(default="B - Scattered fibroglandular", description="ACR Breast Density category (A, B, C, D)")
    symptoms: Optional[str] = Field(default="Routine screening / asymptomatic", description="Presenting symptoms")
    menopause_status: Optional[str] = Field(default="premenopausal", description="Menopausal state")

class Agent1DetectionResult(BaseModel):
    lesion_detected: bool
    bounding_box: Optional[List[int]] = None  # [xmin, ymin, xmax, ymax]
    contour_points: Optional[int] = None
    lesion_area_px: int = 0
    lesion_area_percentage: float = 0.0
    mask_base64: Optional[str] = None
    overlay_detection_base64: Optional[str] = None
    detection_confidence: float = 0.0
    status: str = "completed"

class Agent2ClassificationResult(BaseModel):
    predicted_class: str  # "normal", "benign", "malignant"
    raw_confidence: float
    calibrated_probabilities: Dict[str, float]  # {"normal": 0.02, "benign": 0.88, "malignant": 0.10}
    temperature_scaled: bool = True
    model_name: str = "ConvNeXt-Tiny + TempScaling"
    status: str = "completed"

class MorphologyMetrics(BaseModel):
    shape: str  # "oval", "round", "irregular"
    margin: str  # "circumscribed", "microlobulated", "indistinct", "spiculated"
    aspect_ratio: float
    solidity: float
    acoustic_shadowing_observed: bool
    orientation: str  # "parallel", "non-parallel"

class Agent3XAIResult(BaseModel):
    gradcam_heatmap_base64: Optional[str] = None
    gradcam_overlay_base64: Optional[str] = None
    morphology: MorphologyMetrics
    xai_technique: str = "Grad-CAM++"
    target_layer: str = "stages.3.blocks.2"
    status: str = "completed"

class EvidenceCitation(BaseModel):
    title: str
    source: str
    guideline: str
    relevance_note: str
    match_score: float

class Agent4EvidenceResult(BaseModel):
    matched_guidelines: List[EvidenceCitation]
    total_retrieved: int
    status: str = "completed"

class Agent5RiskResult(BaseModel):
    birads_category: str  # "BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4A", "BI-RADS 4B", "BI-RADS 4C", "BI-RADS 5"
    malignancy_risk_score: float  # 0.0 to 1.0
    risk_level: str  # "Low", "Intermediate / Suspicious", "High"
    fusion_factors: List[str]
    status: str = "completed"

class Agent6ReportResult(BaseModel):
    case_id: str
    timestamp: str
    executive_summary: str
    detailed_findings: str
    birads_clinical_impression: str
    recommended_action: List[str]
    disclaimer: str
    pdf_report_path: Optional[str] = None
    pdf_base64: Optional[str] = None
    status: str = "completed"

class FullAnalysisResponse(BaseModel):
    case_id: str
    timestamp: str
    clinical_inputs: ClinicalDataInput
    agent1_detection: Agent1DetectionResult
    agent2_classification: Agent2ClassificationResult
    agent3_xai: Agent3XAIResult
    agent4_evidence: Agent4EvidenceResult
    agent5_risk: Agent5RiskResult
    agent6_report: Agent6ReportResult
    pipeline_duration_seconds: float
    status: str = "completed"
