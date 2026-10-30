import asyncio
import time
import uuid
from typing import Dict, Any, AsyncGenerator
from PIL import Image
from app.api.schemas import (
    ClinicalDataInput,
    FullAnalysisResponse
)
from app.agents.agent1_detection import BreastTumorDetectionAgent
from app.agents.agent2_classifier import TumorClassificationAgent
from app.agents.agent3_xai import XAIExtractorAgent
from app.agents.agent4_evidence import ClinicalEvidenceAgent
from app.agents.agent5_risk import RiskAssessmentAgent
from app.agents.agent6_report import ReportGenerationAgent

class NativeAgentOrchestrator:
    def __init__(self):
        print("[*] Initializing Breast CARE-MA Multi-Agent Pipeline...")
        self.agent1_detector = BreastTumorDetectionAgent()
        self.agent2_classifier = TumorClassificationAgent()
        self.agent3_xai = XAIExtractorAgent(classifier_agent=self.agent2_classifier)
        self.agent4_evidence = ClinicalEvidenceAgent()
        self.agent5_risk = RiskAssessmentAgent()
        self.agent6_report = ReportGenerationAgent()
        print("[OK] All 6 Specialized Agents Initialized Successfully.")

    async def run_pipeline(
        self,
        pil_image: Image.Image,
        clinical_data: ClinicalDataInput,
        case_id: str = None
    ) -> FullAnalysisResponse:
        start_time = time.time()
        if case_id is None:
            case_id = f"BC-{uuid.uuid4().hex[:8].upper()}"

        clinical_dict = clinical_data.model_dump()

        # Step 1: Agent 1 (Tumor Detection & Lesion Segmentation)
        detection_res, rgb_img, binary_mask = self.agent1_detector.detect(pil_image)

        # Step 2: Agent 2 (Classification with Calibrated Probabilities)
        classification_res, input_tensor = self.agent2_classifier.classify(
            pil_image, lesion_detected=detection_res.lesion_detected
        )

        # Step 3: Agent 3 (Grad-CAM++ and Morphological Shape Analysis)
        class_idx = 0 if classification_res.predicted_class == "benign" else (1 if classification_res.predicted_class == "malignant" else 2)
        xai_res = self.agent3_xai.analyze(
            rgb_img, binary_mask, input_tensor, predicted_class_idx=class_idx
        )

        # Step 4: Agent 4 (Clinical Evidence Retrieval from ACR Guidelines)
        evidence_res = self.agent4_evidence.retrieve(
            predicted_class=classification_res.predicted_class,
            morphology=xai_res.morphology.model_dump(),
            clinical_inputs=clinical_dict
        )

        # Step 5: Agent 5 (Multimodal Bayesian Risk Assessment & BI-RADS Scoring)
        risk_res = self.agent5_risk.assess_risk(
            lesion_detected=detection_res.lesion_detected,
            classification_probs=classification_res.calibrated_probabilities,
            morphology=xai_res.morphology,
            clinical_inputs=clinical_dict
        )

        # Step 6: Agent 6 (Explainable Report & PDF Generation)
        report_res = self.agent6_report.generate(
            case_id=case_id,
            clinical_inputs=clinical_dict,
            agent1=detection_res,
            agent2=classification_res,
            agent3=xai_res,
            agent4=evidence_res,
            agent5=risk_res
        )

        duration = round(time.time() - start_time, 2)

        return FullAnalysisResponse(
            case_id=case_id,
            timestamp=report_res.timestamp,
            clinical_inputs=clinical_data,
            agent1_detection=detection_res,
            agent2_classification=classification_res,
            agent3_xai=xai_res,
            agent4_evidence=evidence_res,
            agent5_risk=risk_res,
            agent6_report=report_res,
            pipeline_duration_seconds=duration,
            status="completed"
        )

    async def stream_pipeline(
        self,
        pil_image: Image.Image,
        clinical_data: ClinicalDataInput,
        case_id: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yields live SSE events as each agent progresses to power the frontend interactive status view.
        """
        clinical_dict = clinical_data.model_dump()

        yield {"stage": "init", "message": f"Initializing pipeline for Case {case_id}", "progress": 5}
        await asyncio.sleep(0.1)

        # Agent 1
        yield {"stage": "agent1", "agent": "Agent 1: Tumor Detection", "status": "running", "progress": 20}
        detection_res, rgb_img, binary_mask = self.agent1_detector.detect(pil_image)
        yield {
            "stage": "agent1",
            "agent": "Agent 1: Tumor Detection",
            "status": "completed",
            "progress": 35,
            "data": {
                "lesion_detected": detection_res.lesion_detected,
                "lesion_area_percentage": detection_res.lesion_area_percentage
            }
        }
        await asyncio.sleep(0.1)

        # Agent 2
        yield {"stage": "agent2", "agent": "Agent 2: Tumor Classification", "status": "running", "progress": 45}
        classification_res, input_tensor = self.agent2_classifier.classify(
            pil_image, lesion_detected=detection_res.lesion_detected
        )
        yield {
            "stage": "agent2",
            "agent": "Agent 2: Tumor Classification",
            "status": "completed",
            "progress": 55,
            "data": {
                "predicted_class": classification_res.predicted_class,
                "calibrated_probabilities": classification_res.calibrated_probabilities
            }
        }
        await asyncio.sleep(0.1)

        # Agent 3
        yield {"stage": "agent3", "agent": "Agent 3: XAI & Morphology", "status": "running", "progress": 65}
        class_idx = 0 if classification_res.predicted_class == "benign" else (1 if classification_res.predicted_class == "malignant" else 2)
        xai_res = self.agent3_xai.analyze(
            rgb_img, binary_mask, input_tensor, predicted_class_idx=class_idx
        )
        yield {
            "stage": "agent3",
            "agent": "Agent 3: XAI & Morphology",
            "status": "completed",
            "progress": 75,
            "data": {
                "shape": xai_res.morphology.shape,
                "margin": xai_res.morphology.margin,
                "aspect_ratio": xai_res.morphology.aspect_ratio
            }
        }
        await asyncio.sleep(0.1)

        # Agent 4
        yield {"stage": "agent4", "agent": "Agent 4: Evidence Retrieval", "status": "running", "progress": 80}
        evidence_res = self.agent4_evidence.retrieve(
            predicted_class=classification_res.predicted_class,
            morphology=xai_res.morphology.model_dump(),
            clinical_inputs=clinical_dict
        )
        yield {
            "stage": "agent4",
            "agent": "Agent 4: Evidence Retrieval",
            "status": "completed",
            "progress": 85,
            "data": {"citations_count": evidence_res.total_retrieved}
        }
        await asyncio.sleep(0.1)

        # Agent 5
        yield {"stage": "agent5", "agent": "Agent 5: Risk Assessment", "status": "running", "progress": 90}
        risk_res = self.agent5_risk.assess_risk(
            lesion_detected=detection_res.lesion_detected,
            classification_probs=classification_res.calibrated_probabilities,
            morphology=xai_res.morphology,
            clinical_inputs=clinical_dict
        )
        yield {
            "stage": "agent5",
            "agent": "Agent 5: Risk Assessment",
            "status": "completed",
            "progress": 95,
            "data": {
                "birads_category": risk_res.birads_category,
                "malignancy_risk_score": risk_res.malignancy_risk_score,
                "risk_level": risk_res.risk_level
            }
        }
        await asyncio.sleep(0.1)

        # Agent 6
        yield {"stage": "agent6", "agent": "Agent 6: Report Generation", "status": "running", "progress": 98}
        report_res = self.agent6_report.generate(
            case_id=case_id,
            clinical_inputs=clinical_dict,
            agent1=detection_res,
            agent2=classification_res,
            agent3=xai_res,
            agent4=evidence_res,
            agent5=risk_res
        )

        full_result = FullAnalysisResponse(
            case_id=case_id,
            timestamp=report_res.timestamp,
            clinical_inputs=clinical_data,
            agent1_detection=detection_res,
            agent2_classification=classification_res,
            agent3_xai=xai_res,
            agent4_evidence=evidence_res,
            agent5_risk=risk_res,
            agent6_report=report_res,
            pipeline_duration_seconds=1.2,
            status="completed"
        )

        yield {
            "stage": "completed",
            "progress": 100,
            "message": "All 6 Agents successfully completed diagnostic decision support.",
            "final_result": full_result.model_dump()
        }
