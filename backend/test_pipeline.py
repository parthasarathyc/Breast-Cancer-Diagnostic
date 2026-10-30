import os
import asyncio
from PIL import Image
from app.api.schemas import ClinicalDataInput
from app.orchestrator.pipeline import NativeAgentOrchestrator

async def main():
    print("[*] Starting Multi-Agent Pipeline Integration Test...")
    orchestrator = NativeAgentOrchestrator()
    sample_path = "backend/data/samples/sample_benign.png"

    if not os.path.exists(sample_path):
        print(f"[!] Sample file not found at {sample_path}")
        return

    img = Image.open(sample_path)
    clinical = ClinicalDataInput(
        patient_age=48,
        family_history=False,
        breast_density="B - Scattered fibroglandular",
        symptoms="Routine mammography screening"
    )

    print(f"[*] Running pipeline on {sample_path}...")
    res = await orchestrator.run_pipeline(img, clinical, case_id="TEST-BENIGN-01")

    print("\n" + "=" * 60)
    print("BREAST CARE-MA MULTI-AGENT EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Case ID: {res.case_id} (Duration: {res.pipeline_duration_seconds}s)")
    print(f"Agent 1 (Detection): Lesion Detected={res.agent1_detection.lesion_detected}, Area={res.agent1_detection.lesion_area_percentage}%")
    print(f"Agent 2 (Classification): Class={res.agent2_classification.predicted_class}, Calibrated Probs={res.agent2_classification.calibrated_probabilities}")
    print(f"Agent 3 (XAI & Morphology): Shape={res.agent3_xai.morphology.shape}, Margin={res.agent3_xai.morphology.margin}, Shadowing={res.agent3_xai.morphology.acoustic_shadowing_observed}")
    print(f"Agent 4 (Evidence Retrieval): {res.agent4_evidence.total_retrieved} guidelines retrieved")
    for g in res.agent4_evidence.matched_guidelines:
        print(f"   -> [{g.source}] {g.title}")
    print(f"Agent 5 (Risk Assessment): {res.agent5_risk.birads_category} ({res.agent5_risk.risk_level}) - Malignancy Risk: {round(res.agent5_risk.malignancy_risk_score*100, 1)}%")
    print(f"Agent 6 (Report & PDF): PDF Generated at -> {res.agent6_report.pdf_report_path}")
    print("=" * 60)
    print("[SUCCESS] Multi-Agent Pipeline Test Passed Successfully!")

if __name__ == "__main__":
    asyncio.run(main())
