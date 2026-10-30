from typing import Dict, Any, List
from app.api.schemas import Agent5RiskResult, MorphologyMetrics

class RiskAssessmentAgent:
    def __init__(self):
        pass

    def assess_risk(
        self,
        lesion_detected: bool,
        classification_probs: Dict[str, float],
        morphology: MorphologyMetrics,
        clinical_inputs: Dict[str, Any]
    ) -> Agent5RiskResult:
        """
        Multimodal Bayesian & BI-RADS risk assessment combining calibrated probabilities,
        morphological descriptors, and clinical risk factors.
        """
        fusion_factors = []
        malignant_prob = classification_probs.get("malignant", 0.0)
        benign_prob = classification_probs.get("benign", 0.0)
        normal_prob = classification_probs.get("normal", 0.0)

        # 1. Base AI Probability
        fusion_factors.append(f"AI Classifier calibrated malignant probability: {round(malignant_prob * 100, 1)}%")

        # 2. Case: Normal / No lesion detected
        if not lesion_detected or normal_prob > 0.85:
            fusion_factors.append("No focal ultrasonic mass or architectural distortion detected.")
            return Agent5RiskResult(
                birads_category="BI-RADS 1",
                malignancy_risk_score=round(max(malignant_prob, 0.001), 3),
                risk_level="Negative / Routine Screening",
                fusion_factors=fusion_factors,
                status="completed"
            )

        # 3. Clinical Adjustments
        clinical_weight = 0.0
        age = clinical_inputs.get("patient_age", 45)
        if age >= 50:
            clinical_weight += 0.03
            fusion_factors.append(f"Age {age} (postmenopausal demographic slight risk elevation)")

        if clinical_inputs.get("family_history", False):
            clinical_weight += 0.08
            fusion_factors.append("Confirmed 1st-degree family history of breast malignancy (+8% risk factor)")

        density = str(clinical_inputs.get("breast_density", "B"))
        if "C" in density or "D" in density:
            clinical_weight += 0.04
            fusion_factors.append(f"Elevated breast density ({density}): increases masking and independent relative risk")

        # 4. Morphological Ultrasound Indicators (ACR BI-RADS descriptors)
        morph_weight = 0.0
        if morphology.shape == "irregular":
            morph_weight += 0.12
            fusion_factors.append("Ultrasound Morphology: Irregular shape strongly correlates with invasive architecture")
        elif morphology.shape == "oval":
            morph_weight -= 0.05
            fusion_factors.append("Ultrasound Morphology: Oval shape favors benign etiology (fibroadenoma/cyst)")

        if morphology.margin in ["spiculated", "indistinct"]:
            morph_weight += 0.15
            fusion_factors.append(f"Ultrasound Morphology: {morphology.margin.capitalize()} margin indicates tissue infiltration")
        elif morphology.margin == "circumscribed":
            morph_weight -= 0.05
            fusion_factors.append("Ultrasound Morphology: Well-circumscribed margin favors benign process")

        if morphology.orientation == "non-parallel":
            morph_weight += 0.08
            fusion_factors.append("Ultrasound Morphology: Non-parallel / taller-than-wide lesion growth")

        if morphology.acoustic_shadowing_observed:
            morph_weight += 0.10
            fusion_factors.append("Acoustic Feature: Posterior acoustic shadowing observed")

        # Fused Calibrated Score
        combined_score = min(max(malignant_prob + clinical_weight + morph_weight, 0.001), 0.999)
        combined_score = round(combined_score, 3)

        # 5. ACR BI-RADS Category Assignment
        if combined_score < 0.02 and morphology.margin == "circumscribed" and morphology.shape in ["oval", "round"]:
            birads = "BI-RADS 2"
            level = "Benign"
        elif combined_score <= 0.05:
            birads = "BI-RADS 3"
            level = "Probably Benign (<2-5% Risk)"
        elif 0.05 < combined_score <= 0.15:
            birads = "BI-RADS 4A"
            level = "Low Suspicion (2% - 15%)"
        elif 0.15 < combined_score <= 0.50:
            birads = "BI-RADS 4B"
            level = "Moderate Suspicion (15% - 50%)"
        elif 0.50 < combined_score < 0.85:
            birads = "BI-RADS 4C"
            level = "High Suspicion (50% - 85%)"
        else:
            birads = "BI-RADS 5"
            level = "Highly Suggestive of Malignancy (>=85%)"

        return Agent5RiskResult(
            birads_category=birads,
            malignancy_risk_score=combined_score,
            risk_level=level,
            fusion_factors=fusion_factors,
            status="completed"
        )
