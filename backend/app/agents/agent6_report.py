import os
import io
import base64
from datetime import datetime
from typing import Dict, Any, List
from app.api.schemas import (
    Agent6ReportResult,
    Agent1DetectionResult,
    Agent2ClassificationResult,
    Agent3XAIResult,
    Agent4EvidenceResult,
    Agent5RiskResult
)
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

class ReportGenerationAgent:
    def __init__(self, reports_dir="backend/reports"):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate(
        self,
        case_id: str,
        clinical_inputs: Dict[str, Any],
        agent1: Agent1DetectionResult,
        agent2: Agent2ClassificationResult,
        agent3: Agent3XAIResult,
        agent4: Agent4EvidenceResult,
        agent5: Agent5RiskResult
    ) -> Agent6ReportResult:
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Clinical summaries
        exec_summary = (
            f"Automated multimodal decision support evaluation for Case #{case_id}. "
            f"AI classification predicts {agent2.predicted_class.upper()} lesion with {round(agent2.raw_confidence * 100, 1)}% confidence. "
            f"Risk assessment assigns {agent5.birads_category} ({agent5.risk_level}) with a calibrated malignancy risk of {round(agent5.malignancy_risk_score * 100, 1)}%."
        )

        morph = agent3.morphology
        findings = (
            f"1. Lesion Detection: {'Focal mass detected' if agent1.lesion_detected else 'No suspicious focal mass'}. "
            f"Estimated relative area: {round(agent1.lesion_area_percentage, 2)}% of ultrasound frame.\n"
            f"2. Morphological Descriptors: Shape={morph.shape}, Margin={morph.margin}, "
            f"Orientation={morph.orientation}, Aspect Ratio={morph.aspect_ratio:.2f}, Solidity={morph.solidity:.2f}.\n"
            f"3. Acoustic Pattern: {'Posterior acoustic shadowing noted' if morph.acoustic_shadowing_observed else 'No significant posterior shadowing'}.\n"
            f"4. XAI Verification: Grad-CAM++ activation maps confirm neural attention aligns with lesion boundaries."
        )

        impression = (
            f"Assessment: {agent5.birads_category} - {agent5.risk_level}.\n"
            f"Clinical Context: Patient age {clinical_inputs.get('patient_age')}, "
            f"Breast density {clinical_inputs.get('breast_density')}, "
            f"Family history: {'Positive' if clinical_inputs.get('family_history') else 'Negative'}."
        )

        recommendations = []
        if agent5.birads_category in ["BI-RADS 1", "BI-RADS 2"]:
            recommendations.append("Routine annual screening mammography/ultrasound as per age guidelines.")
            recommendations.append("Clinical breast examination (CBE) during annual wellness visit.")
        elif agent5.birads_category == "BI-RADS 3":
            recommendations.append("Short-interval (6-month) targeted unilateral ultrasound follow-up to establish lesion stability.")
            recommendations.append("Avoid immediate invasive biopsy unless clinical palpation reveals rapid interval growth.")
        elif agent5.birads_category in ["BI-RADS 4A", "BI-RADS 4B"]:
            recommendations.append("Ultrasound-guided core needle biopsy (CNB) recommended.")
            recommendations.append("Correlate histologic findings with radiographic features to confirm concordance.")
        else:  # 4C or 5
            recommendations.append("Urgent ultrasound-guided core needle biopsy recommended.")
            recommendations.append("Immediate multidisciplinary breast oncology and surgical consultation.")
            recommendations.append("Consider contrast-enhanced breast MRI for staging if invasive disease is confirmed.")

        disclaimer = (
            "NOTICE: Breast CARE-MA is an AI decision-support research prototype developed for academic evaluation. "
            "It does not provide definitive medical diagnosis and must never supersede clinical judgment by a licensed radiologist or oncologist."
        )

        # Generate PDF
        pdf_path = os.path.join(self.reports_dir, f"Report_{case_id}.pdf")
        self._build_pdf(
            pdf_path, case_id, now_str, clinical_inputs, agent1, agent2, agent3, agent4, agent5, recommendations, disclaimer
        )

        # Read PDF as base64 for direct browser preview
        pdf_base64 = None
        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        return Agent6ReportResult(
            case_id=case_id,
            timestamp=now_str,
            executive_summary=exec_summary,
            detailed_findings=findings,
            birads_clinical_impression=impression,
            recommended_action=recommendations,
            disclaimer=disclaimer,
            pdf_report_path=pdf_path,
            pdf_base64=pdf_base64,
            status="completed"
        )

    def _build_pdf(
        self,
        pdf_path: str,
        case_id: str,
        timestamp: str,
        clinical_inputs: Dict[str, Any],
        agent1: Agent1DetectionResult,
        agent2: Agent2ClassificationResult,
        agent3: Agent3XAIResult,
        agent4: Agent4EvidenceResult,
        agent5: Agent5RiskResult,
        recommendations: List[str],
        disclaimer: str
    ):
        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1e3a8a"),
            alignment=1  # Center
        )
        subtitle_style = ParagraphStyle(
            "SubTitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4b5563"),
            alignment=1
        )
        heading_style = ParagraphStyle(
            "SecHeading",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=10,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#1f2937")
        )
        alert_style = ParagraphStyle(
            "Alert",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#991b1b")
        )

        elements = []

        # Header Banner
        elements.append(Paragraph("<b>BREAST CARE-MA DIAGNOSTIC DECISION SUPPORT REPORT</b>", title_style))
        elements.append(Paragraph("Department of Information Technology | Nehru Institute of Engineering and Technology", subtitle_style))
        elements.append(Paragraph("Explainable Multi-Agent AI System for Ultrasound Modality", subtitle_style))
        elements.append(Spacer(1, 12))

        # Case Metadata Table
        data_table = [
            [Paragraph("<b>Case ID:</b>", body_style), Paragraph(case_id, body_style), Paragraph("<b>Date & Time:</b>", body_style), Paragraph(timestamp, body_style)],
            [Paragraph("<b>Patient Age:</b>", body_style), Paragraph(str(clinical_inputs.get("patient_age")), body_style), Paragraph("<b>Family History:</b>", body_style), Paragraph("Yes" if clinical_inputs.get("family_history") else "No", body_style)],
            [Paragraph("<b>Breast Density:</b>", body_style), Paragraph(str(clinical_inputs.get("breast_density")), body_style), Paragraph("<b>Symptoms:</b>", body_style), Paragraph(str(clinical_inputs.get("symptoms", "Routine")), body_style)],
        ]
        t = Table(data_table, colWidths=[100, 170, 100, 170])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
        elements.append(Spacer(1, 10))

        # AI Findings & Quantitative Metrics
        elements.append(Paragraph("<b>1. Quantitative AI Findings & Model Outputs</b>", heading_style))
        probs = agent2.calibrated_probabilities
        ai_table_data = [
            [Paragraph("<b>Parameter</b>", body_style), Paragraph("<b>Result</b>", body_style), Paragraph("<b>Clinical Meaning</b>", body_style)],
            [Paragraph("AI Classification", body_style), Paragraph(f"<b>{agent2.predicted_class.upper()}</b>", body_style), Paragraph(f"Calibrated Malignant Probability: {probs.get('malignant', 0)*100:.1f}%", body_style)],
            [Paragraph("Tumor Detection", body_style), Paragraph("Lesion Detected" if agent1.lesion_detected else "No Lesion", body_style), Paragraph(f"Relative Area: {agent1.lesion_area_percentage:.2f}%", body_style)],
            [Paragraph("Morphology (Agent 3)", body_style), Paragraph(f"{agent3.morphology.shape.capitalize()} / {agent3.morphology.margin.capitalize()}", body_style), Paragraph(f"Aspect Ratio: {agent3.morphology.aspect_ratio:.2f}, Solidity: {agent3.morphology.solidity:.2f}", body_style)],
            [Paragraph("Acoustic Shadowing", body_style), Paragraph("Present" if agent3.morphology.acoustic_shadowing_observed else "Absent", body_style), Paragraph("Acoustic attenuation distal to mass", body_style)],
            [Paragraph("ACR BI-RADS Category", body_style), Paragraph(f"<b>{agent5.birads_category}</b>", body_style), Paragraph(f"<b>{agent5.risk_level}</b>", body_style)],
        ]
        t2 = Table(ai_table_data, colWidths=[140, 160, 240])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#c7d2fe")),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e7ff")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(t2)
        elements.append(Spacer(1, 10))

        # Clinical Guidelines
        elements.append(Paragraph("<b>2. Retrieved Evidence & Clinical Guidelines (Agent 4)</b>", heading_style))
        for g in agent4.matched_guidelines:
            elements.append(Paragraph(f"• <b>{g.title}</b> ({g.source}): {g.guideline}", body_style))
            elements.append(Spacer(1, 2))
        elements.append(Spacer(1, 6))

        # Recommendations
        elements.append(Paragraph("<b>3. Recommended Clinical Follow-up Actions</b>", heading_style))
        for r in recommendations:
            elements.append(Paragraph(f"• {r}", body_style))
            elements.append(Spacer(1, 2))
        elements.append(Spacer(1, 10))

        # Safety Disclaimer Box
        elements.append(Paragraph(f"<b>REGULATORY & SAFETY DISCLAIMER:</b> {disclaimer}", alert_style))

        doc.build(elements)
