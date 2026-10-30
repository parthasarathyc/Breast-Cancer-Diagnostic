import json
import os
from typing import List, Dict, Any
from app.api.schemas import EvidenceCitation, Agent4EvidenceResult

class ClinicalEvidenceAgent:
    def __init__(self, kb_path: str = None):
        if kb_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            kb_path = os.path.join(base_dir, "knowledge_base", "birads_kb.json")
        self.kb_path = kb_path
        self.knowledge_base = self._load_kb()

    def _load_kb(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.kb_path):
            return []
        with open(self.kb_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def retrieve(
        self,
        predicted_class: str,
        morphology: Dict[str, Any],
        clinical_inputs: Dict[str, Any]
    ) -> Agent4EvidenceResult:
        """
        Retrieves matching ACR BI-RADS clinical criteria based on AI features and clinical risk factors.
        """
        scored_entries = []
        
        # Build search tokens
        search_terms = [predicted_class.lower()]
        if morphology.get("shape"):
            search_terms.append(morphology["shape"].lower())
        if morphology.get("margin"):
            search_terms.append(morphology["margin"].lower())
        if morphology.get("acoustic_shadowing_observed"):
            search_terms.append("acoustic shadowing")
        if clinical_inputs.get("family_history"):
            search_terms.append("suspicious")

        for entry in self.knowledge_base:
            score = 0.0
            feature_matches = [m.lower() for m in entry.get("feature_match", [])]
            
            # Exact class match
            if predicted_class.lower() in feature_matches:
                score += 0.5
            
            # Morphological criteria matches
            for term in search_terms:
                if term in feature_matches:
                    score += 0.25
                elif term in entry.get("guideline", "").lower() or term in entry.get("title", "").lower():
                    score += 0.15

            # Clinical adjustment
            if clinical_inputs.get("family_history") and "4" in entry.get("category", ""):
                score += 0.1

            score = min(round(score, 3), 1.0)
            if score > 0.2:
                scored_entries.append((score, entry))

        # Sort by relevance
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        top_matches = scored_entries[:3]

        citations = []
        for score, item in top_matches:
            citations.append(
                EvidenceCitation(
                    title=item["title"],
                    source=item["source"],
                    guideline=item["guideline"],
                    relevance_note=f"Matched clinical criteria for {item['category']}. Action: {item['action']}",
                    match_score=score
                )
            )

        # Fallback if no specific guideline matched
        if not citations and self.knowledge_base:
            fallback = self.knowledge_base[0]
            citations.append(
                EvidenceCitation(
                    title=fallback["title"],
                    source=fallback["source"],
                    guideline=fallback["guideline"],
                    relevance_note=fallback["action"],
                    match_score=0.5
                )
            )

        return Agent4EvidenceResult(
            matched_guidelines=citations,
            total_retrieved=len(citations),
            status="completed"
        )
