"""
FastAPI Dependencies and Model Cache Management
Loads Agent 1 and Agent 2 models once at startup (Singleton pattern).
"""

from typing import Optional
from agent_1_detection.detector import TumorDetectionAgent
from agent_2_classification.classifier import TumorClassificationAgent


class ModelRegistry:
    """Singleton registry holding loaded instances of AI agents."""
    _detection_agent: Optional[TumorDetectionAgent] = None
    _classification_agent: Optional[TumorClassificationAgent] = None

    @classmethod
    def initialize(cls):
        """Pre-load models on application startup."""
        print("[ModelRegistry] Initializing Agent 1 (Tumor Detection)...")
        cls._detection_agent = TumorDetectionAgent()

        print("[ModelRegistry] Initializing Agent 2 (Tumor Classification)...")
        cls._classification_agent = TumorClassificationAgent()
        print("[ModelRegistry] All agents loaded successfully.")

    @classmethod
    def get_detection_agent(cls) -> TumorDetectionAgent:
        if cls._detection_agent is None:
            cls._detection_agent = TumorDetectionAgent()
        return cls._detection_agent

    @classmethod
    def get_classification_agent(cls) -> TumorClassificationAgent:
        if cls._classification_agent is None:
            cls._classification_agent = TumorClassificationAgent()
        return cls._classification_agent
