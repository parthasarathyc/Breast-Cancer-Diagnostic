"""
Agent 1: Tumor Detection Package
"""

from .detector import TumorDetectionAgent
from .schemas import TumorDetectionResponse, BoundingBox

__all__ = ["TumorDetectionAgent", "TumorDetectionResponse", "BoundingBox"]
