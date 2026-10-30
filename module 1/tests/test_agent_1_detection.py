"""
Unit and Integration Tests for Agent 1 – Tumor Detection
"""

import glob
import pytest
import numpy as np
from PIL import Image
from agent_1_detection import TumorDetectionAgent, TumorDetectionResponse


@pytest.fixture(scope="module")
def detector():
    return TumorDetectionAgent()


@pytest.fixture
def sample_ultrasound():
    """Retrieve an actual test image from prepared BUSI dataset."""
    imgs = glob.glob("data/yolo/images/test/*.png")
    if imgs:
        return imgs[0]
    # Fallback synthetic image for standalone testing
    synthetic = np.zeros((256, 256, 3), dtype=np.uint8)
    return synthetic


def test_agent_1_initialization(detector):
    assert detector is not None
    assert detector.model is not None
    assert detector.model_version == "yolov8-tumor-detector-v1.0"


def test_agent_1_predict_valid_image(detector, sample_ultrasound):
    res = detector.predict(sample_ultrasound)
    assert isinstance(res, TumorDetectionResponse)
    assert res.status == "success"
    assert res.agent == "tumor_detection"
    assert isinstance(res.has_tumor, bool)
    assert isinstance(res.tumors_detected, int)
    assert res.inference_time_ms > 0.0
    assert "Academic research prototype" in res.disclaimer


def test_agent_1_normalized_coords(detector, sample_ultrasound):
    res = detector.predict(sample_ultrasound, conf_threshold=0.01)
    if res.detections:
        for det in res.detections:
            xc, yc, w, h = det.normalized_xywh
            assert 0.0 <= xc <= 1.0
            assert 0.0 <= yc <= 1.0
            assert 0.0 <= w <= 1.0
            assert 0.0 <= h <= 1.0


def test_agent_1_invalid_path(detector):
    res = detector.predict("non_existent_image_12345.png")
    assert res.status == "error"
    assert "not exist" in res.message or "failed" in res.message


def test_agent_1_corrupt_bytes(detector):
    res = detector.predict(b"not_an_image_corrupt_data")
    assert res.status == "error"
    assert "Could not decode" in res.message or "failed" in res.message
