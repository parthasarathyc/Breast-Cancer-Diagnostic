"""
Unit and Integration Tests for Agent 2 – Tumor Classification
"""

import glob
import pytest
import numpy as np
from PIL import Image
from agent_2_classification import TumorClassificationAgent, TumorClassificationResponse


@pytest.fixture(scope="module")
def classifier():
    return TumorClassificationAgent()


@pytest.fixture
def sample_ultrasound():
    """Retrieve an actual test image from prepared BUSI dataset."""
    imgs = glob.glob("data/classification/test/*/*.png")
    if imgs:
        return imgs[0]
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


def test_agent_2_initialization(classifier):
    assert classifier is not None
    assert classifier.model is not None
    assert classifier.CLASS_NAMES == ["normal", "benign", "malignant"]


def test_agent_2_predict_valid_image(classifier, sample_ultrasound):
    res = classifier.predict(sample_ultrasound)
    assert isinstance(res, TumorClassificationResponse)
    assert res.status == "success"
    assert res.agent == "tumor_classification"
    assert res.predicted_class in ["normal", "benign", "malignant"]
    assert 0.0 <= res.confidence <= 1.0

    # Probabilities must sum to ~1.0
    probs = res.class_probabilities
    assert "normal" in probs and "benign" in probs and "malignant" in probs
    total_prob = sum(probs.values())
    assert 0.98 <= total_prob <= 1.02
    assert "Academic research prototype" in res.disclaimer


def test_agent_2_invalid_input(classifier):
    res = classifier.predict("invalid_non_existent_path.jpg")
    assert res.status == "error"
    assert res.predicted_class == "unknown"


def test_agent_2_corrupt_bytes(classifier):
    res = classifier.predict(b"corrupt_binary_stream_xyz")
    assert res.status == "error"
    assert res.predicted_class == "unknown"
