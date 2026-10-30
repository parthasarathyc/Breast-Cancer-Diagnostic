"""
Integration Tests for FastAPI Endpoints
Tests /health, /api/v1/agents/detection, /api/v1/agents/classification, and /api/v1/pipeline/analyze.
"""

import glob
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_image_bytes():
    """Create in-memory PNG bytes or load from test split."""
    imgs = glob.glob("data/classification/test/*/*.png")
    if imgs:
        with open(imgs[0], "rb") as f:
            return f.read()

    buf = io.BytesIO()
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "agent_1_detection" in data["agents_ready"]
    assert "agent_2_classification" in data["agents_ready"]


def test_detection_endpoint_valid(client, sample_image_bytes):
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/v1/agents/detection", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["agent"] == "tumor_detection"
    assert data["status"] == "success"
    assert "detections" in data
    assert isinstance(data["tumors_detected"], int)


def test_classification_endpoint_valid(client, sample_image_bytes):
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/v1/agents/classification", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["agent"] == "tumor_classification"
    assert data["status"] == "success"
    assert data["predicted_class"] in ["normal", "benign", "malignant"]
    assert "class_probabilities" in data


def test_pipeline_analyze_endpoint(client, sample_image_bytes):
    files = {"file": ("test.png", sample_image_bytes, "image/png")}
    response = client.post("/api/v1/pipeline/analyze", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "analysis_id" in data
    assert "agent_1_detection" in data
    assert "agent_2_classification" in data
    assert "summary" in data


def test_invalid_mime_type(client):
    files = {"file": ("notes.txt", b"this is text", "text/plain")}
    response = client.post("/api/v1/agents/detection", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_empty_file_upload(client):
    files = {"file": ("empty.png", b"", "image/png")}
    response = client.post("/api/v1/agents/classification", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]
