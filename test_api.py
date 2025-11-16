"""
API tests
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_analyze_invalid_url():
    """Test analyze endpoint with invalid URL"""
    response = client.post(
        "/analyze",
        json={"video_url": "invalid-url"}
    )
    assert response.status_code == 422  # Validation error
