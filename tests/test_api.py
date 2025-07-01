from fastapi.testclient import TestClient
import pytest
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

@patch("app.services.sentiment_service.model_service.predict_sentiment")
def test_analyze_sentiment(mock_predict):
    # Mock the prediction service
    mock_predict.return_value = {
        "text": "This is a great product!",
        "sentiment": "positive",
        "confidence": 0.95,
        "processed_tokens": ["great", "product"]
    }
    
    # Test data
    test_data = {
        "text": "This is a great product!"
    }
    
    # Make request
    response = client.post("/api/v1/analyze", json=test_data)
    
    # Check response
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"
    assert response.json()["confidence"] == 0.95
    assert "processed_tokens" in response.json()
    
    # Verify mock was called with correct data
    mock_predict.assert_called_once_with("This is a great product!")

@patch("app.services.sentiment_service.model_service.predict_batch")
def test_analyze_batch_sentiment(mock_predict_batch):
    # Mock the prediction service
    mock_predict_batch.return_value = [
        {
            "text": "This is a great product!",
            "sentiment": "positive",
            "confidence": 0.95,
            "processed_tokens": ["great", "product"]
        },
        {
            "text": "This is terrible.",
            "sentiment": "negative",
            "confidence": 0.85,
            "processed_tokens": ["terribl"]
        }
    ]
    
    # Test data
    test_data = {
        "texts": ["This is a great product!", "This is terrible."]
    }
    
    # Make request
    response = client.post("/api/v1/analyze/batch", json=test_data)
    
    # Check response
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) == 2
    assert response.json()["results"][0]["sentiment"] == "positive"
    assert response.json()["results"][1]["sentiment"] == "negative"
    
    # Verify mock was called with correct data
    mock_predict_batch.assert_called_once_with(["This is a great product!", "This is terrible."])
