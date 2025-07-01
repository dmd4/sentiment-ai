import requests
import json
import time
import sys

def test_api():
    """Test the sentiment analysis API locally."""
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        health_response = requests.get(f"{base_url}/health")
        print(f"Health check status: {health_response.status_code}")
        print(f"Health check response: {health_response.json()}")
        print("-" * 50)
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API. Make sure it's running on http://localhost:8000")
        sys.exit(1)
    
    # Test single text analysis
    single_text = "This movie was fantastic! I really enjoyed it."
    try:
        sentiment_response = requests.post(
            f"{base_url}/api/v1/analyze",
            json={"text": single_text}
        )
        print(f"Single text analysis status: {sentiment_response.status_code}")
        print(f"Single text analysis response: {json.dumps(sentiment_response.json(), indent=2)}")
        print("-" * 50)
    except Exception as e:
        print(f"Error testing single text analysis: {str(e)}")
    
    # Test batch text analysis
    batch_texts = [
        "This movie was fantastic! I really enjoyed it.",
        "The service was terrible and the food was cold."
    ]
    try:
        batch_response = requests.post(
            f"{base_url}/api/v1/analyze/batch",
            json={"texts": batch_texts}
        )
        print(f"Batch analysis status: {batch_response.status_code}")
        print(f"Batch analysis response: {json.dumps(batch_response.json(), indent=2)}")
    except Exception as e:
        print(f"Error testing batch analysis: {str(e)}")

if __name__ == "__main__":
    print("Waiting for API to start...")
    time.sleep(2)  # Give the API a moment to start
    test_api()
