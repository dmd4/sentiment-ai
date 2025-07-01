#!/usr/bin/env python3
"""
Simple script to test the Sentiment Analysis API locally or remotely.
"""

import argparse
import requests
import json
import sys

def test_sentiment_analysis(url: str, text: str) -> None:
    """Test the sentiment analysis API with the provided text."""
    # Construct the full URL
    if not url.endswith('/'):
        url += '/'
    
    analyze_url = f"{url}api/v1/analyze"
    
    # Prepare the request payload
    payload = {"text": text}
    
    try:
        # Make the request
        print(f"Sending request to {analyze_url}")
        print(f"Payload: {json.dumps(payload)}")
        
        response = requests.post(analyze_url, json=payload)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Print the result
        print("\nSentiment Analysis Result:")
        print(f"Text: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
        if 'processed_tokens' in result and result['processed_tokens']:
            print(f"Processed tokens: {', '.join(result['processed_tokens'])}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        sys.exit(1)

def test_batch_sentiment_analysis(url: str, texts: list) -> None:
    """Test the batch sentiment analysis API with the provided texts."""
    # Construct the full URL
    if not url.endswith('/'):
        url += '/'
    
    batch_url = f"{url}api/v1/analyze/batch"
    
    # Prepare the request payload
    payload = {"texts": texts}
    
    try:
        # Make the request
        print(f"Sending batch request to {batch_url}")
        print(f"Payload: {json.dumps(payload)}")
        
        response = requests.post(batch_url, json=payload)
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Print the results
        print("\nBatch Sentiment Analysis Results:")
        for i, item in enumerate(result['results']):
            print(f"\nResult {i+1}:")
            print(f"Text: {item['text']}")
            print(f"Sentiment: {item['sentiment']}")
            print(f"Confidence: {item['confidence']:.2f}")
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Test the Sentiment Analysis API')
    parser.add_argument('--url', default='http://localhost:8000', help='API URL (default: http://localhost:8000)')
    parser.add_argument('--batch', action='store_true', help='Use batch API endpoint')
    parser.add_argument('text', nargs='+', help='Text to analyze')
    
    args = parser.parse_args()
    
    if args.batch:
        test_batch_sentiment_analysis(args.url, args.text)
    else:
        test_sentiment_analysis(args.url, ' '.join(args.text))

if __name__ == '__main__':
    main()
