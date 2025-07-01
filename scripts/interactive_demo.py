#!/usr/bin/env python3
"""
Sentiment Analysis API Demo
--------------------------
This script demonstrates how to use the Sentiment Analysis API.
It provides a simple command-line interface for analyzing text sentiment.

Usage:
    python demo.py

Author: Your Name
"""

import requests
import json
import sys
import time
from typing import Dict, Any, List, Optional

# Configuration
API_URL = "http://localhost:8000"  # Change this if deployed elsewhere

def print_header():
    """Print a nice header for the demo."""
    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYSIS API DEMO")
    print("=" * 60)
    print("\nThis demo allows you to analyze the sentiment of text using the API.")
    print("Type 'exit' to quit, 'batch' to analyze multiple texts, or enter text to analyze.\n")

def analyze_sentiment(text: str) -> Optional[Dict[str, Any]]:
    """Analyze the sentiment of a single text."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/analyze",
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        return None

def analyze_batch(texts: List[str]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """Analyze the sentiment of multiple texts."""
    try:
        response = requests.post(
            f"{API_URL}/api/v1/analyze/batch",
            json={"texts": texts},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {str(e)}")
        return None

def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        health_data = response.json()
        print(f"API Status: {health_data['status']}")
        print(f"System Info: Python {health_data['system']['python_version']} on {health_data['system']['platform']}")
        return True
    except requests.exceptions.RequestException:
        print("Error: Could not connect to the API.")
        print(f"Make sure the API is running at {API_URL}")
        return False

def display_result(result: Dict[str, Any]):
    """Display the sentiment analysis result in a nice format."""
    if not result:
        return
    
    sentiment = result["sentiment"]
    confidence = result["confidence"]
    
    # Create a visual representation of confidence
    confidence_bar = "█" * int(confidence * 20)
    confidence_empty = "░" * (20 - int(confidence * 20))
    
    print("\n" + "-" * 60)
    print(f"Text: {result['text']}")
    print("-" * 60)
    print(f"Sentiment: {sentiment.upper()}")
    print(f"Confidence: {confidence:.2f} [{confidence_bar}{confidence_empty}] {int(confidence * 100)}%")
    print(f"Processed tokens: {', '.join(result['processed_tokens'])}")
    print("-" * 60 + "\n")

def batch_mode():
    """Enter batch mode to analyze multiple texts."""
    texts = []
    print("\nBATCH MODE: Enter multiple texts (one per line)")
    print("Enter an empty line when finished.\n")
    
    while True:
        line = input("> ").strip()
        if not line:
            break
        texts.append(line)
    
    if not texts:
        print("No texts entered.")
        return
    
    print(f"\nAnalyzing {len(texts)} texts...")
    results = analyze_batch(texts)
    
    if results:
        for i, result in enumerate(results["results"]):
            print(f"\nResult {i+1}/{len(texts)}:")
            display_result(result)

def main():
    """Main function to run the demo."""
    print_header()
    
    # Check if the API is running
    if not check_api_health():
        sys.exit(1)
    
    # Main loop
    while True:
        print("\nEnter text to analyze (or 'exit' to quit, 'batch' for batch mode):")
        user_input = input("> ").strip()
        
        if user_input.lower() == "exit":
            print("\nThank you for using the Sentiment Analysis API Demo!")
            break
        elif user_input.lower() == "batch":
            batch_mode()
        elif user_input:
            print("Analyzing sentiment...")
            result = analyze_sentiment(user_input)
            display_result(result)
        else:
            print("Please enter some text to analyze.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo terminated by user.")
        sys.exit(0)
