#!/usr/bin/env python3
"""
Quick performance test for the improved sentiment model
Tests accuracy on a variety of examples and measures response time
"""

import requests
import time
import json
from typing import List, Dict

API_URL = "http://localhost:8000/api/v1/analyze"

def test_sentiment(text: str) -> Dict:
    """Test a single text and return result with timing"""
    start_time = time.time()
    try:
        response = requests.post(API_URL, json={"text": text}, timeout=5)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            result["response_time"] = end_time - start_time
            return result
        else:
            return {"error": f"HTTP {response.status_code}", "response_time": end_time - start_time}
    except Exception as e:
        return {"error": str(e), "response_time": time.time() - start_time}

def run_performance_tests():
    """Run comprehensive performance tests"""
    
    print("🚀 Testing Improved Sentiment Analysis Model Performance")
    print("=" * 60)
    
    # Test cases with expected results
    test_cases = [
        # Clear Positive Cases
        {"text": "This movie was amazing and fantastic!", "expected": "positive", "category": "Clear Positive"},
        {"text": "I love this product, excellent quality", "expected": "positive", "category": "Clear Positive"},
        {"text": "Great service, highly recommend", "expected": "positive", "category": "Clear Positive"},
        {"text": "Perfect experience, very satisfied", "expected": "positive", "category": "Clear Positive"},
        
        # Clear Negative Cases  
        {"text": "This movie was terrible and awful", "expected": "negative", "category": "Clear Negative"},
        {"text": "I hate this product, horrible quality", "expected": "negative", "category": "Clear Negative"},
        {"text": "Terrible service, very disappointed", "expected": "negative", "category": "Clear Negative"},
        {"text": "Awful experience, complete waste", "expected": "negative", "category": "Clear Negative"},
        
        # Extreme Cases
        {"text": "terrible awful horrible disgusting worst", "expected": "negative", "category": "Extreme Negative"},
        {"text": "amazing fantastic excellent wonderful best", "expected": "positive", "category": "Extreme Positive"},
        
        # Single Words
        {"text": "excellent", "expected": "positive", "category": "Single Word"},
        {"text": "terrible", "expected": "negative", "category": "Single Word"},
        {"text": "good", "expected": "positive", "category": "Single Word"},
        {"text": "bad", "expected": "negative", "category": "Single Word"},
        
        # Negation Cases (more complex)
        {"text": "This movie was not bad", "expected": "positive", "category": "Negation"},
        {"text": "This movie was not good", "expected": "negative", "category": "Negation"},
        {"text": "I don't hate this", "expected": "positive", "category": "Negation"},
        
        # Mixed/Neutral Cases
        {"text": "The movie was okay", "expected": None, "category": "Mixed"},
        {"text": "It was fine, nothing special", "expected": None, "category": "Mixed"},
        {"text": "Average product", "expected": None, "category": "Mixed"},
    ]
    
    # Run tests
    results = []
    response_times = []
    correct_predictions = 0
    total_predictions = 0
    
    print("Running tests...")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        result = test_sentiment(test_case["text"])
        results.append({**test_case, **result})
        
        if "error" not in result:
            response_times.append(result["response_time"])
            
            # Check accuracy (skip mixed cases)
            if test_case["expected"] is not None:
                total_predictions += 1
                if result["sentiment"] == test_case["expected"]:
                    correct_predictions += 1
                    status = "✅"
                else:
                    status = "❌"
            else:
                status = "➖"  # Mixed case, no expected result
            
            print(f"{i:2d}. {status} [{test_case['category']}] '{test_case['text'][:40]}{'...' if len(test_case['text']) > 40 else ''}'")
            print(f"    → {result['sentiment']} ({result['confidence']:.3f}) in {result['response_time']*1000:.1f}ms")
        else:
            print(f"{i:2d}. ❌ ERROR: {result['error']}")
        
        print()
    
    # Calculate statistics
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
    else:
        avg_response_time = max_response_time = min_response_time = 0
    
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Print summary
    print("=" * 60)
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"✅ Accuracy: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
    print(f"⚡ Avg Response Time: {avg_response_time*1000:.1f}ms")
    print(f"📈 Response Time Range: {min_response_time*1000:.1f}ms - {max_response_time*1000:.1f}ms")
    print(f"🎯 Total Tests: {len(test_cases)}")
    print()
    
    # Category breakdown
    categories = {}
    for result in results:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if result.get("expected") and not result.get("error"):
            if result["sentiment"] == result["expected"]:
                categories[cat]["correct"] += 1
    
    print("📋 CATEGORY BREAKDOWN:")
    for cat, stats in categories.items():
        if stats["total"] > 0:
            cat_accuracy = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_accuracy:.1f}%)")
    
    print()
    
    # Performance assessment
    if accuracy >= 85:
        print("🏆 EXCELLENT: Model performance is outstanding!")
    elif accuracy >= 75:
        print("✅ GOOD: Model performance is solid!")
    elif accuracy >= 65:
        print("⚠️  FAIR: Model performance needs improvement")
    else:
        print("❌ POOR: Model performance requires significant work")
    
    if avg_response_time < 0.1:
        print("⚡ FAST: Response times are excellent (<100ms)")
    elif avg_response_time < 0.2:
        print("✅ GOOD: Response times are acceptable (<200ms)")
    else:
        print("⚠️  SLOW: Response times could be improved")
    
    return results

if __name__ == "__main__":
    try:
        results = run_performance_tests()
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        print("Make sure the API is running on http://localhost:8000")
