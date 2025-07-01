#!/bin/bash

API_URL="https://sentiment-ai-iwp2.onrender.com"

echo "🚀 Testing Live Sentiment Analysis API"
echo "======================================"
echo "API URL: $API_URL"
echo ""

# Health check
echo "1. Health Check:"
curl -s $API_URL/health | jq '.'
echo ""

# Basic positive test
echo "2. Positive Sentiment:"
curl -s -X POST $API_URL/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was amazing!"}' | jq '.'
echo ""

# Basic negative test
echo "3. Negative Sentiment:"
curl -s -X POST $API_URL/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was terrible!"}' | jq '.'
echo ""

# Negation test (your special feature)
echo "4. Negation Handling:"
curl -s -X POST $API_URL/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was not bad"}' | jq '.'
echo ""

# Single word test
echo "5. Single Word:"
curl -s -X POST $API_URL/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "excellent"}' | jq '.'
echo ""

echo "✅ API Testing Complete!"
echo "📊 Your API is live at: $API_URL"
echo "📖 API Docs: $API_URL/docs"
