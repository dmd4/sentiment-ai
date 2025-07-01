#!/bin/bash

echo "🚀 Sentiment Analysis API Deployment Script"
echo "==========================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway..."
    railway login
fi

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up

echo "✅ Deployment initiated!"
echo "📊 Check your Railway dashboard for deployment status"
echo "🌐 Your API will be available at: https://your-app.railway.app"
echo ""
echo "🧪 Test your deployed API:"
echo "curl -X POST https://your-app.railway.app/api/v1/analyze \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"text\": \"This movie was amazing!\"}'"
