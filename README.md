# Sentiment AI - Deep Learning Sentiment Analysis

A comprehensive sentiment analysis project featuring both a production-ready REST API and detailed machine learning analysis using LSTM neural networks to classify customer reviews across multiple domains (Amazon, Yelp, IMDB).

## 🎯 Project Overview

This project demonstrates end-to-end machine learning development from research and model training to production deployment. The system successfully classifies sentiment in customer reviews from three different platforms, achieving **82.55% base model accuracy** enhanced by targeted linguistic rules for complex cases like negation handling, with **100% accuracy on comprehensive test scenarios**.

### 🔗 **Dual Purpose:**
1. **Research & Analysis**: Comprehensive deep learning study with detailed performance analysis
2. **Production API**: RESTful web service for real-time sentiment prediction

## 📊 Key Results

- **Base Model Accuracy**: 82.55% (LSTM neural network)
- **Production System Accuracy**: 100% on comprehensive test suite
- **Advanced Features**: Negation handling ("not bad" → positive, "not good" → negative)
- **Cross-Domain Performance**: Successfully generalizes across Amazon, Yelp, and IMDB reviews
- **Response Time**: 32ms average with sub-5ms for rule-based cases
- **Architecture**: Hybrid ML system combining deep learning with targeted linguistic rules

## 🏗️ Model Architecture

- **Embedding Layer**: 4,060 vocabulary → 7-dimensional embeddings
- **LSTM Layer**: 128 units for sequential pattern recognition
- **Dense Layer**: 64 units with ReLU activation
- **Dropout**: Regularization to prevent overfitting
- **Output**: Sigmoid activation for binary classification

![Model Architecture](docs/images/model_architecture.png)

## 🚀 Quick Start

### **Option 1: Use the Live API**
```bash
# Test the deployed API
curl -X POST https://sentiment-ai-iwp2.onrender.com/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!"}'

# Test advanced negation handling
curl -X POST https://sentiment-ai-iwp2.onrender.com/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was not bad"}'

# View interactive API documentation
open https://sentiment-ai-iwp2.onrender.com/docs
```

> **⏱️ Note**: The API may take 30-60 seconds to respond on the first request after periods of inactivity due to Render's free tier sleep behavior. Subsequent requests will be fast (~32ms average response time).

### **Option 2: Run Locally**
```bash
# Clone the repository
git clone https://github.com/yourusername/sentiment-ai.git
cd sentiment-ai

# Install dependencies
pip install -r requirements.txt

# Run the API locally
python run_api.py

# Test locally
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!"}'
```

### **Option 2: Train Your Own Model**
```bash
# Train the model from scratch
python scripts/train_sentiment_model.py

# Generate visualizations
python scripts/generate_visualizations.py

# Run interactive demo
python scripts/interactive_demo.py
```

### **Option 3: Docker Deployment**
```bash
# Build and run with Docker
docker-compose up --build

# API will be available at http://localhost:8000
```

## 🌐 **Live Demo**

**Try the live API:** https://sentiment-ai-iwp2.onrender.com/docs

**Example API calls:**
```bash
# Positive sentiment
curl -X POST https://sentiment-ai-iwp2.onrender.com/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is excellent!"}'

# Negation handling  
curl -X POST https://sentiment-ai-iwp2.onrender.com/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is not bad"}'
```

> **⏱️ Note**: The API may take 30-60 seconds to respond on the first request after periods of inactivity due to Render's free tier sleep behavior. Subsequent requests will be fast (~32ms average response time).

## 📈 Performance Visualizations

### Training History
![Training History](docs/images/training_history.png)

### Performance Metrics
![Performance Metrics](docs/images/performance_metrics.png)

### Confusion Matrix
![Confusion Matrix](docs/images/confusion_matrix_heatmap.png)

### Dataset Overview
![Dataset Overview](docs/images/dataset_overview.png)

## 🔧 Technical Stack

### **Machine Learning**
- **Python 3.12**
- **TensorFlow/Keras** for neural network implementation
- **NLTK** for natural language processing
- **Scikit-learn** for evaluation metrics

### **API & Deployment**
- **Flask** for REST API
- **Docker** for containerization
- **Gunicorn** for production WSGI server
- **pytest** for comprehensive testing

### **Data & Visualization**
- **Pandas** for data manipulation
- **Matplotlib/Seaborn** for visualizations
- **NumPy** for numerical operations

## 📁 Project Structure

```
sentiment-ai/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── Dockerfile                          # Container configuration
├── docker-compose.yml                 # Multi-container setup
├── run_api.py                         # API entry point
├── app/                               # Flask application
│   ├── __init__.py
│   ├── routes.py                      # API endpoints
│   ├── models.py                      # Model loading logic
│   └── utils.py                       # Utility functions
├── scripts/                           # Training & utility scripts
│   ├── train_sentiment_model.py       # Model training pipeline
│   ├── generate_visualizations.py     # Create analysis charts
│   ├── create_tokenizer.py           # Tokenizer generation
│   └── interactive_demo.py           # Interactive model demo
├── models/                            # Trained models
│   ├── sentiment_analysis_model.h5    # Neural network weights
│   ├── tokenizer.pickle               # Text preprocessing
│   └── max_length.txt                 # Sequence length parameter
├── docs/                              # Documentation
│   ├── Sentiment_Analysis_Report.md   # Comprehensive analysis
│   ├── Executive_Summary.md           # Key findings
│   └── images/                        # Visualization assets
├── data/                              # Dataset files
├── tests/                             # Unit tests
└── static/                            # Web interface assets
```

## 🔍 Detailed Analysis

For comprehensive technical analysis including methodology, results interpretation, and performance evaluation:

- 📊 [**Complete Technical Report**](docs/Sentiment_Analysis_Report.md) - In-depth analysis with visualizations
- 📋 [**Executive Summary**](docs/Executive_Summary.md) - Key findings and recommendations
- 🚀 [**API Documentation**](DEPLOYMENT_GUIDE.md) - Production deployment guide

## 📊 API Endpoints

### **Predict Sentiment**
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "text": "Your review text here"
}
```

**Response:**
```json
{
  "text": "Your review text here",
  "sentiment": "positive",
  "confidence": 0.87,
  "processed_tokens": null
}
```

### **Health Check**
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-21T23:39:46.797504",
  "model": {
    "loaded": true,
    "tokenizer_loaded": true
  },
  "system": {
    "cpu_percent": 15.2,
    "memory_percent": 45.8,
    "python_version": "3.10.18"
  }
}
```

### **API Documentation**
```http
GET /docs
```
Interactive Swagger UI documentation available at: https://sentiment-ai-iwp2.onrender.com/docs

## 💡 **Demo Tips**

### **For Live Presentations:**
- **Warm up the API** before demos by making a test request 
- **Mention the free tier** - shows understanding of deployment trade-offs
- **Have backup screenshots** ready in case of connectivity issues

### **Quick Warm-up Command:**
```bash
# Wake up the API before your demo
curl https://sentiment-ai-iwp2.onrender.com/health
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Test API locally
python test_local_api.py

# Load testing
python test_load_performance.py
```

## 📋 Key Findings

### ✅ **Strengths**
- **Cross-domain generalization**: Works across different review types
- **Balanced performance**: Similar accuracy for both sentiment classes
- **Production ready**: Containerized with comprehensive testing
- **Scalable architecture**: Easy to deploy and maintain

### ⚠️ **Areas for Improvement**
- **Pure ML Accuracy**: 82.55% base model could be improved with advanced architectures (BERT/RoBERTa)
- **Rule Dependency**: Current 100% accuracy relies on hand-crafted linguistic rules for edge cases
- **Scalability**: Manual keyword lists require maintenance for new domains

### 🎯 **Business Applications**
- **Customer Feedback Analysis**: Automated sentiment monitoring
- **Product Review Summarization**: Quick sentiment assessment
- **Brand Monitoring**: Cross-platform sentiment tracking
- **Content Moderation**: Initial filtering of user-generated content

## 🚀 Future Enhancements

- [ ] **Advanced Models**: Implement BERT/RoBERTa for improved accuracy
- [ ] **Real-time Processing**: Add streaming data capabilities
- [ ] **Multi-language Support**: Extend beyond English reviews
- [ ] **Confidence Calibration**: Improve probability estimates
- [ ] **A/B Testing Framework**: Compare model versions
- [ ] **Monitoring Dashboard**: Track API performance and model drift

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 82.55% |
| Production Accuracy | 100% (with rules) |
| Precision (Negative) | 85% |
| Recall (Negative) | 66% |
| Precision (Positive) | 67% |
| Recall (Positive) | 85% |
| F1-Score (Overall) | 0.75 |
| Model Size | 415.52 KB |
| Inference Time | <40ms |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates practical application of deep learning for natural language processing, showcasing end-to-end machine learning pipeline development from research and analysis to production deployment.*
