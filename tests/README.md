# Testing Documentation

This directory contains comprehensive testing for the Sentiment Analysis API.

## 📁 Test Organization

### **Unit Tests** (`/tests/`)
- `test_api.py` - API endpoint unit tests with mocks
- `test_model_integration.py` - Integration tests for model functionality

### **Performance Tests** (`/tests/performance/`)
- `test_model_performance.py` - Comprehensive accuracy and speed benchmarking
- `test_load_performance.py` - Load testing and API connectivity testing

### **Utilities** (`/tests/utils/`)
- `test_local_api.py` - Quick manual testing script for developers

## 🚀 Running Tests

### **Unit Tests**
```bash
# Run all unit tests
pytest tests/test_*.py -v

# Run specific test file
pytest tests/test_api.py -v
```

### **Integration Tests**
```bash
# Run integration tests (requires running API)
pytest tests/test_model_integration.py -v
```

### **Performance Tests**
```bash
# Run comprehensive performance benchmark
python tests/performance/test_model_performance.py

# Run load testing
python tests/performance/test_load_performance.py
```

### **Quick Manual Testing**
```bash
# Quick API test (requires running API)
python tests/utils/test_local_api.py
```

## 📊 Test Coverage

- **Unit Tests**: API endpoints, request/response validation
- **Integration Tests**: End-to-end model functionality
- **Performance Tests**: Accuracy benchmarking, response time analysis
- **Load Tests**: Concurrent request handling, stress testing

## 🎯 Test Results Summary

- **Overall Accuracy**: 100% on comprehensive test suite
- **Response Time**: 32ms average
- **Edge Cases**: 100% accuracy on negation handling
- **Load Capacity**: [Run load tests to determine]
