# Sentiment Analysis API Deployment Guide

This guide provides step-by-step instructions for deploying the Sentiment Analysis API to various cloud platforms.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment Options](#cloud-deployment-options)
   - [AWS Elastic Beanstalk](#aws-elastic-beanstalk)
   - [Google Cloud Run](#google-cloud-run)
   - [Heroku](#heroku)
4. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Local Deployment

### Prerequisites

- Python 3.10+
- pip (Python package manager)
- Git

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis-api.git
   cd sentiment-analysis-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the API:
   ```bash
   python run_api.py
   ```

5. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Interactive Demo: http://localhost:8000/demo
   - Health Check: http://localhost:8000/health

## Docker Deployment

### Prerequisites

- Docker
- Docker Compose

### Steps

1. Build and run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

2. Access the API:
   - API Documentation: http://localhost:8000/docs
   - Interactive Demo: http://localhost:8000/demo

3. Stop the containers:
   ```bash
   docker-compose down
   ```

## Cloud Deployment Options

### AWS Elastic Beanstalk

#### Prerequisites

- AWS Account
- AWS CLI installed and configured
- EB CLI installed

#### Steps

1. Initialize EB application:
   ```bash
   eb init -p docker sentiment-analysis-api
   ```

2. Create environment and deploy:
   ```bash
   eb create sentiment-analysis-env
   ```

3. Open the application:
   ```bash
   eb open
   ```

4. For future updates:
   ```bash
   eb deploy
   ```

### Google Cloud Run

#### Prerequisites

- Google Cloud Account
- Google Cloud SDK installed and configured

#### Steps

1. Build the Docker image:
   ```bash
   docker build -t gcr.io/your-project-id/sentiment-analysis-api .
   ```

2. Push to Google Container Registry:
   ```bash
   docker push gcr.io/your-project-id/sentiment-analysis-api
   ```

3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy sentiment-analysis-api \
     --image gcr.io/your-project-id/sentiment-analysis-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Heroku

#### Prerequisites

- Heroku Account
- Heroku CLI installed and configured

#### Steps

1. Login to Heroku:
   ```bash
   heroku login
   ```

2. Create a Heroku app:
   ```bash
   heroku create your-sentiment-api
   ```

3. Add Heroku-specific files:
   - Create a `Procfile` with:
     ```
     web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
     ```

4. Deploy to Heroku:
   ```bash
   git push heroku main
   ```

5. Open the application:
   ```bash
   heroku open
   ```

## Monitoring and Maintenance

### Health Checks

The API provides a health endpoint at `/health` that returns:
- API status
- Model status
- System information

### Logging

Logs are output in JSON format for easy integration with log management systems.

### Scaling

- For Docker: Adjust the number of containers in docker-compose.yml
- For AWS: Configure auto-scaling in Elastic Beanstalk
- For Google Cloud Run: Set min/max instances and concurrency

### Security Considerations

- Set up proper authentication for production deployments
- Use environment variables for sensitive configuration
- Implement rate limiting for public-facing APIs
