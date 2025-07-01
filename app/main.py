from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
from typing import List

from app.api import sentiment, health
from app.core.config import settings
from app.core.logging import configure_logging

# Configure logging
logger = logging.getLogger(__name__)
configure_logging()

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Sentiment Analysis API using LSTM neural networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(sentiment.router, prefix="/api/v1", tags=["sentiment"])

# Mount static files
app.mount("/demo", StaticFiles(directory="static", html=True), name="demo")

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up Sentiment Analysis API")
    # Load the model at startup
    from app.services.sentiment_service import model_service
    model_service.load_model()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Sentiment Analysis API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
