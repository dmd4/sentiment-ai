from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from typing import Dict, Any, List
import logging
import time

from app.schemas.sentiment import SentimentRequest, SentimentResponse, BatchSentimentRequest, BatchSentimentResponse
from app.services.sentiment_service import model_service

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/analyze",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment of text",
    description="Analyze the sentiment of the provided text and return the sentiment classification (positive/negative) with confidence score",
)
async def analyze_sentiment(request: SentimentRequest) -> Dict[str, Any]:
    """
    Analyze the sentiment of the provided text.
    
    This endpoint processes the input text and returns:
    - The sentiment classification (positive or negative)
    - A confidence score for the prediction
    - The processed tokens used for the analysis
    """
    try:
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Make prediction
        result = model_service.predict_sentiment(request.text)
        
        # Record end time and calculate duration
        duration = time.time() - start_time
        
        # Log prediction details
        logger.info(
            f"Sentiment analysis completed in {duration:.4f} seconds",
            extra={
                "props": {
                    "text_length": len(request.text),
                    "sentiment": result["sentiment"],
                    "confidence": result["confidence"],
                    "duration_seconds": duration,
                }
            }
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing sentiment. Please try again later.",
        )

@router.post(
    "/analyze/batch",
    response_model=BatchSentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze sentiment of multiple texts",
    description="Analyze the sentiment of multiple texts in a single request",
)
async def analyze_batch_sentiment(request: BatchSentimentRequest) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze the sentiment of multiple texts in a batch.
    
    This endpoint processes multiple input texts and returns sentiment analysis results for each.
    """
    try:
        # Record start time for performance monitoring
        start_time = time.time()
        
        # Make batch prediction
        results = model_service.predict_batch(request.texts)
        
        # Record end time and calculate duration
        duration = time.time() - start_time
        
        # Log prediction details
        logger.info(
            f"Batch sentiment analysis completed in {duration:.4f} seconds",
            extra={
                "props": {
                    "batch_size": len(request.texts),
                    "avg_text_length": sum(len(text) for text in request.texts) / len(request.texts),
                    "duration_seconds": duration,
                }
            }
        )
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while analyzing sentiment. Please try again later.",
        )
