from fastapi import APIRouter, status
from typing import Dict, Any
import logging
import psutil
import platform
from datetime import datetime

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health of the API and model",
)
async def health_check() -> Dict[str, Any]:
    """
    Check the health of the API.
    
    This endpoint returns information about the API's health and system resources.
    """
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check if model is loaded
        from app.services.sentiment_service import model_service
        model_loaded = model_service.model is not None
        tokenizer_loaded = model_service.tokenizer is not None
        
        return {
            "status": "healthy" if model_loaded and tokenizer_loaded else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "model": {
                "loaded": model_loaded,
                "tokenizer_loaded": tokenizer_loaded
            },
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
            }
        }
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }
