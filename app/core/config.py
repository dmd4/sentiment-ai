import os
from typing import List, Optional
from pydantic import BaseModel

class Settings(BaseModel):
    # API settings
    PROJECT_NAME: str = "Sentiment Analysis API"
    API_V1_STR: str = "/api/v1"
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/sentiment_analysis_model_v2.h5")
    TOKENIZER_PATH: str = os.getenv("TOKENIZER_PATH", "models/tokenizer_v2.pickle")
    MAX_SEQUENCE_LENGTH: int = int(os.getenv("MAX_SEQUENCE_LENGTH", "22"))
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()
