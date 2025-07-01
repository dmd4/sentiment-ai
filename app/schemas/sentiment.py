from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SentimentRequest(BaseModel):
    text: str = Field(..., 
                     description="The text to analyze for sentiment", 
                     min_length=1, 
                     max_length=5000,
                     example="This movie was fantastic! I really enjoyed the plot and the acting was superb.")

class SentimentResponse(BaseModel):
    text: str = Field(..., description="The original text that was analyzed")
    sentiment: str = Field(..., description="The predicted sentiment (positive or negative)")
    confidence: float = Field(..., description="Confidence score of the prediction (0-1)")
    processed_tokens: Optional[List[str]] = Field(None, description="The processed tokens used for prediction")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This movie was fantastic! I really enjoyed the plot and the acting was superb.",
                "sentiment": "positive",
                "confidence": 0.92,
                "processed_tokens": ["movi", "fantast", "realli", "enjoy", "plot", "act", "superb"]
            }
        }

class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., 
                           description="List of texts to analyze for sentiment",
                           min_items=1,
                           max_items=100)
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "This movie was fantastic! I really enjoyed the plot and the acting was superb.",
                    "The service was terrible and the food was cold."
                ]
            }
        }

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse] = Field(..., description="List of sentiment analysis results")
