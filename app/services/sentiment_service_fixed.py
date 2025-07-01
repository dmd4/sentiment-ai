import os
import logging
import pickle
import re
import random
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

class SentimentModelFixed:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_length = settings.MAX_SEQUENCE_LENGTH
        self.stemmer = PorterStemmer()
        self.stop_words = set()
        
        # Download NLTK resources
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not download NLTK resources: {str(e)}")
    
    def load_model(self) -> None:
        """Load the sentiment analysis model and tokenizer."""
        try:
            model_path = settings.MODEL_PATH
            tokenizer_path = settings.TOKENIZER_PATH
            max_length_path = "models/max_length_v2.txt"
            
            # Check if model file exists
            if os.path.exists(model_path):
                logger.info(f"Found model file at {model_path}")
                
                # Load the trained model
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(model_path)
                    logger.info("Successfully loaded TensorFlow model")
                    
                    # Load tokenizer
                    if os.path.exists(tokenizer_path):
                        with open(tokenizer_path, 'rb') as f:
                            self.tokenizer = pickle.load(f)
                        logger.info("Successfully loaded tokenizer")
                    else:
                        logger.warning(f"Tokenizer not found at {tokenizer_path}")
                    
                    # Load correct max_length
                    if os.path.exists(max_length_path):
                        with open(max_length_path, 'r') as f:
                            self.max_length = int(f.read().strip())
                        logger.info(f"Loaded max_length: {self.max_length}")
                    else:
                        logger.warning(f"Max length file not found, using default: {self.max_length}")
                        
                except Exception as e:
                    logger.error(f"Error loading TensorFlow model: {str(e)}")
                    logger.info("Falling back to rule-based approach")
                    self.model = None
                    
            else:
                logger.warning(f"Model file not found at {model_path}. Using rule-based approach.")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
    
    def preprocess_text_improved(self, text: str) -> str:
        """
        Improved preprocessing that handles negation and single words better
        """
        text = text.lower().strip()
        
        # Handle negations more carefully
        negation_patterns = [
            (r'\bnot\s+bad\b', 'POSITIVE_SIGNAL'),
            (r'\bnot\s+terrible\b', 'POSITIVE_SIGNAL'),
            (r'\bnot\s+awful\b', 'POSITIVE_SIGNAL'),
            (r'\bnot\s+horrible\b', 'POSITIVE_SIGNAL'),
            (r'\bnot\s+good\b', 'NEGATIVE_SIGNAL'),
            (r'\bnot\s+great\b', 'NEGATIVE_SIGNAL'),
            (r'\bnot\s+excellent\b', 'NEGATIVE_SIGNAL'),
            (r'\bnot\s+amazing\b', 'NEGATIVE_SIGNAL'),
            (r'\bdon\'t\s+hate\b', 'POSITIVE_SIGNAL'),
            (r'\bdon\'t\s+like\b', 'NEGATIVE_SIGNAL'),
        ]
        
        # Apply negation patterns
        for pattern, replacement in negation_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Handle contractions
        contractions = {
            "don't": "do not",
            "won't": "will not", 
            "can't": "cannot",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "couldn't": "could not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Clean text but preserve important words
        text = re.sub(r'[^a-zA-Z\s_]', '', text)
        
        # Handle single words - add context
        words = text.split()
        if len(words) == 1:
            word = words[0]
            # Add minimal context for single positive/negative words
            if word in ['excellent', 'amazing', 'fantastic', 'wonderful', 'great', 'good']:
                text = f"this is {word}"
            elif word in ['terrible', 'awful', 'horrible', 'bad', 'poor']:
                text = f"this is {word}"
        
        return text
    
    def predict_sentiment(self, text: str) -> Dict[str, Any]:
        """Predict sentiment for a single text with improved logic."""
        
        # Use trained model if available
        if self.model is not None and self.tokenizer is not None:
            try:
                import tensorflow as tf
                
                # Improved preprocessing
                processed_text = self.preprocess_text_improved(text)
                
                # Handle special negation signals
                if 'POSITIVE_SIGNAL' in processed_text:
                    return {
                        "text": text,
                        "sentiment": "positive",
                        "confidence": 0.75,
                        "model_used": "negation_rule"
                    }
                elif 'NEGATIVE_SIGNAL' in processed_text:
                    return {
                        "text": text,
                        "sentiment": "negative", 
                        "confidence": 0.75,
                        "model_used": "negation_rule"
                    }
                
                # Tokenize with the trained tokenizer
                sequences = self.tokenizer.texts_to_sequences([processed_text])
                
                # If tokenization fails, try original text
                if not sequences[0]:
                    sequences = self.tokenizer.texts_to_sequences([text.lower()])
                
                # If still empty, use fallback
                if not sequences[0]:
                    sequences = [[1]]  # Use unknown token
                
                padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=self.max_length)
                
                # Make prediction
                prediction = self.model.predict(padded, verbose=0)[0][0]
                
                # Dynamic threshold based on text characteristics
                threshold = self.get_dynamic_threshold(text, processed_text)
                
                # Convert to sentiment and confidence
                if prediction > threshold:
                    sentiment = "positive"
                    confidence = float(prediction)
                else:
                    sentiment = "negative"
                    confidence = float(1 - prediction)
                
                return {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": confidence,
                    "model_used": "lstm_improved"
                }
                
            except Exception as e:
                logger.error(f"Error using trained model: {str(e)}")
                # Fall back to rule-based approach
        
        # Enhanced rule-based fallback
        return self.rule_based_prediction(text)
    
    def get_dynamic_threshold(self, original_text: str, processed_text: str) -> float:
        """
        Get dynamic threshold based on text characteristics
        """
        words = original_text.lower().split()
        
        # Single word cases - use lower threshold for positive words
        if len(words) == 1:
            positive_words = ['excellent', 'amazing', 'fantastic', 'wonderful', 'great', 'good', 'love', 'perfect']
            negative_words = ['terrible', 'awful', 'horrible', 'bad', 'hate', 'poor', 'worst']
            
            if words[0] in positive_words:
                return 0.3  # Lower threshold for single positive words
            elif words[0] in negative_words:
                return 0.6  # Higher threshold for single negative words
        
        # Default threshold for multi-word texts
        return 0.4
    
    def rule_based_prediction(self, text: str) -> Dict[str, Any]:
        """Enhanced rule-based prediction as fallback"""
        
        # Handle negations first
        text_lower = text.lower()
        
        # Specific negation patterns
        if re.search(r'\bnot\s+(bad|terrible|awful|horrible)', text_lower):
            return {
                "text": text,
                "sentiment": "positive",
                "confidence": 0.7,
                "model_used": "rule_based_negation"
            }
        
        if re.search(r'\bnot\s+(good|great|excellent|amazing)', text_lower):
            return {
                "text": text,
                "sentiment": "negative", 
                "confidence": 0.7,
                "model_used": "rule_based_negation"
            }
        
        # Count positive and negative words
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'happy', 'best', 'fantastic', 'wonderful', 'awesome', 'enjoy', 'nice', 'perfect', 'fun', 'favorite', 'liked', 'beautiful', 'recommend', 'worth', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'poor', 'disappointing', 'disappointed', 'negative', 'boring', 'waste', 'difficult', 'problem', 'issue', 'expensive', 'wrong', 'slow', 'annoying', 'useless']
        
        words = text_lower.split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.5 + (positive_count - negative_count) * 0.1, 0.9)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.5 + (negative_count - positive_count) * 0.1, 0.9)
        else:
            # Default to positive for neutral cases (common in review data)
            sentiment = "positive"
            confidence = 0.5
        
        return {
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "model_used": "rule_based"
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict sentiment for a batch of texts."""
        results = []
        for text in texts:
            result = self.predict_sentiment(text)
            results.append(result)
        return results

# Create a singleton instance
model_service_fixed = SentimentModelFixed()
