from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import os
import tempfile
import asyncio
from typing import List, Dict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Audio-to-Text and 3D Coordinates API",
    description="FastAPI backend for SignAI, audio transcription and text-to-3D coordinates prediction",
    version="0.1.0-beta"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"

if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
    openai.api_type = "azure"
    openai.api_key = AZURE_OPENAI_API_KEY
    openai.api_base = AZURE_OPENAI_ENDPOINT
    openai.api_version = AZURE_OPENAI_API_VERSION
else:
    logger.warning("Azure OpenAI credentials not found in environment variables")

class TextInput(BaseModel):
    text: str

class CoordinateResponse(BaseModel):
    x: float
    y: float
    z: float
    confidence: float

class TranscriptionResponse(BaseModel):
    text: str
    timestamp: str

class CombinedResponse(BaseModel):
    transcribed_text: str
    coordinates: CoordinateResponse
    timestamp: str

class XGBoostCoordinatePredictor:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        
    def create_sample_data(self):
        sample_data = [
            ("move forward", 0, 0, 1),
            ("move backward", 0, 0, -1),
            ("move left", -1, 0, 0),
            ("move right", 1, 0, 0),
            ("move up", 0, 1, 0),
            ("move down", 0, -1, 0),
            ("jump", 0, 2, 0),
            ("crouch", 0, -0.5, 0),
            ("turn left", -0.5, 0, 0),
            ("turn right", 0.5, 0, 0),
            ("stop", 0, 0, 0),
            ("run forward", 0, 0, 2),
            ("walk slowly", 0, 0, 0.5),
            ("step back", 0, 0, -0.5),
            ("rise up", 0, 1.5, 0),
            ("go down", 0, -1.5, 0),
            ("shift left", -2, 0, 0),
            ("shift right", 2, 0, 0),
            ("leap forward", 0, 1, 2),
            ("fall down", 0, -2, 0),
        ]
        
        texts = [item[0] for item in sample_data]
        coordinates = [(item[1], item[2], item[3]) for item in sample_data]
        
        return texts, coordinates
    
    def vectorize_text(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            return self.vectorizer.fit_transform(texts)
        else:
            return self.vectorizer.transform(texts)
    
    def train_model(self):
        try:
            texts, coordinates = self.create_sample_data()
            
            X = self.vectorize_text(texts).toarray()
            
            y_x = [coord[0] for coord in coordinates]
            y_y = [coord[1] for coord in coordinates]
            y_z = [coord[2] for coord in coordinates]
            
            self.model_x = xgb.XGBRegressor(n_estimators=100, random_state=42)
            self.model_y = xgb.XGBRegressor(n_estimators=100, random_state=42)
            self.model_z = xgb.XGBRegressor(n_estimators=100, random_state=42)
            
            self.model_x.fit(X, y_x)
            self.model_y.fit(X, y_y)
            self.model_z.fit(X, y_z)
            
            self.is_trained = True
            logger.info("XGBoost models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {str(e)}")
            raise e
    
    def predict_coordinates(self, text: str):
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            X = self.vectorize_text([text]).toarray()
            
            x_pred = self.model_x.predict(X)[0]
            y_pred = self.model_y.predict(X)[0]
            z_pred = self.model_z.predict(X)[0]
            
            confidence = min(1.0, max(0.0, 1.0 - (abs(x_pred) + abs(y_pred) + abs(z_pred)) / 10))
            
            return CoordinateResponse(
                x=float(x_pred),
                y=float(y_pred),
                z=float(z_pred),
                confidence=float(confidence)
            )
            
        except Exception as e:
            logger.error(f"Error predicting coordinates: {str(e)}")
            raise e

coordinate_predictor = XGBoostCoordinatePredictor()

@app.on_event("startup")
async def startup_event():
    try:
        coordinate_predictor.train_model()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Audio-to-Text and 3D Coordinates API",
        "version": "1.0.0",
        "endpoints": {
            "POST /transcribe": "Convert audio to text using Azure OpenAI",
            "POST /predict-coordinates": "Predict 3D coordinates from text",
            "POST /audio-to-coordinates": "Combined audio transcription and coordinate prediction"
        }
    }

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio_file: UploadFile = File(...)):
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_ENDPOINT:
        raise HTTPException(
            status_code=500, 
            detail="Azure OpenAI credentials not configured"
        )
    
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400, 
                detail="File must be an audio file"
            )
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            with open(temp_file_path, "rb") as audio:
                response = openai.Audio.transcribe(
                    model="whisper-1",
                    file=audio,
                    response_format="text"
                )
            
            transcribed_text = response if isinstance(response, str) else response.get("text", "")
            
            return TranscriptionResponse(
                text=transcribed_text,
                timestamp=datetime.utcnow().isoformat()
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/predict-coordinates", response_model=CoordinateResponse)
async def predict_coordinates(text_input: TextInput):
    try:
        if not text_input.text.strip():
            raise HTTPException(status_code=400, detail="Text input cannot be empty")
        
        coordinates = coordinate_predictor.predict_coordinates(text_input.text)
        return coordinates
        
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting coordinates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Coordinate prediction failed: {str(e)}")

@app.post("/audio-to-coordinates", response_model=CombinedResponse)
async def audio_to_coordinates(audio_file: UploadFile = File(...)):
    try:
        transcription_response = await transcribe_audio(audio_file)
        text_input = TextInput(text=transcription_response.text)
        coordinates = coordinate_predictor.predict_coordinates(text_input.text)
        
        return CombinedResponse(
            transcribed_text=transcription_response.text,
            coordinates=coordinates,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in combined audio-to-coordinates: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Combined processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "xgboost_trained": coordinate_predictor.is_trained,
            "azure_openai_configured": bool(AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)