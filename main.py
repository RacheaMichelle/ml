from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
import tensorflow as tf
import numpy as np
from fastapi.responses import JSONResponse
import joblib
from typing import List
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Load the trained model and preprocessing objects
try:
    model = tf.keras.models.load_model("student_model.h5", compile=False)
    label_encoder = joblib.load("label_encoder.pkl")
    metadata_scaler = joblib.load("scaler.pkl")  # scaler for metadata
except FileNotFoundError as e:
    print(f"Error: Required model or preprocessing file not found: {e}")

app = FastAPI()

templates = Jinja2Templates(directory=".")

class EEGData(BaseModel):
    eeg_signal: List[float] = Field(description="List of 1280 EEG signal values.")
    metadata: List[int] = Field(description="Metadata: [Gender (0 for Female, 1 for Male), Age, MMSE].")

    @field_validator('eeg_signal')
    @classmethod
    def validate_eeg_signal_length(cls, value):
        if len(value) != 1280:
            raise ValueError('eeg_signal must have exactly 1280 values')
        return value

@app.post("/predict/", description="Predicts a neurological condition based on generated EEG data.")
async def predict(data: EEGData):
    try:
        print("Received metadata:", data.metadata)  # Add logging
        # Scale only metadata
        metadata_scaled = metadata_scaler.transform(np.array([data.metadata]))[0]
        # Reshape eeg_signal to match model input
        eeg_input = np.array(data.eeg_signal).reshape(1, 1280, 1)
        # Create metadata input
        metadata_input = np.array(metadata_scaled).reshape(1, -1)
        # Make predictions
        predictions = model.predict([eeg_input, metadata_input])
        predicted_class = np.argmax(predictions, axis=1)
        class_label = label_encoder.inverse_transform(predicted_class)[0]
        return {"predicted_class": class_label, "probabilities": predictions.tolist()}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.mount("/static", StaticFiles(directory="."), name="static")