import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU

import logging
import numpy as np
import mne
from tensorflow import keras
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import pywt

# Configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()



# Load the trained model, label encoder, and scaler
try:
    model = keras.models.load_model('student_model.h5')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    logger.info("Model and components loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model or components: {e}")
    raise  # Raise the exception to stop the application

# Mapping of predicted labels to their meanings
LABEL_MEANINGS = {
    "A": "Alzheimer's Disease",
    "F": "Frontotemporal Dementia",
    "C": "Cognitive Normal/Healthy"
}

def preprocess_eeg_data(file_path):
    """
    Preprocesses the uploaded EEG data to match the training data format.
    """
    try:
        # Read the EEG data
        raw = mne.io.read_raw_eeglab(file_path, preload=True)
        
        # Resample to 128 Hz (matching the training data)
        raw.resample(128)
        
        # Apply bandpass filter (0.5 Hz to 50 Hz)
        raw.filter(0.5, 50, fir_design='firwin')
        
        # Get the data
        data = raw.get_data()
        
        # Ensure the data has 1280 time points
        if data.shape[1] < 1280:
            data = np.pad(data, ((0, 0), (0, 1280 - data.shape[1])), 'constant')
        elif data.shape[1] > 1280:
            data = data[:, :1280]
        
        # Transpose the data to match the model's input shape
        data = data.transpose(1, 0)  # Shape: (1280, 19)
        
        # Apply wavelet denoising
        data_denoised = np.array([wavelet_denoising(signal.squeeze()) for signal in data[np.newaxis, ...]])
        data_denoised = data_denoised[..., np.newaxis]  # Add channel dimension
        
        return data_denoised
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing EEG data: {str(e)}")

def wavelet_denoising(signal, wavelet='db4', level=1):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[1:] = [pywt.threshold(c, value=0.1, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

@app.post("/predict/")
async def predict_eeg(gender: str = Form(...), age: int = Form(...), mmse: int = Form(...), file: UploadFile = File(...)):
    try:
        logger.info("Received prediction request")
        
        # Save the uploaded file temporarily
        temp_file_path = "temp_eeg.set"
        logger.info(f"Saving uploaded file to: {temp_file_path}")
        
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Uploaded file saved successfully. File size: {len(content)} bytes")

        # Verify the file exists and is not empty
        if not os.path.exists(temp_file_path):
            logger.error("Temporary file was not created.")
            raise HTTPException(status_code=500, detail="Temporary file was not created.")
        
        if os.path.getsize(temp_file_path) == 0:
            logger.error("Temporary file is empty.")
            raise HTTPException(status_code=500, detail="Temporary file is empty.")

        # Preprocess the EEG data
        logger.info("Preprocessing EEG data...")
        eeg_data_denoised = preprocess_eeg_data(temp_file_path)
        logger.info("EEG data preprocessed successfully")

        # Prepare metadata
        logger.info("Preparing metadata...")
        metadata = np.array([[gender, age, mmse]])
        metadata[:, 0] = np.where(metadata[:, 0] == 'F', 0, 1)  # Encode gender (F: 0, M: 1)
        metadata = metadata.astype('float32')
        metadata = scaler.transform(metadata)
        logger.info("Metadata processed successfully")

        # Make the prediction
        logger.info("Making prediction...")
        prediction_proba = model.predict([eeg_data_denoised, metadata])
        prediction = np.argmax(prediction_proba, axis=1)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        logger.info(f"Prediction made: {predicted_label}")

        # Map the predicted label to its meaning
        prediction_meaning = LABEL_MEANINGS.get(predicted_label, "Unknown")

        # Clean up the temporary file
        logger.info("Cleaning up temporary file...")
        os.remove(temp_file_path)
        logger.info("Temporary file removed")

        return {"prediction": predicted_label, "prediction_meaning": prediction_meaning}

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        os.remove(temp_file_path) if os.path.exists(temp_file_path) else None
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("index.html", "r") as f:
        return f.read()