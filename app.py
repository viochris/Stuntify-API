from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
import os 

# Initialize the FastAPI application with metadata
app = FastAPI(title="Stunting AI", version="1.0.1")

# --- CORS CONFIGURATION ---
# This middleware is crucial for allowing Frontend applications (like React, Vue, or simple HTML)
# to communicate with this backend API without getting blocked by the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allows all origins (domains) to access this API
    allow_credentials=True,
    allow_methods=["*"],          # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],          # Allows all headers (Authentication, Content-Type, etc.)
)

# --- DATA VALIDATION MODEL (PYDANTIC) ---
# This class defines the expected structure of the input JSON data.
# FastAPI will automatically validate incoming requests against this schema.
class ConditionInput(BaseModel):
    jenis_kelamin: Optional[str] = "Laki-laki"  # Input for Gender (String)
    umur: int  = 19                             # Input for Age (Integer)
    tinggi: float = 91.60                       # Input for Height (Float)
    berat: float = 13.30                        # Input for Weight (Float)

# --- GLOBAL VARIABLES ---
# These variables act as placeholders for the Machine Learning models and scalers.
# They are set to None initially and will be filled when 'load_models()' is called.
model = None
scaler = None
jk_encoder = None
stunting_encoder = None

# --- MODEL LOADING FUNCTION ---
# This function handles the loading of .joblib files (the "brains" of the AI).
# It uses the 'global' keyword to modify the variables defined above.
def load_models():
    global model, scaler, jk_encoder, stunting_encoder
    try:
        # Check if the model is currently empty (None). If so, load it.
        # This prevents reloading the model on every single request (Efficiency).
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        if model is None: 
            print("Mencoba memuat model untuk PERTAMA KALI...")
            
            # Loading the main classifier model
            model = joblib.load(os.path.join(BASE_DIR,"best_model.joblib"))
            
            # Loading the scaler (used to normalize numerical inputs like age, height, weight)
            scaler = joblib.load(os.path.join(BASE_DIR,"scaler.joblib"))
            
            # Loading the encoder for Gender (converts "Laki-laki" to numbers)
            jk_encoder = joblib.load(os.path.join(BASE_DIR,"Jenis Kelamin_encoder.joblib"))
            
            # Loading the encoder for the Target (converts prediction numbers back to "Stunting/Normal")
            stunting_encoder = joblib.load(os.path.join(BASE_DIR,"Stunting_encoder.joblib"))
            
            print("Semua 4 model berhasil dimuat!")
        return True
    except Exception as e:
        # If loading fails (e.g., file not found), print the error and return False
        print(f"Error saat memuat model: {e}")
        return False

# --- HOME ENDPOINT (GET) ---
# This is the entry point of the API. It provides status, documentation links,
# and a guide on how to use the variables (Variable Translation).
@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Stuntify AI API is ready to use",
        "version": "1.0.1",
        "documentation": "/docs",
        "usage_guide": {
            "endpoint": "/predict-stunting",
            "method": "POST",
            "body_format": "JSON",
            "variable_translation": {
                "jenis_kelamin": "Gender (Value: 'Laki-laki' for Male, 'Perempuan' for Female)",
                "umur": "Age (Numeric, typically in months)",
                "tinggi": "Height (Numeric, in cm)",
                "berat": "Weight (Numeric, in kg)"
            }
        },
        "author": "Silvio Christian, Joe"
    }

# --- PREDICTION ENDPOINT (POST) ---
# This is where the actual AI processing happens.
# It accepts JSON data matching the 'ConditionInput' schema.
@app.post("/predict-stunting")
def predict(data: ConditionInput):
    global model, scaler, jk_encoder, stunting_encoder

    # Ensure models are loaded before processing. If loading fails, raise a 500 error.
    if not load_models():
        raise HTTPException(status_code=500, detail="Model gagal dimuat di server. Cek logs.")

    try:
        # 1. Extract data from the Pydantic input model
        jk_string = data.jenis_kelamin
        umur = data.umur
        tinggi = data.tinggi
        berat = data.berat

        # 2. Preprocessing: Encode the Gender string into a number
        # transform() expects a 2D array or list, hence the brackets.
        jk_encoded = jk_encoder.transform([jk_string])[0]

        # 3. Preprocessing: Scale the numerical features (Age, Height, Weight)
        # The scaler expects a 2D array: [[age, height, weight]]
        numerical_features = [[umur, tinggi, berat]]
        scaled_features = scaler.transform(numerical_features)
        
        # Extract the scaled values
        umur_scaled = scaled_features[0][0]
        tinggi_scaled = scaled_features[0][1]
        berat_scaled = scaled_features[0][2]

        # 4. Feature Combination: Combine encoded gender and scaled numerics
        # into a single array formatted for the model.
        final_features_list = [jk_encoded, umur_scaled, tinggi_scaled, berat_scaled]
        final_features = [np.array(final_features_list)]
        
        # 5. Prediction: Ask the model to predict based on the processed features
        prediction_encoded = model.predict(final_features)
        
        # 6. Decoding: Convert the numerical prediction back to a readable string (e.g., "Severely Stunted")
        prediction_string = stunting_encoder.inverse_transform(prediction_encoded)

        # Get the first item of the result
        output = prediction_string[0]
        
        # Return the final result as a JSON response
        return {'prediction': output}

    except KeyError as e:
        # Handle cases where specific keys might be missing (though Pydantic handles most of this)
        raise HTTPException(status_code=400, detail="Key JSON tidak ditemukan: {str(e)}.")
    except Exception as e:
        # Catch-all for any other errors during the prediction process (e.g., math errors)
        raise HTTPException(status_code=400, detail="Terjadi error saat prediksi: {str(e)}")