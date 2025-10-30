#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LATAM Airlines — Flight Delay Prediction API
Deployable on Cloud Run, serving an XGBoost model trained via DelayModel.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# ==========================================================
# Logging Configuration
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================================
# Model Loading
# ==========================================================
MODEL_PATH = "/app/models/xgb_model.pkl"
feature_names = []

try:
    xgb_model = joblib.load(MODEL_PATH)
    feature_names = xgb_model.feature_names_in_
    logger.info(f"✅ Model successfully loaded from {MODEL_PATH}")
    logger.info(f"🔍 Model expects {len(feature_names)} features.")
except Exception as e:
    xgb_model = None
    logger.error(f"❌ Failed to load model from {MODEL_PATH}: {e}")

# ==========================================================
# FastAPI App Initialization
# ==========================================================
app = FastAPI(
    title="LATAM Flight Delay Prediction API",
    description="Predicts flight delay probability based on OPERA, MES, and TIPOVUELO.",
    version="1.0.0"
)

# ==========================================================
# Request Schema
# ==========================================================
class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str

# ==========================================================
# Validation Parameters
# ==========================================================
VALID_OPERAS = {
    "Grupo LATAM",
    "Aerolineas Argentinas",
    "Sky Airline",
    "Copa Air",
    "Latin American Wings"
}
VALID_TIPOVUELOS = {"N", "I"}
VALID_MESES = set(range(1, 13))

# ==========================================================
# Health Check Endpoint
# ==========================================================
@app.get("/health", status_code=200)
def health_check():
    """Simple health check to verify API is running."""
    return {"status": "ok"}

# ==========================================================
# Prediction Endpoint
# ==========================================================
@app.post("/predict", status_code=200)
def predict_delay(flight_data: FlightData):
    """Receives flight data and returns a binary delay prediction."""

    if xgb_model is None:
        logger.error("❌ Model not loaded. Ensure xgb_model.pkl is present.")
        raise HTTPException(status_code=500, detail="Model not available")

    logger.debug(f"📥 Incoming JSON: {flight_data.dict()}")

    # --- Input Validation ---
    if flight_data.OPERA not in VALID_OPERAS:
        logger.warning(f"❌ Invalid airline: {flight_data.OPERA}")
        raise HTTPException(status_code=400, detail="Invalid airline (OPERA)")

    if flight_data.TIPOVUELO not in VALID_TIPOVUELOS:
        logger.warning(f"❌ Invalid flight type: {flight_data.TIPOVUELO}")
        raise HTTPException(status_code=400, detail="Invalid flight type (TIPOVUELO)")

    if flight_data.MES not in VALID_MESES:
        logger.warning(f"❌ Invalid month: {flight_data.MES}")
        raise HTTPException(status_code=400, detail="Invalid month (MES)")

    try:
        # --- Feature Preparation ---
        df = pd.DataFrame([flight_data.dict()])
        df = pd.get_dummies(df, columns=["OPERA", "TIPOVUELO", "MES"], prefix=["OPERA", "TIPOVUELO", "MES"])

        # Ensure all expected features exist
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Align column order
        df = df.reindex(columns=feature_names, fill_value=0)

        logger.debug(f"📊 DataFrame aligned with model features. Shape: {df.shape}")

        # --- Prediction ---
        prediction = xgb_model.predict(df)

        if len(prediction) == 0:
            logger.error("❌ Empty prediction result from model.")
            raise HTTPException(status_code=500, detail="Empty prediction result")

        result = int(prediction[0])
        logger.info(f"✅ Prediction completed successfully: {result}")

        return {
            "delay_prediction": result,
            "details": {
                "airline": flight_data.OPERA,
                "month": flight_data.MES,
                "flight_type": flight_data.TIPOVUELO
            }
        }

    except Exception as e:
        logger.error(f"❌ Error during prediction process: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")

# ==========================================================
# Entry Point for Cloud Run
# ==========================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
