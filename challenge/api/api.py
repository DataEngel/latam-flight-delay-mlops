#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LATAM Airlines — Flight Delay Prediction API
Deployable on Cloud Run, serving an XGBoost model trained via DelayModel.
"""

import io
import os
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
from google.api_core.exceptions import NotFound
from google.cloud import bigquery, storage

# ==========================================================
# Logging Configuration
# ==========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ==========================================================
# Model & BigQuery Configuration
# ==========================================================
GCS_BUCKET_NAME = "latam-challenge-storage"
GCS_MODEL_BLOB_PATH = "latam-model/xgb_model.pkl"
BQ_TABLE_ID = "mlops-latam.latam_model_results.table_preds_model_latam"

xgb_model = None
feature_names = []
bq_client = None


def initialize_model() -> None:
    """Load XGBoost model from Google Cloud Storage."""
    global xgb_model, feature_names
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_MODEL_BLOB_PATH)
        model_bytes = blob.download_as_bytes()
        xgb_model = joblib.load(io.BytesIO(model_bytes))
        feature_names = getattr(xgb_model, "feature_names_in_", [])
        logger.info(
            f"✅ Model successfully loaded from gs://{GCS_BUCKET_NAME}/{GCS_MODEL_BLOB_PATH}"
        )
        logger.info(f"🔍 Model expects {len(feature_names)} features.")
    except Exception as exc:
        xgb_model = None
        feature_names = []
        logger.error(
            "❌ Failed to load model from Google Cloud Storage: %s", exc, exc_info=True
        )


def initialize_bigquery() -> None:
    """Ensure the BigQuery table exists for logging predictions."""
    global bq_client
    try:
        bq_client = bigquery.Client()
        schema = [
            bigquery.SchemaField("prediction_timestamp", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("airline", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("month", "INTEGER", mode="REQUIRED"),
            bigquery.SchemaField("flight_type", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("delay_prediction", "INTEGER", mode="REQUIRED"),
        ]
        try:
            bq_client.get_table(BQ_TABLE_ID)
            logger.info(f"ℹ️ BigQuery table {BQ_TABLE_ID} is available.")
        except NotFound:
            table = bigquery.Table(BQ_TABLE_ID, schema=schema)
            bq_client.create_table(table)
            logger.info(f"✅ BigQuery table {BQ_TABLE_ID} created.")
    except Exception as exc:
        bq_client = None
        logger.error(
            "❌ Failed to initialize BigQuery client or ensure table: %s",
            exc,
            exc_info=True,
        )


def log_prediction_to_bigquery(flight_data: "FlightData", prediction: int) -> None:
    """
    Persist prediction metadata into BigQuery.

    The function is intentionally best-effort: failures are logged but
    do not propagate to the API response.
    """
    if bq_client is None:
        logger.warning("⚠️ BigQuery client not available. Skipping logging.")
        return

    row = {
        "prediction_timestamp": datetime.utcnow().isoformat() + "Z",
        "airline": flight_data.OPERA,
        "month": int(flight_data.MES),
        "flight_type": flight_data.TIPOVUELO,
        "delay_prediction": int(prediction),
    }
    try:
        errors = bq_client.insert_rows_json(BQ_TABLE_ID, [row])
        if errors:
            logger.error("❌ Failed to log prediction to BigQuery: %s", errors)
        else:
            logger.info("📝 Prediction stored in BigQuery.")
    except Exception as exc:
        logger.error(
            "❌ Unexpected error while logging to BigQuery: %s", exc, exc_info=True
        )


# Initialize external dependencies at import time
initialize_model()
initialize_bigquery()

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
        log_prediction_to_bigquery(flight_data, result)

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
