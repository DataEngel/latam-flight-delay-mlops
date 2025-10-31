#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flight delay prediction API for LATAM Airlines."""

from __future__ import annotations

import io
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Union

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "latam-challenge-storage")
GCS_MODEL_BLOB_PATH = os.getenv("GCS_MODEL_BLOB_PATH", "latam-model/xgb_model.pkl")
BQ_TABLE_ID = os.getenv(
    "BQ_TABLE_ID", "mlops-latam.latam_model_results.table_preds_model_latam"
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "xgb_model.pkl"
MODEL_LOCAL_PATH = Path(os.getenv("MODEL_LOCAL_PATH", DEFAULT_MODEL_PATH))
DISABLE_GCP = _env_flag("CHALLENGE_API_DISABLE_GCP", False)
ENABLE_BIGQUERY = _env_flag("CHALLENGE_API_ENABLE_BQ", False)
FAKE_MODEL_MODE = _env_flag("CHALLENGE_API_FAKE_MODEL", False)

if not FAKE_MODEL_MODE:
    import pandas as pd
    from google.api_core.exceptions import NotFound
    from google.cloud import bigquery, storage
else:  # pragma: no cover - optional dependencies are unnecessary under fake mode
    pd = None  # type: ignore
    NotFound = Exception  # type: ignore
    bigquery = storage = None  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


xgb_model = None
feature_names: List[str] = []
bq_client = None


def _extract_feature_names(model) -> List[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names.tolist() if hasattr(names, "tolist") else names)

    booster = getattr(model, "get_booster", lambda: None)()
    booster_names = getattr(booster, "feature_names", None)
    if booster_names:
        return list(booster_names)

    return []


def _load_local_model(path: Path):
    try:
        from joblib import load as joblib_load  # type: ignore

        logger.info("Loading model from local artifact: %s", path)
        return joblib_load(path)
    except Exception as exc:
        logger.debug("joblib.load failed (%s); falling back to pickle.", exc)
        with path.open("rb") as handler:
            return pickle.load(handler)


def initialize_model() -> None:
    global xgb_model, feature_names

    if MODEL_LOCAL_PATH.exists() and not FAKE_MODEL_MODE:
        try:
            model = _load_local_model(MODEL_LOCAL_PATH)
            xgb_model = model
            feature_names[:] = _extract_feature_names(model)
            logger.info("Model loaded from local artifact (%d features).", len(feature_names))
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            xgb_model = None
            feature_names = []
            logger.error("Local model loading failed: %s", exc, exc_info=True)

    if DISABLE_GCP or FAKE_MODEL_MODE:
        if not FAKE_MODEL_MODE:
            logger.warning(
                "GCP integrations disabled; remote model loading skipped and no local artifact available."
            )
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_MODEL_BLOB_PATH)
        model_bytes = blob.download_as_bytes()

        try:
            from joblib import load as joblib_load  # type: ignore

            model = joblib_load(io.BytesIO(model_bytes))
        except Exception:
            model = pickle.loads(model_bytes)

        xgb_model = model
        feature_names[:] = _extract_feature_names(model)
        logger.info(
            "Model loaded from gs://%s/%s (%d features).",
            GCS_BUCKET_NAME,
            GCS_MODEL_BLOB_PATH,
            len(feature_names),
        )
    except Exception as exc:  # pragma: no cover - depends on remote services
        xgb_model = None
        feature_names = []
        logger.error("Remote model loading failed: %s", exc, exc_info=True)


def initialize_bigquery() -> None:
    global bq_client

    if not ENABLE_BIGQUERY or FAKE_MODEL_MODE:
        bq_client = None
        logger.info("BigQuery logging disabled via configuration.")
        return

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
            logger.info("BigQuery table %s already exists.", BQ_TABLE_ID)
        except NotFound:
            table = bigquery.Table(BQ_TABLE_ID, schema=schema)
            bq_client.create_table(table)
            logger.info("BigQuery table %s created.", BQ_TABLE_ID)
    except Exception as exc:  # pragma: no cover - depends on remote services
        bq_client = None
        logger.error("BigQuery initialisation failed: %s", exc, exc_info=True)


def log_prediction_to_bigquery(flight_data: "FlightData", prediction: int) -> None:
    if bq_client is None or FAKE_MODEL_MODE:
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
            logger.error("Failed to log prediction to BigQuery: %s", errors)
    except Exception as exc:  # pragma: no cover - depends on remote services
        logger.error("Unexpected BigQuery error: %s", exc, exc_info=True)


class _FakeModel:
    def predict(self, flights: Sequence["FlightData"]) -> List[int]:
        return [0 for _ in flights]


def _initialize_fake_model() -> None:
    global xgb_model, feature_names
    feature_names[:] = []
    xgb_model = _FakeModel()
    logger.info("Fake model initialised for test mode.")


if FAKE_MODEL_MODE:
    _initialize_fake_model()
else:
    initialize_model()
    initialize_bigquery()


app = FastAPI(
    title="LATAM Flight Delay Prediction API",
    description="Predicts flight delay probability based on OPERA, MES and TIPOVUELO.",
    version="1.0.0",
)


class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str


class BatchRequest(BaseModel):
    flights: List[FlightData]


VALID_OPERAS = {
    "Grupo LATAM",
    "Aerolineas Argentinas",
    "Sky Airline",
    "Copa Air",
    "Latin American Wings",
}
VALID_TIPOVUELOS = {"N", "I"}
VALID_MESES = set(range(1, 13))


def _validate_flight(flight: FlightData) -> None:
    if flight.OPERA not in VALID_OPERAS:
        raise HTTPException(status_code=400, detail="Invalid airline (OPERA)")
    if flight.TIPOVUELO not in VALID_TIPOVUELOS:
        raise HTTPException(status_code=400, detail="Invalid flight type (TIPOVUELO)")
    if flight.MES not in VALID_MESES:
        raise HTTPException(status_code=400, detail="Invalid month (MES)")


def _build_features(flights: Sequence[FlightData]):
    if FAKE_MODEL_MODE:
        return flights

    if xgb_model is None or not feature_names:
        raise HTTPException(status_code=500, detail="Model not available")

    payload = [flight.dict() for flight in flights]
    df = pd.DataFrame(payload)
    df = pd.get_dummies(
        df,
        columns=["OPERA", "TIPOVUELO", "MES"],
        prefix=["OPERA", "TIPOVUELO", "MES"],
    )

    for column in feature_names:
        if column not in df.columns:
            df[column] = 0

    df = df.reindex(columns=feature_names, fill_value=0)
    return df


@app.get("/health", status_code=200)
def health_check():
    return {"status": "ok"}


@app.post("/predict", status_code=200)
def predict_delay(payload: Union[BatchRequest, FlightData]):
    flights = payload.flights if isinstance(payload, BatchRequest) else [payload]

    for flight in flights:
        _validate_flight(flight)

    if FAKE_MODEL_MODE:
        predictions = [0 for _ in flights]
    else:
        features_df = _build_features(flights)
        try:
            raw_predictions = xgb_model.predict(features_df)
        except Exception as exc:
            logger.error("Model inference failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal prediction error") from exc
        predictions = [int(value) for value in raw_predictions.tolist()]

    if isinstance(payload, BatchRequest):
        return {"predict": predictions}

    result = predictions[0]
    log_prediction_to_bigquery(payload, result)
    return {
        "delay_prediction": result,
        "details": {
            "airline": payload.OPERA,
            "month": payload.MES,
            "flight_type": payload.TIPOVUELO,
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual execution
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
