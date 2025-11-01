import importlib

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def api_module(monkeypatch):
    """El módulo de la API es recargado con el modo simulado habilitado para garantizar determinismo."""
    monkeypatch.setenv("CHALLENGE_API_FAKE_MODEL", "1")
    monkeypatch.setenv("CHALLENGE_API_DISABLE_GCP", "1")
    monkeypatch.setenv("CHALLENGE_API_ENABLE_BQ", "0")

    from challenge.api import api

    importlib.reload(api)
    return api


@pytest.fixture()
def client(api_module):
    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_endpoint_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_delay_details(client):
    payload = {"OPERA": "Grupo LATAM", "MES": 5, "TIPOVUELO": "N"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["delay_prediction"] == 0
    assert body["details"] == {
        "airline": "Grupo LATAM",
        "month": 5,
        "flight_type": "N",
    }


def test_predict_rejects_invalid_airline(client):
    payload = {"OPERA": "Unknown Airline", "MES": 5, "TIPOVUELO": "N"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid airline (OPERA)"


def test_predict_batch_request(client):
    payload = {
        "flights": [
            {"OPERA": "Grupo LATAM", "MES": 1, "TIPOVUELO": "N"},
            {"OPERA": "Sky Airline", "MES": 12, "TIPOVUELO": "I"},
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"predict": [0, 0]}
