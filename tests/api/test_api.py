import os
import unittest

import importlib
from fastapi.testclient import TestClient


class TestBatchPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        os.environ["CHALLENGE_API_FAKE_MODEL"] = "1"
        os.environ["ANYIO_BACKEND"] = "asyncio"
        importlib.reload(__import__("challenge.api.api"))
        from challenge import app
        cls._app = app if app is not None else importlib.import_module("challenge.api.api").app

    def setUp(self):
        self.client = TestClient(self._app)
        
    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N", 
                    "MES": 3
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})
    

    def test_should_failed_unkown_column_1(self):
        data = {       
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "N",
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
    
    def test_should_failed_unkown_column_3(self):
        data = {        
            "flights": [
                {
                    "OPERA": "Argentinas", 
                    "TIPOVUELO": "O", 
                    "MES": 13
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
