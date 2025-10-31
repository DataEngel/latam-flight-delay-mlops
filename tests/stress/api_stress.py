from locust import HttpUser, task, between

class StressUser(HttpUser):

    wait_time = between(0.1, 0.5)
    
    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict",
            json={
                "OPERA": "Aerolineas Argentinas",
                "TIPOVUELO": "N",
                "MES": 3
            }
        )


    @task
    def predict_latam(self):
        self.client.post(
            "/predict",
            json={
                "OPERA": "Grupo LATAM",
                "TIPOVUELO": "N",
                "MES": 3
            }
        )
