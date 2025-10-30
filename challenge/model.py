import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
import logging

# Se configura el registro de logs para el seguimiento del proceso.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class DelayModel:
    """
    Clase encargada del modelado y predicción de retrasos de vuelos en el aeropuerto SCL.
    El código fue estructurado a partir del notebook del Data Scientist y adaptado para su uso en producción.
    """

    def __init__(self):
        """Se inicializa la clase con los atributos del modelo y las columnas de características."""
        self._model = None
        self._feature_columns = None

    # ==============================================================
    # PREPROCESAMIENTO
    # ==============================================================
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Se preparan los datos crudos para entrenamiento o inferencia.

        Args:
            data (pd.DataFrame): conjunto de datos de entrada.
            target_column (str, opcional): nombre de la columna objetivo si se dispone.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: características y variable objetivo si target_column está definido.
            pd.DataFrame: únicamente características si no se especifica variable objetivo.
        """

        def get_period_day(date: str) -> str:
            """Determina la franja horaria correspondiente a la fecha programada."""
            try:
                date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
            except Exception:
                return "noche"

            if datetime.strptime("05:00", "%H:%M").time() <= date_time <= datetime.strptime("11:59", "%H:%M").time():
                return "mañana"
            elif datetime.strptime("12:00", "%H:%M").time() <= date_time <= datetime.strptime("18:59", "%H:%M").time():
                return "tarde"
            else:
                return "noche"

        def is_high_season(fecha: str) -> int:
            """Identifica si la fecha corresponde a un período de alta demanda."""
            try:
                year = int(fecha.split("-")[0])
                fecha_dt = datetime.strptime(fecha, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return 0

            ranges = [
                ("15-Dec", "31-Dec"),
                ("1-Jan", "3-Mar"),
                ("15-Jul", "31-Jul"),
                ("11-Sep", "30-Sep")
            ]
            for start, end in ranges:
                rmin = datetime.strptime(start, "%d-%b").replace(year=year)
                rmax = datetime.strptime(end, "%d-%b").replace(year=year)
                if rmin <= fecha_dt <= rmax:
                    return 1
            return 0

        def get_min_diff(row) -> float:
            """Calcula la diferencia en minutos entre la hora real y la programada."""
            try:
                f_o = datetime.strptime(row["Fecha-O"], "%Y-%m-%d %H:%M:%S")
                f_i = datetime.strptime(row["Fecha-I"], "%Y-%m-%d %H:%M:%S")
                return (f_o - f_i).total_seconds() / 60
            except Exception:
                return 0

        data = data.copy()
        logging.info("Se inicia el proceso de generación de características...")

        # Se generan las variables derivadas a partir de las fechas y horarios.
        data["period_day"] = data["Fecha-I"].apply(get_period_day)
        data["high_season"] = data["Fecha-I"].apply(is_high_season)
        data["min_diff"] = data.apply(get_min_diff, axis=1)

        # En caso de no existir la variable objetivo, se crea con base en el umbral de 15 minutos.
        if "delay" not in data.columns:
            data["delay"] = np.where(data["min_diff"] > 15, 1, 0)

        # Se codifican las variables categóricas mediante one-hot encoding.
        features = pd.concat([
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES")
        ], axis=1)

        # Se guarda el orden de las columnas para mantener consistencia durante la inferencia.
        self._feature_columns = features.columns.tolist()

        if target_column:
            target = data[target_column]
            logging.info("El preprocesamiento se completó con la variable objetivo incluida.")
            return features, target

        logging.info("El preprocesamiento se completó para datos de predicción.")
        return features

    # ==============================================================
    # ENTRENAMIENTO
    # ==============================================================
    def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
        """
        Se entrena el modelo XGBoost utilizando los datos preprocesados.

        Args:
            features (pd.DataFrame): conjunto de características.
            target (pd.Series): variable objetivo.
        """
        logging.info("Se inicia el entrenamiento del modelo...")
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.33, random_state=42
        )

        # Se entrena el modelo con parámetros controlados y balanceados.
        model = xgb.XGBClassifier(
            random_state=1,
            learning_rate=0.01,
            n_estimators=200,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        self._model = model
        joblib.dump(model, "xgb_model.pkl")
        logging.info("El modelo fue entrenado y almacenado en disco como xgb_model.pkl.")

    # ==============================================================
    # PREDICCIÓN
    # ==============================================================
    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Se generan predicciones a partir de un conjunto de características preprocesadas.

        Args:
            features (pd.DataFrame): datos listos para inferencia.

        Returns:
            List[int]: lista de valores binarios indicando presencia (1) o ausencia (0) de retraso.
        """
        if self._model is None:
            logging.info("No se encontró el modelo en memoria; se cargará desde disco.")
            self._model = joblib.load("xgb_model.pkl")

        # Se garantiza la alineación de las columnas respecto al modelo entrenado.
        for col in self._feature_columns:
            if col not in features.columns:
                features[col] = 0
        features = features[self._feature_columns]

        preds = self._model.predict(features)
        logging.info(f"Se generaron {len(preds)} predicciones.")
        return preds.tolist()
