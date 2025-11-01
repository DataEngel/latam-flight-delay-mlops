#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
El pipeline completo de predicción de retrasos de LATAM es orquestado desde este script.
Las operaciones de entrenamiento, predicción o ambas quedan disponibles según el modo indicado.

Ejemplos de uso en terminal:

python run_pipeline.py --mode train
python run_pipeline.py --mode predict --predict_data ../data/data.csv
python run_pipeline.py --mode both
"""

import argparse
import pandas as pd
import logging
from model import DelayModel


# El registro de logs es configurado para permitir el seguimiento del proceso.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


def main():
    """
    El flujo principal del pipeline es definido y se habilitan los modos de entrenamiento, predicción o ambos.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline LATAM - Entrenamiento y predicción de retrasos"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict", "both"],
        required=True,
        help="Modo de ejecución disponible: train / predict / both"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="../data/data.csv",
        help="Ruta al archivo CSV con los datos de entrenamiento"
    )
    parser.add_argument(
        "--predict_data",
        type=str,
        default="../data/data.csv",
        help="Ruta al archivo CSV con los datos de predicción"
    )
    args = parser.parse_args()

    model = DelayModel()

    # ==========================================================
    # ENTRENAMIENTO
    # ==========================================================
    if args.mode in ["train", "both"]:
        logging.info("=== MODO ENTRENAMIENTO ===")
        df_train = pd.read_csv(args.train_data)
        X, y = model.preprocess(df_train, target_column="delay")
        model.fit(X, y)
        logging.info("El entrenamiento fue completado correctamente.")

    # ==========================================================
    # PREDICCIÓN
    # ==========================================================
    if args.mode in ["predict", "both"]:
        logging.info("=== MODO PREDICCIÓN ===")
        df_pred = pd.read_csv(args.predict_data)
        X_pred = model.preprocess(df_pred)
        preds = model.predict(X_pred)
        df_pred["predicted_delay"] = preds

        output_file = "predictions_output.csv"
        df_pred.to_csv(output_file, index=False)
        logging.info(f"Las predicciones fueron generadas y guardadas en {output_file}.")

    logging.info("✅ La ejecución del pipeline finalizó exitosamente.")


if __name__ == "__main__":
    main()
