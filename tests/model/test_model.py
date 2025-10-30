import unittest
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

from challenge.model import DelayModel


class TestDelayModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._data_path = Path(__file__).resolve().parents[2] / "data" / "data.csv"
        cls._artifact_path = Path(__file__).resolve().parents[2] / "xgb_model.pkl"
        cls._raw_data = pd.read_csv(cls._data_path, low_memory=False)

    def setUp(self) -> None:
        super().setUp()
        self.model = DelayModel()
        self.addCleanup(self._cleanup_artifact)

    def _expected_features(self) -> pd.DataFrame:
        return pd.concat(
            [
                pd.get_dummies(self._raw_data["OPERA"], prefix="OPERA"),
                pd.get_dummies(self._raw_data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(self._raw_data["MES"], prefix="MES"),
            ],
            axis=1,
        )

    def _expected_target(self) -> pd.Series:
        scheduled = pd.to_datetime(self._raw_data["Fecha-I"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        operated = pd.to_datetime(self._raw_data["Fecha-O"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        min_diff = (operated - scheduled).dt.total_seconds() / 60.0
        expected = (min_diff > 15).astype(int)
        expected.name = "delay"
        return expected

    def _cleanup_artifact(self) -> None:
        if self._artifact_path.exists():
            self._artifact_path.unlink()

    def test_preprocess_for_training_matches_expected_engineering(self) -> None:
        features, target = self.model.preprocess(data=self._raw_data, target_column="delay")

        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(target, pd.Series)

        expected_features = self._expected_features()
        expected_target = self._expected_target()

        pdt.assert_frame_equal(features, expected_features, check_dtype=False)
        pdt.assert_series_equal(target, expected_target, check_dtype=False)

    def test_preprocess_without_target_preserves_feature_columns(self) -> None:
        features = self.model.preprocess(data=self._raw_data)

        self.assertIsInstance(features, pd.DataFrame)
        expected_features = self._expected_features()

        pdt.assert_frame_equal(features, expected_features, check_dtype=False)
        self.assertIsNotNone(self.model._feature_columns)
        self.assertListEqual(self.model._feature_columns, list(features.columns))

    def test_fit_and_predict_end_to_end(self) -> None:
        features, target = self.model.preprocess(data=self._raw_data, target_column="delay")

        self.model.fit(features=features, target=target)

        self.assertIsNotNone(self.model._model)
        self.assertTrue(self._artifact_path.exists())

        inference_batch = features.head(32).copy()
        predictions = self.model.predict(features=inference_batch)

        self.assertEqual(len(predictions), inference_batch.shape[0])
        self.assertTrue(all(pred in (0, 1) for pred in predictions))

        self.model._model = None
        reload_predictions = self.model.predict(features=inference_batch.copy())
        self.assertEqual(len(reload_predictions), inference_batch.shape[0])
        self.assertTrue(all(pred in (0, 1) for pred in reload_predictions))
