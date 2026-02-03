import typing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def rolling_zscore(
    series: pd.DataFrame, window_size: int = 60, threshold: float = 3.0
) -> tuple:
    """Calculate rolling z-scores for a time series and identify anomalies."""
    rolling_mean = series.rolling(window=window_size).mean()
    rolling_std = series.rolling(window=window_size).std()
    z_scores = (series - rolling_mean) / rolling_std
    anomalies = np.abs(z_scores) > threshold
    return z_scores, anomalies


class IsolationForestModel:
    def __init__(self, n_estimators: int = 200, contamination: str = "auto", random_state: int = 42) -> None:
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X: np.ndarray) -> typing.Self:
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> bool:
        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> typing.Any:
        """Higher score = more normal, lower = more anomalous"""
        return self.model.decision_function(X)

    def detect_anomalies(self, X: np.ndarray, return_scores: bool = False) -> tuple:
        preds = self.model.predict(X)  # +1 normal, -1 anomaly
        mask = preds == -1  # boolean mask

        if return_scores:
            scores = self.decision_function(X)
            return mask, scores

        return mask

    def save(self, path: str | Path) -> None:
        import joblib
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "IsoForestDetector":
        import joblib
        loaded = joblib.load(path)
        obj = cls()
        obj.model = loaded
        return obj


