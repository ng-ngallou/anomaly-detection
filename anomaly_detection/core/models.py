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


class IsoForestDetector:
    def __init__(self, contamination: float = 0.01, random_state: int = 42) -> None:
        self.model = IsolationForest(
            contamination=contamination, random_state=random_state
        )

    def fit(self, series: pd.DataFrame) -> None:
        X = np.array(series).reshape(-1, 1)
        self.model.fit(X)

    def predict(self, series: pd.DataFrame) -> bool:
        X = np.array(series).reshape(-1, 1)
        preds = self.model.predict(X)
        return preds == -1
