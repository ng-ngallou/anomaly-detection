import pandas as pd


def load_timeseries(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df.set_index("timestamp")


def normalize_series(series: pd.DataFrame) -> pd.DataFrame:
    return (series - series.mean()) / series.std()
