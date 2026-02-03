import pandas as pd

def build_features(series: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    feat_df = pd.DataFrame({
        "value": series,
    })

    feat_df["roll_mean"]  = series.rolling(window).mean()
    feat_df["roll_std"]   = series.rolling(window).std()
    feat_df["roll_min"]   = series.rolling(window).min()
    feat_df["roll_max"]   = series.rolling(window).max()

    df =  feat_df.dropna().copy()
    return df
