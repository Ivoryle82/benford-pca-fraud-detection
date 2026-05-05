from fastapi import FastAPI
from pathlib import Path
import pandas as pd
import os

app = FastAPI()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
USER_FEATURES = DATA_DIR / "user_features.csv"
REMOTE_DATA_URL = os.getenv("REMOTE_DATA_URL")


def load_user_features():
    # Prefer remote data when REMOTE_DATA_URL is provided (useful on Vercel)
    if REMOTE_DATA_URL:
        try:
            return pd.read_csv(REMOTE_DATA_URL)
        except Exception:
            pass

    if USER_FEATURES.exists():
        return pd.read_csv(USER_FEATURES)
    return pd.DataFrame()


@app.get('/api/users')
def users(limit: int = 100):
    df = load_user_features()
    return df.head(limit).to_dict(orient='records')


@app.get('/api/stats')
def stats():
    df = load_user_features()
    if df.empty:
        return {}
    return df.describe().to_dict()


@app.get('/api/benford')
def benford():
    df = load_user_features()
    if df.empty:
        return {}
    if 'isFraud' in df.columns and 'benford_ratio' in df.columns:
        return df.groupby('isFraud')['benford_ratio'].mean().to_dict()
    numeric = df.select_dtypes('number')
    return numeric.mean().to_dict()
