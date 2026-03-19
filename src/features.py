from typing import List, Dict
import numpy as np
import pandas as pd

def create_lag_features(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """Создает лаги для каждого ряда."""
    result = []
    for uid in df['unique_id'].unique():
        series = df[df['unique_id'] == uid].copy().sort_values('ds')
        for lag in lags:
            series[f'lag_{lag}'] = series['y'].shift(lag)
        result.append(series)
    return pd.concat(result).dropna()

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет календарные признаки (месяц, квартал, год)."""
    df = df.copy()
    df['month'] = (df['ds'] - 1) % 12 + 1  # Для monthly данных
    df['quarter'] = ((df['ds'] - 1) // 3) % 4 + 1
    df['is_year_end'] = ((df['ds'] - 1) % 12 == 11).astype(int)
    return df

def create_fourier_features(df: pd.DataFrame, seasonal_period: int = 12, n_harmonics: int = 3) -> pd.DataFrame:
    """Создает ряды Фурье для моделирования сезонности."""
    df = df.copy()
    for i in range(1, n_harmonics + 1):
        df[f'fourier_sin_{i}'] = np.sin(2 * np.pi * i * df['ds'] / seasonal_period)
        df[f'fourier_cos_{i}'] = np.cos(2 * np.pi * i * df['ds'] / seasonal_period)
    return df

def prepare_features(df: pd.DataFrame, feature_set: str, seasonal_lag: int = 12) -> pd.DataFrame:
    """
    Подготавливает разные наборы признаков.
    
    feature_set: 'regular', 'seasonal', 'calendar', 'fourier', 
                 'combined_1', 'combined_2'
    """
    df = df.copy()
    
    if feature_set == 'regular':
        df = create_lag_features(df, lags=[1, 2, 3, 4, 5, 6])
    elif feature_set == 'seasonal':
        df = create_lag_features(df, lags=[seasonal_lag, seasonal_lag*2, seasonal_lag*3])
    elif feature_set == 'calendar':
        df = create_lag_features(df, lags=[1, 2, 3])
        df = create_calendar_features(df)
    elif feature_set == 'fourier':
        df = create_lag_features(df, lags=[1, 2, 3])
        df = create_fourier_features(df, seasonal_period=seasonal_lag, n_harmonics=3)
    elif feature_set == 'combined_1':  # regular + seasonal
        df = create_lag_features(df, lags=[1, 2, 3, 4, 5, 6, 12, 24, 36])
    elif feature_set == 'combined_2':  # all features
        df = create_lag_features(df, lags=[1, 2, 3, 4, 5, 6, 12, 24, 36])
        df = create_calendar_features(df)
        df = create_fourier_features(df, seasonal_period=seasonal_lag, n_harmonics=3)
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    return df