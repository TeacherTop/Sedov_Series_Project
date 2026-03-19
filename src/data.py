import pandas as pd
import numpy as np
import json
from pathlib import Path
from statsmodels.tsa.stattools import acf


def load_tsf_robust(filepath: str) -> pd.DataFrame:
    """Загружает TSF файл в DataFrame"""
    long_data = []
    series_count = 0
    
    with open(filepath, 'rb') as f:
        lines = f.readlines()
    
    for line in lines:
        try:
            line_str = line.decode('utf-8', errors='ignore').strip()
        except Exception:
            line_str = line.decode('latin-1', errors='ignore').strip()
            
        if not line_str or line_str.startswith('@'):
            continue
        
        parts = line_str.split(':')
        
        data_part = None
        for part in parts:
            if ',' in part:
                data_part = part
                break
        
        if not data_part:
            if len(parts) > 1:
                data_part = parts[0]
            else:
                continue

        if data_part.endswith('.'):
            data_part = data_part[:-1]
            
        try:
            values = [float(x.strip()) for x in data_part.split(',') if x.strip()]
        except ValueError:
            continue
        
        if not values:
            continue
            
        series_count += 1
        unique_id = f'M{series_count}'
        
        for t, value in enumerate(values):
            long_data.append({
                'unique_id': unique_id,
                'ds': t,
                'y': value
            })
    
    return pd.DataFrame(long_data)


def has_strong_seasonality(series, seasonal_lag=12, threshold=0.3):
    """Проверяет есть ли сильная сезонность в ряде"""
    if len(series.dropna()) < seasonal_lag * 2:
        return False
    acf_values = acf(series.dropna(), nlags=seasonal_lag * 2, fft=False)
    return acf_values[seasonal_lag] > threshold


def load_selected_series(filepath: str = './results/selected_series_ids.json'):
    """Загружает список отобранных ID рядов из JSON файла"""
    if Path(filepath).exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


if __name__ == '__main__':
    df = load_tsf_robust('m4_monthly_dataset.tsf')
    print(f"Загружено {df['unique_id'].nunique()} рядов")
    
    # Берем подвыборку и проверяем сезонность
    np.random.seed(42)
    sample_ids = np.random.choice(df['unique_id'].unique(), size=200, replace=False)
    sample_df = df[df['unique_id'].isin(sample_ids)].copy()
    
    seasonal_series = []
    for uid in sample_df['unique_id'].unique():
        series = sample_df[sample_df['unique_id'] == uid]['y'].reset_index(drop=True)
        if has_strong_seasonality(series):
            seasonal_series.append(uid)
    
    print(f"Рядов с сезонностью: {len(seasonal_series)}")
    
    # Берем финальную выборку
    final_series_ids = seasonal_series[:100]
    print(f"Выбрано: {len(final_series_ids)} рядов")
    
    # Сохраняем результаты
    Path('./results').mkdir(parents=True, exist_ok=True)
    
    with open('./results/selected_series_ids.json', 'w', encoding='utf-8') as f:
        json.dump(final_series_ids, f, ensure_ascii=False, indent=2)
    
    print(f"Сохранено {len(final_series_ids)} ID")