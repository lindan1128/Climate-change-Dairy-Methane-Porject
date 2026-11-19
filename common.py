from pathlib import Path
import json
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / 'data'

CLIMATE = ['temperature', 'vapour', 'precipitation', 'wind']

def read_table(path, sheet_name=None):
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        if sheet_name is None:
            raise ValueError('Specify sheet_name when reading a workbook')
        return pd.read_excel(path, sheet_name=sheet_name, engine='openpyxl')
    return pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path)

def require(frame, columns):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError('Missing columns: ' + ', '.join(sorted(missing)))

def config(path):
    path = Path(path).resolve()
    result = json.loads(path.read_text())
    for key in ['annual_panel', 'context', 'sem_data', 'covariances', 'split', 'climate_annual', 'population', 'output']:
        if result.get(key):
            p = Path(result[key]).expanduser()
            result[key] = str(p if p.is_absolute() else path.parent / p)
    return result

def output_dir(cfg, name):
    path = Path(cfg['output']) / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def standardize(x):
    x = np.asarray(x, dtype=float)
    mean, sd = x.mean(axis=0), x.std(axis=0, ddof=1)
    if not np.isfinite(x).all() or np.any(sd <= 0):
        raise ValueError('Standardization requires finite, nonconstant variables')
    return (x - mean) / sd, mean, sd
