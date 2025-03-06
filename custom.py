import numpy as np
import pandas as pd

def transform_mjd(X):
    mjd = X[:, 0]
    min_mjd = np.min(mjd)
    sequential_days = mjd - min_mjd
    mjd_epoch = pd.Timestamp('1858-11-17')
    dates = pd.to_datetime(mjd, unit='D', origin=mjd_epoch)
    months = dates.month
    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)
    return np.column_stack([sequential_days, month_sin, month_cos])

def convert_ra_dec(X):
    ra_rad = np.radians(X[:, 0])
    dec_rad = np.radians(X[:, 1])
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.column_stack([x, y, z])

def clean_class_column(X):
    X = pd.DataFrame(X, columns=["class"])
    mapping = {"G": "GALAXY", "S": "STAR", "QSO": "QUASAR"}
    X["class"] = X["class"].replace(mapping)
    return X.to_numpy()
