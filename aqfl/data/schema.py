"""Dataset schema and feature definitions."""

from __future__ import annotations

RAW_COLUMNS = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "wd",
    "WSPM",
    "station",
]

CONTINUOUS_COLUMNS = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

NONNEGATIVE_COLUMNS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "RAIN", "WSPM"]
MISSING_SOURCE_COLUMNS = [*CONTINUOUS_COLUMNS, "wd"]
TIME_FEATURE_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]
FEATURE_COLUMNS = [
    *CONTINUOUS_COLUMNS,
    "wind_dir_sin",
    "wind_dir_cos",
    *TIME_FEATURE_COLUMNS,
    *[f"{name}_missing" for name in MISSING_SOURCE_COLUMNS],
]

TARGET_COLUMN = "PM2.5"
EXPECTED_FEATURE_DIM = 31

WIND_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

if len(FEATURE_COLUMNS) != EXPECTED_FEATURE_DIM:
    raise RuntimeError(f"Feature schema has {len(FEATURE_COLUMNS)} dimensions, expected 31")
