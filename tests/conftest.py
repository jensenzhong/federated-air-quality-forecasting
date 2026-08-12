from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aqfl.data.schema import RAW_COLUMNS


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2013-03-01", periods=72, freq="h")
    frame = pd.DataFrame(
        {
            "No": np.arange(1, 73),
            "year": timestamps.year,
            "month": timestamps.month,
            "day": timestamps.day,
            "hour": timestamps.hour,
            "PM2.5": np.linspace(10, 81, 72),
            "PM10": np.linspace(20, 91, 72),
            "SO2": np.linspace(2, 10, 72),
            "NO2": np.linspace(4, 20, 72),
            "CO": np.linspace(100, 900, 72),
            "O3": np.linspace(5, 40, 72),
            "TEMP": np.linspace(-3, 12, 72),
            "PRES": np.linspace(1010, 1020, 72),
            "DEWP": np.linspace(-9, 4, 72),
            "RAIN": np.zeros(72),
            "wd": ["N"] * 72,
            "WSPM": np.linspace(0.5, 3, 72),
            "station": ["Aotizhongxin"] * 72,
        }
    )
    frame["timestamp"] = timestamps
    return frame


def write_prsa(path: Path, frame: pd.DataFrame) -> None:
    frame[RAW_COLUMNS].to_csv(path, index=False)
