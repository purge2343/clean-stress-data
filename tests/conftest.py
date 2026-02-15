"""Shared pytest fixtures for DDR stress data cleaning tests."""

import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def clean_df():
    """A small, well-formed DataFrame with canonical column names."""
    return pd.DataFrame({
        "chip_id": ["DDR5-0001", "DDR5-0001", "DDR5-0002", "DDR5-0002", "DDR5-0003"],
        "test_type": ["burn_in", "retention", "burn_in", "endurance", "htol"],
        "temperature_c": [125.0, 85.0, 130.0, 25.0, 150.0],
        "voltage_v": [1.4, 1.2, 1.35, 1.2, 1.5],
        "cycles": [1000, 500000, 1200, 1000000, 5000],
        "duration_s": [86400, 43200, 90000, 36000, 360000],
        "pass_fail": [True, True, False, True, False],
        "timestamp": pd.to_datetime([
            "2024-01-15 10:00:00",
            "2024-02-20 14:30:00",
            "2024-03-10 08:15:00",
            "2024-04-05 16:45:00",
            "2024-05-01 09:00:00",
        ]),
    })


@pytest.fixture
def messy_df():
    """DataFrame with column name variations and mixed formats."""
    return pd.DataFrame({
        "Chip ID": ["DDR5-0001", "DDR5-0002", "", "DDR5-0003", "DDR5-0004"],
        "Test Type": ["Burn In", "RETENTION", "thermal-cycling", "endurance", "HTOL"],
        "Temp (C)": ["125", "85", "50", "25", "150"],
        "Voltage (V)": ["1.4", "1.2", "1.35", "1.2", "1.5"],
        "Cycle Count": ["1000", "500000", "1200", "1000000", "5000"],
        "Duration (s)": ["86400", "43200", "90000", "36000", "360000"],
        "Pass/Fail": ["Pass", "1", "FAIL", "true", "F"],
        "Timestamp": [
            "2024-01-15 10:00:00",
            "01/20/2024 14:30:00",
            "2024-03-10T08:15:00",
            "2024-04-05",
            "05-01-2024 09:00:00",
        ],
    })


@pytest.fixture
def tmp_csv(tmp_path):
    """Helper to write a DataFrame to a temporary CSV and return the path."""
    def _write(df, filename="test_data.csv", **kwargs):
        path = tmp_path / filename
        df.to_csv(path, index=False, **kwargs)
        return path
    return _write
