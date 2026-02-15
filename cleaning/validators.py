"""Validation functions for DDR stress test data.

Each validator returns boolean masks and/or violation descriptions,
without modifying the input DataFrame.
"""

import pandas as pd
import numpy as np

from .constants import PHYSICAL_LIMITS, CANONICAL_COLUMNS, VALID_TEST_TYPES


def validate_physical_limits(df):
    """Check each numeric column against hard physical limits.

    Returns:
        tuple: (valid_mask, violations_list)
            - valid_mask: boolean Series, True for rows within all limits
            - violations_list: list of dicts with column, row indices, and bounds
    """
    valid_mask = pd.Series(True, index=df.index)
    violations = []

    for col, (lo, hi) in PHYSICAL_LIMITS.items():
        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        if col == "voltage_v":
            # voltage_v: (0, 5.0] — 0 is exclusive
            col_invalid = (series <= lo) | (series > hi)
        else:
            col_invalid = (series < lo) | (series > hi)

        # NaN values are not violations (they're missing, not impossible)
        col_invalid = col_invalid.fillna(False)

        if col_invalid.any():
            bad_indices = df.index[col_invalid].tolist()
            violations.append({
                "column": col,
                "count": int(col_invalid.sum()),
                "indices": bad_indices,
                "bounds": (lo, hi),
            })
            valid_mask &= ~col_invalid

    return valid_mask, violations


def validate_required_columns(df):
    """Check which canonical columns are missing from the DataFrame.

    Returns:
        list[str]: names of missing canonical columns
    """
    return [col for col in CANONICAL_COLUMNS if col not in df.columns]


def validate_test_types(series):
    """Check for unrecognized test type values.

    Args:
        series: pandas Series of test_type values

    Returns:
        tuple: (valid_mask, unrecognized_values)
            - valid_mask: boolean Series, True for recognized or NaN values
            - unrecognized_values: set of string values not in VALID_TEST_TYPES
    """
    cleaned = series.dropna().astype(str).str.strip().str.lower()
    # Normalize underscores/hyphens/spaces
    cleaned = cleaned.str.replace(r"[\s\-]+", "_", regex=True)

    unrecognized = set(cleaned.unique()) - VALID_TEST_TYPES
    valid_mask = pd.Series(True, index=series.index)

    if unrecognized:
        normalized_full = series.astype(str).str.strip().str.lower().str.replace(
            r"[\s\-]+", "_", regex=True
        )
        valid_mask = normalized_full.isin(VALID_TEST_TYPES) | series.isna()

    return valid_mask, unrecognized
