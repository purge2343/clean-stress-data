"""Stage 3: Aggregate and summarize cleaned stress test data."""

import warnings

import pandas as pd
import numpy as np


def aggregate(df):
    """Aggregate cleaned data by (chip_id, test_type).

    For each group computes:
    - total_tests, pass_count, fail_count, pass_rate
    - Per numeric column: mean, median, std, min, max

    Args:
        df: Cleaned DataFrame from Stage 2.

    Returns:
        summary_df: Aggregated summary DataFrame.
    """
    if df.empty:
        return pd.DataFrame()

    group_cols = [c for c in ["chip_id", "test_type"] if c in df.columns]
    if not group_cols:
        # No grouping columns available — treat everything as one group
        group_cols = []

    numeric_cols = [
        col for col in ["temperature_c", "voltage_v", "cycles", "duration_s"]
        if col in df.columns
    ]

    rows = []

    if group_cols:
        grouped = df.groupby(group_cols, dropna=False)
    else:
        grouped = [("all", df)]

    for group_key, group_df in grouped:
        row = {}

        # Unpack group key
        if group_cols:
            if len(group_cols) == 1:
                row[group_cols[0]] = group_key
            else:
                for col, val in zip(group_cols, group_key):
                    row[col] = val

        row["total_tests"] = len(group_df)

        # Pass/fail counts
        if "pass_fail" in group_df.columns:
            pf = group_df["pass_fail"]
            # Handle boolean and mixed types
            pass_count = pf.apply(
                lambda x: x is True or x == 1 or (isinstance(x, str) and x.lower() in ("true", "1"))
            ).sum()
            fail_count = pf.apply(
                lambda x: x is False or x == 0 or (isinstance(x, str) and x.lower() in ("false", "0"))
            ).sum()
            row["pass_count"] = int(pass_count)
            row["fail_count"] = int(fail_count)
            total_pf = row["pass_count"] + row["fail_count"]
            row["pass_rate"] = row["pass_count"] / total_pf if total_pf > 0 else np.nan
        else:
            row["pass_count"] = 0
            row["fail_count"] = 0
            row["pass_rate"] = np.nan

        # Numeric column statistics
        for col in numeric_cols:
            vals = pd.to_numeric(group_df[col], errors="coerce")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_median"] = vals.median()
                row[f"{col}_std"] = vals.std()
                row[f"{col}_min"] = vals.min()
                row[f"{col}_max"] = vals.max()

        rows.append(row)

    return pd.DataFrame(rows)
