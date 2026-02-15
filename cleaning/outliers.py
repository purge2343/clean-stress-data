"""Stage 2: Filter outliers and remove duplicates."""

from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from .constants import CANONICAL_COLUMNS, PHYSICAL_LIMITS
from .validators import validate_physical_limits


@dataclass
class OutlierReport:
    """Summary of what was removed during outlier filtering."""
    header_rows: int = 0
    invalid_test_types: int = 0
    physical_violations: int = 0
    physical_details: list = field(default_factory=list)
    iqr_outliers: int = 0
    iqr_bounds: dict = field(default_factory=dict)
    duplicates_removed: int = 0
    rows_before: int = 0
    rows_after: int = 0

    def summary(self):
        lines = [
            f"Outlier Report:",
            f"  Rows before: {self.rows_before}",
            f"  Header rows removed: {self.header_rows}",
            f"  Invalid test_type rows removed: {self.invalid_test_types}",
            f"  Physical limit violations removed: {self.physical_violations}",
            f"  IQR outliers removed: {self.iqr_outliers}",
            f"  Duplicates removed: {self.duplicates_removed}",
            f"  Rows after: {self.rows_after}",
        ]
        if self.iqr_bounds:
            lines.append("  IQR bounds per (test_type, column):")
            for key, bounds in sorted(self.iqr_bounds.items()):
                lines.append(f"    {key}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
        return "\n".join(lines)


def _remove_header_rows(df, report):
    """Remove rows where chip_id equals a column name (header repeats)."""
    if df.empty or "chip_id" not in df.columns:
        return df

    col_names = {c.lower() for c in CANONICAL_COLUMNS}
    is_header = df["chip_id"].astype(str).str.strip().str.lower().isin(col_names)
    removed = is_header.sum()
    report.header_rows = removed
    if removed:
        return df[~is_header].copy()
    return df


def _remove_invalid_test_types(df, report):
    """Remove rows where test_type is NaN (set by normalizer for invalid values)."""
    if df.empty or "test_type" not in df.columns:
        return df

    invalid = df["test_type"].isna()
    removed = invalid.sum()
    report.invalid_test_types = removed
    if removed:
        return df[~invalid].copy()
    return df


def _remove_physical_violations(df, report):
    """Remove rows with physically impossible values."""
    if df.empty:
        return df

    valid_mask, violations = validate_physical_limits(df)
    removed = (~valid_mask).sum()
    report.physical_violations = removed
    report.physical_details = violations
    return df[valid_mask].copy()


def _iqr_filter(df, iqr_multiplier, report):
    """IQR-based outlier detection per numeric column, grouped by test_type."""
    if df.empty or iqr_multiplier is None:
        return df

    numeric_cols = [
        col for col in ["temperature_c", "voltage_v", "cycles", "duration_s"]
        if col in df.columns
    ]

    if not numeric_cols:
        return df

    valid_mask = pd.Series(True, index=df.index)

    group_col = "test_type" if "test_type" in df.columns else None

    if group_col and df[group_col].notna().any():
        groups = df.groupby(group_col, dropna=True)
    else:
        # Treat entire DataFrame as one group
        groups = [(None, df)]

    for group_name, group_df in groups:
        for col in numeric_cols:
            series = pd.to_numeric(group_df[col], errors="coerce").dropna()
            if len(series) < 4:
                # Too few values for meaningful IQR
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            col_values = pd.to_numeric(group_df[col], errors="coerce")
            outlier_mask = (col_values < lower) | (col_values > upper)
            # NaN is not an outlier
            outlier_mask = outlier_mask.fillna(False)

            valid_mask.loc[group_df.index] &= ~outlier_mask

            key = (str(group_name), col) if group_name else ("all", col)
            report.iqr_bounds[key] = (lower, upper)

    removed = (~valid_mask).sum()
    report.iqr_outliers = removed
    return df[valid_mask].copy()


def _remove_duplicates(df, report):
    """Remove duplicate rows based on (chip_id, test_type, timestamp)."""
    if df.empty:
        return df

    dup_cols = [c for c in ["chip_id", "test_type", "timestamp"] if c in df.columns]
    if not dup_cols:
        return df

    before = len(df)
    result = df.drop_duplicates(subset=dup_cols, keep="first")
    report.duplicates_removed = before - len(result)
    return result


def filter_outliers(df, iqr_multiplier=1.5, remove_duplicates=True):
    """Stage 2: Filter outliers and duplicates.

    Args:
        df: Normalized DataFrame from Stage 1.
        iqr_multiplier: IQR multiplier for outlier detection.
            Use None to skip IQR filtering.
        remove_duplicates: Whether to remove duplicate rows.

    Returns:
        tuple: (cleaned_df, OutlierReport)
    """
    report = OutlierReport(rows_before=len(df))

    if df.empty:
        report.rows_after = 0
        return df.copy(), report

    # 1. Remove header-repeat rows first
    result = _remove_header_rows(df, report)

    # 2. Remove rows with invalid (NaN) test_type
    result = _remove_invalid_test_types(result, report)

    # 3. Remove physical limit violations
    result = _remove_physical_violations(result, report)

    # 4. IQR-based outlier detection (after physical limits removed)
    result = _iqr_filter(result, iqr_multiplier, report)

    # 5. Duplicate removal
    if remove_duplicates:
        result = _remove_duplicates(result, report)

    report.rows_after = len(result)
    return result, report
