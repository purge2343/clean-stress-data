"""Stage 1: Parse and normalize raw stress test data."""

import re

import pandas as pd
import numpy as np

from .constants import (
    COLUMN_ALIASES,
    CANONICAL_COLUMNS,
    GARBAGE_VALUES,
    PASS_FAIL_MAP,
    TIMESTAMP_FORMATS,
    VALID_TEST_TYPES,
)


def _to_snake_case(name):
    """Convert an arbitrary string to snake_case."""
    s = name.strip()
    # Replace common separators with underscore
    s = re.sub(r"[\s\-/]+", "_", s)
    # Insert underscore before uppercase letters (camelCase → camel_case)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = s.lower()
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    s = s.strip("_")
    return s


def _map_columns(df):
    """Map column names to canonical names via COLUMN_ALIASES.

    Returns:
        tuple: (renamed_df, warnings)
    """
    warnings = []
    rename_map = {}
    canonical_used = {}

    for col in df.columns:
        lookup_key = col.strip().lower()
        # Also try with whitespace collapsed
        lookup_key_collapsed = re.sub(r"\s+", " ", lookup_key)

        canonical = COLUMN_ALIASES.get(lookup_key) or COLUMN_ALIASES.get(lookup_key_collapsed)

        if canonical:
            if canonical in canonical_used:
                raise ValueError(
                    f"Duplicate mapping to '{canonical}': columns "
                    f"'{canonical_used[canonical]}' and '{col}' both map to it."
                )
            canonical_used[canonical] = col
            rename_map[col] = canonical
        else:
            # Preserve unrecognized columns, lowercased to snake_case
            snake = _to_snake_case(col)
            if snake in canonical_used:
                # Avoid collision with already-mapped canonical column
                snake = f"{snake}_extra"
            rename_map[col] = snake

    result = df.rename(columns=rename_map)
    return result, warnings


def _parse_timestamps(series):
    """Try multiple timestamp formats, fallback to pd.to_datetime inference."""
    if series.isna().all():
        return series

    parsed = pd.Series(pd.NaT, index=series.index)
    remaining = series.notna()

    for fmt in TIMESTAMP_FORMATS:
        if not remaining.any():
            break
        attempt = pd.to_datetime(
            series[remaining], format=fmt, errors="coerce"
        )
        newly_parsed = attempt.notna()
        parsed.loc[remaining & series.index.isin(attempt[newly_parsed].index)] = attempt[newly_parsed]
        # Update remaining: keep only rows where parsed is still NaT
        remaining = remaining & parsed.isna()

    # Fallback: pd.to_datetime with inference for anything still unparsed
    if remaining.any():
        fallback = pd.to_datetime(series[remaining], errors="coerce", dayfirst=False)
        parsed.loc[remaining] = fallback

    return parsed


def _normalize_temperature(df):
    """Detect and convert Kelvin/Fahrenheit to Celsius, mV to V.

    Heuristics:
    - Values > 200 that seem like Kelvin (typical offset ~273) → subtract 273.15
    - No reliable auto-detect for F vs C in DDR testing range, so we only
      convert obviously-Kelvin values (> 200 and plausible after conversion)
    """
    warnings = []
    if "temperature_c" not in df.columns:
        return df, warnings

    temps = pd.to_numeric(df["temperature_c"], errors="coerce")

    # Detect Kelvin: values in [208, 573] range (maps to [-65, 300] C)
    kelvin_mask = (temps > 200) & (temps - 273.15 >= -65) & (temps - 273.15 <= 300)
    # But only if the majority of high values look Kelvin
    # Simple heuristic: if more than half of values > 200 look Kelvin
    high_vals = temps[temps > 200].dropna()
    if len(high_vals) > 0:
        kelvin_candidate = ((high_vals - 273.15 >= -65) & (high_vals - 273.15 <= 300))
        if kelvin_candidate.sum() > len(high_vals) * 0.5:
            temps[kelvin_mask] = temps[kelvin_mask] - 273.15
            if kelvin_mask.any():
                warnings.append(
                    f"Converted {kelvin_mask.sum()} temperature values from Kelvin to Celsius"
                )

    df["temperature_c"] = temps
    return df, warnings


def _normalize_voltage(df):
    """Detect millivolt values and convert to volts."""
    warnings = []
    if "voltage_v" not in df.columns:
        return df, warnings

    volts = pd.to_numeric(df["voltage_v"], errors="coerce")

    # Heuristic: values > 100 are likely millivolts (DDR voltage is typically 0.6-3.6V)
    mv_mask = volts > 100
    if mv_mask.any():
        volts[mv_mask] = volts[mv_mask] / 1000.0
        warnings.append(
            f"Converted {mv_mask.sum()} voltage values from mV to V (were > 100)"
        )

    df["voltage_v"] = volts
    return df, warnings


def _coerce_pass_fail(series):
    """Map pass/fail string variants to boolean via PASS_FAIL_MAP."""
    if series.isna().all():
        return series

    result = series.copy()
    notna_mask = series.notna()
    lowered = series[notna_mask].astype(str).str.strip().str.lower()
    mapped = lowered.map(PASS_FAIL_MAP)
    result.loc[notna_mask] = mapped
    return result


def _normalize_test_types(series):
    """Standardize test type strings; set invalid types to NaN."""
    if series.isna().all():
        return series, []

    warnings = []
    result = series.copy()
    notna_mask = series.notna()
    cleaned = series[notna_mask].astype(str).str.strip().str.lower()
    cleaned = cleaned.str.replace(r"[\s\-]+", "_", regex=True)
    result.loc[notna_mask] = cleaned

    # Set values not in VALID_TEST_TYPES to NaN
    notna_after = result.notna()
    invalid_mask = notna_after & ~result.isin(VALID_TEST_TYPES)
    if invalid_mask.any():
        invalid_values = set(result[invalid_mask].unique())
        warnings.append(
            f"Set {invalid_mask.sum()} invalid test_type values to NaN: "
            f"{sorted(invalid_values)[:10]}"
        )
        result.loc[invalid_mask] = np.nan

    return result, warnings


def _remove_garbage_rows(df):
    """Remove rows where all canonical fields are garbage/NaN.

    Catches header-repeat rows, full hex-dump rows, and full controller-log rows
    where every meaningful field is missing or invalid.
    """
    canonical_fields = [
        c for c in CANONICAL_COLUMNS if c in df.columns and c != "timestamp"
    ]
    if not canonical_fields:
        return df, []

    col_names = {c.lower() for c in CANONICAL_COLUMNS}
    remove_mask = pd.Series(False, index=df.index)

    # Detect header-repeat rows: chip_id matches a column name
    if "chip_id" in df.columns:
        is_header = df["chip_id"].astype(str).str.strip().str.lower().isin(col_names)
        remove_mask |= is_header

    # Detect all-NaN rows (every canonical field is missing)
    all_na = df[canonical_fields].isna().all(axis=1)
    remove_mask |= all_na

    if remove_mask.any():
        warnings = [f"Removed {remove_mask.sum()} garbage/header-repeat rows"]
        return df[~remove_mask].copy(), warnings
    return df, []


def _fill_missing_chip_ids(df):
    """Fill missing chip_id with UNKNOWN_<row>."""
    warnings = []
    if "chip_id" not in df.columns:
        df["chip_id"] = [f"UNKNOWN_{i}" for i in df.index]
        warnings.append("No chip_id column found; generated placeholder IDs")
        return df, warnings

    missing = df["chip_id"].isna() | (df["chip_id"].astype(str).str.strip() == "")
    if missing.any():
        df.loc[missing, "chip_id"] = [f"UNKNOWN_{i}" for i in df.index[missing]]
        warnings.append(f"Filled {missing.sum()} missing chip_id values with placeholders")

    return df, warnings


def normalize(df):
    """Stage 1: Normalize raw DataFrame.

    Returns:
        tuple: (normalized_df, warnings_list)
    """
    all_warnings = []

    if df.empty:
        return df.copy(), ["Input DataFrame is empty"]

    # Work on a copy
    result = df.copy()

    # 1. Map column names
    result, w = _map_columns(result)
    all_warnings.extend(w)

    # 2. Standardize text columns to lowercase/stripped
    for col in result.columns:
        if result[col].dtype == object:
            result[col] = result[col].astype(str).str.strip()
            # Don't lowercase chip_id (preserve original identifiers) or timestamp
            if col not in ("chip_id", "timestamp"):
                result[col] = result[col].str.lower()
            # Restore NaN for garbage/null-like strings from astype(str)
            na_vals = result[col].str.lower().str.strip().isin(GARBAGE_VALUES)
            result.loc[na_vals, col] = np.nan

    # 3. Parse timestamps
    if "timestamp" in result.columns:
        result["timestamp"] = _parse_timestamps(result["timestamp"])

    # 4. Normalize test types
    if "test_type" in result.columns:
        result["test_type"], w = _normalize_test_types(result["test_type"])
        all_warnings.extend(w)

    # 5. Coerce numeric columns
    for col in ["temperature_c", "voltage_v", "cycles", "duration_s"]:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors="coerce")

    # 6. Normalize units
    result, w = _normalize_temperature(result)
    all_warnings.extend(w)
    result, w = _normalize_voltage(result)
    all_warnings.extend(w)

    # 7. Coerce pass/fail
    if "pass_fail" in result.columns:
        result["pass_fail"] = _coerce_pass_fail(result["pass_fail"])

    # 8. Remove all-garbage rows (before filling missing chip IDs)
    result, w = _remove_garbage_rows(result)
    all_warnings.extend(w)

    # 9. Fill missing chip IDs
    result, w = _fill_missing_chip_ids(result)
    all_warnings.extend(w)

    return result, all_warnings
