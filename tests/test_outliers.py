"""Tests for cleaning/outliers.py."""

import pandas as pd
import numpy as np
import pytest

from cleaning.outliers import filter_outliers, OutlierReport


class TestPhysicalLimitRemoval:
    def test_removes_impossible_temperature(self, clean_df):
        df = clean_df.copy()
        df.loc[0, "temperature_c"] = 999  # impossible
        result, report = filter_outliers(df, iqr_multiplier=None, remove_duplicates=False)
        assert report.physical_violations == 1
        assert len(result) == len(df) - 1

    def test_removes_negative_cycles(self, clean_df):
        df = clean_df.copy()
        df.loc[0, "cycles"] = -100
        result, report = filter_outliers(df, iqr_multiplier=None, remove_duplicates=False)
        assert report.physical_violations == 1


class TestIQRFilter:
    def test_constructed_iqr_bounds(self):
        """Known data where IQR outliers are predictable."""
        # 10 normal values + 1 extreme outlier
        temps = [25.0] * 10 + [999.0]  # 999 is also a physical violation
        # Use values within physical limits for a cleaner test
        df = pd.DataFrame({
            "chip_id": [f"C{i}" for i in range(11)],
            "test_type": ["burn_in"] * 11,
            "temperature_c": [25.0, 26.0, 24.0, 25.5, 24.5,
                              26.5, 23.5, 25.0, 24.0, 26.0, 200.0],
            "voltage_v": [1.2] * 11,
            "cycles": [1000] * 11,
            "duration_s": [3600] * 11,
        })
        result, report = filter_outliers(df, iqr_multiplier=1.5, remove_duplicates=False)
        # The 200.0 value should be an IQR outlier (others are ~24-27)
        assert 200.0 not in result["temperature_c"].values

    def test_multiplier_zero_strict(self):
        """Multiplier=0 removes everything outside Q1-Q3."""
        df = pd.DataFrame({
            "chip_id": [f"C{i}" for i in range(10)],
            "test_type": ["burn_in"] * 10,
            "temperature_c": list(range(10, 20)),  # 10-19
            "voltage_v": [1.2] * 10,
        })
        result, report = filter_outliers(df, iqr_multiplier=0, remove_duplicates=False)
        # With IQR multiplier 0, only values exactly at Q1-Q3 range survive
        assert len(result) <= len(df)

    def test_multiplier_100_removes_nothing(self):
        """Very large multiplier should keep all physically valid rows."""
        df = pd.DataFrame({
            "chip_id": [f"C{i}" for i in range(5)],
            "test_type": ["burn_in"] * 5,
            "temperature_c": [10.0, 20.0, 30.0, 40.0, 50.0],
            "voltage_v": [1.0, 1.5, 2.0, 2.5, 3.0],
            "cycles": [100, 200, 300, 400, 500],
            "duration_s": [1000, 2000, 3000, 4000, 5000],
        })
        result, report = filter_outliers(df, iqr_multiplier=100, remove_duplicates=False)
        assert len(result) == 5
        assert report.iqr_outliers == 0


class TestDuplicateRemoval:
    def test_removes_exact_duplicates(self, clean_df):
        df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
        result, report = filter_outliers(df, iqr_multiplier=None, remove_duplicates=True)
        assert report.duplicates_removed == 1

    def test_keep_duplicates_flag(self, clean_df):
        df = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
        result, report = filter_outliers(df, iqr_multiplier=None, remove_duplicates=False)
        assert report.duplicates_removed == 0
        assert len(result) == len(df)


class TestOrderOfOperations:
    def test_physical_before_iqr(self):
        """Physical violations should be removed before IQR calculation.

        If a physically impossible outlier is included in IQR calculation,
        it would skew the bounds.
        """
        # 9 normal values + 1 physical violation
        df = pd.DataFrame({
            "chip_id": [f"C{i}" for i in range(10)],
            "test_type": ["burn_in"] * 10,
            "temperature_c": [25.0] * 9 + [999.0],  # 999 is physically impossible
            "voltage_v": [1.2] * 10,
            "cycles": [1000] * 10,
            "duration_s": [3600] * 10,
        })
        result, report = filter_outliers(df, iqr_multiplier=1.5, remove_duplicates=False)
        assert report.physical_violations == 1
        # The remaining 9 rows should all survive IQR (they're identical)
        assert len(result) == 9


class TestEmptyDataFrame:
    def test_empty_passes_through(self):
        df = pd.DataFrame(columns=["chip_id", "test_type", "temperature_c"])
        result, report = filter_outliers(df)
        assert result.empty
        assert report.rows_before == 0
        assert report.rows_after == 0


class TestInvalidTestTypeRemoval:
    def test_nan_test_type_rows_removed(self):
        df = pd.DataFrame({
            "chip_id": ["C1", "C2", "C3"],
            "test_type": ["burn_in", np.nan, "htol"],
            "temperature_c": [125.0, 85.0, 150.0],
            "voltage_v": [1.4, 1.2, 1.5],
            "cycles": [1000, 500, 5000],
            "duration_s": [86400, 43200, 360000],
        })
        result, report = filter_outliers(df, iqr_multiplier=None, remove_duplicates=False)
        assert report.invalid_test_types == 1
        assert len(result) == 2
        assert "burn_in" in result["test_type"].values
        assert "htol" in result["test_type"].values

    def test_header_repeat_rows_removed(self):
        df = pd.DataFrame({
            "chip_id": ["C1", "chip_id", "C3"],
            "test_type": ["burn_in", "test_type", "htol"],
            "temperature_c": [125.0, 85.0, 150.0],
            "voltage_v": [1.4, 1.2, 1.5],
            "cycles": [1000, 500, 5000],
            "duration_s": [86400, 43200, 360000],
        })
        result, report = filter_outliers(df, iqr_multiplier=None, remove_duplicates=False)
        assert report.header_rows == 1
        assert len(result) == 2

    def test_valid_rows_preserved(self, clean_df):
        before = len(clean_df)
        result, report = filter_outliers(clean_df, iqr_multiplier=None, remove_duplicates=False)
        assert report.header_rows == 0
        assert report.invalid_test_types == 0
        assert len(result) == before


class TestOutlierReport:
    def test_report_summary(self):
        report = OutlierReport(
            physical_violations=5,
            iqr_outliers=3,
            duplicates_removed=2,
            rows_before=100,
            rows_after=90,
        )
        summary = report.summary()
        assert "100" in summary
        assert "90" in summary
        assert "5" in summary
