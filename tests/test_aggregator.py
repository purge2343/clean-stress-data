"""Tests for cleaning/aggregator.py."""

import pandas as pd
import numpy as np
import pytest

from cleaning.aggregator import aggregate


class TestSingleGroup:
    def test_single_group_statistics(self):
        df = pd.DataFrame({
            "chip_id": ["C1"] * 4,
            "test_type": ["burn_in"] * 4,
            "temperature_c": [100.0, 120.0, 130.0, 150.0],
            "voltage_v": [1.2, 1.3, 1.4, 1.5],
            "cycles": [1000, 2000, 3000, 4000],
            "duration_s": [3600, 7200, 10800, 14400],
            "pass_fail": [True, True, True, False],
        })
        result = aggregate(df)
        assert len(result) == 1
        row = result.iloc[0]
        assert row["total_tests"] == 4
        assert row["pass_count"] == 3
        assert row["fail_count"] == 1
        assert abs(row["pass_rate"] - 0.75) < 0.001
        assert abs(row["temperature_c_mean"] - 125.0) < 0.001
        assert abs(row["temperature_c_min"] - 100.0) < 0.001
        assert abs(row["temperature_c_max"] - 150.0) < 0.001


class TestMultiGroup:
    def test_correct_grouping(self, clean_df):
        result = aggregate(clean_df)
        # clean_df has 5 rows with different chip_id/test_type combos
        assert len(result) >= 3  # at least 3 distinct groups

    def test_groups_have_correct_columns(self, clean_df):
        result = aggregate(clean_df)
        assert "chip_id" in result.columns
        assert "test_type" in result.columns
        assert "total_tests" in result.columns
        assert "pass_rate" in result.columns


class TestPassRate:
    def test_all_pass(self):
        df = pd.DataFrame({
            "chip_id": ["C1"] * 3,
            "test_type": ["burn_in"] * 3,
            "pass_fail": [True, True, True],
        })
        result = aggregate(df)
        assert result.iloc[0]["pass_rate"] == 1.0

    def test_all_fail(self):
        df = pd.DataFrame({
            "chip_id": ["C1"] * 3,
            "test_type": ["burn_in"] * 3,
            "pass_fail": [False, False, False],
        })
        result = aggregate(df)
        assert result.iloc[0]["pass_rate"] == 0.0


class TestSingleRowGroup:
    def test_single_row_std_is_nan(self):
        df = pd.DataFrame({
            "chip_id": ["C1"],
            "test_type": ["burn_in"],
            "temperature_c": [125.0],
            "pass_fail": [True],
        })
        result = aggregate(df)
        assert pd.isna(result.iloc[0]["temperature_c_std"])


class TestEmptyDataFrame:
    def test_empty_df(self):
        df = pd.DataFrame()
        result = aggregate(df)
        assert result.empty
