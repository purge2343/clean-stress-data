"""Tests for cleaning/validators.py."""

import pandas as pd
import numpy as np
import pytest

from cleaning.validators import (
    validate_physical_limits,
    validate_required_columns,
    validate_test_types,
)


class TestValidatePhysicalLimits:
    def test_all_within_limits(self, clean_df):
        mask, violations = validate_physical_limits(clean_df)
        assert mask.all()
        assert len(violations) == 0

    @pytest.mark.parametrize("temp,expected_valid", [
        (-65, True),    # at lower limit
        (-66, False),   # below lower limit
        (300, True),    # at upper limit
        (301, False),   # above upper limit
        (25, True),     # normal value
    ])
    def test_temperature_boundaries(self, temp, expected_valid):
        df = pd.DataFrame({"temperature_c": [temp]})
        mask, _ = validate_physical_limits(df)
        assert mask.iloc[0] == expected_valid

    @pytest.mark.parametrize("voltage,expected_valid", [
        (0.0, False),   # at lower bound (exclusive)
        (0.001, True),  # just above 0
        (5.0, True),    # at upper limit
        (5.1, False),   # above upper limit
        (1.2, True),    # normal value
    ])
    def test_voltage_boundaries(self, voltage, expected_valid):
        df = pd.DataFrame({"voltage_v": [voltage]})
        mask, _ = validate_physical_limits(df)
        assert mask.iloc[0] == expected_valid

    @pytest.mark.parametrize("cycles,expected_valid", [
        (0, True),               # at lower limit
        (-1, False),             # below lower limit
        (10_000_000_000, True),  # at upper limit
        (10_000_000_001, False), # above upper limit
    ])
    def test_cycles_boundaries(self, cycles, expected_valid):
        df = pd.DataFrame({"cycles": [cycles]})
        mask, _ = validate_physical_limits(df)
        assert mask.iloc[0] == expected_valid

    @pytest.mark.parametrize("duration,expected_valid", [
        (0, True),            # at lower limit
        (-1, False),          # below lower limit
        (31_536_000, True),   # at upper limit (1 year)
        (31_536_001, False),  # above upper limit
    ])
    def test_duration_boundaries(self, duration, expected_valid):
        df = pd.DataFrame({"duration_s": [duration]})
        mask, _ = validate_physical_limits(df)
        assert mask.iloc[0] == expected_valid

    def test_nan_values_are_not_violations(self):
        df = pd.DataFrame({"temperature_c": [np.nan, 25.0, np.nan]})
        mask, violations = validate_physical_limits(df)
        assert mask.all()

    def test_all_nan_column(self):
        df = pd.DataFrame({"temperature_c": [np.nan, np.nan, np.nan]})
        mask, violations = validate_physical_limits(df)
        assert mask.all()
        assert len(violations) == 0

    def test_violations_report_details(self):
        df = pd.DataFrame({
            "temperature_c": [25, 999, -200],
            "voltage_v": [1.2, 1.2, 1.2],
        })
        mask, violations = validate_physical_limits(df)
        assert not mask.all()
        temp_viols = [v for v in violations if v["column"] == "temperature_c"]
        assert len(temp_viols) == 1
        assert temp_viols[0]["count"] == 2

    def test_missing_column_no_error(self):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        mask, violations = validate_physical_limits(df)
        assert mask.all()


class TestValidateRequiredColumns:
    def test_all_present(self, clean_df):
        missing = validate_required_columns(clean_df)
        assert missing == []

    def test_some_missing(self):
        df = pd.DataFrame({"chip_id": [1], "test_type": ["burn_in"]})
        missing = validate_required_columns(df)
        assert "temperature_c" in missing
        assert "voltage_v" in missing
        assert "chip_id" not in missing

    def test_empty_df(self):
        df = pd.DataFrame()
        missing = validate_required_columns(df)
        assert len(missing) == 8  # All canonical columns missing


class TestValidateTestTypes:
    def test_all_valid(self):
        series = pd.Series(["burn_in", "retention", "htol"])
        mask, unrec = validate_test_types(series)
        assert mask.all()
        assert len(unrec) == 0

    def test_unrecognized_values(self):
        series = pd.Series(["burn_in", "invalid_test", "htol"])
        mask, unrec = validate_test_types(series)
        assert not mask.all()
        assert "invalid_test" in unrec

    def test_nan_values_accepted(self):
        series = pd.Series(["burn_in", np.nan, "htol"])
        mask, unrec = validate_test_types(series)
        assert mask.all()

    def test_normalized_variants(self):
        """Hyphens and spaces should be normalized to underscores."""
        series = pd.Series(["burn-in", "burn in", "thermal cycling"])
        mask, unrec = validate_test_types(series)
        assert mask.all()
        assert len(unrec) == 0
