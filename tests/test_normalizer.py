"""Tests for cleaning/normalizer.py."""

import pandas as pd
import numpy as np
import pytest

from cleaning.normalizer import normalize


class TestColumnMapping:
    @pytest.mark.parametrize("input_col,expected_canonical", [
        ("ChipID", "chip_id"),
        ("Chip ID", "chip_id"),
        ("chip_id", "chip_id"),
        ("Device ID", "chip_id"),
        ("DUT", "chip_id"),
        ("Test Type", "test_type"),
        ("TestType", "test_type"),
        ("stress_type", "test_type"),
        ("Temp (C)", "temperature_c"),
        ("Temperature", "temperature_c"),
        ("temp_celsius", "temperature_c"),
        ("Voltage (V)", "voltage_v"),
        ("VDD", "voltage_v"),
        ("Cycle Count", "cycles"),
        ("num_cycles", "cycles"),
        ("Duration (s)", "duration_s"),
        ("elapsed_time", "duration_s"),
        ("Pass/Fail", "pass_fail"),
        ("Result", "pass_fail"),
        ("Timestamp", "timestamp"),
        ("test_date", "timestamp"),
    ])
    def test_alias_mapping(self, input_col, expected_canonical):
        df = pd.DataFrame({input_col: ["test_value"]})
        result, _ = normalize(df)
        assert expected_canonical in result.columns

    def test_unrecognized_columns_preserved(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "My Custom Column": ["some_value"],
        })
        result, _ = normalize(df)
        assert "my_custom_column" in result.columns

    def test_duplicate_mapping_raises(self):
        """Two columns that map to the same canonical name should raise."""
        df = pd.DataFrame({
            "Chip ID": ["DDR5-0001"],
            "ChipID": ["DDR5-0002"],
        })
        with pytest.raises(ValueError, match="Duplicate mapping"):
            normalize(df)


class TestTimestampParsing:
    @pytest.mark.parametrize("ts_str,expected_year", [
        ("2024-01-15 10:00:00", 2024),
        ("2024-01-15T10:00:00", 2024),
        ("01/15/2024 10:00:00", 2024),
        ("2024-01-15", 2024),
    ])
    def test_various_formats(self, ts_str, expected_year):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "timestamp": [ts_str],
        })
        result, _ = normalize(df)
        assert result["timestamp"].iloc[0].year == expected_year

    def test_unparseable_timestamp_becomes_nat(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "timestamp": ["not_a_date"],
        })
        result, _ = normalize(df)
        assert pd.isna(result["timestamp"].iloc[0])


class TestUnitConversion:
    def test_millivolt_to_volt(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "voltage_v": ["1200"],  # mV
        })
        result, warnings = normalize(df)
        assert abs(result["voltage_v"].iloc[0] - 1.2) < 0.01
        assert any("mV" in w for w in warnings)

    def test_normal_voltage_unchanged(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "voltage_v": ["1.2"],
        })
        result, _ = normalize(df)
        assert abs(result["voltage_v"].iloc[0] - 1.2) < 0.001


class TestPassFailCoercion:
    @pytest.mark.parametrize("input_val,expected", [
        ("Pass", True),
        ("FAIL", False),
        ("pass", True),
        ("fail", False),
        ("P", True),
        ("F", False),
        ("1", True),
        ("0", False),
        ("true", True),
        ("false", False),
    ])
    def test_pass_fail_variants(self, input_val, expected):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "pass_fail": [input_val],
        })
        result, _ = normalize(df)
        assert result["pass_fail"].iloc[0] == expected


class TestMissingChipId:
    def test_missing_chip_id_filled(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001", "", None],
            "test_type": ["burn_in", "burn_in", "burn_in"],
        })
        result, warnings = normalize(df)
        assert "UNKNOWN_" in str(result["chip_id"].iloc[1])
        assert "UNKNOWN_" in str(result["chip_id"].iloc[2])

    def test_no_chip_id_column(self):
        df = pd.DataFrame({
            "test_type": ["burn_in"],
            "temperature_c": [125],
        })
        result, warnings = normalize(df)
        assert "chip_id" in result.columns
        assert "UNKNOWN_" in str(result["chip_id"].iloc[0])


class TestEmptyDataFrame:
    def test_empty_df(self):
        df = pd.DataFrame()
        result, warnings = normalize(df)
        assert result.empty
        assert len(warnings) > 0


class TestTestTypeNormalization:
    def test_normalizes_spaces_and_hyphens(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001", "DDR5-0002", "DDR5-0003"],
            "test_type": ["Burn In", "thermal-cycling", "HTOL"],
        })
        result, _ = normalize(df)
        assert result["test_type"].iloc[0] == "burn_in"
        assert result["test_type"].iloc[1] == "thermal_cycling"
        assert result["test_type"].iloc[2] == "htol"


class TestGarbageHandling:
    @pytest.mark.parametrize("garbage_val", ["nil", "null", "n/a", "undefined", "N/A"])
    def test_nil_becomes_nan(self, garbage_val):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "test_type": [garbage_val],
        })
        result, _ = normalize(df)
        assert pd.isna(result["test_type"].iloc[0])

    @pytest.mark.parametrize("error_val", ["#REF!", "#VALUE!", "#DIV/0!", "#N/A"])
    def test_spreadsheet_errors_become_nan(self, error_val):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "test_type": [error_val],
        })
        result, _ = normalize(df)
        assert pd.isna(result["test_type"].iloc[0])

    def test_hex_dump_test_type_becomes_nan(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "test_type": ["de_ad_be_ef_ca_fe_ba_be"],
        })
        result, _ = normalize(df)
        assert pd.isna(result["test_type"].iloc[0])

    def test_ate_code_test_type_becomes_nan(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "test_type": ["test_aborted@step117"],
        })
        result, _ = normalize(df)
        assert pd.isna(result["test_type"].iloc[0])

    def test_ate_measurement_test_type_becomes_nan(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001"],
            "test_type": ["vmin=1.04v"],
        })
        result, _ = normalize(df)
        assert pd.isna(result["test_type"].iloc[0])

    def test_garbage_rows_removed(self):
        df = pd.DataFrame({
            "chip_id": ["DDR5-0001", "chip_id", "---"],
            "test_type": ["burn_in", "test_type", "---"],
            "temperature_c": ["125", "temperature_c", "---"],
            "voltage_v": ["1.2", "voltage_v", "---"],
        })
        result, warnings = normalize(df)
        # Header-repeat and all-garbage rows should be removed
        assert len(result) == 1
        assert result["chip_id"].iloc[0] == "DDR5-0001"

    def test_valid_test_types_preserved(self):
        valid_types = ["burn_in", "htol", "retention", "endurance",
                       "thermal_cycling", "electromigration"]
        df = pd.DataFrame({
            "chip_id": [f"DDR5-{i:04d}" for i in range(len(valid_types))],
            "test_type": valid_types,
        })
        result, _ = normalize(df)
        for i, tt in enumerate(valid_types):
            assert result["test_type"].iloc[i] == tt
