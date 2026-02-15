"""Tests for cleaning/constants.py."""

from cleaning.constants import (
    COLUMN_ALIASES,
    CANONICAL_COLUMNS,
    PHYSICAL_LIMITS,
    PASS_FAIL_MAP,
    VALID_TEST_TYPES,
    TIMESTAMP_FORMATS,
)


class TestColumnAliases:
    def test_every_canonical_column_has_identity_alias(self):
        """Each canonical column name should map to itself."""
        for col in CANONICAL_COLUMNS:
            assert col in COLUMN_ALIASES
            assert COLUMN_ALIASES[col] == col

    def test_all_alias_values_are_canonical(self):
        """Every alias must map to a canonical column."""
        for alias, canonical in COLUMN_ALIASES.items():
            assert canonical in CANONICAL_COLUMNS, (
                f"Alias '{alias}' maps to '{canonical}' which is not canonical"
            )

    def test_aliases_cover_common_variations(self):
        """Spot check that common variations are present."""
        assert COLUMN_ALIASES["chipid"] == "chip_id"
        assert COLUMN_ALIASES["chip id"] == "chip_id"
        assert COLUMN_ALIASES["temp (c)"] == "temperature_c"
        assert COLUMN_ALIASES["pass/fail"] == "pass_fail"
        assert COLUMN_ALIASES["voltage"] == "voltage_v"


class TestPhysicalLimits:
    def test_covers_numeric_canonical_columns(self):
        """PHYSICAL_LIMITS should have entries for all numeric columns."""
        numeric_cols = {"temperature_c", "voltage_v", "cycles", "duration_s"}
        assert set(PHYSICAL_LIMITS.keys()) == numeric_cols

    def test_limits_are_tuples_of_two(self):
        for col, limits in PHYSICAL_LIMITS.items():
            assert len(limits) == 2, f"{col} limits should be a 2-tuple"
            lo, hi = limits
            assert lo < hi, f"{col}: lo ({lo}) should be < hi ({hi})"


class TestPassFailMap:
    def test_has_true_and_false_entries(self):
        trues = [k for k, v in PASS_FAIL_MAP.items() if v is True]
        falses = [k for k, v in PASS_FAIL_MAP.items() if v is False]
        assert len(trues) >= 3
        assert len(falses) >= 3

    def test_all_keys_are_lowercase(self):
        for key in PASS_FAIL_MAP:
            assert key == key.lower()


class TestValidTestTypes:
    def test_is_set(self):
        assert isinstance(VALID_TEST_TYPES, set)

    def test_expected_types_present(self):
        expected = {"retention", "endurance", "burn_in", "htol"}
        assert expected.issubset(VALID_TEST_TYPES)


class TestTimestampFormats:
    def test_is_nonempty_list(self):
        assert isinstance(TIMESTAMP_FORMATS, list)
        assert len(TIMESTAMP_FORMATS) > 0

    def test_iso_format_first(self):
        assert TIMESTAMP_FORMATS[0] == "%Y-%m-%d %H:%M:%S"
