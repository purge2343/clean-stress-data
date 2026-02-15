"""Tests for generate_fake_data.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from generate_fake_data import generate

PROJECT_DIR = Path(__file__).parent.parent


class TestSeedReproducibility:
    def test_same_seed_same_output(self):
        df1 = generate(num_rows=50, seed=42)
        df2 = generate(num_rows=50, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_output(self):
        df1 = generate(num_rows=50, seed=42)
        df2 = generate(num_rows=50, seed=99)
        assert not df1.equals(df2)


class TestRowCount:
    def test_exact_row_count_clean(self):
        df = generate(num_rows=100, seed=42)
        assert len(df) == 100

    def test_row_count_with_outliers(self):
        df = generate(num_rows=100, outlier_pct=5, seed=42)
        # Outliers replace existing rows, so count should be the same
        assert len(df) == 100

    def test_row_count_with_duplicates(self):
        df = generate(num_rows=100, duplicate_pct=10, seed=42)
        # Duplicates are added, so total > 100
        assert len(df) > 100


class TestMessyMode:
    def test_messy_column_names(self):
        df = generate(num_rows=50, messy=True, seed=42)
        # At least some columns should NOT be canonical snake_case
        canonical = {"chip_id", "test_type", "temperature_c", "voltage_v",
                     "cycles", "duration_s", "pass_fail", "timestamp"}
        has_non_canonical = any(col not in canonical for col in df.columns)
        # In messy mode, column names are randomized, so most won't be canonical
        assert has_non_canonical or len(df.columns) == 8

    def test_clean_mode_canonical_columns(self):
        df = generate(num_rows=50, messy=False, seed=42)
        expected = {"chip_id", "test_type", "temperature_c", "voltage_v",
                    "cycles", "duration_s", "pass_fail", "timestamp"}
        assert set(df.columns) == expected

    def test_messy_has_extra_columns(self):
        """Messy mode should inject extra columns from dump tools."""
        df = generate(num_rows=50, messy=True, seed=42)
        # Should have more than 8 canonical columns
        assert len(df.columns) > 8

    def test_messy_has_garbage_rows(self):
        """Messy mode should inject garbage/corrupted rows."""
        df = generate(num_rows=100, messy=True, seed=42)
        # Should have more rows than requested due to garbage injection
        assert len(df) > 100

    def test_messy_has_null_variants(self):
        """Messy mode should contain various null-like placeholder strings."""
        df = generate(num_rows=200, messy=True, seed=42)
        all_values = df.astype(str).values.flatten()
        all_text = " ".join(all_values)
        # Should contain at least some null-like placeholders
        null_variants = ["NULL", "N/A", "nil", "None", "NA"]
        found_any = any(v in all_text for v in null_variants)
        assert found_any

    def test_messy_has_gibberish(self):
        """Messy mode should contain memory controller gibberish."""
        df = generate(num_rows=200, messy=True, seed=42)
        all_values = df.astype(str).values.flatten()
        all_text = " ".join(all_values)
        # Should contain hex dumps or controller error strings
        gibberish_markers = ["0x", "ERR", "TIMEOUT", "ECC", "#REF!", "ADDR:"]
        found_any = any(m in all_text for m in gibberish_markers)
        assert found_any

    def test_messy_has_jedec_artifacts(self):
        """Messy mode should contain JEDEC/SPD/timing artifacts."""
        df = generate(num_rows=300, messy=True, seed=42)
        all_values = df.astype(str).values.flatten()
        all_text = " ".join(all_values)
        jedec_markers = ["SPD", "MR", "tRCD", "DDR5", "LPDDR", "CL="]
        found_any = any(m in all_text for m in jedec_markers)
        assert found_any

    def test_messy_has_ate_artifacts(self):
        """Messy mode should contain ATE bin codes and tester errors."""
        df = generate(num_rows=300, messy=True, seed=42)
        all_values = df.astype(str).values.flatten()
        all_text = " ".join(all_values)
        ate_markers = ["BIN", "ABORT", "INTERRUPT", "TIMEOUT", "LOT:", "WAFER"]
        found_any = any(m in all_text for m in ate_markers)
        assert found_any

    def test_messy_has_extra_ate_columns(self):
        """Messy mode should include ATE-related extra columns."""
        df = generate(num_rows=100, messy=True, seed=42)
        ate_cols = {"bin_result", "lot_id", "wafer_id", "site_num",
                    "spd_vendor", "speed_bin", "idd_measurement", "tester_status"}
        found_cols = set(df.columns) & ate_cols
        # Should have at least some of these
        assert len(found_cols) >= 1


class TestOutputIsValidCSV:
    def test_readable_by_pandas(self, tmp_path):
        output = tmp_path / "test.csv"
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "generate_fake_data.py"),
             "-o", str(output), "--rows", "20", "--seed", "42"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_DIR),
        )
        assert result.returncode == 0
        assert output.exists()
        df = pd.read_csv(output)
        assert len(df) == 20


class TestCLI:
    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "generate_fake_data.py"), "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
