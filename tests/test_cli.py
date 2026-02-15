"""End-to-end CLI tests for clean_stress_data.py."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_DIR = Path(__file__).parent.parent


def run_cli(*args, check=True):
    """Run the CLI as a subprocess."""
    result = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "clean_stress_data.py"), *args],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
    )
    return result


class TestHappyPath:
    def test_creates_cleaned_and_summary(self, clean_df, tmp_csv, tmp_path):
        input_csv = tmp_csv(clean_df, "stress_test.csv")
        output_csv = tmp_path / "cleaned.csv"
        summary_csv = tmp_path / "summary.csv"

        result = run_cli(
            str(input_csv),
            "-o", str(output_csv),
            "--summary-output", str(summary_csv),
        )
        assert result.returncode == 0
        assert output_csv.exists()
        assert summary_csv.exists()

        cleaned = pd.read_csv(output_csv)
        assert len(cleaned) > 0
        assert "chip_id" in cleaned.columns

        summary = pd.read_csv(summary_csv)
        assert "total_tests" in summary.columns

    def test_default_output_names(self, clean_df, tmp_csv, tmp_path):
        input_csv = tmp_csv(clean_df, "mydata.csv")

        result = run_cli(str(input_csv))
        assert result.returncode == 0

        expected_cleaned = tmp_path / "mydata_cleaned.csv"
        expected_summary = tmp_path / "mydata_summary.csv"
        assert expected_cleaned.exists()
        assert expected_summary.exists()


class TestNoSummary:
    def test_no_summary_flag(self, clean_df, tmp_csv, tmp_path):
        input_csv = tmp_csv(clean_df, "test.csv")
        output_csv = tmp_path / "cleaned.csv"

        result = run_cli(
            str(input_csv),
            "-o", str(output_csv),
            "--no-summary",
        )
        assert result.returncode == 0
        assert output_csv.exists()
        # Summary should use default name, but with --no-summary it shouldn't exist
        assert not (tmp_path / "test_summary.csv").exists()


class TestEmptyCSV:
    def test_empty_csv(self, tmp_path):
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("chip_id,test_type,temperature_c\n")

        output_csv = tmp_path / "cleaned.csv"
        result = run_cli(
            str(empty_csv),
            "-o", str(output_csv),
        )
        assert result.returncode == 0
        assert output_csv.exists()


class TestStrictMode:
    def test_strict_with_warnings_exits_1(self, tmp_path):
        """Messy data should produce warnings, causing strict mode to exit 1."""
        messy_csv = tmp_path / "messy.csv"
        messy_csv.write_text(
            "chip_id,test_type,temperature_c,voltage_v,cycles,duration_s,pass_fail,timestamp\n"
            "DDR5-0001,burn_in,999,1.2,1000,3600,pass,2024-01-01\n"
        )

        result = run_cli(str(messy_csv), "--strict", "-o", str(tmp_path / "out.csv"))
        assert result.returncode == 1

    def test_strict_clean_data_exits_0(self, clean_df, tmp_csv, tmp_path):
        input_csv = tmp_csv(clean_df, "clean.csv")
        result = run_cli(
            str(input_csv), "--strict",
            "-o", str(tmp_path / "out.csv"),
        )
        assert result.returncode == 0


class TestHelp:
    def test_help_exits_0(self):
        result = run_cli("--help", check=False)
        assert result.returncode == 0
        assert "clean" in result.stdout.lower() or "stress" in result.stdout.lower()
