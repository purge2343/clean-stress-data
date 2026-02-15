#!/usr/bin/env python3
"""CLI tool to clean DDR memory stress test data from CSV files.

Usage:
    python clean_stress_data.py input.csv [-o output.csv] [--summary-output summary.csv]
                                          [--no-summary] [--iqr-multiplier 1.5]
                                          [--keep-duplicates] [--strict]
                                          [--log-level INFO]
"""

import argparse
import csv
import io
import logging
import sys
from pathlib import Path

import pandas as pd

from cleaning.normalizer import normalize
from cleaning.outliers import filter_outliers
from cleaning.aggregator import aggregate

logger = logging.getLogger("clean_stress_data")


def _read_csv(path):
    """Read CSV with auto-detected delimiter, fallback encoding."""
    path = Path(path)

    # Try to detect delimiter by reading a sample
    for encoding in ("utf-8", "latin-1"):
        try:
            raw = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        logger.error("Could not decode file with utf-8 or latin-1")
        sys.exit(1)

    if encoding != "utf-8":
        logger.warning("File was not UTF-8; decoded with %s", encoding)

    # Use CSV sniffer to detect delimiter
    try:
        sample = raw[:8192]
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    df = pd.read_csv(
        io.StringIO(raw),
        sep=delimiter,
        dtype=str,  # Read everything as string initially
        keep_default_na=True,
    )

    return df


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Clean DDR memory stress test data from CSV files."
    )
    parser.add_argument("input", help="Path to input CSV file")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Path to cleaned output CSV (default: <input_stem>_cleaned.csv)",
    )
    parser.add_argument(
        "--summary-output", default=None,
        help="Path to summary output CSV (default: <input_stem>_summary.csv)",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip summary generation",
    )
    parser.add_argument(
        "--iqr-multiplier", type=float, default=1.5,
        help="IQR multiplier for outlier detection (default: 1.5)",
    )
    parser.add_argument(
        "--keep-duplicates", action="store_true",
        help="Do not remove duplicate rows",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Exit with code 1 on any validation warnings",
    )
    parser.add_argument(
        "--temp-min", type=float, default=None,
        help="Override minimum temperature limit (default: -65)",
    )
    parser.add_argument(
        "--temp-max", type=float, default=None,
        help="Override maximum temperature limit (default: 300)",
    )
    parser.add_argument(
        "--voltage-max", type=float, default=None,
        help="Override maximum voltage limit (default: 5.0)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    # Determine output paths
    stem = input_path.stem
    parent = input_path.parent
    output_path = Path(args.output) if args.output else parent / f"{stem}_cleaned.csv"
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else parent / f"{stem}_summary.csv"
    )

    # Apply custom physical limits if provided
    from cleaning import constants
    if args.temp_min is not None:
        lo, hi = constants.PHYSICAL_LIMITS["temperature_c"]
        constants.PHYSICAL_LIMITS["temperature_c"] = (args.temp_min, hi)
    if args.temp_max is not None:
        lo, hi = constants.PHYSICAL_LIMITS["temperature_c"]
        constants.PHYSICAL_LIMITS["temperature_c"] = (lo, args.temp_max)
    if args.voltage_max is not None:
        lo, hi = constants.PHYSICAL_LIMITS["voltage_v"]
        constants.PHYSICAL_LIMITS["voltage_v"] = (lo, args.voltage_max)

    # Read input
    logger.info("Reading %s", input_path)
    df = _read_csv(input_path)
    logger.info("Read %d rows, %d columns", len(df), len(df.columns))

    had_warnings = False

    # Stage 1: Normalize
    logger.info("Stage 1: Normalizing...")
    df_norm, norm_warnings = normalize(df)
    if norm_warnings:
        had_warnings = True
        for w in norm_warnings:
            logger.warning("Normalize: %s", w)

    # Stage 2: Filter outliers
    logger.info("Stage 2: Filtering outliers...")
    df_clean, outlier_report = filter_outliers(
        df_norm,
        iqr_multiplier=args.iqr_multiplier,
        remove_duplicates=not args.keep_duplicates,
    )
    logger.info(outlier_report.summary())
    if outlier_report.physical_violations > 0 or outlier_report.iqr_outliers > 0:
        had_warnings = True

    # Stage 3: Aggregate (optional)
    summary_df = None
    if not args.no_summary:
        logger.info("Stage 3: Aggregating...")
        summary_df = aggregate(df_clean)

    # Write outputs
    df_clean.to_csv(output_path, index=False)
    logger.info("Wrote cleaned data to %s (%d rows)", output_path, len(df_clean))

    if summary_df is not None:
        summary_df.to_csv(summary_path, index=False)
        logger.info("Wrote summary to %s (%d rows)", summary_path, len(summary_df))

    if args.strict and had_warnings:
        logger.error("Strict mode: exiting with code 1 due to warnings")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
