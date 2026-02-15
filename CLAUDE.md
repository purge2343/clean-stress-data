# CLAUDE.md

## Build & test commands

- Install dependencies: `pip install -r requirements.txt`
- Run all tests: `pytest tests/ -v`
- Run a single test file: `pytest tests/test_normalizer.py -v`
- Run the cleaner: `python clean_stress_data.py <input.csv> -o <output.csv>`
- Generate test data: `python generate_fake_data.py -o sample.csv --messy --rows 50 --seed 42`

## Architecture

Three-stage pipeline: **normalize** → **filter outliers** → **aggregate**.

- `cleaning/constants.py` — All domain constants: `CANONICAL_COLUMNS`, `COLUMN_ALIASES`, `VALID_TEST_TYPES`, `GARBAGE_VALUES`, `PHYSICAL_LIMITS`, `PASS_FAIL_MAP`, `TIMESTAMP_FORMATS`
- `cleaning/normalizer.py` — Stage 1. Entry point: `normalize(df) → (df, warnings)`. Maps columns, detects garbage/null values, normalizes test types (rejects invalid ones), converts units, parses timestamps.
- `cleaning/outliers.py` — Stage 2. Entry point: `filter_outliers(df) → (df, OutlierReport)`. Removes header rows, invalid test types, physical violations, IQR outliers, duplicates — in that order.
- `cleaning/aggregator.py` — Stage 3. Entry point: `aggregate(df) → summary_df`. Groups by (chip_id, test_type), computes statistics.
- `cleaning/validators.py` — Pure validation functions used by outliers.py. `validate_physical_limits()`, `validate_test_types()`, `validate_required_columns()`.
- `clean_stress_data.py` — CLI that wires the three stages together.
- `generate_fake_data.py` — Generates realistic messy DDR stress test CSVs with ATE artifacts, JEDEC dumps, garbage rows, etc.

## Code conventions

- All text columns lowercased during normalization except `chip_id` and `timestamp`.
- Garbage detection is centralized in `GARBAGE_VALUES` (constants.py). Add new null-like patterns there.
- Invalid test_type values are set to NaN by the normalizer, then removed by the outlier stage.
- Tests use pytest fixtures from `tests/conftest.py`: `clean_df` (well-formed data), `messy_df` (mixed formats), `tmp_csv` (temp file helper).
- Test classes are organized by feature: `TestColumnMapping`, `TestGarbageHandling`, `TestIQRFilter`, etc.

## Key invariants

- After cleaning, `test_type` only contains values from `VALID_TEST_TYPES`.
- `GARBAGE_VALUES` strings in any column become NaN during normalization.
- Physical limits in `PHYSICAL_LIMITS` are hard boundaries — rows outside them are dropped.
- Order of operations in outlier stage matters: header rows → invalid test types → physical limits → IQR → duplicates.
