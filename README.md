# clean-stress-data

A Python CLI tool for cleaning DDR memory stress test CSV data. Handles messy real-world exports from ATE equipment, memory controllers, and spreadsheet tools — normalizing column names, detecting garbage values, filtering outliers, removing duplicates, and producing per-chip summaries.

## Quick start

```bash
pip install -r requirements.txt

# Clean a CSV file
python clean_stress_data.py input.csv -o cleaned.csv --summary-output summary.csv

# Generate sample messy data for testing
python generate_fake_data.py -o sample.csv --messy --rows 200 --seed 42
```

## Pipeline

The cleaner runs three stages in order:

### Stage 1 — Normalize (`cleaning/normalizer.py`)

- Maps column name variants (e.g. `Chip ID`, `DUT`, `Device ID`) to canonical names
- Converts garbage/null-like strings (`nil`, `null`, `#REF!`, `---`, etc.) to NaN
- Parses timestamps across multiple formats
- Normalizes test type strings and rejects invalid ones (hex dumps, ATE codes, measurement strings)
- Converts millivolt values to volts, Kelvin to Celsius
- Coerces pass/fail variants (`Pass`, `1`, `true`, `Y`) to boolean
- Fills missing chip IDs with `UNKNOWN_<row>` placeholders
- Removes all-garbage and header-repeat rows

### Stage 2 — Filter outliers (`cleaning/outliers.py`)

- Removes header-repeat rows (chip_id equals a column name)
- Removes rows with invalid (NaN) test types
- Removes physically impossible values (temperature > 300C, negative cycles, etc.)
- IQR-based outlier detection per numeric column, grouped by test type
- Duplicate row removal based on (chip_id, test_type, timestamp)

### Stage 3 — Aggregate (`cleaning/aggregator.py`)

- Groups by (chip_id, test_type)
- Computes pass/fail counts, pass rate
- Per numeric column: mean, median, std, min, max

## Valid test types

`retention`, `endurance`, `burn_in`, `thermal_cycling`, `htol`, `electromigration`

## CLI options

```
python clean_stress_data.py input.csv [options]

  -o, --output          Output CSV path (default: <input>_cleaned.csv)
  --summary-output      Summary CSV path (default: <input>_summary.csv)
  --no-summary          Skip summary generation
  --iqr-multiplier      IQR multiplier for outlier detection (default: 1.5)
  --keep-duplicates     Do not remove duplicate rows
  --strict              Exit with code 1 on any validation warnings
  --temp-min/--temp-max Override temperature limits
  --voltage-max         Override voltage limit
  --log-level           DEBUG, INFO, WARNING, ERROR (default: INFO)
```

## Fake data generator

```
python generate_fake_data.py [options]

  -o, --output       Output path (default: fake_stress_data.csv)
  --rows             Number of rows (default: 200)
  --chips            Number of unique chip IDs (default: 10)
  --messy            Add column name variations, garbage rows, ATE/JEDEC artifacts
  --outlier-pct      Percentage of physically impossible values (default: 0)
  --duplicate-pct    Percentage of duplicate rows (default: 0)
  --seed             Random seed for reproducibility
```

## Testing

```bash
pytest tests/ -v
```

## Project structure

```
clean_stress_data.py          # CLI entry point
generate_fake_data.py         # Fake data generator
cleaning/
  constants.py                # Column aliases, valid types, physical limits, garbage values
  normalizer.py               # Stage 1: parse and normalize
  outliers.py                 # Stage 2: filter outliers, duplicates
  aggregator.py               # Stage 3: aggregate summaries
  validators.py               # Validation helpers (physical limits, test types)
tests/
  conftest.py                 # Shared fixtures
  test_normalizer.py          # Normalizer tests (column mapping, garbage handling, etc.)
  test_outliers.py            # Outlier filter tests (physical, IQR, duplicates, invalid types)
  test_aggregator.py          # Aggregation tests
  test_validators.py          # Validator tests
  test_cli.py                 # End-to-end CLI tests
  test_constants.py           # Constants sanity checks
  test_generate_fake_data.py  # Data generator tests
```
