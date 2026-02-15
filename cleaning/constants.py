"""Domain constants for DDR memory stress test data cleaning."""

CANONICAL_COLUMNS = [
    "chip_id",
    "test_type",
    "temperature_c",
    "voltage_v",
    "cycles",
    "duration_s",
    "pass_fail",
    "timestamp",
]

# Maps various column name variations to canonical names.
# Lookup is done case- and whitespace-insensitively.
COLUMN_ALIASES = {
    # chip_id
    "chip_id": "chip_id",
    "chipid": "chip_id",
    "chip id": "chip_id",
    "chip": "chip_id",
    "chip_number": "chip_id",
    "chip number": "chip_id",
    "device_id": "chip_id",
    "device id": "chip_id",
    "dut_id": "chip_id",
    "dut id": "chip_id",
    "dut": "chip_id",
    "sample_id": "chip_id",
    "sample id": "chip_id",
    "id": "chip_id",
    # test_type
    "test_type": "test_type",
    "testtype": "test_type",
    "test type": "test_type",
    "test": "test_type",
    "type": "test_type",
    "stress_type": "test_type",
    "stress type": "test_type",
    "test_name": "test_type",
    "test name": "test_type",
    # temperature_c
    "temperature_c": "temperature_c",
    "temperature": "temperature_c",
    "temp_c": "temperature_c",
    "temp (c)": "temperature_c",
    "temp(c)": "temperature_c",
    "temp": "temperature_c",
    "temperature (c)": "temperature_c",
    "temperature(c)": "temperature_c",
    "temp_celsius": "temperature_c",
    "temperature_celsius": "temperature_c",
    # voltage_v
    "voltage_v": "voltage_v",
    "voltage": "voltage_v",
    "volt": "voltage_v",
    "voltage (v)": "voltage_v",
    "voltage(v)": "voltage_v",
    "volt_v": "voltage_v",
    "vdd": "voltage_v",
    "supply_voltage": "voltage_v",
    "supply voltage": "voltage_v",
    # cycles
    "cycles": "cycles",
    "cycle_count": "cycles",
    "cycle count": "cycles",
    "num_cycles": "cycles",
    "num cycles": "cycles",
    "cycle": "cycles",
    "total_cycles": "cycles",
    "total cycles": "cycles",
    # duration_s
    "duration_s": "duration_s",
    "duration": "duration_s",
    "duration (s)": "duration_s",
    "duration(s)": "duration_s",
    "time_s": "duration_s",
    "time (s)": "duration_s",
    "time(s)": "duration_s",
    "elapsed_s": "duration_s",
    "elapsed_time": "duration_s",
    "elapsed time": "duration_s",
    "test_duration": "duration_s",
    "test duration": "duration_s",
    # pass_fail
    "pass_fail": "pass_fail",
    "pass/fail": "pass_fail",
    "passfail": "pass_fail",
    "pass fail": "pass_fail",
    "result": "pass_fail",
    "status": "pass_fail",
    "test_result": "pass_fail",
    "test result": "pass_fail",
    "outcome": "pass_fail",
    # timestamp
    "timestamp": "timestamp",
    "time_stamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "date time": "timestamp",
    "test_date": "timestamp",
    "test date": "timestamp",
    "test_time": "timestamp",
    "tested_at": "timestamp",
}

VALID_TEST_TYPES = {
    "retention",
    "endurance",
    "burn_in",
    "thermal_cycling",
    "htol",
    "electromigration",
}

# Strings that should always be treated as NaN/missing.
GARBAGE_VALUES = {
    # Null-like
    "null", "nil", "na", "n/a", "nan", "none", "",
    "undefined", "unknown", "missing",
    # Placeholders
    "-", "--", "---", ".", "???",
    # Spreadsheet errors
    "#ref!", "#value!", "#n/a", "#div/0!", "#name?", "#num!", "#null!",
    "=sum()", "=#ref!",
}

# Hard physical limits: values outside these are physically impossible.
# Format: (min_inclusive, max_inclusive) — None means unbounded on that side.
PHYSICAL_LIMITS = {
    "temperature_c": (-65, 300),
    "voltage_v": (0, 5.0),       # 0 is exclusive (handled in validator)
    "cycles": (0, 10_000_000_000),
    "duration_s": (0, 31_536_000),
}

PASS_FAIL_MAP = {
    "pass": True,
    "fail": False,
    "p": True,
    "f": False,
    "1": True,
    "0": False,
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "y": True,
    "n": False,
}

TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y%m%d%H%M%S",
    "%Y%m%d",
]
