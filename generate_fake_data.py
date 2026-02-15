#!/usr/bin/env python3
"""Generate fake DDR memory stress test CSV data for testing.

Usage:
    python generate_fake_data.py [-o output.csv] [--rows 200] [--chips 10]
                                 [--messy] [--outlier-pct 5] [--duplicate-pct 3]
                                 [--seed 42]
"""

import argparse
import random
import string
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from cleaning.constants import VALID_TEST_TYPES, COLUMN_ALIASES


# Realistic distributions per test type
TEST_PROFILES = {
    "burn_in": {
        "temperature_c": (125, 5),     # mean, std
        "voltage_v": (1.4, 0.05),      # elevated voltage
        "cycles": (1000, 200),
        "duration_s": (86400, 7200),    # ~24 hours
        "pass_rate": 0.92,
    },
    "retention": {
        "temperature_c": (85, 10),
        "voltage_v": (1.2, 0.03),      # nominal
        "cycles": (500000, 100000),     # high cycle counts
        "duration_s": (43200, 3600),    # ~12 hours
        "pass_rate": 0.95,
    },
    "thermal_cycling": {
        "temperature_c": (50, 80),      # wide range
        "voltage_v": (1.2, 0.02),
        "cycles": (2000, 500),
        "duration_s": (72000, 14400),   # ~20 hours
        "pass_rate": 0.88,
    },
    "endurance": {
        "temperature_c": (25, 5),       # nominal temp
        "voltage_v": (1.2, 0.03),
        "cycles": (1000000, 200000),    # high cycles
        "duration_s": (36000, 7200),    # ~10 hours
        "pass_rate": 0.90,
    },
    "htol": {
        "temperature_c": (150, 10),
        "voltage_v": (1.5, 0.05),
        "cycles": (5000, 1000),
        "duration_s": (360000, 36000),  # ~100 hours
        "pass_rate": 0.85,
    },
    "electromigration": {
        "temperature_c": (200, 15),
        "voltage_v": (1.8, 0.1),
        "cycles": (10000, 2000),
        "duration_s": (172800, 28800),  # ~48 hours
        "pass_rate": 0.80,
    },
}

# Column name variations for messy mode
MESSY_COLUMN_NAMES = {
    "chip_id": ["Chip ID", "ChipID", "chip_id", "Device ID", "DUT_ID", "Sample ID"],
    "test_type": ["Test Type", "TestType", "test_type", "Stress Type", "Test Name"],
    "temperature_c": ["Temp (C)", "Temperature", "temp_c", "Temperature (C)", "Temp"],
    "voltage_v": ["Voltage (V)", "Voltage", "voltage_v", "VDD", "Supply Voltage"],
    "cycles": ["Cycles", "Cycle Count", "cycles", "Num Cycles", "Total Cycles"],
    "duration_s": ["Duration (s)", "Duration", "duration_s", "Time (s)", "Elapsed Time"],
    "pass_fail": ["Pass/Fail", "Result", "pass_fail", "Status", "Outcome"],
    "timestamp": ["Timestamp", "Date", "timestamp", "DateTime", "Test Date"],
}

MESSY_PASS_VALUES = ["Pass", "PASS", "pass", "P", "p", "1", "true", "True", "YES"]
MESSY_FAIL_VALUES = ["Fail", "FAIL", "fail", "F", "f", "0", "false", "False", "NO"]

MESSY_TIMESTAMP_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%Y-%m-%d",
]

# --- Memory controller gibberish and dump artifacts ---

# Hex register dumps that might leak into CSV exports from memory controllers
MC_REGISTER_DUMPS = [
    "0xDEADBEEF",
    "0x00FF00FF",
    "0xCAFEBABE",
    "0xBAADF00D",
    "0x8BADF00D",
    "0xFEEDFACE",
    "0x0D15EA5E",
    "0xC0FFEE00",
    "0xFFFFFFFF",
    "0x00000000",
    "REG[0x3A]=0x1F",
    "MR4=0x05",
    "MR5=0x00",
    "DRAM_CFG:0x4012A",
    "ERR_STS:0x0000",
]

# JEDEC/LPDDR log artifacts — SPD, mode registers, timing params
JEDEC_ARTIFACTS = [
    # SPD register dumps (DDR5 SPD is 1024 bytes, these are typical field reads)
    "SPD:BYTE0=0x51",
    "SPD:BYTE1=0x11",
    "SPD[2]=0x12",       # DDR5 device type
    "SPD[17]=0x08",      # module type
    "SPD_VENDOR=0x2C00",  # Micron
    "SPD_VENDOR=0xAD00",  # SK Hynix
    "SPD_VENDOR=0xCE00",  # Samsung
    "SPD_CRC_FAIL",
    "SPD_READ_TIMEOUT",
    # Mode register reads (DDR5 has MR0-MR63+)
    "MR0=0x0034",
    "MR1=0x0001",
    "MR2=0x0000",
    "MR3=0x0031",
    "MR4=0x0000",
    "MR5=0x0240",
    "MR8=0x18",           # die density / IO width
    "MR12=0x004D",        # Vref training
    "MR13=0x0000",
    "MR14=0x004D",
    "MR32=0x00",
    "MR40=0x00",
    "MR46=DQS_INTERVAL=0x10",
    "MR63=0xFF",          # LPDDR ZQ reset
    # Timing parameter strings
    "tCK=0.625ns",
    "tRCD=13.75ns",
    "tRP=13.75ns",
    "tRAS=32ns",
    "tRC=45.75ns",
    "tRFC=295ns",
    "tRFC2=160ns",
    "tREFI=3.9us",
    "tWR=30ns",
    "tWTR_L=12ns",
    "tRTP=12ns",
    "tCCD_L=8tCK",
    "tFAW=20ns",
    "tXP=7.5ns",
    "CL=22",
    "CWL=20",
    "CL-nRCD-nRP=22-22-22",
    "SPEED_BIN=DDR5-4800",
    "SPEED_BIN=DDR5-5600",
    "SPEED_BIN=DDR5-6400",
    "SPEED_BIN=LPDDR5-6400",
    # LPDDR-specific
    "LPDDR5_WCK2DQI_FAIL",
    "LPDDR5_WCK2DQO_FAIL",
    "LPDDR_FSP0_ACTIVE",
    "LPDDR_FSP1_SWITCH_ERR",
    "DVFSC_MODE=DISABLED",
    "DVFSC_TRANSITION_ERR",
]

# ATE (Automatic Test Equipment) error codes and artifacts
ATE_ARTIFACTS = [
    # Bin classifications from tester
    "BIN1_PASS",
    "BIN2_PASS_DOWNGRADE",
    "BIN3_FAIL_FUNC",
    "BIN4_FAIL_PARAM",
    "BIN5_FAIL_LEAK",
    "BIN6_FAIL_IDDQ",
    "BIN7_FAIL_SCAN",
    "BIN8_FAIL_CONTACT",
    "BIN9_RETEST",
    "BIN99_ABORT",
    # Lot/wafer identifiers that bleed into wrong columns
    "LOT:T24A012.3",
    "LOT:M23K456.1",
    "WAFER:W12",
    "WAFER:W03-EDGE",
    "WAFER:W25_CENTER",
    "DIE:X12Y34",
    "DIE:X03Y45_CORNER",
    "SITE:S1",
    "SITE:S8",
    "SITE:S16_RETEST",
    # Tester error/status messages
    "TESTER_TIMEOUT_300s",
    "TESTER_OVERFLOW",
    "FORCE_V_CLAMP",
    "MEASURE_I_OVERRANGE",
    "DUT_SHORT_DETECTED",
    "CONTACT_FAIL_PIN_A5",
    "CONTACT_FAIL_PIN_DQ7",
    "PROBER_ALIGN_ERR",
    "HANDLER_JAM",
    "SOCKET_CONTACT_WARN",
    # Partial test results from interrupted runs
    "TEST_ABORTED@STEP42",
    "TEST_ABORTED@STEP117",
    "INTERRUPTED:POWER_GLITCH",
    "INTERRUPTED:THERMAL_SHUTDOWN",
    "INCOMPLETE:TIMEOUT",
    "SKIP:PREV_FAIL",
    "SKIP:SAMPLE_PLAN",
    "RETEST:MARGINAL",
    # ATE measurement strings that leak in
    "Idd0=45.2mA",
    "Idd2p=1.8mA",
    "Idd4r=230mA",
    "Idd6=12.5mA",
    "Vmin=0.95V",
    "Vmin=FAIL@0.90V",
    "Shmoo:PASS@(1.1V,125C)",
    "Shmoo:FAIL@(0.85V,125C)",
]

# Garbled strings from memory controller UART/serial log leaking into data
MC_GIBBERISH = [
    "READ_TIMEOUT",
    "ECC_CORRECTED",
    "ECC_UNCORRECTABLE",
    "REFRESH_ERR",
    "CMD_BUS_ERR",
    "DQ_CALIBRATION_FAIL",
    "tRFC_VIOLATION",
    "PHY_INIT_INCOMPLETE",
    "WRITE_LEVELING_ERR",
    "ZQ_CAL_TIMEOUT",
    "ROW_HAMMER_DETECTED",
    "BANK_CONFLICT",
    "PAGE_MISS",
    "AUTO_PRECHARGE",
    "---",
    "N/A",
    "???",
    "#REF!",
    "#VALUE!",
    "#N/A",
    "ERR",
    "NULL",
    "nil",
    "-",
    ".",
    "--",
    "n/a",
    "NA",
    "NaN",
]

# Raw memory dump hex lines that might end up as cell values
MEMORY_DUMP_LINES = [
    "FF FF FF FF FF FF FF FF",
    "00 A3 5C 12 00 00 FF 01",
    "DE AD BE EF CA FE BA BE",
    "41 42 43 44 00 00 00 00",
    "7F 45 4C 46 01 01 01 00",
    "ADDR:0x7FFF0000 DATA:0x12345678",
    "BIT_FLIP@0x0040_2C00",
    "CE@RANK0_BG1_BA2_ROW0x1A3F_COL0x120",
    "DIMM0_CH0_CS0",
]

# Extra columns that might appear from memory dump tools, ATE, or JEDEC logs
DUMP_EXTRA_COLUMNS = {
    "mc_status": lambda rng: rng.choice(
        ["OK", "WARN", "ERR", "0x00", "0xFF", "IDLE", "BUSY", "REFRESH", ""]
    ),
    "ecc_count": lambda rng: rng.choice(
        [0, 0, 0, 0, 1, 2, "", "N/A", None, 0, 0, 3, 15, "0x0F"]
    ),
    "die_location": lambda rng: rng.choice(
        ["TOP_LEFT", "TOP_RIGHT", "BOT_LEFT", "BOT_RIGHT", "", None,
         "WAFER_23_DIE_4A", "LOT#A2B3", "X12Y34", "X03Y45_CORNER"]
    ),
    "raw_ber": lambda rng: rng.choice(
        [1e-15, 3.2e-12, 0, "", "< 1e-18", "BELOW_THRESHOLD", None, 5.1e-10, "1.2E-14"]
    ),
    "fw_version": lambda rng: rng.choice(
        ["v2.1.3", "v2.1.4", "2.1.3-rc1", "", None, "UNKNOWN", "v2.0.0-dirty"]
    ),
    "bin_result": lambda rng: rng.choice(
        ["BIN1_PASS", "BIN1_PASS", "BIN1_PASS", "BIN2_PASS_DOWNGRADE",
         "BIN3_FAIL_FUNC", "BIN4_FAIL_PARAM", "BIN8_FAIL_CONTACT",
         "BIN9_RETEST", "", None, "SKIP:SAMPLE_PLAN"]
    ),
    "lot_id": lambda rng: rng.choice(
        [f"T24{rng.choice('ABCDEF')}{rng.randint(100,999)}.{rng.randint(1,9)}",
         f"M23{rng.choice('JKLM')}{rng.randint(100,999)}.{rng.randint(1,9)}",
         "", None, "N/A", f"LOT:T24A{rng.randint(100,999)}.3"]
    ),
    "wafer_id": lambda rng: rng.choice(
        [f"W{rng.randint(1,25):02d}", f"W{rng.randint(1,25):02d}-EDGE",
         f"W{rng.randint(1,25):02d}_CENTER", "", None]
    ),
    "site_num": lambda rng: rng.choice(
        [f"S{rng.randint(1,16)}", f"S{rng.randint(1,16)}_RETEST",
         "", None, "N/A", 0, rng.randint(1, 16)]
    ),
    "spd_vendor": lambda rng: rng.choice(
        ["0x2C00", "0xAD00", "0xCE00", "Micron", "SK_Hynix", "Samsung",
         "", None, "SPD_CRC_FAIL", "SPD_READ_TIMEOUT"]
    ),
    "speed_bin": lambda rng: rng.choice(
        ["DDR5-4800", "DDR5-5600", "DDR5-6400", "LPDDR5-6400",
         "DDR5-4800B", "DDR5-5600A", "", None, "UNKNOWN"]
    ),
    "idd_measurement": lambda rng: rng.choice(
        ["Idd0=45.2mA", "Idd2p=1.8mA", "Idd4r=230mA", "Idd6=12.5mA",
         "", None, "OVERRANGE", "MEASURE_FAIL",
         f"Idd0={rng.uniform(30, 60):.1f}mA"]
    ),
    "tester_status": lambda rng: rng.choice(
        ["OK", "OK", "OK", "TIMEOUT", "CLAMP", "OVERRANGE",
         "CONTACT_WARN", "", None, "HANDLER_JAM", "PROBER_ALIGN_ERR"]
    ),
}

# Whitespace patterns that plague real CSV exports
WHITESPACE_JUNK = [
    "  ",           # leading spaces
    "\t",           # tab
    " \t ",         # mixed
    "  \n",         # trailing newline (will be stripped by CSV reader usually)
    "\u00A0",       # non-breaking space
    "\u200B",       # zero-width space
]


def _random_hex_string(rng, length=8):
    """Generate a random hex string like a register dump."""
    return "0x" + "".join(rng.choices("0123456789ABCDEF", k=length))


def _corrupt_value(rng, value):
    """Randomly corrupt a value to simulate memory controller / ATE artifacts."""
    corruption = rng.choice([
        "hex_prefix",
        "gibberish",
        "register_dump",
        "memory_dump",
        "partial_garbage",
        "whitespace_wrap",
        "spreadsheet_error",
        "unicode_junk",
        "jedec_artifact",
        "ate_artifact",
        "ate_measurement",
    ])

    if corruption == "hex_prefix":
        return _random_hex_string(rng)
    elif corruption == "gibberish":
        return rng.choice(MC_GIBBERISH)
    elif corruption == "register_dump":
        return rng.choice(MC_REGISTER_DUMPS)
    elif corruption == "memory_dump":
        return rng.choice(MEMORY_DUMP_LINES)
    elif corruption == "partial_garbage":
        s = str(value)
        garbage = "".join(rng.choices(string.ascii_letters + string.digits + "!@#$%", k=rng.randint(2, 8)))
        return rng.choice([f"{s}{garbage}", f"{garbage}{s}", f"{s}_{garbage}"])
    elif corruption == "whitespace_wrap":
        return rng.choice(WHITESPACE_JUNK) + str(value) + rng.choice(WHITESPACE_JUNK)
    elif corruption == "spreadsheet_error":
        return rng.choice(["#REF!", "#VALUE!", "#N/A", "#DIV/0!", "#NAME?", "=SUM()", "=#REF!"])
    elif corruption == "unicode_junk":
        junk_chars = rng.choices(["\u00B0", "\u2103", "\u00B5", "\u03A9", "\u2126"], k=rng.randint(1, 3))
        return str(value) + "".join(junk_chars)
    elif corruption == "jedec_artifact":
        return rng.choice(JEDEC_ARTIFACTS)
    elif corruption == "ate_artifact":
        return rng.choice(ATE_ARTIFACTS)
    elif corruption == "ate_measurement":
        # ATE measurement string leaking into the wrong column
        return rng.choice([
            f"Idd0={rng.uniform(30, 60):.1f}mA",
            f"Vmin={rng.uniform(0.85, 1.1):.2f}V",
            f"Shmoo:PASS@({rng.uniform(0.9, 1.2):.1f}V,{rng.randint(25, 150)}C)",
            f"Shmoo:FAIL@({rng.uniform(0.7, 0.9):.2f}V,{rng.randint(25, 150)}C)",
        ])

    return value


def _generate_rows(rng, np_rng, num_rows, num_chips, messy):
    """Generate the base data rows."""
    test_types = list(VALID_TEST_TYPES)
    chip_ids = [f"DDR5-{i:04d}" for i in range(1, num_chips + 1)]

    base_time = datetime(2024, 1, 1)
    rows = []

    for row_idx in range(num_rows):
        test_type = rng.choice(test_types)
        chip_id = rng.choice(chip_ids)
        profile = TEST_PROFILES[test_type]

        temp = max(-60, np_rng.normal(profile["temperature_c"][0], profile["temperature_c"][1]))
        voltage = max(0.3, np_rng.normal(profile["voltage_v"][0], profile["voltage_v"][1]))
        cycles = max(0, int(np_rng.normal(profile["cycles"][0], profile["cycles"][1])))
        duration = max(0, np_rng.normal(profile["duration_s"][0], profile["duration_s"][1]))
        passed = rng.random() < profile["pass_rate"]
        ts = base_time + timedelta(seconds=rng.randint(0, 31536000))

        if messy:
            # Randomize pass/fail format
            if passed:
                pass_fail = rng.choice(MESSY_PASS_VALUES)
            else:
                pass_fail = rng.choice(MESSY_FAIL_VALUES)

            # Randomize timestamp format
            ts_fmt = rng.choice(MESSY_TIMESTAMP_FORMATS)
            ts_str = ts.strftime(ts_fmt)

            # Occasionally use alternate test type formatting
            if rng.random() < 0.3:
                test_type = test_type.replace("_", " ")
            elif rng.random() < 0.3:
                test_type = test_type.replace("_", "-")
            elif rng.random() < 0.2:
                test_type = test_type.upper()
        else:
            pass_fail = "pass" if passed else "fail"
            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "chip_id": chip_id,
            "test_type": test_type,
            "temperature_c": round(temp, 2),
            "voltage_v": round(voltage, 4),
            "cycles": cycles,
            "duration_s": round(duration, 1),
            "pass_fail": pass_fail,
            "timestamp": ts_str,
        }

        if messy:
            # ~5% chance: leave fields blank (empty string)
            if rng.random() < 0.05:
                blank_col = rng.choice(["chip_id", "duration_s", "cycles", "voltage_v"])
                row[blank_col] = ""

            # ~4% chance: inject NULL-like placeholders instead of actual missing
            if rng.random() < 0.04:
                null_col = rng.choice(["chip_id", "test_type", "temperature_c", "duration_s", "cycles"])
                row[null_col] = rng.choice(["NULL", "null", "None", "N/A", "n/a", "NA", "-", "nil", ""])

            # ~5% chance: inject memory controller gibberish into a field
            if rng.random() < 0.05:
                gibberish_col = rng.choice(["test_type", "pass_fail", "chip_id", "temperature_c", "voltage_v"])
                row[gibberish_col] = _corrupt_value(rng, row[gibberish_col])

            # ~3% chance: pad values with whitespace junk
            if rng.random() < 0.03:
                ws_col = rng.choice(list(row.keys()))
                ws = rng.choice(WHITESPACE_JUNK)
                row[ws_col] = f"{ws}{row[ws_col]}{ws}"

            # ~3% chance: voltage in millivolts instead of volts
            if rng.random() < 0.03:
                row["voltage_v"] = round(voltage * 1000, 1)

            # ~2% chance: temperature in Kelvin
            if rng.random() < 0.02:
                row["temperature_c"] = round(temp + 273.15, 2)

            # ~2% chance: hex register value leaking into a numeric field
            if rng.random() < 0.02:
                hex_col = rng.choice(["cycles", "duration_s", "temperature_c"])
                row[hex_col] = _random_hex_string(rng)

            # ~2% chance: entire row is a garbled memory dump line
            if rng.random() < 0.02:
                dump_line = rng.choice(MEMORY_DUMP_LINES)
                for col in row:
                    row[col] = dump_line

            # ~3% chance: JEDEC/SPD artifact leaks into a field
            if rng.random() < 0.03:
                jedec_col = rng.choice(["test_type", "chip_id", "pass_fail", "temperature_c", "voltage_v"])
                row[jedec_col] = rng.choice(JEDEC_ARTIFACTS)

            # ~3% chance: ATE error/bin code leaks into a field
            if rng.random() < 0.03:
                ate_col = rng.choice(["pass_fail", "test_type", "chip_id", "duration_s"])
                row[ate_col] = rng.choice(ATE_ARTIFACTS)

            # ~2% chance: lot/wafer ID bleeds into chip_id or test_type
            if rng.random() < 0.02:
                row["chip_id"] = rng.choice([
                    f"LOT:T24A{rng.randint(100,999)}.{rng.randint(1,9)}",
                    f"W{rng.randint(1,25):02d}_X{rng.randint(0,30)}Y{rng.randint(0,30)}",
                    f"DIMM0_CH{rng.randint(0,1)}_CS{rng.randint(0,1)}",
                ])

            # ~2% chance: partial row from interrupted test (some fields truncated)
            if rng.random() < 0.02:
                cutoff = rng.randint(2, 5)
                keys = list(row.keys())
                for k in keys[cutoff:]:
                    row[k] = rng.choice(["", None, "INCOMPLETE:TIMEOUT",
                                         "TEST_ABORTED", "---"])

        rows.append(row)

    return rows


def _inject_outliers(rows, rng, np_rng, outlier_pct):
    """Inject physically impossible values into a percentage of rows."""
    num_outliers = max(1, int(len(rows) * outlier_pct / 100))
    indices = rng.sample(range(len(rows)), min(num_outliers, len(rows)))

    for idx in indices:
        anomaly = rng.choice(["temp", "voltage", "cycles", "duration"])
        if anomaly == "temp":
            rows[idx]["temperature_c"] = rng.choice([999, -200, 500, -100])
        elif anomaly == "voltage":
            rows[idx]["voltage_v"] = rng.choice([-1.0, 0, 15.0, 100.0])
        elif anomaly == "cycles":
            rows[idx]["cycles"] = rng.choice([-1000, 99_999_999_999])
        elif anomaly == "duration":
            rows[idx]["duration_s"] = rng.choice([-500, 999_999_999])

    return rows


def _inject_duplicates(rows, rng, duplicate_pct):
    """Inject exact duplicate rows."""
    num_dupes = max(1, int(len(rows) * duplicate_pct / 100))
    for _ in range(num_dupes):
        src = rng.choice(rows)
        rows.append(dict(src))
    rng.shuffle(rows)
    return rows


def _inject_extra_columns(df, rng):
    """Add extra unrecognized columns that might come from memory dump tools."""
    # Pick 2-4 random extra columns
    extras = rng.sample(list(DUMP_EXTRA_COLUMNS.keys()), k=rng.randint(2, min(4, len(DUMP_EXTRA_COLUMNS))))
    for col_name in extras:
        generator = DUMP_EXTRA_COLUMNS[col_name]
        df[col_name] = [generator(rng) for _ in range(len(df))]
    return df


def _inject_garbage_rows(rows, rng, count):
    """Inject completely garbled rows from memory dumps, ATE logs, JEDEC artifacts."""
    garbage_types = [
        "all_hex", "dump_line", "controller_log", "empty_ish",
        "header_repeat", "jedec_spd_row", "ate_status_row",
        "ate_partial_abort", "spd_register_block",
    ]

    for _ in range(count):
        garbage_type = rng.choice(garbage_types)

        if garbage_type == "all_hex":
            row = {k: _random_hex_string(rng) for k in rows[0].keys()}
        elif garbage_type == "dump_line":
            dump = rng.choice(MEMORY_DUMP_LINES)
            row = {k: dump for k in rows[0].keys()}
        elif garbage_type == "controller_log":
            log_msg = rng.choice(MC_GIBBERISH)
            row = {k: log_msg for k in rows[0].keys()}
        elif garbage_type == "empty_ish":
            row = {k: rng.choice(["", " ", "  ", "\t", "-", ".", None]) for k in rows[0].keys()}
        elif garbage_type == "header_repeat":
            row = {k: k for k in rows[0].keys()}
        elif garbage_type == "jedec_spd_row":
            # An entire row that's SPD/timing data pasted in
            jedec = rng.choice(JEDEC_ARTIFACTS)
            row = {k: jedec for k in rows[0].keys()}
        elif garbage_type == "ate_status_row":
            # ATE tester status line dumped as a data row
            ate = rng.choice(ATE_ARTIFACTS)
            row = {k: ate for k in rows[0].keys()}
        elif garbage_type == "ate_partial_abort":
            # Partial row from an ATE abort — first few fields valid, rest is error
            keys = list(rows[0].keys())
            valid_src = rng.choice(rows)
            row = {}
            cutoff = rng.randint(1, 3)
            for i, k in enumerate(keys):
                if i < cutoff:
                    row[k] = valid_src[k]
                else:
                    row[k] = rng.choice([
                        "TEST_ABORTED@STEP42", "INTERRUPTED:POWER_GLITCH",
                        "INTERRUPTED:THERMAL_SHUTDOWN", "INCOMPLETE:TIMEOUT",
                        "SKIP:PREV_FAIL", "", None,
                    ])
        elif garbage_type == "spd_register_block":
            # SPD register block pasted as a row
            keys = list(rows[0].keys())
            row = {}
            for i, k in enumerate(keys):
                row[k] = f"SPD[{i*16}..{(i+1)*16-1}]=" + " ".join(
                    f"{rng.randint(0,255):02X}" for _ in range(min(4, 16))
                ) + "..."

        pos = rng.randint(0, len(rows))
        rows.insert(pos, row)

    return rows


def _apply_messy_columns(df, rng):
    """Rename columns to random variations."""
    rename_map = {}
    for canonical_col in df.columns:
        if canonical_col in MESSY_COLUMN_NAMES:
            rename_map[canonical_col] = rng.choice(MESSY_COLUMN_NAMES[canonical_col])
        else:
            rename_map[canonical_col] = canonical_col
    return df.rename(columns=rename_map)


def generate(
    num_rows=200,
    num_chips=10,
    messy=False,
    outlier_pct=0,
    duplicate_pct=0,
    seed=None,
):
    """Generate fake DDR stress test data.

    Returns:
        pd.DataFrame
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    rows = _generate_rows(rng, np_rng, num_rows, num_chips, messy)

    if outlier_pct > 0:
        rows = _inject_outliers(rows, rng, np_rng, outlier_pct)

    if duplicate_pct > 0:
        rows = _inject_duplicates(rows, rng, duplicate_pct)

    # In messy mode, inject garbage rows (~3% of total)
    if messy and len(rows) > 10:
        garbage_count = max(2, int(len(rows) * 0.03))
        rows = _inject_garbage_rows(rows, rng, garbage_count)

    df = pd.DataFrame(rows)

    # In messy mode, add extra unrecognized columns from dump tools
    if messy:
        df = _inject_extra_columns(df, rng)
        df = _apply_messy_columns(df, rng)

    return df


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Generate fake DDR memory stress test CSV data."
    )
    parser.add_argument(
        "-o", "--output", default="fake_stress_data.csv",
        help="Output CSV path (default: fake_stress_data.csv)",
    )
    parser.add_argument(
        "--rows", type=int, default=200,
        help="Number of base rows to generate (default: 200)",
    )
    parser.add_argument(
        "--chips", type=int, default=10,
        help="Number of unique chip IDs (default: 10)",
    )
    parser.add_argument(
        "--messy", action="store_true",
        help="Generate messy data with column name variations and mixed formats",
    )
    parser.add_argument(
        "--outlier-pct", type=float, default=0,
        help="Percentage of rows with physically impossible values (default: 0)",
    )
    parser.add_argument(
        "--duplicate-pct", type=float, default=0,
        help="Percentage of exact duplicate rows to inject (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    df = generate(
        num_rows=args.rows,
        num_chips=args.chips,
        messy=args.messy,
        outlier_pct=args.outlier_pct,
        duplicate_pct=args.duplicate_pct,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows → {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
