"""main/engine/process/DB_process_config.py

Config + logging setup for the DB_DATA_PROCESS refactor.

This module intentionally preserves the original logging.basicConfig() side-effects
from DB_DATA_PROCESS.py so running the new runner yields identical log formatting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict


# ---------------------------------------------------------------------
# Logging config (must match original)
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%I:%M:%S %p",
    force=True,
)


def cfg() -> Dict[str, Any]:
    """Central configuration and logging toggles.

    NOTE: This is copied 1:1 from DB_DATA_PROCESS.py to preserve defaults.
    """

    c: Dict[str, Any] = {
        # Warm-start / history requirements
        # You requested >=240min so bell-curve PV detection can see prior extrema.
        "WARM_START_MINUTES": 240,
        # Output for downstream consumers (DB_DATA_TREND)
        "TREND_OUT_JSON": (Path(__file__).resolve().parent / "../ts/json/process_trend_latest.json").resolve(),

        # Gaussian dump reversal-catch v1.0 (feature only; override disabled by default)
        "REV_CATCH_ENABLED": True,
        "REV_OVERRIDE_ENABLED": False,
        "REV_TAIL_SECONDS": 75,              # seconds to estimate "tail slope" from end of window
        "REV_RECENT_POINTS_MAX": 25,         # extremum must be within this many points of end
        "REV_RATIO_STRONG": 1.30,            # last-leg strength vs previous-leg strength
        "REV_MIN_LASTLEG_SLOPE": 0.020,      # price units / sec (on smoothed series)
        "REV_MIN_LASTLEG_DELTA": 4.0,        # price units
        "REV_MIN_SIGMA_CONFIRM": 2,          # min sigmas (among 8/23) agreeing for "strong" reversal
        "PACK_RANGE_MAX": 30.0,              # max spread between last sigma mids to be considered "packed"
        "G83_R2_MIN": 0.95,                  # regression quality gate for sigma 83 tail



        # Timing
        "WAKE_OFFSET_SECONDS": 12,  # target_sleep = max(0, seconds_until - 12)

        # NOTE: Legacy fixed tail snapshot windows are deprecated.
        # We now use PV-leg → NOW bell segments as the primary diagnostic window.

        # Bell-curve (Peak/Valley) tailing snapshots
        "BELL_CURVE_LOOKBACK_MINUTES": 240,   # PV search window length ending at last_ts
        "BELL_CURVE_FALLBACK_TO_FIXED": False, # if PV pair not found, print fixed tails
        "PV_REF_SIGMA": 23,                  # reference sigma used to detect PV turning points
        "PV_MIN_SEP_SECONDS": 10.0,          # minimum separation between extrema (anti-wiggle)

        # Gaussian Channel params
        "GAUSS_CHANNEL_K": 2.0,              # channel spread multiplier (robust_std)
        "LOG_GAUSS_CHANNEL_SNAPSHOT": True,  # A) snapshot style
        "LOG_GAUSS_CHANNELS": True,          # alias used by orchestrator; keep True so snapshot stays on
        "LOG_GAUSS_CHANNEL_PVTAIL": True,    # B) PV-leg vs Tail style
        "LOG_GAUSS_CHANNEL_PV_TAIL": True,  # alias-friendly: PV-leg vs Tail Gaussian channels
        # Turn FULLSTACK on by default so we can immediately verify the per-sigma PV/TAIL print path.
        "LOG_GAUSS_CHANNEL_PV_TAIL_FULLSTACK": True,  # if True, print full sigma stack per leg; else σ=23 only

        # Diagnostics / guardrails
        "DEBUG_IMPORT_PATHS": True,   # print which file each process module was imported from
        "DEBUG_PV_TAIL_STATUS": True, # emit a one-line PV-TAIL STATUS per epoch: printed / skipped / failed

        # Output paths
        "OUTPUT_DIR": Path(__file__).resolve().parent / "logs",
        "WIDE_CSV": "gauss_epoch_snapshots_wide.csv",
        "LONG_CSV": "gauss_epoch_snapshots_long.csv",

        "MODEL_hyster": {"MODEL_ID": "method_hyster_v1.0.json", "ENABLED": True},

        "SAVE_FORECAST_LOG": {"ENABLED": True},

        "PRINT": {
            "HEADER": True,
            "PERF": {"ENABLED": True},
            "TAIL_ANCHOR": {"ENABLED": True},
            "PV_REF": {"ENABLED": True},
            "MSBC": {
                "LEG1": {
                    "ENABLED": False,
                    "SUMMARY": {"ENABLED": False},
                    "SERIES": {"ENABLED": False, "MAX_POINTS": 0, "DECIMATE": False},
                },
                "LEG2": {
                    "ENABLED": True,
                    "SUMMARY": {"ENABLED": True},
                    "SERIES": {"ENABLED": False, "MAX_POINTS": 0, "DECIMATE": True},
                },
                "DIAGNOSTICS": {"ENABLED": True},
            },
            "GCS": {"ENABLED": True},

            "PV_TAIL_CHANNELS": {
                "ENABLED": True,
                "PER_SIGMA": {
                    "ENABLED": True,
                    "SUMMARY": {"ENABLED": True},
                    "SERIES": {"ENABLED": True},
                    "MAX_POINTS": 0,
                    "DECIMATE": True,
                },
            },

            "CSD_DCSD": {
                "ENABLED": True,
                "LEG1": {"CSD": {"ENABLED": False}, "DCSD": {"ENABLED": False}},
                "LEG2": {"CSD": {"ENABLED": True}, "DCSD": {"ENABLED": True}},
            },

            "GBC": {
                "ENABLED": True,
                "SD": {"ENABLED": False},
                "DIAG": {"ENABLED": True},
                "BC_DIAG": {"ENABLED": True},
            },

            "HYSTERESIS": {
                "ENABLED": True,
                "HEADER": {"ENABLED": True},
                "EPISODE": {"ENABLED": True},
                "PROBE": {"ENABLED": True},
                "ETA": {"ENABLED": True},
                "LADDER": {"ENABLED": True},
                "DEBUG": {"ENABLED": False},
            },

            "TREND": {
                "DECISION": True,
                "SCORES": True,
                "FEATURES": True,
                "CALC": True
            },

        },

        # -------------------------
        # LEGACY LOG TOGGLES (kept for backward compatibility)
        # -------------------------
        "LOG_SLEEP_STATUS": True,            # printed once per cycle (as you wanted)
        "LOG_HEADER": True,                  # trigger header
        "LOG_TAILING_SNAPSHOTS": True,       # master toggle for snapshot section
        "LOG_BELL_CURVE_SNAPSHOTS": True,    # PV-leg + Tail-leg (new default)
        # LOG_FIXED_TAIL_SNAPSHOTS deprecated (removed)

        # Trend method selection
        # - "base": original multi-sigma vote
        # - "g83_ema_flip": adds EMA alignment + flip detection on σ=83
        "TREND_METHOD": "base",

        # EMA+flip params used by TREND_METHOD="g83_ema_flip"
        "EMA_FAST_SPAN": 8,
        "EMA_SLOW_SPAN": 21,
        "EMA_TREND_SPAN": 55,
        "EMA_CROSS_LOOKBACK": 12,
        "EMA_MIN_SEPARATION_MAD": 0.05,
        "G83_SLOPE_MIN_MAD": 0.03,
        "G83_FLIP_G2_MIN_MAD": 0.02,
        "ALLOW_NEUTRAL": True,
        "LOG_EPOCH_SERIES_DUMP": True,     # σ=8 | v1, v2, v3... (full 5-min epoch)

        # Bell-curve per-leg series dumps (single-line, per-sigma)
        # When enabled, prints the full (or capped) raw Gaussian values for each leg immediately beneath
        # the corresponding PV-leg / TAIL metrics line.
        "LOG_BELL_CURVE_LEG_SERIES_DUMP": True,
        "BELL_CURVE_LEG_SERIES_DECIMALS": 2,
        "BELL_CURVE_LEG_SERIES_MAX_POINTS": 0,  # 0 => no cap (print full leg)

        "LOG_TREND_DECISION": True,        # print calculate_trend() output

        # Optional future toggles
        "LOG_PATTERN_CUES": False,
        "LOG_CLOSE_SERIES": False,
        "LOG_RAW_TS": False,
    }

    _normalize_print_toggles(c)
    return c


def _normalize_print_toggles(c: Dict[str, Any]) -> None:
    """Map legacy flat toggles into the nested PRINT dictionary.

    PRINT is canonical.
    Legacy keys remain supported for older modules / configs.
    """

    p = c.setdefault("PRINT", {})

    # Header
    if "LOG_HEADER" in c:
        p["HEADER"] = bool(c.get("LOG_HEADER", p.get("HEADER", True)))

    # MSBC master (from legacy bell toggle)
    if "LOG_BELL_CURVE_SNAPSHOTS" in c:
        on = bool(c.get("LOG_BELL_CURVE_SNAPSHOTS", True))
        p.setdefault("MSBC", {})
        p.setdefault("PV_REF", {})
        p.setdefault("TAIL_ANCHOR", {})
        # treat as an overall bell enable: leg summaries + pv_ref + tail anchor
        p["PV_REF"]["ENABLED"] = on
        p["TAIL_ANCHOR"]["ENABLED"] = on
        p["MSBC"].setdefault("LEG1", {}).setdefault("ENABLED", on)
        p["MSBC"].setdefault("LEG2", {}).setdefault("ENABLED", on)

    # MSBC per-leg series dump (legacy was a single switch for both legs)
    if "LOG_BELL_CURVE_LEG_SERIES_DUMP" in c:
        dump_on = bool(c.get("LOG_BELL_CURVE_LEG_SERIES_DUMP", False))
        p.setdefault("MSBC", {})
        p["MSBC"].setdefault("LEG1", {})
        p["MSBC"].setdefault("LEG2", {})
        p["MSBC"]["LEG1"].setdefault("SERIES", {})
        p["MSBC"]["LEG2"].setdefault("SERIES", {})
        p["MSBC"]["LEG1"]["SERIES"]["ENABLED"] = dump_on
        p["MSBC"]["LEG2"]["SERIES"]["ENABLED"] = dump_on

    # Gaussian channels snapshot
    if "LOG_GAUSS_CHANNEL_SNAPSHOT" in c or "LOG_GAUSS_CHANNELS" in c:
        on = bool(c.get("LOG_GAUSS_CHANNEL_SNAPSHOT", c.get("LOG_GAUSS_CHANNELS", True)))
        p.setdefault("GCS", {})
        p["GCS"]["ENABLED"] = on

    if "LOG_GAUSS_CHANNEL_PV_TAIL" in c or "LOG_GAUSS_CHANNEL_PVTAIL" in c:
        on = bool(c.get("LOG_GAUSS_CHANNEL_PV_TAIL", c.get("LOG_GAUSS_CHANNEL_PVTAIL", True)))
        p.setdefault("PV_TAIL_CHANNELS", {})
        p["PV_TAIL_CHANNELS"]["ENABLED"] = on

    # Trend decision
    if "LOG_TREND_DECISION" in c:
        p.setdefault("TREND", {})
        p["TREND"]["DECISION"] = bool(c.get("LOG_TREND_DECISION", True))

    # Legacy 5-min epoch dump
    if "LOG_EPOCH_SERIES_DUMP_LEGACY" in c:
        p["LEGACY_EPOCH_DUMP"] = bool(c.get("LOG_EPOCH_SERIES_DUMP_LEGACY", False))


def tail_print_seconds(cfg_dict: Dict[str, Any]) -> list[int]:
    """Deprecated.

    Fixed multi-window tail snapshots (60/30/20/10/5s) are no longer used.
    We keep this function so orchestrator imports remain stable.

    Returns an empty list.
    """
    return []


def log_if(enabled: bool, msg: str) -> None:
    if enabled:
        logging.info(msg)