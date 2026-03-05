# main/engine/process/DB_process_printing.py
"""DB_process_printing

This module is **pure formatting**: it turns already-computed catalog/calc outputs
into human-readable log blocks.

Design goals:
 - No business logic / no calculations that change the model output.
 - Be resilient to missing keys (log placeholders instead of crashing).
 - Match the user's preferred log style (see Log_example.txt).

NOTE ON TIMESTAMPS
The user's Log_example.txt shows a *double timestamp* on the first header line
(logger prefix + an embedded timestamp). We keep that behavior for that first
line only to match the contract.
"""

from __future__ import annotations

import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# =====================================================================================================================
# Forecast log writer (append-only, last 100)
# =====================================================================================================================

BASE_DIR = Path(__file__).resolve().parent
TS_DIR = (BASE_DIR / ".." / ".." / "ts").resolve()
TREND_LOG_FILE = TS_DIR / "json" / "DB_rounds_trend.json"
save_log_history = 600

def save_forecast_log(trend_label: str, confidence: float, next_epoch: int, model_version: str, mode: str) -> None:
    """Append a single forecast decision to DB_rounds_trend.json (keeps last 100 entries)."""
    try:
        conf_val = float(confidence) if confidence is not None else 0.0
    except Exception:
        conf_val = 0.0

    entry = {
        "timestamp": time.strftime("%m/%d/%Y %I:%M:%S %p"),
        "trend": str(trend_label) if trend_label is not None else "Neutral",
        "confidence": round(conf_val, 3),
        "next_epoch": next_epoch,
        "model_version": str(model_version) if model_version is not None else "",
        "mode": str(mode) if mode is not None else "",
    }

    history = []
    if TREND_LOG_FILE.exists():
        try:
            with open(TREND_LOG_FILE, "r", encoding="utf-8") as f:
                history = json.load(f) or []
        except Exception:
            history = []

    history.append(entry)
    history = history[-save_log_history:]

    try:
        TREND_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TREND_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info(f"[forecast_log] wrote next_epoch={entry.get('next_epoch')} trend={entry.get('trend')} -> {TREND_LOG_FILE}")
    except Exception as e:
        logger.warning(f"[forecast_log] write failed -> {TREND_LOG_FILE}: {e}")


# =====================================================================================================================
# Helpers
# =====================================================================================================================
from main.engine.process.printing.DB_process_printing_utils import (
    _safe_get, _fmt_time, _fmt_float, _line, _fmt_price, _series_preview, _as_mapping, _fmt_iso,
    _arrow, _print_cfg
)
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger


# =====================================================================================================================
# Public API
# =====================================================================================================================

def print_header(timing: Any, windows: Any, decision_dt: datetime, config: Any = None) -> None:
    """Epoch capture header.

    Note: `config` is accepted for compatibility with orchestrator calls, even if
    this printer doesn't currently need it.
    """
    slog = get_section_logger(logger, config)

    import inspect
    print("[debug] slog class:", type(slog))
    print("[debug] slog module:", inspect.getmodule(type(slog)).__file__)
    print("[debug] has TREND_FEATURES:", hasattr(slog, "TREND_FEATURES"))
    print("[debug] dir TREND*:", [x for x in dir(slog) if x.startswith("TREND")])

    # timing/windows are dataclasses in your pipeline, but keep dict fallback.
    try:
        curr_epoch = getattr(timing, "curr_epoch", None)
        next_epoch = getattr(timing, "next_epoch", None)
        dt_next = getattr(timing, "dt_next", None)
        full_start = getattr(windows, "full_start", None)
        full_end = getattr(windows, "full_end", None)
    except Exception:
        t = timing if isinstance(timing, dict) else {}
        w = windows if isinstance(windows, dict) else {}
        curr_epoch = t.get("curr_epoch")
        next_epoch = t.get("next_epoch")
        dt_next = t.get("dt_next")
        full_start = w.get("full_start")
        full_end = w.get("full_end")

    ts_now = _fmt_time(decision_dt)

    # 1) Contract line (double timestamp as in Log_example.txt)
    slog.HEADER(
        f"{ts_now} - GAUSS EPOCH CAPTURE: (Epoch {curr_epoch}) for EPOCH ANALYSIS ({next_epoch}) "
        f"from time: {full_start} to {full_end} | Predict Next Epoch: (Epoch {next_epoch}) at {_fmt_iso(dt_next)} | "
        f"decision_time={ts_now}"
    )

    # 2) Extra lines used in the example
    slog.HEADER(f"TRIGGER: Next Epoch {next_epoch} @ {_fmt_iso(dt_next)} | decision_time={ts_now}")
    slog.HEADER(f"ANALYZE EPOCH: {curr_epoch}")
    slog.HEADER(f"FULL WINDOW : {_fmt_time(full_start)} {_arrow()} {_fmt_time(full_end)}")
    slog.HEADER(_line("-", 155))


def print_epoch_dump(
        timing: Any,
        windows: Any,
        per_sigma_full: Dict[int, Dict[str, Any]],
        config: Dict[str, Any],
) -> None:
    """Legacy 5-minute epoch dump (pre-catalog behavior).

    This is only called when orchestrator enables LOG_EPOCH_SERIES_DUMP_LEGACY.

    per_sigma_full format:
      { sigma: {"ts": [...], "values": [...] } }
    """
    if not bool(config.get("LOG_EPOCH_SERIES_DUMP", True)):
        return

    curr_epoch = _safe_get(timing, "curr_epoch", default="?")
    next_epoch = _safe_get(timing, "next_epoch", default="?")

    logger.info("")
    logger.info(
        f"Epoch {curr_epoch} Gaussian Series Dump (LEGACY 5-min window) | "
        f"(Epoch {curr_epoch} used for Epoch {next_epoch})"
    )

    ws0 = _safe_get(windows, "full_start", default=None)
    ws1 = _safe_get(windows, "full_end", default=None)
    logger.info(f"Window: {_fmt_time(ws0)} → {_fmt_time(ws1)}")

    for sigma in sorted((per_sigma_full or {}).keys()):
        pack = per_sigma_full.get(sigma, {}) or {}
        ts = pack.get("ts", []) or []
        values = pack.get("values", []) or []

        if ts and values:
            logger.info(
                f"σ={int(sigma):>3} | ({_fmt_time(ts[0])} → {_fmt_time(ts[-1])}) "
                f"({len(values)} Data Points) | {_series_preview(values, max_items=9999, nd=2)}"
            )
        else:
            logger.info(f"σ={int(sigma):>3} | (empty)")


def print_trend_decision(
        timing: Any = None,
        trend_out: Any = None,
        config: Any = None,
        decision_dt: datetime | None = None,
        **_kwargs,
) -> None:
    """Print the final trend decision block (Log_example-style).

    This function is intentionally *formatting-only*.
    """

    if decision_dt is None:
        decision_dt = _safe_get(timing, "decision_dt", default=None)

    p = _print_cfg(config)
    slog = get_section_logger(logger, config)

    td = _as_mapping(trend_out)

    # TrendDecision may carry richer fields under .extras (do not change business logic; just improve visibility)
    extras = None
    try:
        extras = getattr(trend_out, "extras", None)
    except Exception:
        extras = None
    if extras is None and isinstance(td, dict):
        extras = td.get("extras")
    extras = extras if isinstance(extras, dict) else {}

    # Backfill common fields from extras when TrendDecision is a dataclass/object
    if td.get("model_id") is None and td.get("model") is not None:
        td["model_id"] = td.get("model")
    if td.get("raw") is None and "raw_trend" in extras:
        td["raw"] = extras.get("raw_trend")
    if td.get("reason") is None and "reason" in extras:
        td["reason"] = extras.get("reason")
    if not td.get("scores") and isinstance(extras.get("scores"), dict):
        td["scores"] = extras.get("scores")
    if not td.get("features") and isinstance(extras.get("features"), dict):
        td["features"] = extras.get("features")
    if td.get("calc") is None and isinstance(extras.get("calc"), dict):
        td["calc"] = extras.get("calc")

    model_id = td.get("model_id")
    trend = td.get("trend")
    conf = td.get("confidence")
    reason = td.get("reason")
    scores = td.get("scores") or {}
    notes = td.get("notes")
    raw = td.get("raw")
    feats = td.get("features") or {}

    curr_epoch = _safe_get(timing, "curr_epoch", default="?")
    next_epoch = _safe_get(timing, "next_epoch", default="?")

    slog.TREND_DECISION(
        f"Epoch data from {curr_epoch} --→ Predict Next Epoch {next_epoch} | "
        f"trend={trend} | confidence={_fmt_float(conf, nd=3)} | model={model_id}"
        + (f" | notes={notes}" if notes is not None else "")
        + (f" | raw={raw}" if raw is not None else "")
    )

    # Calc / gates block (explicit equations + guardrail gates)
    calc = td.get("calc") or {}
    if p.get("TREND", {}).get("CALC", True) and isinstance(calc, dict) and calc:
        guard = (calc.get("guardrail") or {}) if isinstance(calc.get("guardrail"), dict) else {}
        gdbg = (guard.get("debug") or {}) if isinstance(guard.get("debug"), dict) else {}
        vote = (calc.get("vote") or {}) if isinstance(calc.get("vote"), dict) else {}

        fired = int(bool(guard.get("fired", False)))
        gname = guard.get("name", "GUARD")
        gnote = guard.get("note") or "-"
        greason = guard.get("reason") or "-"

        # compact gates (show pass/fail style)
        gates = []
        for k in ["probe_warn", "probe_mismatch", "collapse_ok", "disorder_ok"]:
            if k in gdbg:
                gates.append(f"{k}={int(bool(gdbg.get(k)))}")
        gates_s = ",".join(gates) if gates else "n/a"

        slog.TREND_CALC(
            f"[td_calc] vote_raw={_fmt_float(vote.get('vote_raw'), nd=4)} | vote_norm={_fmt_float(vote.get('vote_norm'), nd=4)} | "
            f"guard={gname} fired={fired} | note={gnote} | reason={greason} | gates={gates_s}"
        )

        # extra guardrail diagnostics when fired or when debug is on
        if fired or bool(config.get("TREND_DEBUG", False)):
            slog.TREND_CALC(
                f"[td_calc_guard] ep_sign={gdbg.get('ep_sign', '?')} pr_sign={gdbg.get('pr_sign', '?')} | "
                f"flip_watch={gdbg.get('flip_watch', '?')} fast_collapse={gdbg.get('fast_collapse', '?')} | "
                f"d_abs_slope_tail={_fmt_float(gdbg.get('d_abs_slope_tail'), nd=6)} | "
                f"S0:dW_norm={_fmt_float(gdbg.get('s0_dW_norm'), nd=4)} eff_order={_fmt_float(gdbg.get('s0_eff_order'), nd=4)}"
            )

    # Scores block
    if p.get("TREND", {}).get("SCORES", True) and isinstance(scores, dict) and scores:
        slog.TREND_SCORES(
            f"[td_scores] neutral={_fmt_float(scores.get('neutral'), nd=4)} | bull={_fmt_float(scores.get('bull'), nd=4)} | "
            f"bear={_fmt_float(scores.get('bear'), nd=4)} | reversal={_fmt_float(scores.get('reversal'), nd=4)} | "
            f"reason={reason} | model={model_id}"
        )

    # Features block
    if p.get("TREND", {}).get("FEATURES", True) and isinstance(feats, dict) and feats:
        preferred = [
            "g83_delta_mid",
            "g83_delta_width",
            "g83_slope_mid",
            "g83_slope_width",
            "g83_mid_start",
            "g83_mid_last",
            "g83_width_start",
            "g83_width_last",
        ]
        parts = []
        for k in preferred:
            if k in feats:
                parts.append(f"{k}={_fmt_float(feats.get(k), nd=6)}")
        if not parts:
            # fallback: show a compact subset
            for k in sorted(feats.keys())[:12]:
                parts.append(f"{k}={_fmt_float(feats.get(k), nd=6)}")
        slog.TREND_FEATURES(f"[td_features] " + " | ".join(parts))


    # -------------------------------------------------------------------------------------------------
    # Optional forecast log save (controlled by config)
    # -------------------------------------------------------------------------------------------------
    try:
        save_enabled = True
        if isinstance(config, dict):
            sf = config.get("SAVE_FORECAST_LOG", True)
            if isinstance(sf, dict):
                save_enabled = bool(sf.get("ENABLED", True))
            else:
                save_enabled = bool(sf)

        logger.info(f"[forecast_log] save_enabled={save_enabled} next_epoch={next_epoch} model={model_id} trend={trend}")
        if save_enabled:
            save_forecast_log(
                trend_label=trend,
                confidence=conf,
                next_epoch=next_epoch,
                model_version=model_id,
                mode=notes,
            )
    except Exception as e:
        logger.warning(f"[forecast_log] save failed: {e}")

    slog.TREND_DECISION(_line("=", 155))
