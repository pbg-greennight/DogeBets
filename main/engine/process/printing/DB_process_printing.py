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
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

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

    # Single, non-duplicated decision line
    slog.TREND_DECISION(f"{_fmt_time(decision_dt)} - Trend Decision ({model_id})")
    slog.TREND_DECISION(
        f"Epoch data from {curr_epoch} --→ Predict Next Epoch {next_epoch} | "
        f"trend={trend} | confidence={_fmt_float(conf, nd=3)} | model={model_id}"
        + (f" | notes={notes}" if notes is not None else "")
        + (f" | raw={raw}" if raw is not None else "")
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

    slog.TREND_DECISION(_line("=", 155))
