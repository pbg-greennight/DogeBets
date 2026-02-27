# main/engine/process/DB_process_trend.py

from __future__ import annotations

import math
import statistics
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from main.engine.process.DB_process_calc import data_process_calc
from main.engine.process.DB_process_types import TrendDecision


# Default model location (you confirmed this folder structure)
DEFAULT_MODEL_FILE = "dev_method_v1_0_hookdown_neu.json"
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "models" / DEFAULT_MODEL_FILE)


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _median_abs_dev(x):
    """MAD around median (robust scale)."""
    if not x:
        return 0.0
    med = statistics.median(x)
    dev = [abs(v - med) for v in x]
    return statistics.median(dev)


def _safe_get_values(blob: Any) -> list[float]:
    """
    per_sigma_full[s] might be:
      - {'values': [...], ...}
      - or directly a list/tuple of floats
    """
    if blob is None:
        return []
    if isinstance(blob, dict):
        vals = blob.get("values") or blob.get("vals") or blob.get("series") or []
        return list(vals) if isinstance(vals, (list, tuple)) else []
    if isinstance(blob, (list, tuple)):
        return list(blob)
    return []


def _compute_sign_to(values: list[float], k: int = 21) -> int:
    """Direction sign using a short tail slope (last vs last-k)."""
    if len(values) < 2:
        return 0
    kk = min(k, len(values))
    a = values[-kk]
    b = values[-1]
    d = b - a
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0


def _compute_hook(values: list[float], short_k: int = 10, long_k: int = 50) -> int:
    """
    Hook = short-term slope sign disagrees with longer-term slope sign
           AND short-term move is meaningfully strong vs tail noise.
    Returns 1 if hook detected else 0.
    """
    n = len(values)
    if n < 12:
        return 0

    sk = min(short_k, n - 1)
    lk = min(long_k, n - 1)

    # slopes over tail windows
    short_slope = (values[-1] - values[-1 - sk]) / max(1, sk)
    long_slope = (values[-1] - values[-1 - lk]) / max(1, lk)

    s_sign = 1 if short_slope > 0 else (-1 if short_slope < 0 else 0)
    l_sign = 1 if long_slope > 0 else (-1 if long_slope < 0 else 0)

    if s_sign == 0 or l_sign == 0:
        return 0
    if s_sign == l_sign:
        return 0

    # robust noise from last ~sk diffs
    tail = values[-(sk + 1) :]
    diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    mad = _median_abs_dev(diffs) + 1e-9

    # require short slope magnitude to be non-trivial vs noise
    # (tuned to avoid constant "hook=1" spam)
    strength = abs(short_slope) / mad
    if strength >= 2.0:
        return 1
    return 0


def _compute_flat(values: list[float], k: int = 21) -> float:
    """
    Flatness score in [0..1], higher means flatter.
    Uses |tail slope| relative to robust tail noise (MAD of diffs).
    """
    if len(values) < 3:
        return 1.0

    kk = min(k, len(values) - 1)
    tail = values[-(kk + 1) :]
    diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    mad = _median_abs_dev(diffs) + 1e-9

    slope = (tail[-1] - tail[0]) / max(1, kk)

    # Convert to a flatness score:
    # big slope vs noise -> flat ~ 0
    # tiny slope vs noise -> flat ~ 1
    ratio = abs(slope) / (5.0 * mad)  # 5x MAD is "pretty active"
    flat = 1.0 - ratio
    return _clamp01(flat)


def _build_per_sigma_inputs(
    per_sigma_full: Dict[int, Dict[str, Any]],
    wanted_sigmas,
    diag_chunk: int = 12,
    tail_n: int = 21,
    debug: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Build the exact per-sigma payload DB_process_calc expects.

    REQUIRED keys per sigma for scoring:
      - values: list[float]
      - sign_to: int (-1/0/+1)
      - hook: int (0/1)
      - flat: float [0..1]
      - diag: small tail list for debugging/printing (optional but useful)
    """
    out: Dict[int, Dict[str, Any]] = {}

    if debug:
        logging.info("[trend_debug] ---- BUILD PER SIGMA INPUTS START ----")
        logging.info(f"[trend_debug] per_sigma_full keys: {sorted(list(per_sigma_full.keys()))}")

    for s in wanted_sigmas:
        blob = per_sigma_full.get(s, {})
        values = _safe_get_values(blob)

        if not values:
            out[s] = {"values": [], "sign_to": 0, "hook": 0, "flat": 1.0, "diag": []}
            if debug:
                logging.info(f"[trend_debug] sigma={s} MISSING/EMPTY series")
            continue

        sign_to = _compute_sign_to(values, k=tail_n)
        hook = _compute_hook(values, short_k=max(8, tail_n // 2), long_k=max(30, tail_n * 2))
        flat = _compute_flat(values, k=tail_n)
        diag = values[-diag_chunk:] if diag_chunk and len(values) >= 1 else []

        out[s] = {
            "values": values,
            "sign_to": int(sign_to),
            "hook": int(hook),
            "flat": float(flat),
            "diag": diag,
        }

        if debug:
            logging.info(
                f"[trend_debug] sigma={s} n_vals={len(values)} first={values[0]:.4f} last={values[-1]:.4f} "
                f"sign_to={sign_to} hook={hook} flat={flat:.3f}"
            )

    if debug:
        logging.info("[trend_debug] ---- BUILD PER SIGMA INPUTS END ----")

    return out


def calculate_trend(
    curr_epoch: int,
    next_epoch: int,
    windows,
    per_sigma_full: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    model_path: Optional[str] = None,
) -> TrendDecision:
    """
    Orchestrator entry point. Keeps existing signature used in DB_process_orchestrator.py.

    - Builds model-ready per_sigma inputs (values + sign_to/hook/flat)
    - Loads json model config
    - Returns TrendDecision for printing + TREND_OUT_JSON
    """
    try:
        # Resolve model path robustly (NEVER let it be None)
        mp = model_path or (config.get("MODEL_PATH") if isinstance(config, dict) else None) or DEFAULT_MODEL_PATH
        mp = str(mp)

        # Optional debug toggle via config
        debug = bool(config.get("TREND_DEBUG", False)) if isinstance(config, dict) else False

        # Confirm file exists early (gives you immediate, readable diagnostics)
        p = Path(mp)
        if debug:
            logging.info(f"[trend_debug] model_path resolved: {mp}")
            logging.info(f"[trend_debug] model_path exists: {p.exists()}")
            logging.info(f"[trend_debug] this_file_dir: {Path(__file__).resolve().parent}")

        wanted_sigmas = (config.get("GAUSS_SIGMAS") if isinstance(config, dict) else None) or [8, 23, 38, 53, 68, 83]
        diag_chunk = int(config.get("DUMP_DIAG_CHUNK", 12)) if isinstance(config, dict) else 12
        tail_n = int(config.get("TAIL_FEATURE_POINTS", 21)) if isinstance(config, dict) else 21

        # Build model inputs (THIS is what fixes the "always neutral" problem)
        per_sigma = _build_per_sigma_inputs(
            per_sigma_full=per_sigma_full,
            wanted_sigmas=wanted_sigmas,
            diag_chunk=diag_chunk,
            tail_n=tail_n,
            debug=debug,
        )

        processed = data_process_calc(per_sigma=per_sigma, model_path=mp)

        raw_trend = str(processed.get("trend", "Neutral"))
        scores = processed.get("scores", {}) or {}
        neutral_score = float(scores.get("neutral", 0.0) or 0.0)
        bull_score = float(scores.get("bull", 0.0) or 0.0)
        bear_score = float(scores.get("bear", 0.0) or 0.0)
        rev_score = float(scores.get("reversal", 0.0) or 0.0)

        # outward trend contract: Bull/Bear/Neutral only
        if raw_trend == "Bear→Bull":
            trend = "Bull"
        elif raw_trend in ("Bull", "Bear", "Neutral"):
            trend = raw_trend
        else:
            trend = "Neutral"

        # confidence (keep your existing contract)
        if trend == "Neutral":
            confidence = max(0.0, min(1.0, 1.0 - neutral_score))
        elif trend == "Bull":
            confidence = max(0.0, min(1.0, max(bull_score, rev_score)))
        else:  # Bear
            confidence = max(0.0, min(1.0, bear_score))

        model_id = str(processed.get("model_id", "DEV_METHOD_v1.0_HOOKDOWN_NEU"))
        reason = str(processed.get("reason", ""))

        extras = {
            "raw_trend": raw_trend,
            "reason": reason,
            "scores": {"neutral": neutral_score, "bull": bull_score, "bear": bear_score, "reversal": rev_score},
            "features": processed.get("features", {}) or {},
            "per_sigma_inputs": {int(k): {kk: (vv if kk != "values" else f"n={len(vv)}") for kk, vv in v.items()} for k, v in per_sigma.items()},
        }

        notes = f"{reason} | raw={raw_trend}"

        td = TrendDecision(
            trend=trend,
            confidence=float(confidence),
            model=model_id,
            notes=notes,
        )
        setattr(td, "extras", extras)

        if debug:
            logging.info(
                f"[trend_debug] result trend={trend} raw={raw_trend} "
                f"scores: neu={neutral_score:.3f} bull={bull_score:.3f} bear={bear_score:.3f} rev={rev_score:.3f} "
                f"reason={reason}"
            )

        return td

    except Exception as e:
        # Add much more useful failure diagnostics
        mp = (model_path or (config.get("MODEL_PATH") if isinstance(config, dict) else None) or DEFAULT_MODEL_PATH)
        try:
            mp_str = str(mp)
        except Exception:
            mp_str = "<unstringable>"

        logging.exception(f"[calculate_trend] failed: {e}")
        logging.error(f"[calculate_trend] model_path attempted: {mp_str}")
        logging.error(f"[calculate_trend] default_model_path: {DEFAULT_MODEL_PATH}")
        logging.error(f"[calculate_trend] this_file_dir: {Path(__file__).resolve().parent}")
        logging.error(f"[calculate_trend] model_exists: {Path(mp_str).exists() if mp_str not in ('<unstringable>', '') else False}")
        logging.error(f"[calculate_trend] per_sigma_full keys: {sorted(list(per_sigma_full.keys())) if isinstance(per_sigma_full, dict) else '<not a dict>'}")

        td = TrendDecision(
            trend="Neutral",
            confidence=0.0,
            model="DEV_METHOD_v1.0_HOOKDOWN_NEU",
            notes="ERROR_FALLBACK",
        )
        setattr(td, "extras", {"error": str(e), "model_path": mp_str})
        return td
