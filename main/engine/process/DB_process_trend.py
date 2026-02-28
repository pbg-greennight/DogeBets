# main/engine/process/DB_process_trend.py

from __future__ import annotations

import math
import statistics
import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional

from main.engine.process.DB_process_types import TrendDecision


# Default model location (you confirmed this folder structure)
DEFAULT_MODEL_FILE = "method_hyster_v1.0.json"
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


def _load_hyster_method_config(model_path: str) -> Dict[str, Any]:
    p = Path(model_path)
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("[method_hyster] config must be a JSON object")
    if "method_id" not in cfg:
        raise ValueError("[method_hyster] missing required key: method_id")
    return cfg


def _mean(x: list[float]) -> float:
    return float(sum(x) / max(len(x), 1)) if x else 0.0


def _safe_tail_slope(x: list[float], k: int) -> float:
    if len(x) < 2:
        return 0.0
    kk = min(max(k, 1), len(x) - 1)
    return float((x[-1] - x[-1 - kk]) / max(kk, 1))


def _score_hyster_method(per_sigma: Dict[int, Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    stacks = cfg.get("stacks", {}).get("ladder", []) or []
    vote_weights = cfg.get("stacks", {}).get("vote_weights", {}) or {}
    dyn = cfg.get("dynamic_thresholds", {}) or {}
    drift_cfg = dyn.get("drift", {}) or {}

    flat_abs_m_norm_max = float(drift_cfg.get("flat_abs_m_norm_max", 0.5) or 0.5)
    trend_abs_m_norm_min = float(drift_cfg.get("trend_abs_m_norm_min", 1.0) or 1.0)

    per_stack: Dict[str, Dict[str, float]] = {}
    vote_raw = 0.0
    vote_denom = 0.0
    collapse_hits = 0
    expand_hits = 0
    disorder_hits = 0

    for sdef in stacks:
        sid = str(sdef.get("id", ""))
        sigs = [int(x) for x in (sdef.get("sigmas") or [])]
        if not sid or not sigs:
            continue

        rows = [per_sigma.get(s, {}) for s in sigs]
        if any(not r or not r.get("values") for r in rows):
            continue

        sign_votes = [int(r.get("sign_to", 0)) for r in rows]
        hooks = [int(r.get("hook", 0)) for r in rows]
        flats = [float(r.get("flat", 1.0)) for r in rows]
        values_last = [float(r["values"][-1]) for r in rows]

        slope_tail = _mean([_safe_tail_slope(list(r["values"]), 12) for r in rows])
        slope_long = _mean([_safe_tail_slope(list(r["values"]), 36) for r in rows])
        m_norm = slope_tail / (abs(slope_long) + 1e-9)

        direction = 1 if sum(sign_votes) > 0 else (-1 if sum(sign_votes) < 0 else 0)
        spread = max(values_last) - min(values_last)
        spread_ref = max(abs(_mean(values_last)), 1e-9)
        spread_norm = spread / spread_ref
        dW_norm = _mean([abs(_safe_tail_slope(list(r["values"]), 8)) for r in rows])
        dW_norm *= -1.0 if _mean(hooks) > 0.5 else 1.0
        eff_order = max(abs(_mean(sign_votes)), 0.0) * (1.0 - min(1.0, _mean(flats)))
        cross_rate = float(sum(1 for h in hooks if h > 0)) / max(len(hooks), 1)

        if dW_norm < -1.0:
            collapse_hits += 1
        if dW_norm > 1.0:
            expand_hits += 1
        if cross_rate >= 0.5 or eff_order < 0.2:
            disorder_hits += 1

        weight = float(vote_weights.get(sid, 0.0) or 0.0)
        quality = max(0.0, min(1.0, abs(m_norm) * (0.6 + 0.4 * eff_order)))
        vote_raw += weight * direction * quality
        vote_denom += abs(weight * quality)

        per_stack[sid] = {
            "direction": float(direction),
            "m_norm": float(m_norm),
            "spread_norm": float(spread_norm),
            "dW_norm": float(dW_norm),
            "eff_order": float(eff_order),
            "cross_rate": float(cross_rate),
            "weight": float(weight),
            "quality": float(quality),
        }

    vote_norm = abs(vote_raw) / max(vote_denom, 1e-9)
    signed_vote = 1 if vote_raw > 0 else (-1 if vote_raw < 0 else 0)

    if disorder_hits >= 2:
        regime = "DISORDER"
        trend = "Neutral"
        confidence = min(0.5, vote_norm)
    elif collapse_hits >= 1 and signed_vote != 0:
        regime = "FAN_COLLAPSE"
        trend = "Bull" if signed_vote > 0 else "Bear"
        confidence = min(1.0, vote_norm * 0.8)
    elif expand_hits >= 1 and signed_vote != 0:
        regime = "FAN_EXPAND"
        trend = "Bull" if signed_vote > 0 else "Bear"
        confidence = min(1.0, vote_norm)
    elif abs(vote_raw) > 0 and vote_norm >= trend_abs_m_norm_min:
        regime = "TIGHT_DRIFT"
        trend = "Bull" if signed_vote > 0 else "Bear"
        confidence = min(1.0, vote_norm * 1.05)
    elif vote_norm <= flat_abs_m_norm_max:
        regime = "FLAT_NEUTRAL"
        trend = "Neutral"
        confidence = max(0.0, 1.0 - vote_norm)
    else:
        regime = "NEUTRAL_GUARD"
        trend = "Neutral"
        confidence = max(0.0, 0.7 - vote_norm)

    return {
        "trend": trend,
        "confidence": max(0.0, min(1.0, float(confidence))),
        "regime": regime,
        "scores": {
            "neutral": float(1.0 - confidence) if trend == "Neutral" else float(max(0.0, 1.0 - vote_norm)),
            "bull": float(confidence if trend == "Bull" else 0.0),
            "bear": float(confidence if trend == "Bear" else 0.0),
            "reversal": 0.0,
        },
        "vote_raw": float(vote_raw),
        "vote_norm": float(vote_norm),
        "per_stack": per_stack,
    }


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

        method_cfg = _load_hyster_method_config(mp)
        processed = _score_hyster_method(per_sigma=per_sigma, cfg=method_cfg)

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

        model_id = str(method_cfg.get("method_id", "method_hyster_v1.0"))
        reason = str(processed.get("regime", "NEUTRAL_GUARD"))

        extras = {
            "raw_trend": raw_trend,
            "reason": reason,
            "scores": {"neutral": neutral_score, "bull": bull_score, "bear": bear_score, "reversal": rev_score},
            "features": {
                "vote_raw": processed.get("vote_raw", 0.0),
                "vote_norm": processed.get("vote_norm", 0.0),
                "per_stack": processed.get("per_stack", {}),
            },
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
