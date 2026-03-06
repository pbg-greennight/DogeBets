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
DEFAULT_MODEL_FILE = "trend_method_v2.json"
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
    tail = values[-(sk + 1):]
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
    tail = values[-(kk + 1):]
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


def evaluate_pv_channel_guardrail(hyst_obj: dict | None, model_cfg: dict) -> tuple[bool, str | None, str | None, dict]:
    """PV channel guardrail (legacy behavior) extracted into a function for unified guardrails system.

    Returns: (fired, note, reason, debug)
    """
    guard_cfg = model_cfg.get("PV_CHANNEL_GUARDRAIL", {}) if isinstance(model_cfg, dict) else {}
    guard_enabled = bool(guard_cfg.get("enabled", False))

    guard_fired = False
    guard_note: str | None = None
    guard_reason: str | None = None
    guard_debug: dict = {}

    if guard_enabled and isinstance(hyst_obj, dict) and not hyst_obj.get("meta", {}).get("skipped"):
        ep = hyst_obj.get("episode") or {}
        pr = hyst_obj.get("probe") or {}
        stacks = hyst_obj.get("stacks") or {}

        ep_sign = int(ep.get("sign_now") or 0)
        pr_sign = int(pr.get("sign_now") or 0)

        flip_watch = bool(pr.get("flip_watch", False))
        fast_collapse = bool(pr.get("fast_collapse", False))

        # Prefer the highest stack's slope sign as a 'macro' reference.
        s3 = stacks.get("S3") or {}
        macro = int(s3.get("direction") or 0)

        # Channel width proxy: W_pct from S1 if present, else S0.
        s1 = stacks.get("S1") or {}
        s0 = stacks.get("S0") or {}
        W_pct = float(s1.get("W_pct", s0.get("W_pct", 0.0)))

        # Disagreement signature: probe sign vs episode/macro sign
        disagree = (pr_sign != ep_sign) or (macro != 0 and pr_sign != macro)

        # Guard policy (config driven)
        min_w_pct = float(guard_cfg.get("min_w_pct", 0.0))
        require_disagree = bool(guard_cfg.get("require_disagree", True))
        require_flip_watch = bool(guard_cfg.get("require_flip_watch", True))
        require_fast_collapse = bool(guard_cfg.get("require_fast_collapse", False))

        conds = []
        conds.append(W_pct <= min_w_pct)
        conds.append(disagree if require_disagree else True)
        conds.append(flip_watch if require_flip_watch else True)
        conds.append(fast_collapse if require_fast_collapse else True)

        guard_debug = {
            "W_pct": W_pct,
            "min_w_pct": min_w_pct,
            "ep_sign": ep_sign,
            "pr_sign": pr_sign,
            "macro": macro,
            "disagree": disagree,
            "flip_watch": flip_watch,
            "fast_collapse": fast_collapse,
            "conds": conds,
        }

        if all(conds):
            guard_fired = True
            guard_note = "PV_CHANNEL_GUARDRAIL"
            guard_reason = "PV_CHANNEL_GUARDRAIL"

    return guard_fired, guard_note, guard_reason, guard_debug


# ----------------------------------------------------------------------------------------------------------------------
# Unified Guardrails System (config-driven)
# ----------------------------------------------------------------------------------------------------------------------

def _get_guardrails_config(model_cfg: dict) -> dict:
    """Return unified guardrails config. Falls back to legacy PV_CHANNEL_GUARDRAIL if present."""
    gr = model_cfg.get("guardrails")
    if isinstance(gr, dict):
        return gr

    # Legacy fallback (do not remove - keeps older configs working)
    legacy = model_cfg.get("PV_CHANNEL_GUARDRAIL", {})
    return {
        "enabled": bool(legacy.get("enabled", False)),
        "order": ["pv_channel_guardrail"],
        "rules": {
            "pv_channel_guardrail": {
                "type": "pv_channel_guardrail",
                "enabled": bool(legacy.get("enabled", False)),
                "params": legacy,
            }
        },
    }


def _guardrail_result(name: str, fired: bool, note: str = "", meta: dict | None = None) -> dict:
    return {
        "name": name,
        "fired": bool(fired),
        "note": note or "",
        "meta": meta or {},
    }


def _eval_gr_flat(decision: dict, vote_raw: float, per_sigma_inputs: dict, rule: dict) -> dict:
    p = rule.get("params", {})
    sigmas = p.get("sigmas", [23, 38, 53, 68, 83])
    flat_hi = float(p.get("flat_hi", 0.92))
    min_frac = float(p.get("min_frac", 0.6))

    only_abs_vote = float(p.get("only_if_abs_vote_raw_below", 0.25))
    only_conf_below = float(p.get("only_if_confidence_below", 0.90))

    diag = per_sigma_inputs.get("diag", {})
    flats = []
    for s in sigmas:
        rec = diag.get(int(s), {})
        if "flat" in rec:
            flats.append(float(rec["flat"]))

    if not flats:
        return _guardrail_result("gr_flat", False, note=p.get("note", "GR_FLAT"), meta={"reason": "NO_FLAT_DATA"})

    frac = sum(1 for x in flats if x >= flat_hi) / float(len(flats))
    cond = (
            frac >= min_frac
            and abs(float(vote_raw)) <= only_abs_vote
            and float(decision.get("confidence", 0.0)) <= only_conf_below
    )

    meta = {"frac_flat_hi": frac, "flat_hi": flat_hi, "min_frac": min_frac, "abs_vote_raw": abs(float(vote_raw))}
    return _guardrail_result("gr_flat", cond, note=p.get("note", "GR_FLAT"), meta=meta)


def _eval_gr_rtc(decision: dict, vote_raw: float, per_sigma_inputs: dict, rule: dict) -> dict:
    """Regime Transition Cluster guardrail (heuristic, config-driven)."""
    p = rule.get("params", {})
    diag = per_sigma_inputs.get("diag", {})

    sig_fast = [int(x) for x in p.get("sigmas_fast", [8, 23])]
    sig_mid = [int(x) for x in p.get("sigmas_mid", [23, 38, 53])]
    sig_slow = [int(x) for x in p.get("sigmas_slow", [68, 83])]

    min_sign_flips = int(p.get("min_sign_flips", 1))
    min_hook_count = int(p.get("min_hook_count", 1))

    mid_flat_hi = float(p.get("mid_flat_hi", 0.80))
    min_mid_flat_frac = float(p.get("min_mid_flat_frac", 0.67))

    require_fast_slow_disagree = bool(p.get("require_fast_slow_disagree", True))

    only_abs_vote = float(p.get("only_if_abs_vote_raw_below", 0.35))
    only_conf_below = float(p.get("only_if_confidence_below", 0.90))

    # sign flips + hook counts
    flips = 0
    hook_count = 0
    for s, rec in diag.items():
        s_i = int(s)
        if "sign_from" in rec and "sign_to" in rec:
            if int(rec["sign_from"]) != int(rec["sign_to"]):
                flips += 1
        if int(rec.get("hook", 0)) == 1:
            hook_count += 1

    # mid flatness fraction
    mid_flats = []
    for s in sig_mid:
        rec = diag.get(int(s), {})
        if "flat" in rec:
            mid_flats.append(float(rec["flat"]))
    if mid_flats:
        mid_flat_frac = sum(1 for x in mid_flats if x >= mid_flat_hi) / float(len(mid_flats))
    else:
        mid_flat_frac = 0.0

    # fast vs slow disagreement (based on sign_to)
    def _sign_to(s: int) -> int | None:
        rec = diag.get(int(s), {})
        if "sign_to" in rec:
            return int(rec["sign_to"])
        return None

    fast_signs = [x for x in (_sign_to(s) for s in sig_fast) if x is not None]
    slow_signs = [x for x in (_sign_to(s) for s in sig_slow) if x is not None]
    fast_slow_disagree = False
    if fast_signs and slow_signs:
        fast_major = 1 if sum(fast_signs) >= 0 else -1
        slow_major = 1 if sum(slow_signs) >= 0 else -1
        fast_slow_disagree = (fast_major != slow_major)

    cond = (
            flips >= min_sign_flips
            and hook_count >= min_hook_count
            and mid_flat_frac >= min_mid_flat_frac
            and abs(float(vote_raw)) <= only_abs_vote
            and float(decision.get("confidence", 0.0)) <= only_conf_below
            and (fast_slow_disagree if require_fast_slow_disagree else True)
    )

    meta = {
        "sign_flips": flips,
        "hook_count": hook_count,
        "mid_flat_frac": mid_flat_frac,
        "mid_flat_hi": mid_flat_hi,
        "min_mid_flat_frac": min_mid_flat_frac,
        "fast_slow_disagree": fast_slow_disagree,
        "abs_vote_raw": abs(float(vote_raw)),
    }
    return _guardrail_result("gr_rtc", cond, note=p.get("note", "GR_RTC"), meta=meta)


def _apply_guardrails(decision: dict, vote_raw: float, per_sigma_inputs: dict, hyst_obj: dict | None, model_cfg: dict,
                      log: logging.Logger) -> dict:
    """Apply config-driven guardrails without changing base model scoring logic."""
    gr = _get_guardrails_config(model_cfg)
    if not gr.get("enabled", False):
        return decision

    rules = gr.get("rules", {})
    order = gr.get("order", []) or []

    fired_any = False
    fired_list: list[dict] = []

    for name in order:
        rule = rules.get(name, {})
        if not rule or not rule.get("enabled", False):
            continue

        gtype = rule.get("type", name)

        if gtype == "pv_channel_guardrail":
            fired, gnote, greason, gdebug = evaluate_pv_channel_guardrail(hyst_obj, model_cfg)
            res = _guardrail_result(name, fired, note=str(gnote or ''), meta={'reason': greason, 'debug': gdebug})
        elif gtype == "gr_flat":
            res = _eval_gr_flat(decision, vote_raw, per_sigma_inputs, rule)
        elif gtype == "gr_rtc":
            res = _eval_gr_rtc(decision, vote_raw, per_sigma_inputs, rule)
        else:
            # Unknown guardrail type; ignore safely.
            res = _guardrail_result(name, False, note="UNKNOWN_GUARDRAIL_TYPE", meta={"type": gtype})

        fired_list.append(res)

        if res.get("fired"):
            fired_any = True
            p = rule.get("params", {})

            # Override policy: force to configured trend (default Neutral), but do NOT change underlying
            # vote_raw / features. Only adjust the returned decision fields.
            force_trend = p.get("force_trend", "Neutral")
            decision["guard"] = decision.get("guard", gtype)
            decision["guardrail"] = name
            decision["guardrail_note"] = res.get("note", "")
            decision["trend"] = force_trend

            if force_trend == "Neutral":
                decision["confidence"] = float(p.get("force_neutral_confidence", 0.15))

            # Stop at first firing guardrail (order is intentional)
            break

    decision.setdefault("guardrails", {})
    decision["guardrails"]["enabled"] = True
    decision["guardrails"]["order"] = order
    decision["guardrails"]["fired_any"] = fired_any
    decision["guardrails"]["results"] = fired_list

    return decision


def calculate_trend(
        curr_epoch: int,
        next_epoch: int,
        windows,
        per_sigma_full: Dict[int, Dict[str, Any]],
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        hyst_obj: Optional[Any] = None,
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
            "per_sigma_inputs": {int(k): {kk: (vv if kk != "values" else f"n={len(vv)}") for kk, vv in v.items()} for
                                 k, v in per_sigma.items()},
        }

        # Apply unified, config-driven guardrails (does not change base scoring; only post-processes output)
        try:
            per_sigma_inputs = {
                "diag": {
                    str(s): {
                        "flat": float(per_sigma.get(s, {}).get("flat", 0.0) or 0.0),
                        "hook": int(per_sigma.get(s, {}).get("hook", 0) or 0),
                        "sign_to": int(per_sigma.get(s, {}).get("sign_to", 0) or 0),
                    }
                    for s in (method_cfg.get("sigmas") or [])
                }
            }
        except Exception:
            per_sigma_inputs = {"diag": {}}

        hyst_obj_local = hyst_obj if hyst_obj is not None else {"per_stack": processed.get("per_stack", {}), "vote_raw": processed.get("vote_raw", 0.0)}

        decision = {"trend": trend, "confidence": confidence, "reason": reason}
        decision = _apply_guardrails(
            decision=decision,
            vote_raw=processed.get("vote_raw", 0.0),
            per_sigma_inputs=per_sigma_inputs,
            hyst_obj=hyst_obj_local,
            model_cfg=method_cfg,
            log=logging.getLogger(__name__),
        )

        trend = str(decision.get("trend", trend))
        confidence = float(decision.get("confidence", confidence))
        # Preserve base reason but allow guardrail-specific reason if provided
        guard_reason = str(decision.get("guardrail", "") or "")
        reason = guard_reason or reason

        guard_note = str(decision.get("guardrail_note", "") or "")
        guard_debug = decision.get("guardrail_debug", {}) or {}
        notes = f"{reason} | raw={raw_trend}" + (f" | {guard_note}" if guard_note else "")

        td = TrendDecision(
            trend=trend,
            confidence=float(confidence),
            model=model_id,
            notes=notes,
        )
        extras["guardrails"] = decision.get("guardrails", {})
        extras["guardrail_note"] = guard_note
        extras["guardrail_debug"] = guard_debug
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
        logging.error(
            f"[calculate_trend] model_exists: {Path(mp_str).exists() if mp_str not in ('<unstringable>', '') else False}")
        logging.error(
            f"[calculate_trend] per_sigma_full keys: {sorted(list(per_sigma_full.keys())) if isinstance(per_sigma_full, dict) else '<not a dict>'}")

        td = TrendDecision(
            trend="Neutral",
            confidence=0.0,
            model="DEV_METHOD_v1.0_HOOKDOWN_NEU",
            notes="ERROR_FALLBACK",
        )
        setattr(td, "extras", {"error": str(e), "model_path": mp_str})
        return td


# ==================================================================================================
# Engine execution layer (v2 wiring)
# --------------------------------------------------------------------------------------------------
# Goal:
#   - Read merged config toggles
#   - Run enabled engines (gaussian / hysteresis / guardrails, etc.)
#   - Build one TrendDecision
#   - Save once
#   - Return diagnostics for printing
#
# This layer is intentionally self-contained so DB_process_orchestrator does NOT need to change.
#
# Expected merged config shape (examples):
#   config["MODEL_gaussian"] = {"MODEL_ID": "trend_method_gaussian_v2.json", "ENABLED": True}
#   config["MODEL_hyster"]   = {"MODEL_ID": "method_hyster_v1.0.json",       "ENABLED": True}
#   config["MODEL_guardrails"]= {"ENABLED": True}  # optional (defaults to True)
#   config["TREND_OUT_JSON"] = "E:\\...\\ts\\json\\TREND_OUT.json"           # optional
#
# If TREND_OUT_JSON is not provided, this layer will NOT write files (but will still return the decision).

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

# Conservative, deterministic engine priority (highest wins if multiple produce non-neutral)
DEFAULT_ENGINE_ORDER = ("MODEL_hyster", "MODEL_gaussian")

@dataclass
class EngineResult:
    engine_key: str
    enabled: bool
    model_id: str
    decision: TrendDecision
    diagnostics: Dict[str, Any]


def _cfg_get_model_toggle(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = config.get(key)
    return v if isinstance(v, dict) else {}


def _resolve_model_path(model_id: str, config: Dict[str, Any]) -> str:
    """
    Resolve a model ID into an on-disk path.
    Supports:
      - absolute paths
      - paths relative to config["MODEL_DIR"] (if provided)
      - paths relative to this file's directory (fallback)
    """
    model_id = (model_id or "").strip()
    if not model_id:
        return ""

    p = Path(model_id)
    if p.is_absolute():
        return str(p)

    base_dir = config.get("MODEL_DIR")
    if isinstance(base_dir, str) and base_dir.strip():
        cand = Path(base_dir) / model_id
        if cand.exists():
            return str(cand)

    # Fallback: relative to engine/process directory (common pattern in this project)
    cand2 = Path(__file__).resolve().parent / model_id
    if cand2.exists():
        return str(cand2)

    # Last resort: return as-is (caller may handle)
    return model_id


def _engine_gaussian(per_sigma_full: Dict[str, Any],
                     timing: Dict[str, Any],
                     config: Dict[str, Any],
                     model_path: str,
                     slog: Any = None) -> Tuple[TrendDecision, Dict[str, Any]]:
    """
    Minimal Gaussian engine stub:
      - Uses sign of last minus first of G83 midline (or any available sigma) as direction
      - Confidence scales with |delta| normalized by robust band width if present
    NOTE: This is intentionally simple wiring so the engine layer is functional immediately.
          You can replace this with your full trend_method_gaussian JSON-driven logic later.
    """
    sigmas_pref = [83, 68, 53, 38, 23, 8]
    series = None
    used_sigma = None

    # Attempt to use per_sigma_full["gauss_series"][sigma]["midline"] or similar
    # Be permissive; we only need a numeric series list.
    for s in sigmas_pref:
        for path_keys in (
            ("gauss_series", str(s), "midline"),
            ("gauss", str(s), "series"),
            ("gauss", str(s), "midline"),
            (str(s), "series"),
            (str(s),),
        ):
            cur = per_sigma_full
            ok = True
            for k in path_keys:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            if ok and isinstance(cur, list) and len(cur) >= 2 and all(isinstance(x, (int, float)) for x in cur[-10:]):
                series = cur
                used_sigma = s
                break
        if series is not None:
            break

    if not series:
        td = TrendDecision(trend="Neutral", confidence=0.0, model="GAUSSIAN_V2", notes="NO_SERIES")
        return td, {"engine": "gaussian", "status": "no_series"}

    first = float(series[0])
    last = float(series[-1])
    delta = last - first
    trend = "Bull" if delta > 0 else "Bear" if delta < 0 else "Neutral"

    # Optional: use channel width if present
    width = None
    try:
        gcs = per_sigma_full.get("gaussian_channels", {})
        w = gcs.get(str(used_sigma), {}).get("width")
        if isinstance(w, (int, float)) and w > 0:
            width = float(w)
    except Exception:
        width = None

    denom = width if (width and width > 0) else max(1.0, abs(delta))
    conf = min(1.0, abs(delta) / denom)

    td = TrendDecision(trend=trend, confidence=float(conf), model="GAUSSIAN_V2", notes=f"SIGMA_{used_sigma}")
    return td, {"engine": "gaussian", "sigma": used_sigma, "delta": delta, "width": width, "conf": float(conf)}


def _engine_hysteresis(per_sigma_full: Dict[str, Any],
                       timing: Dict[str, Any],
                       config: Dict[str, Any],
                       model_path: str,
                       slog: Any = None) -> Tuple[TrendDecision, Dict[str, Any]]:
    """
    Hysteresis engine wrapper.
    Uses existing calculate_trend() implementation (DEV_METHOD_v1.0_HOOKDOWN_NEU / method_hyster family).
    """
    td = calculate_trend(per_sigma_full, timing=timing, config=config, model_path=model_path)

    diag = {}
    extras = getattr(td, "extras", None)
    if isinstance(extras, dict):
        diag.update(extras)

    diag.update({"engine": "hysteresis", "model_path": model_path})
    return td, diag


def _apply_optional_guardrails(td: TrendDecision,
                               features: Dict[str, Any],
                               config: Dict[str, Any],
                               slog: Any = None) -> TrendDecision:
    """
    Apply guardrails if enabled in config.
    """
    gcfg = _cfg_get_model_toggle(config, "MODEL_guardrails")
    enabled = gcfg.get("ENABLED", True)
    if not enabled:
        return td

    try:
        return _apply_guardrails(td, features, config=config, slog=slog)
    except Exception as e:
        if slog:
            slog.error(f"[trend_guardrails] failed: {e}")
        return td


def _choose_final_decision(engine_results: Dict[str, EngineResult],
                           config: Dict[str, Any],
                           slog: Any = None) -> Tuple[TrendDecision, str]:
    """
    Deterministic arbiter:
      1) Any non-neutral wins by highest confidence
      2) Tie-break: DEFAULT_ENGINE_ORDER priority
      3) Else Neutral (highest confidence neutral)
    Returns: (decision, winner_engine_key)
    """
    enabled = [er for er in engine_results.values() if er.enabled and isinstance(er.decision, TrendDecision)]
    if not enabled:
        td = TrendDecision(trend="Neutral", confidence=0.0, model="ENGINE_LAYER", notes="NO_ENGINES_ENABLED")
        return td, ""

    non_neu = [er for er in enabled if getattr(er.decision, "trend", "Neutral") != "Neutral"]
    pool = non_neu if non_neu else enabled

    # highest confidence
    max_conf = max(float(getattr(er.decision, "confidence", 0.0) or 0.0) for er in pool)
    top = [er for er in pool if float(getattr(er.decision, "confidence", 0.0) or 0.0) == max_conf]
    if len(top) == 1:
        return top[0].decision, top[0].engine_key

    # tie-break by order
    for k in DEFAULT_ENGINE_ORDER:
        for er in top:
            if er.engine_key == k:
                return er.decision, er.engine_key

    # final fallback
    return top[0].decision, top[0].engine_key


def _save_trend_out_json(td: TrendDecision,
                         timing: Dict[str, Any],
                         config: Dict[str, Any],
                         slog: Any = None) -> Optional[str]:
    """
    Save TrendDecision once to TREND_OUT_JSON if configured.
    Returns path written, or None if skipped.
    """
    out_path = config.get("TREND_OUT_JSON")
    if not isinstance(out_path, str) or not out_path.strip():
        return None

    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "decision_time": timing.get("decision_time") or timing.get("decision_time_str"),
            "epoch_analyze": timing.get("epoch_analyze"),
            "epoch_predict": timing.get("epoch_predict"),
            "trend": getattr(td, "trend", "Neutral"),
            "confidence": float(getattr(td, "confidence", 0.0) or 0.0),
            "model": getattr(td, "model", ""),
            "notes": getattr(td, "notes", ""),
            "extras": getattr(td, "extras", None),
        }

        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)

    except Exception as e:
        if slog:
            slog.error(f"[trend_save] failed writing TREND_OUT_JSON: {e}")
        return None


def run_trend_layer(per_sigma_full: Dict[str, Any],
                    timing: Dict[str, Any],
                    windows: Optional[Dict[str, Any]] = None,
                    config: Optional[Dict[str, Any]] = None,
                    slog: Any = None) -> Tuple[TrendDecision, Dict[str, Any]]:
    """
    Public entry point for orchestrator:
      - reads config toggles
      - runs enabled engines
      - chooses final decision
      - applies guardrails (optional)
      - saves TREND_OUT_JSON (optional)
      - returns (TrendDecision, diagnostics dict for printing)
    """
    config = config or {}
    windows = windows or {}
    engine_results: Dict[str, EngineResult] = {}

    # Hysteresis
    hcfg = _cfg_get_model_toggle(config, "MODEL_hyster")
    hen = bool(hcfg.get("ENABLED", False))
    hmid = str(hcfg.get("MODEL_ID", "") or "")
    hpath = _resolve_model_path(hmid, config) if hmid else ""
    if hen:
        td, diag = _engine_hysteresis(per_sigma_full, timing, config, hpath, slog=slog)
        engine_results["MODEL_hyster"] = EngineResult("MODEL_hyster", True, hmid, td, diag)
    else:
        engine_results["MODEL_hyster"] = EngineResult("MODEL_hyster", False, hmid, TrendDecision("Neutral", 0.0, "ENGINE_LAYER", "DISABLED"), {"engine":"hysteresis","status":"disabled"})

    # Gaussian
    gcfg = _cfg_get_model_toggle(config, "MODEL_gaussian")
    gen = bool(gcfg.get("ENABLED", False))
    gmid = str(gcfg.get("MODEL_ID", "") or "")
    gpath = _resolve_model_path(gmid, config) if gmid else ""
    if gen:
        td, diag = _engine_gaussian(per_sigma_full, timing, config, gpath, slog=slog)
        engine_results["MODEL_gaussian"] = EngineResult("MODEL_gaussian", True, gmid, td, diag)
    else:
        engine_results["MODEL_gaussian"] = EngineResult("MODEL_gaussian", False, gmid, TrendDecision("Neutral", 0.0, "ENGINE_LAYER", "DISABLED"), {"engine":"gaussian","status":"disabled"})

    # Choose winner
    td_final, winner = _choose_final_decision(engine_results, config, slog=slog)

    # Attach engine diagnostics on extras
    diag_out: Dict[str, Any] = {
        "winner_engine": winner,
        "engines": {k: {"enabled": er.enabled, "model_id": er.model_id, "decision": {
            "trend": getattr(er.decision, "trend", "Neutral"),
            "confidence": float(getattr(er.decision, "confidence", 0.0) or 0.0),
            "model": getattr(er.decision, "model", ""),
            "notes": getattr(er.decision, "notes", ""),
        }, "diagnostics": er.diagnostics} for k, er in engine_results.items()},
    }

    # Best-effort features for guardrails:
    # - If the hysteresis engine provided features in extras, reuse them.
    features = {}
    try:
        w_extras = engine_results.get(winner, EngineResult("", False, "", td_final, {})).diagnostics
        if isinstance(w_extras, dict) and isinstance(w_extras.get("features"), dict):
            features = w_extras["features"]
    except Exception:
        features = {}

    td_final = _apply_optional_guardrails(td_final, features=features, config=config, slog=slog)

    # Save once
    written = _save_trend_out_json(td_final, timing=timing, config=config, slog=slog)
    if written:
        diag_out["trend_out_json"] = written

    # Also attach on decision
    setattr(td_final, "extras", {"engine_layer": diag_out, "extras": getattr(td_final, "extras", None)})

    return td_final, diag_out
