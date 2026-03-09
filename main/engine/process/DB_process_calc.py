# main/engine/process/DB_process_calc.py

import json
from pathlib import Path
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# CONFIG LOAD + VALIDATE
# -----------------------------
# ----------------------------------------------------------------------------------------------------------------------
# Trend method config loader (v2): supports includes + engine_library injection.
# DB_process_trend.py should call: load_trend_method_config_v2(model_path)
#
# Rules:
# - Load root JSON
# - Merge includes (deep merge; dicts merge recursively, scalars/lists overwrite)
# - "includes" entries can be relative paths or legacy names; we try a few fallback resolutions.
# - Inject engine_library if still missing (trend_method_engine_library.json in same models dir).
# ----------------------------------------------------------------------------------------------------------------------

from pathlib import Path
import copy
import json
import logging

_log_cfg = logging.getLogger(__name__)

def _deep_merge_dicts(base: dict, extra: dict) -> dict:
    """Deep merge `extra` into `base` (mutates & returns base). Lists/scalars overwrite."""
    for k, v in (extra or {}).items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge_dicts(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base

def _resolve_include_path(model_path: Path, include_ref: str) -> "Path | None":
    """Resolve an include reference to an on-disk JSON path using a few compatibility rules."""
    if not include_ref:
        return None
    inc = str(include_ref).replace("\\", "/").lstrip("/")
    models_dir = model_path.parent

    # 1) direct relative
    cand = (models_dir / inc)
    if cand.exists():
        return cand

    # 2) drop leading 'parts/' if present
    if inc.startswith("parts/"):
        cand = (models_dir / inc[len("parts/"):])
        if cand.exists():
            return cand

    # 3) legacy prefix rewrite: 'trend_method_v2.<name>.json' -> 'trend_method_<name>.json'
    if inc.startswith("trend_method_v2.") and inc.endswith(".json"):
        tail = inc[len("trend_method_v2."):]
        cand = (models_dir / f"trend_method_{tail}")
        if cand.exists():
            return cand

    # 4) known alias mapping (your repo naming)
    alias_map = {
        "trend_method_gaussian_logic.json": "trend_method_gaussian_engine.json",
        "trend_method_v2.episode_logic.json": "trend_method_hysteresis_engine.json",
        "trend_method_v2.regimes_fusion_probe.json": "trend_method_engine_layer.json",
        "trend_method_v2.guardrails.json": "trend_method_guardrails.json",
        "trend_method_v2.time_inputs_pairs.json": "trend_method_time_inputs_pairs.json",
        "trend_method_v2.features_thresholds.json": "trend_method_features_thresholds.json",
        "trend_method_v2.stacks.json": "trend_method_stacks.json",
        "trend_method_v2.engine_library.json": "trend_method_engine_library.json",
    }
    if inc in alias_map:
        cand = (models_dir / alias_map[inc])
        if cand.exists():
            return cand

    # 5) basename-only in same dir
    cand = (models_dir / Path(inc).name)
    if cand.exists():
        return cand

    return None

def load_trend_method_config_v2(model_path: str) -> dict:
    """
    Loads and returns the final merged trend method config.

    Flow:
    - loads root JSON
    - merges "includes"
    - injects engine_library (if missing)
    - returns final merged dict
    """
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Trend method config not found: {p}")

    root = json.loads(p.read_text(encoding="utf-8"))
    merged = copy.deepcopy(root)

    includes = list(root.get("includes") or [])
    for inc in includes:
        inc_path = _resolve_include_path(p, inc)
        if not inc_path or not inc_path.exists():
            _log_cfg.warning("[trend_cfg] include missing: %s (from %s)", inc, p)
            continue
        try:
            inc_cfg = json.loads(inc_path.read_text(encoding="utf-8"))
        except Exception as e:
            _log_cfg.warning("[trend_cfg] include read failed: %s (%s)", inc_path, e)
            continue
        _deep_merge_dicts(merged, inc_cfg)

    # engine_library injection (late, so includes can override)
    if "engine_library" not in merged or not isinstance(merged.get("engine_library"), dict):
        lib_path = _resolve_include_path(p, "trend_method_engine_library.json") or (p.parent / "trend_method_engine_library.json")
        if lib_path and lib_path.exists():
            try:
                merged["engine_library"] = json.loads(lib_path.read_text(encoding="utf-8"))
                _log_cfg.info("[trend_cfg] injected engine_library from %s", lib_path)
            except Exception as e:
                _log_cfg.warning("[trend_cfg] engine_library injection failed: %s (%s)", lib_path, e)

    # Optional: keep the original include list for traceability
    merged.setdefault("_meta", {})
    merged["_meta"].update({
        "root_path": str(p),
        "includes_attempted": includes,
    })

    return merged

# Back-compat shim (older callers)
def load_trend_method_config(model_path: str) -> dict:
    return load_trend_method_config_v2(model_path)


@lru_cache(maxsize=16)
def load_model_config(model_path: str) -> dict:
    p = Path(model_path)
    cfg = json.loads(p.read_text(encoding="utf-8"))
    validate_model_config(cfg)
    return cfg

def validate_model_config(cfg: dict) -> None:
    required_top = ["model_id", "sigmas", "neutral", "scores", "decision", "overrides"]
    for k in required_top:
        if k not in cfg:
            raise ValueError(f"[model_config] missing required key: {k}")

    # basic sanity checks
    neu = cfg["neutral"]
    if not (0.0 <= neu["hard_gate_threshold"] <= 1.0):
        raise ValueError("[model_config] neutral.hard_gate_threshold out of range")

    w = neu["weights"]
    wsum = float(w["flat"]) + float(w["disagree"]) + float(w["hook"])
    if wsum <= 0.0:
        raise ValueError("[model_config] neutral.weights sum must be > 0")

    dec = cfg["decision"]
    if dec["best_min_threshold"] < 0.0 or dec["best_min_threshold"] > 1.0:
        raise ValueError("[model_config] decision.best_min_threshold out of range")
    if dec["margin_threshold"] < 0.0 or dec["margin_threshold"] > 1.0:
        raise ValueError("[model_config] decision.margin_threshold out of range")


# -----------------------------
# HELPERS
# -----------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def sign(x: float) -> int:
    if x > 0: return 1
    if x < 0: return -1
    return 0

def majority_sign(vals: list[int]) -> int:
    s = sum(vals)
    return sign(s)  # +1, -1, or 0


# -----------------------------
# FEATURE COMPUTATION (MODEL NEEDS)
# -----------------------------
# NOTE: This assumes you already have per-sigma dump_diag outputs:
# sign_to, hook, flat for each sigma in cfg["sigmas"]["wanted"]

def build_model_features(per_sigma: dict, cfg: dict) -> dict:
    """
    per_sigma example:
      {
        8:  {"sign_to": -1, "hook": 1, "flat": 0.0},
        23: {"sign_to": +1, "hook": 0, "flat": 0.0},
        ...
      }
    """
    wanted = cfg["sigmas"]["wanted"]
    long_sig = cfg["sigmas"]["long"]

    feats = {}

    # flatten per-sigma values into key style sign_to_s8, hook_s8, flat_s8
    flats = []
    signs_all = []
    for s in wanted:
        d = per_sigma.get(s, {})
        feats[f"sign_to_s{s}"] = int(d.get("sign_to", 0))
        feats[f"hook_s{s}"] = int(d.get("hook", 0))
        feats[f"flat_s{s}"] = float(d.get("flat", 0.0))
        flats.append(feats[f"flat_s{s}"])
        signs_all.append(feats[f"sign_to_s{s}"])

    feats["flat_mean"] = sum(flats) / max(1, len(flats))
    feats["hook_any"] = 1 if any(feats[f"hook_s{s}"] == 1 for s in wanted) else 0

    # disagreement_ratio vs long majority
    long_signs = [feats[f"sign_to_s{s}"] for s in long_sig]
    maj_long = majority_sign(long_signs)
    feats["maj_long"] = maj_long

    if maj_long == 0:
        feats["disagree_ratio"] = 0.0
    else:
        disagree = 0
        total = 0
        for v in signs_all:
            if v == 0:
                continue
            total += 1
            if sign(v) != maj_long:
                disagree += 1
        feats["disagree_ratio"] = (disagree / total) if total else 0.0

    # hook-down flag for override
    od = cfg["overrides"]["hook_down_force_bear"]
    s = int(od["sigma"])
    feats["hook_down_s8"] = 1 if (feats.get(f"hook_s{s}", 0) == 1 and feats.get(f"sign_to_s{s}", 0) == -1) else 0

    return feats


# -----------------------------
# NEUTRALITY
# -----------------------------

def data_neutrality(feats: dict, cfg: dict) -> dict:
    neu = cfg["neutral"]

    flat_start = float(neu["flat"]["start"])
    flat_span  = float(neu["flat"]["span"])
    dis_start  = float(neu["disagree"]["start"])
    dis_span   = float(neu["disagree"]["span"])

    flat_score = clamp((feats["flat_mean"] - flat_start) / flat_span)
    disagree_score = clamp((feats["disagree_ratio"] - dis_start) / dis_span)
    hook_score = 1.0 if feats["hook_any"] == 1 else 0.0

    w = neu["weights"]
    neutral_score = clamp(
        float(w["flat"]) * flat_score +
        float(w["disagree"]) * disagree_score +
        float(w["hook"]) * hook_score
    )

    is_neutral_hard = neutral_score >= float(neu["hard_gate_threshold"])

    reason = "NEU_HARD_GATE" if is_neutral_hard else "NEU_OK"
    return {
        "neutral_score": neutral_score,
        "neutral_flat_score": flat_score,
        "neutral_disagree_score": disagree_score,
        "neutral_hook_score": hook_score,
        "is_neutral_hard": is_neutral_hard,
        "reason": reason
    }


# -----------------------------
# REVERSAL (Bear→Bull)
# -----------------------------

def data_reversal(feats: dict, neutral_score: float, cfg: dict) -> dict:
    mid_sig = cfg["sigmas"]["mid"]
    long_sig = cfg["sigmas"]["long"]

    long_bear_strength = sum(1 for s in long_sig if feats[f"sign_to_s{s}"] == -1) / len(long_sig)
    mid_bull_strength  = sum(1 for s in mid_sig  if feats[f"sign_to_s{s}"] == +1) / len(mid_sig)

    raw = long_bear_strength * mid_bull_strength
    penalty = float(cfg["scores"]["reversal"]["neutral_penalty"])
    reversal_score = clamp(raw * (1.0 - penalty * neutral_score))

    return {
        "long_bear_strength": long_bear_strength,
        "mid_bull_strength": mid_bull_strength,
        "reversal_score_raw": raw,
        "reversal_score": reversal_score
    }


# -----------------------------
# CONTINUATION (Bull/Bear)
# -----------------------------

def data_continuation(feats: dict, neutral_score: float, cfg: dict) -> dict:
    core = cfg["sigmas"]["core"]

    bull_support = sum(1 for s in core if feats[f"sign_to_s{s}"] == +1) / len(core)
    bear_support = sum(1 for s in core if feats[f"sign_to_s{s}"] == -1) / len(core)

    bull_pen = float(cfg["scores"]["bull"]["neutral_penalty"])
    bear_pen = float(cfg["scores"]["bear"]["neutral_penalty"])
    bear_boost = float(cfg["scores"]["bear"]["hook_down_boost"]) if feats["hook_down_s8"] == 1 else 0.0

    bull_score = clamp(bull_support * (1.0 - bull_pen * neutral_score))
    bear_score = clamp(bear_support * (1.0 - bear_pen * neutral_score) + bear_boost)

    return {
        "bull_support": bull_support,
        "bear_support": bear_support,
        "bull_score": bull_score,
        "bear_score": bear_score,
        "bear_hook_boost": bear_boost
    }


# -----------------------------
# FINAL DECISION (LOCKED ORDER)
# -----------------------------

def data_process_calc(per_sigma: dict, model_path: str) -> dict:
    cfg = load_model_config(model_path)
    feats = build_model_features(per_sigma, cfg)

    neu = data_neutrality(feats, cfg)
    neutral_score = neu["neutral_score"]

    # Step 1: Neutral hard gate
    if neu["is_neutral_hard"]:
        return {
            "model_id": cfg["model_id"],
            "trend": "Neutral",
            "confidence": 0.0,  # placeholder
            "scores": {"neutral": neutral_score, "bull": 0.0, "bear": 0.0, "reversal": 0.0},
            "features": feats,
            "reason": "NEU_HARD_GATE"
        }

    rev = data_reversal(feats, neutral_score, cfg)
    cont = data_continuation(feats, neutral_score, cfg)

    # Step 2: Hook-down override forces Bear
    if cfg["overrides"]["hook_down_force_bear"]["enabled"] and feats["hook_down_s8"] == 1:
        return {
            "model_id": cfg["model_id"],
            "trend": "Bear",
            "confidence": 0.0,
            "scores": {"neutral": neutral_score, "bull": cont["bull_score"], "bear": cont["bear_score"], "reversal": rev["reversal_score"]},
            "features": feats,
            "reason": "HOOK_DOWN_OVERRIDE"
        }

    # Step 3: Score compare
    scores = {
        "Bear→Bull": rev["reversal_score"],
        "Bull": cont["bull_score"],
        "Bear": cont["bear_score"]
    }

    best_label = max(scores, key=scores.get)
    best_val = scores[best_label]
    second_val = sorted(scores.values(), reverse=True)[1]

    best_min = float(cfg["decision"]["best_min_threshold"])
    margin = float(cfg["decision"]["margin_threshold"])

    if best_val < best_min:
        trend = "Neutral"
        reason = "NEU_LOW_BEST"
    elif (best_val - second_val) < margin:
        trend = "Neutral"
        reason = "NEU_TIE_MARGIN"
    else:
        trend = best_label
        reason = f"WIN_{best_label}"

    return {
        "model_id": cfg["model_id"],
        "trend": trend,
        "confidence": 0.0,  # placeholder for future confidence_level()
        "scores": {"neutral": neutral_score, "bull": cont["bull_score"], "bear": cont["bear_score"], "reversal": rev["reversal_score"]},
        "features": feats,
        "reason": reason
    }


# =============================================================================
# STAGE 1 TRUTH ENGINE OUTPUT: Bell + Channels (moved out of printing)
# =============================================================================

def _resolve_global_last_ts(per_sigma_hist: Dict[int, Dict[str, Any]], decision_dt: datetime) -> Optional[datetime]:
    last_ts_candidates: List[datetime] = []
    for _sigma, _pack in (per_sigma_hist or {}).items():
        ts_all = _pack.get("ts", []) or []
        eligible = [t for t in ts_all if t <= decision_dt]
        if eligible:
            last_ts_candidates.append(max(eligible))
    return max(last_ts_candidates) if last_ts_candidates else None


def build_bell_out(
    *,
    timing: Any,
    decision_dt: datetime,
    per_sigma_hist: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    close_series: Optional[Tuple[List[datetime], List[float]]] = None,
) -> Dict[str, Any]:
    """
    Truth-source bell structure:
      - tail anchor (global last_ts)
      - pv_ref pair (sigma=PV_REF_SIGMA)
      - per-sigma PV pairs
      - leg1/leg2 series + snapshot_metrics per sigma
      - combined PV→NOW series per sigma
    """
    from DB_process_metrics import snapshot_metrics, find_last_extrema_pair
    from DB_process_slicing import slice_by_window, slice_tail_window_with_fallback

    out: Dict[str, Any] = {"status": "ok"}

    last_ts = _resolve_global_last_ts(per_sigma_hist, decision_dt)
    if not last_ts:
        return {"status": "empty", "reason": "no_samples", "bell": {}, "bell_curve_series": {}}

    last_sample_age = (decision_dt - last_ts).total_seconds()

    pv_ref_sigma = int(config.get("PV_REF_SIGMA", 23))
    pv_min_sep = float(config.get("PV_MIN_SEP_SECONDS", 10.0))
    lookback_minutes = int(config.get("BELL_CURVE_LOOKBACK_MINUTES", 240))
    start_dt = last_ts - timedelta(minutes=lookback_minutes)

    # pv_ref pair
    ref_pack = per_sigma_hist.get(pv_ref_sigma, {})
    ref_ts_all = ref_pack.get("ts", []) or []
    ref_vals_all = ref_pack.get("values", []) or []
    ref_ts_win, ref_vals_win = slice_by_window(ref_ts_all, ref_vals_all, start_dt, last_ts)
    pv_pair_ref = find_last_extrema_pair(ref_ts_win, ref_vals_win, last_ts, min_sep_seconds=pv_min_sep)

    # per-sigma pairs
    per_sigma_pairs: Dict[int, Optional[Dict[str, Any]]] = {}
    for sigma in sorted(per_sigma_hist.keys()):
        ts_all = per_sigma_hist[sigma].get("ts", []) or []
        vals_all = per_sigma_hist[sigma].get("values", []) or []
        ts_win, vals_win = slice_by_window(ts_all, vals_all, start_dt, last_ts)
        per_sigma_pairs[sigma] = find_last_extrema_pair(ts_win, vals_win, last_ts, min_sep_seconds=pv_min_sep)

    bell_curve_series: Dict[int, Dict[str, Any]] = {}
    bell: Dict[str, Any] = {
        "last_ts": last_ts,
        "last_sample_age": float(last_sample_age),
        "pv_ref_sigma": pv_ref_sigma,
        "pv_pair_ref": pv_pair_ref,
        "per_sigma_pairs": per_sigma_pairs,
        "lookback_minutes": lookback_minutes,
        "start_dt": start_dt,
        "leg1": {"sigmas": {}},
        "leg2": {"sigmas": {}},
        "diagnostics": {"per_sigma": {}},
    }

    # BTC close context for the pv_ref audit line (optional)
    btc_start = btc_current = None
    try:
        if close_series and close_series[1]:
            btc_start = close_series[1][0]
            btc_current = close_series[1][-1]
    except Exception:
        pass
    bell["btc_close"] = {"start": btc_start, "current": btc_current}

    # build per-sigma legs + metrics + combined
    for sigma in sorted(per_sigma_hist.keys()):
        ts_all = per_sigma_hist[sigma].get("ts", []) or []
        vals_all = per_sigma_hist[sigma].get("values", []) or []
        pair = per_sigma_pairs.get(sigma)

        # choose PV bounds (per-sigma, else pv_ref, else empty)
        if pair is not None:
            t_prev = pair["prev"]["ts"]
            t_last = pair["last"]["ts"]
        elif pv_pair_ref is not None:
            t_prev = pv_pair_ref["prev"]["ts"]
            t_last = pv_pair_ref["last"]["ts"]
        else:
            bell["leg1"]["sigmas"][sigma] = {"ts": [], "values": [], "metrics": {}, "t0": None, "t1": None}
            bell["leg2"]["sigmas"][sigma] = {"ts": [], "values": [], "metrics": {}, "t0": None, "t1": None}
            bell_curve_series[sigma] = {"ts": [], "values": []}
            continue

        ts_leg1, vals_leg1 = slice_by_window(ts_all, vals_all, t_prev, t_last)
        ts_leg2, vals_leg2 = slice_tail_window_with_fallback(ts_all, vals_all, t_last, last_ts)

        m1 = snapshot_metrics(vals_leg1, ts_leg1)
        m2 = snapshot_metrics(vals_leg2, ts_leg2)

        bell["leg1"]["sigmas"][sigma] = {"t0": t_prev, "t1": t_last, "ts": ts_leg1, "values": vals_leg1, "metrics": m1}
        bell["leg2"]["sigmas"][sigma] = {"t0": t_last, "t1": last_ts, "ts": ts_leg2, "values": vals_leg2, "metrics": m2}

        # combined PV→NOW (avoid duplicate point at join)
        ts_comb = list(ts_leg1 or [])
        vals_comb = list(vals_leg1 or [])
        if ts_leg2 and ts_comb and ts_leg2[0] == ts_comb[-1]:
            ts_leg2 = ts_leg2[1:]
            vals_leg2 = vals_leg2[1:]
        ts_comb.extend(list(ts_leg2 or []))
        vals_comb.extend(list(vals_leg2 or []))
        bell_curve_series[sigma] = {"ts": ts_comb, "values": vals_comb}

        # Diagnostics ("extra brain" lines)
        try:
            tail_vals = list(vals_comb[-max(20, min(120, len(vals_comb))):])
            tail_ts = list(ts_comb[-len(tail_vals):])
            mt = snapshot_metrics(tail_vals, tail_ts)
        
            # --- helper: slope over last k steps (in seconds domain) ---
            def _seg_slope(vals: List[float], ts: List[datetime], k: int) -> float:
                if len(vals) < 2:
                    return 0.0
                kk = max(1, min(int(k), len(vals) - 1))
                dt = max((ts[-1] - ts[-1 - kk]).total_seconds(), 1e-9)
                return (float(vals[-1]) - float(vals[-1 - kk])) / dt
        
            # prev vs last "abs slope" (for shrink ratio and sign->sign)
            k_seg = max(6, min(20, len(tail_vals) // 6))
            prev_slice_vals = tail_vals[:-k_seg] if len(tail_vals) > 2 * k_seg else tail_vals
            prev_slice_ts = tail_ts[:-k_seg] if len(tail_ts) > 2 * k_seg else tail_ts
        
            prev_slope = _seg_slope(prev_slice_vals, prev_slice_ts, k=k_seg)
            last_slope = _seg_slope(tail_vals, tail_ts, k=k_seg)
        
            prev_abs = abs(prev_slope)
            last_abs = abs(last_slope)
            shrink = (last_abs / (prev_abs + 1e-9)) if prev_abs > 0 else 0.0
        
            sign_from = 0 if prev_slope == 0 else (1 if prev_slope > 0 else -1)
            sign_to = 0 if last_slope == 0 else (1 if last_slope > 0 else -1)
        
            # hook: sign disagreement short vs long
            short_s = _seg_slope(tail_vals, tail_ts, k=max(4, k_seg // 2))
            long_s = _seg_slope(tail_vals, tail_ts, k=max(12, k_seg * 2))
            hook = 1 if (short_s == 0.0 or long_s == 0.0) else (1 if (short_s > 0) != (long_s > 0) else 0)
        
            # eps: MAD of instantaneous slopes (robust noise floor) + flat score
            diffs = []
            for i in range(1, len(tail_vals)):
                dt_i = max((tail_ts[i] - tail_ts[i - 1]).total_seconds(), 1e-9)
                diffs.append((float(tail_vals[i]) - float(tail_vals[i - 1])) / dt_i)
        
            eps = 0.0
            if len(diffs) >= 3:
                import statistics
                med = statistics.median(diffs)
                mad = statistics.median([abs(d - med) for d in diffs]) + 1e-9
                eps = float(mad)
        
                # flat score from tail diffs MAD (1=flat, 0=not flat)
                flat = max(0.0, min(1.0, 1.0 - (abs(mt.get("slope", 0.0)) / (5.0 * mad))))
            else:
                flat = 0.0
        
            bell["diagnostics"]["per_sigma"][sigma] = {
                "shrink": float(shrink),
                "flat": float(flat),
                "hook": int(hook),
                "prev_abs": float(prev_abs),
                "last_abs": float(last_abs),
                "eps": float(eps),
                "sign_from": int(sign_from),
                "sign_to": int(sign_to),
            }
        except Exception:
            bell["diagnostics"]["per_sigma"][sigma] = {
                "shrink": 0.0,
                "flat": 0.0,
                "hook": 0,
                "prev_abs": 0.0,
                "last_abs": 0.0,
                "eps": 0.0,
                "sign_from": 0,
                "sign_to": 0,
            }
        
    out["bell"] = bell
    out["bell_curve_series"] = bell_curve_series
    return out


def build_channels_out(
    *,
    timing: Any,
    windows: Any,
    per_sigma_full: Dict[int, Dict[str, Any]],
    per_sigma_hist: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Truth-source channel structures:
      - snapshot: build_channel_snapshot(per_sigma_full)
      - pv_tail: build_channel_pv_tail(per_sigma_hist)
    """
    from DB_process_gauss_channel import build_channel_snapshot, build_channel_pv_tail

    k = float(config.get("GAUSS_CHANNEL_K", 2.0))

    snapshot = build_channel_snapshot(per_sigma_full, k=k)

    # pv_tail uses pv_ref sigma extrema internally (your printing used it directly);
    # keep the ref sigma consistent with config.
    pv_ref_sigma = int(config.get("PV_REF_SIGMA", 23))
    pv_tail = build_channel_pv_tail(
        timing=timing,
        windows=windows,
        per_sigma_hist=per_sigma_hist,
        pv_ref_sigma=pv_ref_sigma,
        config=config,
    )

    return {"snapshot": snapshot, "pv_tail": pv_tail}


def build_calc_out(
    *,
    timing: Any,
    windows: Any,
    decision_dt: datetime,
    per_sigma_full: Dict[int, Dict[str, Any]],
    per_sigma_hist: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    close_series: Optional[Tuple[List[datetime], List[float]]] = None,
) -> Dict[str, Any]:
    """
    Stage 1: single truth object consumed by printing and (later) trend.
    """
    bell_out = build_bell_out(
        timing=timing,
        decision_dt=decision_dt,
        per_sigma_hist=per_sigma_hist,
        config=config,
        close_series=close_series,
    )
    channels_out = build_channels_out(
        timing=timing,
        windows=windows,
        per_sigma_full=per_sigma_full,
        per_sigma_hist=per_sigma_hist,
        config=config,
    )

    return {
        "status": "ok",
        "decision_dt": decision_dt,
        "bell": bell_out.get("bell", {}) or {},
        "bell_curve_series": bell_out.get("bell_curve_series", {}) or {},
        "channels": channels_out,
    }

from pathlib import Path
import importlib.util
import traceback

MODELS_DIR = Path(__file__).resolve().parent / "models"
MODEL_CONFIG_GLOB = "trend_method_v2_0.json"
PREDICTIONS_LATEST_FILE = MODELS_DIR / "model_predictions" / "model_predictions_latest.json"
PREDICTIONS_HISTORY_FILE = MODELS_DIR / "model_predictions_history" / "model_predictions_history.jsonl"

def _safe_number(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_json_config(path: Path) -> Dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"model config must be object: {path}")
    cfg.setdefault("enabled", True)
    cfg.setdefault("description", "")
    cfg.setdefault("thresholds", {})
    return cfg


def _load_model_module(model_py_path: Path):
    name = f"dogebets_model_{model_py_path.stem}"
    spec = importlib.util.spec_from_file_location(name, model_py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load model module: {model_py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def _error_result(model_id: str, err: Exception) -> Dict[str, Any]:
    return {
        "model_id": model_id,
        "trend": "Neutral",
        "confidence": 1.0,
        "score": 0.0,
        "reason": "model_error",
        "debug": {"error": str(err), "trace": traceback.format_exc(limit=3)},
        "raw_features_used": {},
    }

def _normalize_model_result(model_id: str, result: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(result or {})
    out["model_id"] = str(out.get("model_id") or model_id)
    out["trend"] = str(out.get("trend") or "Neutral")
    if out["trend"] not in ("Bull", "Bear", "Neutral"):
        out["trend"] = "Neutral"
    out["confidence"] = _safe_number(out.get("confidence", 1.0), 1.0)
    out["score"] = _safe_number(out.get("score", 0.0), 0.0)
    out["reason"] = str(out.get("reason") or "")
    out["debug"] = out.get("debug") if isinstance(out.get("debug"), dict) else {}
    out["raw_features_used"] = out.get("raw_features_used") if isinstance(out.get("raw_features_used"),
                                                                              dict) else {}
    return out

def load_enabled_model_configs(models_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    base = models_dir or MODELS_DIR
    out: List[Dict[str, Any]] = []
    for cfg_path in sorted(base.glob(MODEL_CONFIG_GLOB)):
        try:
            cfg = _load_json_config(cfg_path)
        except Exception as exc:
            logging.warning("[models] failed loading config %s: %s", cfg_path, exc)
            continue
        if not bool(cfg.get("enabled", True)):
            continue
        cfg["_config_path"] = str(cfg_path)
        out.append(cfg)
    return out

def persist_model_predictions(epoch_payload: Dict[str, Any]) -> None:
    PREDICTIONS_LATEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PREDICTIONS_LATEST_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(epoch_payload, indent=2), encoding="utf-8")
    tmp.replace(PREDICTIONS_LATEST_FILE)
    with PREDICTIONS_HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(epoch_payload) + "\n")


def run_enabled_models(
        *,
        epoch: int,
        next_epoch: int,
        timestamp: str,
        features: Dict[str, Any],
        models_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    base = models_dir or MODELS_DIR
    configs = load_enabled_model_configs(base)
    results: List[Dict[str, Any]] = []

    for cfg in configs:
        model_id = str(cfg.get("model_id") or "")
        if not model_id:
            continue
        model_path = base / f"model_{model_id.replace('trend_method_', '')}.py"
        if not model_path.exists():
            logging.warning("[models] model file missing for %s (%s)", model_id, model_path)
            results.append(_error_result(model_id, FileNotFoundError(f"model file missing: {model_path}")))
            continue

        try:
            module = _load_model_module(model_path)
            if not hasattr(module, "run_model"):
                raise AttributeError("run_model(features, config) not found")
            raw = module.run_model(features, cfg)
            results.append(_normalize_model_result(model_id, raw))
        except Exception as exc:
            logging.exception("[models] %s failed", model_id)
            results.append(_error_result(model_id, exc))

    primary = results[0] if results else {
        "trend": "Neutral",
        "score": 0.0,
        "confidence": 1.0,
    }

    payload = {
        "epoch": int(epoch),
        "next_epoch": int(next_epoch),
        "timestamp": str(timestamp),
        "models": results,
        "v2.0_trend": str(primary.get("trend", "Neutral")),
        "v2.0_score": _safe_number(primary.get("score", 0.0), 0.0),
        "v2.0_confidence": _safe_number(primary.get("confidence", 1.0), 1.0),
        "v2.0_correct": None,
    }

    persist_model_predictions(payload)
    return payload