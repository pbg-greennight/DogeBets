from __future__ import annotations

from typing import Any

from .feature_mapper_v21_live import build_v21_feature_vector, mapped_context_is_active


def _f(x, default: float | None = 0.0) -> float | None:
    if x is None and default is None:
        return None
    try:
        return float(x if x is not None else default)
    except Exception:
        return default


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _flat_pick(src_row: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in src_row and src_row[key] is not None:
            return src_row[key]
    return default


def _extract_src_summary(td_features: dict) -> dict:
    src_row = _as_dict(td_features.get("src_row") or td_features.get("live_src_row"))
    if not src_row:
        return {}

    keep = [
        "src_meta_epoch",
        "src_meta_next_epoch",
        "src_meta_decision_time",
        "src_meta_close_now",
        "src_meta_hist_points",
        "src_live_has_bell",
        "src_live_has_channel_snapshot",
        "src_live_has_pv_tail",
        "src_live_has_hyst",
        "src_live_sigma_count",
    ]
    return {k: src_row.get(k) for k in keep if k in src_row}


def _extract_gcs_signals(td_features: dict, src_row: dict) -> dict:
    gcs = _as_dict(td_features.get("gcs"))
    out = {
        "px_mid_8": None,
        "px_mid_23": None,
        "zpos_83": None,
        "contracting_count": 0,
        "contracting_persist_max": 0,
        "transfer_depth": None,
        "transfer_dir": None,
        "transfer_state": None,
    }

    # Nested live payload path first
    pricepos = _as_dict(gcs.get("pricepos"))
    for key in ("s8", "8"):
        blob = _as_dict(pricepos.get(key))
        if blob:
            out["px_mid_8"] = _f(blob.get("px_mid"), None)
            break
    for key in ("s23", "23"):
        blob = _as_dict(pricepos.get(key))
        if blob:
            out["px_mid_23"] = _f(blob.get("px_mid"), None)
            break
    for key in ("s83", "83"):
        blob = _as_dict(pricepos.get(key))
        if blob:
            out["zpos_83"] = _f(blob.get("zpos"), None)
            break

    regime = _as_dict(gcs.get("regime"))
    count = 0
    persist_max = 0
    for _, blob in regime.items():
        blob = _as_dict(blob)
        if str(blob.get("regime", "")).lower() == "contracting":
            count += 1
            try:
                persist_max = max(persist_max, int(blob.get("persist") or 0))
            except Exception:
                pass
    out["contracting_count"] = count
    out["contracting_persist_max"] = persist_max

    transfer = _as_dict(gcs.get("transfer"))
    if transfer:
        out["transfer_depth"] = _f(transfer.get("depth"), None)
        out["transfer_dir"] = transfer.get("dir")
        out["transfer_state"] = transfer.get("state")

    # Flat src_row fallback / override when nested payload unavailable
    if src_row:
        out["px_mid_8"] = _f(_flat_pick(src_row,
            "src_gcs_pricepos_s8_px_mid", "src_gcs_s8_px_mid", "src_gcs_px_mid_8"), out["px_mid_8"])
        out["px_mid_23"] = _f(_flat_pick(src_row,
            "src_gcs_pricepos_s23_px_mid", "src_gcs_s23_px_mid", "src_gcs_px_mid_23"), out["px_mid_23"])
        out["zpos_83"] = _f(_flat_pick(src_row,
            "src_gcs_pricepos_s83_zpos", "src_gcs_s83_zpos", "src_gcs_zpos_83"), out["zpos_83"])
        out["contracting_count"] = int(_flat_pick(src_row,
            "src_gcs_contracting_count", default=out["contracting_count"]) or 0)
        out["contracting_persist_max"] = int(_flat_pick(src_row,
            "src_gcs_contracting_persist_max", default=out["contracting_persist_max"]) or 0)
        out["transfer_depth"] = _f(_flat_pick(src_row,
            "src_gcs_transfer_depth", default=out["transfer_depth"]), out["transfer_depth"])
        out["transfer_dir"] = _flat_pick(src_row,
            "src_gcs_transfer_dir", default=out["transfer_dir"])
        out["transfer_state"] = _flat_pick(src_row,
            "src_gcs_transfer_state", default=out["transfer_state"])

    return out


def _extract_gbc_signals(td_features: dict, src_row: dict) -> dict:
    gbc = _as_dict(td_features.get("gbc"))
    diag = _as_dict(gbc.get("diag") or td_features.get("gbc_diag"))
    out = {
        "shrink_s8": None,
        "shrink_s23": None,
        "shrink_s38": None,
        "hook_s8": None,
        "hook_s23": None,
        "hook_s38": None,
    }
    for sigma in (8, 23, 38):
        blob = _as_dict(diag.get(f"s{sigma}") or diag.get(str(sigma)))
        if blob:
            out[f"shrink_s{sigma}"] = _f(blob.get("shrink"), None)
            out[f"hook_s{sigma}"] = blob.get("hook")

    if src_row:
        for sigma in (8, 23, 38):
            out[f"shrink_s{sigma}"] = _f(_flat_pick(src_row,
                f"src_gbc_s{sigma}_shrink", f"src_gbc_shrink_s{sigma}"), out[f"shrink_s{sigma}"])
            hook = _flat_pick(src_row, f"src_gbc_s{sigma}_hook", f"src_gbc_hook_s{sigma}")
            if hook is not None:
                out[f"hook_s{sigma}"] = hook
    return out


def _extract_hyst_signals(td_features: dict, src_row: dict) -> dict:
    hyst = _as_dict(td_features.get("hysteresis"))
    spread = _as_dict(hyst.get("spread_state"))
    probe = _as_dict(hyst.get("probe"))
    out = {
        "spread_risk": spread.get("risk"),
        "spread_state": spread.get("spread"),
        "probe_flip_watch": probe.get("flip_watch"),
        "probe_fast_collapse": probe.get("fast_collapse"),
    }
    if src_row:
        out["spread_risk"] = _flat_pick(src_row, "src_hyst_spread_risk", default=out["spread_risk"])
        out["spread_state"] = _flat_pick(src_row, "src_hyst_spread_state", default=out["spread_state"])
        out["probe_flip_watch"] = _flat_pick(src_row, "src_hyst_probe_flip_watch", default=out["probe_flip_watch"])
        out["probe_fast_collapse"] = _flat_pick(src_row, "src_hyst_probe_fast_collapse", default=out["probe_fast_collapse"])
    return out


def run_model_v21_live(td_features: dict, config: dict | None = None) -> dict:
    td_features = _as_dict(td_features)
    config = _as_dict(config)

    gauss = _as_dict(td_features.get("gauss"))
    latest = _as_dict(gauss.get("latest"))
    slopes = _as_dict(gauss.get("slopes"))
    hyst = _as_dict(td_features.get("hysteresis"))

    mapped_features = build_v21_feature_vector(td_features)
    mapped_context_active = mapped_context_is_active(mapped_features)
    src_summary = _extract_src_summary(td_features)
    src_row = _as_dict(td_features.get("src_row") or td_features.get("live_src_row"))
    gcs_sig = _extract_gcs_signals(td_features, src_row)
    gbc_sig = _extract_gbc_signals(td_features, src_row)
    hyst_sig = _extract_hyst_signals(td_features, src_row)

    s8 = float(_f(slopes.get("s8"), 0.0))
    s23 = float(_f(slopes.get("s23"), 0.0))
    s53 = float(_f(slopes.get("s53"), 0.0))
    s83 = float(_f(slopes.get("s83"), 0.0))

    fast_min = float(_f(config.get("V21_LIVE_FAST_MIN"), 0.05))
    reversal_min = float(_f(config.get("V21_LIVE_REVERSAL_MIN"), 0.05))
    mid_dom_ratio = float(_f(config.get("V21_LIVE_MID_DOM_RATIO"), 1.0))
    decision_min = float(_f(config.get("V21_LIVE_DECISION_MIN"), 0.0))

    bear_exhaustion_reversal_min = float(_f(config.get("V21_LIVE_BEAR_EXHAUST_REV_MIN"), reversal_min))
    bear_exhaust_contracting_count_min = int(_f(config.get("V21_LIVE_BEAR_EXHAUST_CONTRACTING_MIN"), 4))
    bear_exhaust_s8_shrink_max = float(_f(config.get("V21_LIVE_BEAR_EXHAUST_S8_SHRINK_MAX"), 0.60))
    bear_exhaust_s23_shrink_max = float(_f(config.get("V21_LIVE_BEAR_EXHAUST_S23_SHRINK_MAX"), 0.70))
    bear_exhaust_slow_zpos83_max = float(_f(config.get("V21_LIVE_BEAR_EXHAUST_ZPOS83_MAX"), -1.0))

    bull_exhaustion_reversal_min = float(_f(config.get("V21_LIVE_BULL_EXHAUST_REV_MIN"), reversal_min))
    bull_exhaust_contracting_count_min = int(_f(config.get("V21_LIVE_BULL_EXHAUST_CONTRACTING_MIN"), 4))
    bull_exhaust_s8_shrink_max = float(_f(config.get("V21_LIVE_BULL_EXHAUST_S8_SHRINK_MAX"), 0.60))
    bull_exhaust_s23_shrink_max = float(_f(config.get("V21_LIVE_BULL_EXHAUST_S23_SHRINK_MAX"), 0.70))
    bull_exhaust_slow_zpos83_min = float(_f(config.get("V21_LIVE_BULL_EXHAUST_ZPOS83_MIN"), 1.0))

    fast_score = s8 + (0.5 * s83)
    mid_score = (1.5 * s23) + (0.75 * s53)
    continuation_score = fast_score
    reversal_score = fast_score - mid_score

    fast_agree = (_sign(s8) == _sign(s83)) and (_sign(s8) != 0)
    mid_dom = abs(mid_score) >= (mid_dom_ratio * abs(fast_score))
    mid_contradicts_fast = (_sign(mid_score) != 0) and (_sign(mid_score) != _sign(fast_score))

    px_mid_8 = gcs_sig["px_mid_8"]
    px_mid_23 = gcs_sig["px_mid_23"]
    zpos_83 = gcs_sig["zpos_83"]
    contracting_count = int(gcs_sig["contracting_count"] or 0)
    shrink_s8 = gbc_sig["shrink_s8"]
    shrink_s23 = gbc_sig["shrink_s23"]

    bear_fast_reclaim = (px_mid_8 is not None and px_mid_8 > 0)
    bear_front_reclaim = bear_fast_reclaim and px_mid_23 is not None and px_mid_23 >= 0
    bear_shrink_exhaust = ((shrink_s8 is not None and shrink_s8 <= bear_exhaust_s8_shrink_max) or
                           (shrink_s23 is not None and shrink_s23 <= bear_exhaust_s23_shrink_max))
    bear_slow_stretch = (zpos_83 is not None and zpos_83 <= bear_exhaust_slow_zpos83_max)

    bull_fast_reclaim = (px_mid_8 is not None and px_mid_8 < 0)
    bull_front_reclaim = bull_fast_reclaim and px_mid_23 is not None and px_mid_23 <= 0
    bull_shrink_exhaust = ((shrink_s8 is not None and shrink_s8 <= bull_exhaust_s8_shrink_max) or
                           (shrink_s23 is not None and shrink_s23 <= bull_exhaust_s23_shrink_max))
    bull_slow_stretch = (zpos_83 is not None and zpos_83 >= bull_exhaust_slow_zpos83_min)

    same_sign_bear_exhaustion_reversal_risk = bool(
        fast_agree and fast_score < 0 and mid_score < 0 and mid_dom
        and reversal_score >= bear_exhaustion_reversal_min
        and bear_front_reclaim
        and contracting_count >= bear_exhaust_contracting_count_min
        and bear_shrink_exhaust and bear_slow_stretch
    )

    same_sign_bull_exhaustion_downside_reversion_risk = bool(
        fast_agree and fast_score > 0 and mid_score > 0 and mid_dom
        and (-reversal_score) >= bull_exhaustion_reversal_min
        and bull_front_reclaim
        and contracting_count >= bull_exhaust_contracting_count_min
        and bull_shrink_exhaust and bull_slow_stretch
    )

    branch = "continuation" if fast_agree else "reversal_or_neutral"
    decision_score = 0.0
    trend = "Neutral"
    reason = "UNSET"

    if fast_agree:
        if same_sign_bear_exhaustion_reversal_risk:
            trend = "Neutral"
            reason = "BEAR_EXHAUSTION_REVERSION_RISK"
            branch = "same_sign_exhaustion_reversion_risk"
        elif same_sign_bull_exhaustion_downside_reversion_risk:
            trend = "Neutral"
            reason = "BULL_EXHAUSTION_DOWNSIDE_REVERSION_RISK"
            branch = "same_sign_exhaustion_reversion_risk"
        elif abs(fast_score) < fast_min:
            trend = "Neutral"
            reason = "FAST_WEAK"
        elif mid_contradicts_fast and mid_dom:
            trend = "Neutral"
            reason = "MID_CONTRADICTION_NEUTRAL"
        else:
            decision_score = continuation_score
            trend = "Bull" if decision_score > 0 else "Bear"
            reason = "BULL_CONTINUATION_BRANCH" if decision_score > 0 else "BEAR_CONTINUATION_BRANCH"
    else:
        if mid_dom and abs(reversal_score) >= reversal_min:
            decision_score = reversal_score
            trend = "Bull" if decision_score > 0 else "Bear"
            reason = "BULL_REVERSAL_BRANCH" if decision_score > 0 else "BEAR_REVERSAL_BRANCH"
        else:
            trend = "Neutral"
            reason = "FAST_CONTRADICTION_NEUTRAL"

    if trend != "Neutral" and abs(decision_score) < decision_min:
        trend = "Neutral"
        reason = "WEAK_DECISION_SCORE"

    confidence = 1.0
    bull_score = float(decision_score)
    bear_score = float(-decision_score)
    separation = float(abs(bull_score - bear_score))

    diagnostics = {
        "equation": {
            "fast_score_eq": "s8 + 0.5*s83",
            "mid_score_eq": "1.5*s23 + 0.75*s53",
            "continuation_score_eq": "fast_score",
            "reversal_score_eq": "fast_score - mid_score",
            "branch_rule": "continuation if sign(s8)==sign(s83)!=0 else reversal_or_neutral",
            "neutral_rule": "Neutral when fast is weak, when mid contradicts+dominates, or when contradiction lacks reversal strength",
            "decision_rule": "use src_row/full-log context for same-sign exhaustion neutralization before plain continuation",
            "bull_score_eq": "decision_score",
            "bear_score_eq": "-decision_score",
            "separation_eq": "abs(decision_score - (-decision_score))",
            "confidence_eq": "fixed 1.0 until tuned",
        },
        "inputs": {
            "s8": s8,
            "s23": s23,
            "s53": s53,
            "s83": s83,
            "latest_gauss": {
                "s8": _f(latest.get("s8"), 0.0),
                "s23": _f(latest.get("s23"), 0.0),
                "s38": _f(latest.get("s38"), 0.0),
                "s53": _f(latest.get("s53"), 0.0),
                "s68": _f(latest.get("s68"), 0.0),
                "s83": _f(latest.get("s83"), 0.0),
            },
            "fast_min": fast_min,
            "reversal_min": reversal_min,
            "mid_dom_ratio": mid_dom_ratio,
            "decision_min": decision_min,
        },
        "scores": {
            "fast_score": float(fast_score),
            "mid_score": float(mid_score),
            "continuation_score": float(continuation_score),
            "reversal_score": float(reversal_score),
            "decision_score": float(decision_score),
            "bull_score": float(bull_score),
            "bear_score": float(bear_score),
            "separation": float(separation),
            "confidence": float(confidence),
        },
        "flags": {
            "branch": branch,
            "fast_agree": bool(fast_agree),
            "mid_dom": bool(mid_dom),
            "mid_contradicts_fast": bool(mid_contradicts_fast),
            "mapped_context_active": bool(mapped_context_active),
            "src_row_available": bool(src_summary),
            "same_sign_bear_exhaustion_reversal_risk": bool(same_sign_bear_exhaustion_reversal_risk),
            "same_sign_bull_exhaustion_downside_reversion_risk": bool(same_sign_bull_exhaustion_downside_reversion_risk),
            "front_reclaim": bool(bear_front_reclaim or bull_front_reclaim),
            "fast_reclaim": bool(bear_fast_reclaim or bull_fast_reclaim),
            "shrink_exhaust": bool(bear_shrink_exhaust or bull_shrink_exhaust),
            "slow_stretch": bool(bear_slow_stretch or bull_slow_stretch),
        },
        "context": {
            "px_mid_8": px_mid_8,
            "px_mid_23": px_mid_23,
            "zpos_83": zpos_83,
            "contracting_count": contracting_count,
            "contracting_persist_max": int(gcs_sig["contracting_persist_max"] or 0),
            "transfer_depth": gcs_sig["transfer_depth"],
            "transfer_dir": gcs_sig["transfer_dir"],
            "transfer_state": gcs_sig["transfer_state"],
            "shrink_s8": shrink_s8,
            "shrink_s23": shrink_s23,
            "shrink_s38": gbc_sig["shrink_s38"],
            "hook_s8": gbc_sig["hook_s8"],
            "hook_s23": gbc_sig["hook_s23"],
            "hook_s38": gbc_sig["hook_s38"],
            "hyst_spread_risk": hyst_sig["spread_risk"],
            "hyst_spread_state": hyst_sig["spread_state"],
            "hyst_probe_flip_watch": hyst_sig["probe_flip_watch"],
            "hyst_probe_fast_collapse": hyst_sig["probe_fast_collapse"],
        },
        "mapped_features": mapped_features,
        "src_summary": src_summary,
        "hyst_meta": _as_dict(hyst.get("meta")),
        "raw_td_features": td_features,
    }

    return {
        "trend": trend,
        "confidence": float(confidence),
        "reason": reason,
        "model": "v21_live",
        "diagnostics": diagnostics,
    }
