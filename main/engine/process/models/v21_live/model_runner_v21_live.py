from __future__ import annotations

from typing import Any, Dict

try:
    from main.engine.process.models.v21_live.feature_mapper_v21_live import (
        build_v21_feature_bundle,
        mapped_context_is_active,
    )
    from main.engine.process.models.v21_live.v21_live_debug_report import (
        build_v21_debug_report,
    )
    from main.engine.process.models.v21_live.v21_live_validator import (
        normalize_src_row,
        validate_src_row,
    )
except Exception:  # pragma: no cover - standalone fallback
    from process.models.v21_live.feature_mapper_v21_live import (
        build_v21_feature_bundle,
        mapped_context_is_active,
    )
    from process.models.v21_live.v21_live_debug_report import (
        build_v21_debug_report,
    )
    from process.models.v21_live.v21_live_validator import (
        normalize_src_row,
        validate_src_row,
    )


def _f(x: Any, default: float | None = 0.0) -> float | None:
    if x is None and default is None:
        return None
    try:
        return float(x if x is not None else default)
    except Exception:
        return default


def _sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dir_sign(value: Any) -> int:
    text = str(value or "").strip().lower()
    if text in {"up", "bull", "positive"}:
        return 1
    if text in {"down", "bear", "negative"}:
        return -1
    return 0


def _mean(*values: Any) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _extract_src_summary(src_row: dict) -> dict:
    if not src_row:
        return {}
    keep = [
        "src_contract_version",
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
        "src_debug_missing_required_count",
        "src_debug_missing_optional_count",
        "src_debug_cov_msbc",
        "src_debug_cov_gcs",
        "src_debug_cov_csd",
        "src_debug_cov_dcsd",
        "src_debug_cov_gbc",
        "src_debug_cov_hyst",
        "src_debug_cov_overall",
    ]
    return {k: src_row.get(k) for k in keep if k in src_row}


def _extract_nested_gcs(td_features: dict) -> dict:
    gcs = _as_dict(td_features.get("gcs"))
    out = {
        "px_mid_8": None,
        "px_mid_23": None,
        "zpos_83": None,
        "contracting_count": None,
        "transfer_depth": None,
        "transfer_dir": None,
        "transfer_state": None,
    }
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
    if regime:
        count = 0
        for _, blob in regime.items():
            blob = _as_dict(blob)
            if str(blob.get("regime", "")).lower() == "contracting":
                count += 1
        out["contracting_count"] = int(count)
    transfer = _as_dict(gcs.get("transfer"))
    if transfer:
        out["transfer_depth"] = _f(transfer.get("depth"), None)
        out["transfer_dir"] = transfer.get("dir")
        out["transfer_state"] = transfer.get("state")
    return out


def _extract_nested_gbc(td_features: dict) -> dict:
    gbc = _as_dict(td_features.get("gbc"))
    diag = _as_dict(gbc.get("diag") or td_features.get("gbc_diag"))
    out: Dict[str, Any] = {}
    for sigma in (8, 23, 38):
        blob = _as_dict(diag.get(f"s{sigma}") or diag.get(str(sigma)))
        if blob:
            out[f"shrink_s{sigma}"] = _f(blob.get("shrink"), None)
            out[f"hook_s{sigma}"] = blob.get("hook")
    return out


def _extract_nested_hyst(td_features: dict) -> dict:
    hyst = _as_dict(td_features.get("hysteresis"))
    spread = _as_dict(hyst.get("spread_state"))
    probe = _as_dict(hyst.get("probe"))
    return {
        "hyst_spread_risk": spread.get("risk"),
        "hyst_spread_state": spread.get("spread"),
        "hyst_pressure_state": spread.get("pressure"),
        "hyst_stack_stability": _f(spread.get("stability"), None),
        "hyst_stack_stability_state": spread.get("stability_state"),
        "hyst_near_cross_state": spread.get("near_cross_state"),
        "hyst_fan_tightness": _f(spread.get("fan_tightness"), None),
        "hyst_stack_alignment": _f(spread.get("stack_alignment"), None),
        "hyst_probe_flip_watch": probe.get("flip_watch"),
        "hyst_probe_fast_collapse": probe.get("fast_collapse"),
        "hyst_probe_recovery": probe.get("recovery"),
    }


def build_v21_live_feature_context(src_row: dict, td_features: dict, config: dict | None = None) -> dict:
    td_features = _as_dict(td_features)
    src_row_norm = normalize_src_row(_as_dict(src_row)) if src_row else {}
    src_audit = _as_dict(td_features.get("src_audit")) or (validate_src_row(src_row_norm) if src_row_norm else {})
    src_summary = _extract_src_summary(src_row_norm)

    gauss = _as_dict(td_features.get("gauss"))
    slopes = _as_dict(gauss.get("slopes"))
    curvature = _as_dict(gauss.get("curvature"))
    latest = _as_dict(gauss.get("latest"))

    mapped_bundle = build_v21_feature_bundle({**td_features, "src_row": src_row_norm}, config=config)
    src_ctx = _as_dict(mapped_bundle.get("context"))
    mapped_features = _as_dict(mapped_bundle.get("vector"))
    mapped_missing = list(mapped_bundle.get("missing") or [])
    mapped_aliases = list(mapped_bundle.get("used_aliases") or [])

    nested_gcs = _extract_nested_gcs(td_features)
    nested_gbc = _extract_nested_gbc(td_features)
    nested_hyst = _extract_nested_hyst(td_features)

    coverage = _as_dict(src_audit.get("family_coverage"))

    ctx: Dict[str, Any] = {
        "s8": _f(slopes.get("s8"), 0.0),
        "s23": _f(slopes.get("s23"), 0.0),
        "s38": _f(slopes.get("s38"), 0.0),
        "s53": _f(slopes.get("s53"), 0.0),
        "s68": _f(slopes.get("s68"), 0.0),
        "s83": _f(slopes.get("s83"), 0.0),
        "curvature_s23": _f(_first_non_null(curvature.get("s23"), src_ctx.get("curvature_s23")), 0.0),
        "curvature_s53": _f(_first_non_null(curvature.get("s53"), src_ctx.get("curvature_s53")), 0.0),
        "latest_gauss": {
            "s8": _f(latest.get("s8"), 0.0),
            "s23": _f(latest.get("s23"), 0.0),
            "s38": _f(latest.get("s38"), 0.0),
            "s53": _f(latest.get("s53"), 0.0),
            "s68": _f(latest.get("s68"), 0.0),
            "s83": _f(latest.get("s83"), 0.0),
        },
        # primary mapped context + nested backstops
        "px_mid_8": _first_non_null(src_ctx.get("px_mid_8"), nested_gcs.get("px_mid_8")),
        "px_mid_23": _first_non_null(src_ctx.get("px_mid_23"), nested_gcs.get("px_mid_23")),
        "px_mid_38": src_ctx.get("px_mid_38"),
        "px_mid_53": src_ctx.get("px_mid_53"),
        "px_mid_68": src_ctx.get("px_mid_68"),
        "px_mid_83": src_ctx.get("px_mid_83"),
        "zpos_83": _first_non_null(src_ctx.get("zpos_83"), nested_gcs.get("zpos_83")),
        "contracting_count": int(_first_non_null(src_ctx.get("contracting_count"), nested_gcs.get("contracting_count"), 0) or 0),
        "expanding_count": int(_first_non_null(src_ctx.get("expanding_count"), 0) or 0),
        "transfer_depth": _first_non_null(src_ctx.get("transfer_depth"), nested_gcs.get("transfer_depth")),
        "transfer_dir": _first_non_null(src_ctx.get("transfer_dir"), nested_gcs.get("transfer_dir")),
        "transfer_state": _first_non_null(src_ctx.get("transfer_state"), nested_gcs.get("transfer_state")),
        "front_back_disagreement": _first_non_null(src_ctx.get("front_back_disagreement"), 0.0),
        "reclaim_front": _first_non_null(src_ctx.get("reclaim_front"), 0.0),
        "reclaim_slow": _first_non_null(src_ctx.get("reclaim_slow"), 0.0),
        "shrink_s8": _first_non_null(src_ctx.get("shrink_s8"), nested_gbc.get("shrink_s8")),
        "shrink_s23": _first_non_null(src_ctx.get("shrink_s23"), nested_gbc.get("shrink_s23")),
        "shrink_s38": _first_non_null(src_ctx.get("shrink_s38"), nested_gbc.get("shrink_s38")),
        "release_score": _first_non_null(src_ctx.get("release_score"), 0.0),
        "reversal_stage": _first_non_null(src_ctx.get("reversal_stage"), 0.0),
        "same_sign_exhaustion": _first_non_null(src_ctx.get("same_sign_exhaustion"), 0.0),
        "csd_spread_now": _first_non_null(src_ctx.get("csd_spread_now"), 0.0),
        "csd_spread_delta": _first_non_null(src_ctx.get("csd_spread_delta"), 0.0),
        # MSBC / GBC structure
        "msbc_transfer_dir": src_ctx.get("msbc_transfer_dir"),
        "msbc_transfer_depth": src_ctx.get("msbc_transfer_depth"),
        "msbc_transfer_state": src_ctx.get("msbc_transfer_state"),
        "msbc_continuity_mean": src_ctx.get("msbc_continuity_mean"),
        "msbc_propagation_agree_ratio": src_ctx.get("msbc_propagation_agree_ratio"),
        "msbc_propagation_disagree_ratio": src_ctx.get("msbc_propagation_disagree_ratio"),
        "msbc_pair_rows": src_ctx.get("msbc_pair_rows"),
        "msbc_slope_mono": src_ctx.get("msbc_slope_mono"),
        "msbc_tan_mono": src_ctx.get("msbc_tan_mono"),
        "msbc_curve_decay": src_ctx.get("msbc_curve_decay"),
        "msbc_linm_mono": src_ctx.get("msbc_linm_mono"),
        "msbc_override_count": src_ctx.get("msbc_override_count"),
        "msbc_fast_override_count": src_ctx.get("msbc_fast_override_count"),
        "msbc_slow_override_count": src_ctx.get("msbc_slow_override_count"),
        "msbc_disorder_count": src_ctx.get("msbc_disorder_count"),
        "msbc_age_seconds": src_ctx.get("msbc_age_seconds"),
        "msbc_age_label": src_ctx.get("msbc_age_label"),
        "hook_s8": _first_non_null(src_ctx.get("hook_s8"), nested_gbc.get("hook_s8")),
        "hook_s23": _first_non_null(src_ctx.get("hook_s23"), nested_gbc.get("hook_s23")),
        "hook_s38": _first_non_null(src_ctx.get("hook_s38"), nested_gbc.get("hook_s38")),
        "flat_s8": src_ctx.get("flat_s8"),
        "flat_s23": src_ctx.get("flat_s23"),
        "flat_s38": src_ctx.get("flat_s38"),
        "turn_age_s8": src_ctx.get("turn_age_s8"),
        "turn_age_s23": src_ctx.get("turn_age_s23"),
        "turn_age_s38": src_ctx.get("turn_age_s38"),
        "hook_persist_s8": src_ctx.get("hook_persist_s8"),
        "hook_persist_s23": src_ctx.get("hook_persist_s23"),
        "hook_persist_s38": src_ctx.get("hook_persist_s38"),
        "front_flat_mean": src_ctx.get("front_flat_mean"),
        "front_turn_age_min": src_ctx.get("front_turn_age_min"),
        "front_hook_persist_max": src_ctx.get("front_hook_persist_max"),
        "front_sign_persist_max": src_ctx.get("front_sign_persist_max"),
        "front_norm_ctx_mean": src_ctx.get("front_norm_ctx_mean"),
        "front_hook_active": src_ctx.get("front_hook_active"),
        "gbc_active_sigma_count": src_ctx.get("gbc_active_sigma_count"),
        "gbc_series_available": src_ctx.get("gbc_series_available"),
        # HYST
        "hyst_spread_risk": _first_non_null(src_ctx.get("hyst_spread_risk"), nested_hyst.get("hyst_spread_risk")),
        "hyst_spread_state": _first_non_null(src_ctx.get("hyst_spread_state"), nested_hyst.get("hyst_spread_state")),
        "hyst_pressure_state": _first_non_null(src_ctx.get("hyst_pressure_state"), nested_hyst.get("hyst_pressure_state")),
        "hyst_stack_stability": _first_non_null(src_ctx.get("hyst_stack_stability"), nested_hyst.get("hyst_stack_stability")),
        "hyst_stack_stability_state": _first_non_null(src_ctx.get("hyst_stack_stability_state"), nested_hyst.get("hyst_stack_stability_state")),
        "hyst_near_cross_state": _first_non_null(src_ctx.get("hyst_near_cross_state"), nested_hyst.get("hyst_near_cross_state")),
        "hyst_fan_tightness": _first_non_null(src_ctx.get("hyst_fan_tightness"), nested_hyst.get("hyst_fan_tightness")),
        "hyst_stack_alignment": _first_non_null(src_ctx.get("hyst_stack_alignment"), nested_hyst.get("hyst_stack_alignment")),
        "hyst_probe_flip_watch": _first_non_null(src_ctx.get("hyst_probe_flip_watch"), nested_hyst.get("hyst_probe_flip_watch")),
        "hyst_probe_fast_collapse": _first_non_null(src_ctx.get("hyst_probe_fast_collapse"), nested_hyst.get("hyst_probe_fast_collapse")),
        "hyst_probe_recovery": _first_non_null(src_ctx.get("hyst_probe_recovery"), nested_hyst.get("hyst_probe_recovery")),
        "hyst_cross_rate": _first_non_null(src_ctx.get("hyst_cross_rate"), 0.0),
        "hyst_order_stability": _first_non_null(src_ctx.get("hyst_order_stability"), 0.0),
        "hyst_ladder_compression": _first_non_null(src_ctx.get("hyst_ladder_compression"), 0.0),
        # debug/meta
        "mapped_features": mapped_features,
        "mapped_missing": mapped_missing,
        "mapped_aliases": mapped_aliases,
        "mapped_context_active": mapped_context_is_active(mapped_features),
        "src_summary": src_summary,
        "src_audit": src_audit,
        "src_row_available": bool(src_row_norm),
        "src_cov_overall": _f(coverage.get("overall"), 0.0),
        "src_cov_msbc": _f(coverage.get("msbc"), 0.0),
        "src_cov_gcs": _f(coverage.get("gcs"), 0.0),
        "src_cov_csd": _f(coverage.get("csd"), 0.0),
        "src_cov_dcsd": _f(coverage.get("dcsd"), 0.0),
        "src_cov_gbc": _f(coverage.get("gbc"), 0.0),
        "src_cov_hyst": _f(coverage.get("hyst"), 0.0),
        "src_missing_required_count": len(src_audit.get("missing_required") or []),
        "src_missing_optional_count": len(src_audit.get("missing_optional") or []),
    }
    return ctx


def score_v21_live_context(ctx: dict, config: dict | None = None) -> dict:
    ctx = _as_dict(ctx)
    config = _as_dict(config)

    s8 = float(_f(ctx.get("s8"), 0.0))
    s23 = float(_f(ctx.get("s23"), 0.0))
    s53 = float(_f(ctx.get("s53"), 0.0))
    s83 = float(_f(ctx.get("s83"), 0.0))

    fast_min = float(_f(config.get("V21_FAST_MIN"), 0.05))
    reversal_min = float(_f(config.get("V21_REVERSAL_MIN"), 0.05))
    mid_dom_ratio = float(_f(config.get("V21_MID_DOM_RATIO"), 1.0))
    decision_min = float(_f(config.get("V21_DECISION_MIN"), 0.0))
    low_stability_max = float(_f(config.get("V21_LOW_STABILITY_MAX"), 0.45))
    release_score_min = float(_f(config.get("V21_RELEASE_SCORE_MIN"), 0.35))
    reversal_stage_min = float(_f(config.get("V21_REVERSAL_STAGE_MIN"), 0.50))
    same_sign_exhaustion_min = float(_f(config.get("V21_SAME_SIGN_EXHAUSTION_MIN"), 0.50))
    bear_exhaust_contracting_count_min = int(_f(config.get("V21_BEAR_EXHAUST_CONTRACTING_MIN"), 2.0) or 2)
    bull_exhaust_contracting_count_min = int(_f(config.get("V21_BULL_EXHAUST_CONTRACTING_MIN"), 2.0) or 2)
    bear_exhaust_fast_reclaim_min = float(_f(config.get("V21_BEAR_EXHAUST_FAST_RECLAIM_MIN"), -0.15))
    bull_exhaust_fast_reclaim_max = float(_f(config.get("V21_BULL_EXHAUST_FAST_RECLAIM_MAX"), 0.15))
    bear_exhaust_s8_shrink_max = float(_f(config.get("V21_BEAR_EXHAUST_S8_SHRINK_MAX"), 0.95))
    bear_exhaust_s23_shrink_max = float(_f(config.get("V21_BEAR_EXHAUST_S23_SHRINK_MAX"), 0.95))
    bull_exhaust_s8_shrink_max = float(_f(config.get("V21_BULL_EXHAUST_S8_SHRINK_MAX"), 0.95))
    bull_exhaust_s23_shrink_max = float(_f(config.get("V21_BULL_EXHAUST_S23_SHRINK_MAX"), 0.95))
    bear_exhaust_slow_zpos83_max = float(_f(config.get("V21_BEAR_EXHAUST_SLOW_ZPOS83_MAX"), -0.55))
    bull_exhaust_slow_zpos83_min = float(_f(config.get("V21_BULL_EXHAUST_SLOW_ZPOS83_MIN"), 0.55))
    bear_exhaustion_reversal_min = float(_f(config.get("V21_BEAR_EXHAUSTION_REVERSAL_MIN"), 0.10))
    bull_exhaustion_reversal_min = float(_f(config.get("V21_BULL_EXHAUSTION_REVERSAL_MIN"), 0.10))
    coverage_floor = float(_f(config.get("V21_LIVE_COVERAGE_FLOOR"), 0.85))

    fast_score = s8 + 0.5 * s83
    mid_score = 1.5 * s23 + 0.75 * s53
    continuation_score_base = fast_score
    reversal_score_base = fast_score - mid_score

    fast_sign = _sign(fast_score)
    mid_sign = _sign(mid_score)
    continuation_sign = _sign(continuation_score_base)
    reversal_sign = _sign(reversal_score_base)
    fast_agree = _sign(s8) != 0 and _sign(s8) == _sign(s83)
    mid_dom = abs(mid_score) >= (mid_dom_ratio * abs(fast_score))
    mid_contradicts_fast = fast_sign != 0 and mid_sign != 0 and fast_sign != mid_sign

    px_mid_8 = _f(ctx.get("px_mid_8"), None)
    px_mid_23 = _f(ctx.get("px_mid_23"), None)
    zpos_83 = _f(ctx.get("zpos_83"), None)
    contracting_count = int(_f(ctx.get("contracting_count"), 0.0) or 0)
    transfer_depth = _f(ctx.get("transfer_depth"), 0.0) or 0.0
    transfer_dir = ctx.get("transfer_dir") or ctx.get("msbc_transfer_dir")
    transfer_dir_sign = _dir_sign(transfer_dir)
    shrink_s8 = _f(ctx.get("shrink_s8"), None)
    shrink_s23 = _f(ctx.get("shrink_s23"), None)
    release_score = float(_f(ctx.get("release_score"), 0.0))
    reversal_stage = float(_f(ctx.get("reversal_stage"), 0.0))
    same_sign_exhaustion = float(_f(ctx.get("same_sign_exhaustion"), 0.0))

    continuity_mean = _clamp(float(_f(ctx.get("msbc_continuity_mean"), 0.0)), 0.0, 1.0)
    propagation_agree = _clamp(float(_f(ctx.get("msbc_propagation_agree_ratio"), 0.0)), 0.0, 1.0)
    propagation_disagree = _clamp(float(_f(ctx.get("msbc_propagation_disagree_ratio"), 0.0)), 0.0, 1.0)
    pair_rows = max(int(_f(ctx.get("msbc_pair_rows"), 0.0) or 0), 1)
    override_ratio = _clamp(float(_f(ctx.get("msbc_override_count"), 0.0)) / 6.0, 0.0, 1.0)
    disorder_ratio = _clamp(float(_f(ctx.get("msbc_disorder_count"), 0.0)) / max(4.0, 4.0 * pair_rows), 0.0, 1.0)
    slope_mono = _clamp(float(_f(ctx.get("msbc_slope_mono"), 0.0)), 0.0, 1.0)
    tan_mono = _clamp(float(_f(ctx.get("msbc_tan_mono"), 0.0)), 0.0, 1.0)
    curve_decay = _clamp(float(_f(ctx.get("msbc_curve_decay"), 0.0)), 0.0, 1.0)

    front_hook_active = bool(ctx.get("front_hook_active"))
    front_hook_score = 1.0 if front_hook_active else 0.0
    front_hook_persist_max = _clamp(float(_f(ctx.get("front_hook_persist_max"), 0.0)) / 4.0, 0.0, 1.0)
    front_turn_age_min = _f(ctx.get("front_turn_age_min"), None)
    front_recent_turn = 1.0 if (front_turn_age_min is not None and front_turn_age_min <= 3.0) else 0.0
    front_flat_mean = _clamp(float(_f(ctx.get("front_flat_mean"), 0.0)), 0.0, 1.0)
    front_norm_ctx_mean = _clamp(float(_f(ctx.get("front_norm_ctx_mean"), 0.0)) / 2.0, 0.0, 1.0)

    transfer_depth_norm = _clamp(float(transfer_depth) / 4.0, 0.0, 1.0)
    transfer_align = 1.0 if continuation_sign != 0 and transfer_dir_sign == continuation_sign else 0.0
    transfer_oppose = 1.0 if continuation_sign != 0 and transfer_dir_sign != 0 and transfer_dir_sign != continuation_sign else 0.0

    structure_continuation_support = _clamp(
        0.30 * transfer_align * transfer_depth_norm
        + 0.20 * continuity_mean
        + 0.15 * propagation_agree
        + 0.10 * slope_mono
        + 0.10 * tan_mono
        + 0.05 * curve_decay,
        0.0,
        1.0,
    )
    structure_continuation_drag = _clamp(
        0.20 * disorder_ratio
        + 0.20 * override_ratio
        + 0.15 * front_hook_score
        + 0.10 * front_recent_turn
        + 0.10 * front_flat_mean,
        0.0,
        1.0,
    )
    structure_reversal_support = _clamp(
        0.20 * front_hook_score
        + 0.15 * front_hook_persist_max
        + 0.15 * front_recent_turn
        + 0.15 * disorder_ratio
        + 0.15 * override_ratio
        + 0.10 * propagation_disagree
        + 0.10 * front_norm_ctx_mean
        + 0.10 * transfer_oppose * transfer_depth_norm,
        0.0,
        1.0,
    )

    continuation_adjust = structure_continuation_support - structure_continuation_drag
    reversal_adjust = structure_reversal_support + 0.10 * max(0.0, release_score) + 0.10 * max(0.0, reversal_stage)

    continuation_score = continuation_score_base
    if continuation_sign != 0:
        continuation_score = continuation_score_base + continuation_sign * continuation_adjust

    reversal_score = reversal_score_base
    if reversal_sign != 0:
        reversal_score = reversal_score_base + reversal_sign * reversal_adjust

    hyst_spread_risk = ctx.get("hyst_spread_risk")
    hyst_spread_state = ctx.get("hyst_spread_state")
    hyst_pressure_state = ctx.get("hyst_pressure_state")
    hyst_stack_stability = _f(ctx.get("hyst_stack_stability"), None)
    hyst_probe_fast_collapse = bool(ctx.get("hyst_probe_fast_collapse"))

    bear_fast_reclaim = px_mid_8 is not None and px_mid_8 >= bear_exhaust_fast_reclaim_min
    bear_front_reclaim = bear_fast_reclaim and px_mid_23 is not None and px_mid_23 >= 0
    bear_shrink_exhaust = (
        (shrink_s8 is not None and shrink_s8 <= bear_exhaust_s8_shrink_max)
        or (shrink_s23 is not None and shrink_s23 <= bear_exhaust_s23_shrink_max)
    )
    bear_slow_stretch = zpos_83 is not None and zpos_83 <= bear_exhaust_slow_zpos83_max

    bull_fast_reclaim = px_mid_8 is not None and px_mid_8 <= bull_exhaust_fast_reclaim_max
    bull_front_reclaim = bull_fast_reclaim and px_mid_23 is not None and px_mid_23 <= 0
    bull_shrink_exhaust = (
        (shrink_s8 is not None and shrink_s8 <= bull_exhaust_s8_shrink_max)
        or (shrink_s23 is not None and shrink_s23 <= bull_exhaust_s23_shrink_max)
    )
    bull_slow_stretch = zpos_83 is not None and zpos_83 >= bull_exhaust_slow_zpos83_min

    release_ready = (release_score >= release_score_min) or (reversal_stage >= reversal_stage_min)
    same_sign_exhaust_ready = same_sign_exhaustion >= same_sign_exhaustion_min
    collapse_watch = hyst_spread_risk in {"collapse_watch", "reversal_watch"} or hyst_probe_fast_collapse
    whipsaw_risk = hyst_spread_risk == "whipsaw_risk"
    low_stability = hyst_stack_stability is not None and hyst_stack_stability <= low_stability_max
    pressure_active = hyst_pressure_state in {"approaching", "pressuring"}

    same_sign_bear_exhaustion_reversal_risk = bool(
        fast_agree
        and fast_score < 0
        and mid_score < 0
        and mid_dom
        and (reversal_score >= bear_exhaustion_reversal_min or release_ready)
        and bear_front_reclaim
        and contracting_count >= bear_exhaust_contracting_count_min
        and bear_slow_stretch
        and (bear_shrink_exhaust or same_sign_exhaust_ready)
        and (collapse_watch or low_stability or pressure_active or same_sign_exhaust_ready or front_hook_active)
    )

    same_sign_bull_exhaustion_downside_reversion_risk = bool(
        fast_agree
        and fast_score > 0
        and mid_score > 0
        and mid_dom
        and ((-reversal_score) >= bull_exhaustion_reversal_min or release_ready)
        and bull_front_reclaim
        and contracting_count >= bull_exhaust_contracting_count_min
        and bull_slow_stretch
        and (bull_shrink_exhaust or same_sign_exhaust_ready)
        and (collapse_watch or low_stability or pressure_active or same_sign_exhaust_ready or front_hook_active)
    )

    validation_penalty = 0.0
    if int(ctx.get("src_missing_required_count") or 0) > 0:
        validation_penalty += 1.0
    if float(_f(ctx.get("src_cov_overall"), 0.0)) < coverage_floor:
        validation_penalty += 0.30

    neutral_pressure = 0.0
    if abs(fast_score) < fast_min:
        neutral_pressure += 0.35
    if mid_contradicts_fast and mid_dom:
        neutral_pressure += 0.65
    if collapse_watch:
        neutral_pressure += 0.25
    if whipsaw_risk:
        neutral_pressure += 0.25
    if low_stability:
        neutral_pressure += 0.20
    if same_sign_bear_exhaustion_reversal_risk or same_sign_bull_exhaustion_downside_reversion_risk:
        neutral_pressure += 1.0
    neutral_pressure += validation_penalty

    exhaustion_score = max(
        1.0 if (same_sign_bear_exhaustion_reversal_risk or same_sign_bull_exhaustion_downside_reversion_risk) else 0.0,
        same_sign_exhaustion,
        0.5 * release_score + 0.5 * reversal_stage,
        structure_reversal_support,
    )

    return {
        "fast_score": float(fast_score),
        "mid_score": float(mid_score),
        "continuation_score_base": float(continuation_score_base),
        "reversal_score_base": float(reversal_score_base),
        "continuation_score": float(continuation_score),
        "reversal_score": float(reversal_score),
        "continuation_adjust": float(continuation_adjust),
        "reversal_adjust": float(reversal_adjust),
        "structure_continuation_support": float(structure_continuation_support),
        "structure_continuation_drag": float(structure_continuation_drag),
        "structure_reversal_support": float(structure_reversal_support),
        "structure_disorder_ratio": float(disorder_ratio),
        "front_hook_score": float(front_hook_score),
        "decision_min": float(decision_min),
        "fast_min": float(fast_min),
        "reversal_min": float(reversal_min),
        "mid_dom_ratio": float(mid_dom_ratio),
        "neutral_pressure": float(neutral_pressure),
        "validation_penalty": float(validation_penalty),
        "exhaustion_score": float(exhaustion_score),
        "fast_agree": bool(fast_agree),
        "mid_dom": bool(mid_dom),
        "mid_contradicts_fast": bool(mid_contradicts_fast),
        "same_sign_bear_exhaustion_reversal_risk": bool(same_sign_bear_exhaustion_reversal_risk),
        "same_sign_bull_exhaustion_downside_reversion_risk": bool(same_sign_bull_exhaustion_downside_reversion_risk),
        "collapse_watch": bool(collapse_watch),
        "whipsaw_risk": bool(whipsaw_risk),
        "low_stability": bool(low_stability),
        "pressure_active": bool(pressure_active),
        "hyst_spread_risk": hyst_spread_risk,
        "hyst_spread_state": hyst_spread_state,
        "hyst_pressure_state": hyst_pressure_state,
    }


def decide_v21_live(score_obj: dict, config: dict | None = None) -> dict:
    score_obj = _as_dict(score_obj)

    fast_score = float(_f(score_obj.get("fast_score"), 0.0))
    continuation_score = float(_f(score_obj.get("continuation_score"), 0.0))
    reversal_score = float(_f(score_obj.get("reversal_score"), 0.0))
    neutral_pressure = float(_f(score_obj.get("neutral_pressure"), 0.0))
    decision_min = float(_f(score_obj.get("decision_min"), 0.0))
    fast_min = float(_f(score_obj.get("fast_min"), 0.0))
    reversal_min = float(_f(score_obj.get("reversal_min"), 0.0))

    bull_continuation = max(0.0, continuation_score)
    bear_continuation = max(0.0, -continuation_score)
    bull_reversal = max(0.0, reversal_score)
    bear_reversal = max(0.0, -reversal_score)

    branch = "continuation" if score_obj.get("fast_agree") else "reversal_or_neutral"
    decision_score = 0.0
    trend = "Neutral"
    reason = "UNSET"

    if score_obj.get("fast_agree"):
        if score_obj.get("same_sign_bear_exhaustion_reversal_risk"):
            trend = "Neutral"
            reason = "BEAR_EXHAUSTION_REVERSION_RISK"
            branch = "same_sign_exhaustion_reversion_risk"
        elif score_obj.get("same_sign_bull_exhaustion_downside_reversion_risk"):
            trend = "Neutral"
            reason = "BULL_EXHAUSTION_DOWNSIDE_REVERSION_RISK"
            branch = "same_sign_exhaustion_reversion_risk"
        elif abs(fast_score) < fast_min:
            trend = "Neutral"
            reason = "FAST_WEAK"
        elif score_obj.get("collapse_watch") and score_obj.get("low_stability"):
            trend = "Neutral"
            reason = "HYST_COLLAPSE_WATCH"
        elif score_obj.get("whipsaw_risk") and score_obj.get("mid_contradicts_fast"):
            trend = "Neutral"
            reason = "HYST_WHIPSAW_RISK"
        elif score_obj.get("mid_contradicts_fast") and score_obj.get("mid_dom"):
            trend = "Neutral"
            reason = "MID_CONTRADICTION_NEUTRAL"
        else:
            decision_score = continuation_score
            trend = "Bull" if decision_score > 0 else "Bear"
            reason = "BULL_CONTINUATION_BRANCH" if decision_score > 0 else "BEAR_CONTINUATION_BRANCH"
    else:
        if score_obj.get("mid_dom") and abs(reversal_score) >= reversal_min:
            decision_score = reversal_score
            trend = "Bull" if decision_score > 0 else "Bear"
            reason = "BULL_REVERSAL_BRANCH" if decision_score > 0 else "BEAR_REVERSAL_BRANCH"
        else:
            trend = "Neutral"
            reason = "FAST_CONTRADICTION_NEUTRAL"

    if trend != "Neutral" and abs(decision_score) < decision_min:
        trend = "Neutral"
        decision_score = 0.0
        reason = "WEAK_DECISION_SCORE"

    # Public directional scores should align to the active branch.
    if branch == "continuation":
        bull_score = bull_continuation
        bear_score = bear_continuation
    elif branch == "reversal_or_neutral":
        bull_score = bull_reversal
        bear_score = bear_reversal
    else:
        bull_score = max(bull_continuation, bull_reversal)
        bear_score = max(bear_continuation, bear_reversal)

    neutral_score = float(neutral_pressure) + (1.0 if trend == "Neutral" else 0.0)

    ranked = sorted([bull_score, bear_score, neutral_score], reverse=True)
    top = ranked[0] if ranked else 0.0
    second = ranked[1] if len(ranked) > 1 else 0.0
    total = sum(ranked) + 1e-9
    separation = float(top - second)
    confidence = _clamp(separation / total)
    if trend != "Neutral":
        directional_total = abs(continuation_score) + abs(reversal_score) + neutral_score + 1e-9
        confidence = max(confidence, _clamp(abs(decision_score) / directional_total))

    decision_score_norm = float(decision_score / (abs(continuation_score) + abs(reversal_score) + neutral_score + 1e-9))

    out = dict(score_obj)
    out.update(
        {
            "branch": branch,
            "decision_score": float(decision_score),
            "decision_score_norm": decision_score_norm,
            "trend": trend,
            "reason": reason,
            "bull_score": float(bull_score),
            "bear_score": float(bear_score),
            "neutral_score": float(neutral_score),
            "bull_continuation": float(bull_continuation),
            "bear_continuation": float(bear_continuation),
            "bull_reversal": float(bull_reversal),
            "bear_reversal": float(bear_reversal),
            "separation": float(separation),
            "confidence": float(confidence),
        }
    )
    return out


def run_model_v21_live(td_features: dict, config: dict | None = None) -> dict:
    td_features = _as_dict(td_features)
    config = _as_dict(config)
    src_row = _as_dict(td_features.get("src_row") or td_features.get("live_src_row"))

    ctx = build_v21_live_feature_context(src_row, td_features, config=config)
    score_obj = score_v21_live_context(ctx, config=config)
    decision = decide_v21_live(score_obj, config=config)
    debug_report = build_v21_debug_report(src_row, ctx, score_obj, decision, config=config)

    diagnostics = {
        "equation": {
            "fast_score_eq": "s8 + 0.5*s83",
            "mid_score_eq": "1.5*s23 + 0.75*s53",
            "continuation_score_eq": "fast_score + sign(fast_score)*continuation_adjust",
            "reversal_score_eq": "(fast_score - mid_score) + sign(reversal_score_base)*reversal_adjust",
            "confidence_eq": "max(separation/(bull+bear+neutral), abs(decision_score)/(abs(cont)+abs(rev)+neutral))",
            "branch_rule": "continuation if fast_agree else reversal_or_neutral",
            "decision_rule": "use src_row-derived GCS/CSD/DCSD/HYST/MSBC/GBC context before plain continuation",
        },
        "inputs": {
            "s8": ctx.get("s8"),
            "s23": ctx.get("s23"),
            "s53": ctx.get("s53"),
            "s83": ctx.get("s83"),
            "latest_gauss": ctx.get("latest_gauss"),
            "fast_min": score_obj.get("fast_min"),
            "reversal_min": score_obj.get("reversal_min"),
            "mid_dom_ratio": score_obj.get("mid_dom_ratio"),
            "decision_min": score_obj.get("decision_min"),
        },
        "scores": {
            "fast_score": decision.get("fast_score"),
            "mid_score": decision.get("mid_score"),
            "continuation_score": decision.get("continuation_score"),
            "reversal_score": decision.get("reversal_score"),
            "continuation_adjust": decision.get("continuation_adjust"),
            "reversal_adjust": decision.get("reversal_adjust"),
            "exhaustion_score": decision.get("exhaustion_score"),
            "neutral_score": decision.get("neutral_score"),
            "decision_score": decision.get("decision_score"),
            "decision_score_norm": decision.get("decision_score_norm"),
            "bull_score": decision.get("bull_score"),
            "bear_score": decision.get("bear_score"),
            "bull_continuation": decision.get("bull_continuation"),
            "bear_continuation": decision.get("bear_continuation"),
            "bull_reversal": decision.get("bull_reversal"),
            "bear_reversal": decision.get("bear_reversal"),
            "separation": decision.get("separation"),
            "confidence": decision.get("confidence"),
        },
        "flags": {
            "branch": decision.get("branch"),
            "fast_agree": decision.get("fast_agree"),
            "mid_dom": decision.get("mid_dom"),
            "mid_contradicts_fast": decision.get("mid_contradicts_fast"),
            "mapped_context_active": bool(ctx.get("mapped_context_active")),
            "src_row_available": bool(ctx.get("src_row_available")),
            "same_sign_bear_exhaustion_reversal_risk": decision.get("same_sign_bear_exhaustion_reversal_risk"),
            "same_sign_bull_exhaustion_downside_reversion_risk": decision.get("same_sign_bull_exhaustion_downside_reversion_risk"),
            "collapse_watch": decision.get("collapse_watch"),
            "whipsaw_risk": decision.get("whipsaw_risk"),
            "low_stability": decision.get("low_stability"),
            "pressure_active": decision.get("pressure_active"),
        },
        "context": {
            "px_mid_8": ctx.get("px_mid_8"),
            "px_mid_23": ctx.get("px_mid_23"),
            "zpos_83": ctx.get("zpos_83"),
            "contracting_count": ctx.get("contracting_count"),
            "transfer_depth": ctx.get("transfer_depth"),
            "transfer_dir": ctx.get("transfer_dir"),
            "transfer_state": ctx.get("transfer_state"),
            "msbc_continuity_mean": ctx.get("msbc_continuity_mean"),
            "msbc_propagation_agree_ratio": ctx.get("msbc_propagation_agree_ratio"),
            "msbc_override_count": ctx.get("msbc_override_count"),
            "msbc_disorder_count": ctx.get("msbc_disorder_count"),
            "front_hook_active": ctx.get("front_hook_active"),
            "front_turn_age_min": ctx.get("front_turn_age_min"),
            "release_score": ctx.get("release_score"),
            "reversal_stage": ctx.get("reversal_stage"),
            "same_sign_exhaustion": ctx.get("same_sign_exhaustion"),
            "hyst_spread_risk": ctx.get("hyst_spread_risk"),
            "hyst_spread_state": ctx.get("hyst_spread_state"),
            "hyst_pressure_state": ctx.get("hyst_pressure_state"),
            "hyst_stack_stability": ctx.get("hyst_stack_stability"),
            "hyst_probe_flip_watch": ctx.get("hyst_probe_flip_watch"),
            "hyst_probe_fast_collapse": ctx.get("hyst_probe_fast_collapse"),
            "hyst_probe_recovery": ctx.get("hyst_probe_recovery"),
        },
        "mapped_features": ctx.get("mapped_features"),
        "mapped_missing": ctx.get("mapped_missing"),
        "mapped_aliases": ctx.get("mapped_aliases"),
        "src_summary": ctx.get("src_summary"),
        "src_audit": ctx.get("src_audit"),
        "validation": debug_report.get("validation"),
        "scorecard": debug_report.get("scorecard"),
        "score_obj": decision,
        "raw_td_features": td_features,
    }

    return {
        "trend": decision.get("trend", "Neutral"),
        "confidence": float(decision.get("confidence", 0.0)),
        "reason": decision.get("reason", "UNSET"),
        "model": "v21_live",
        "diagnostics": diagnostics,
    }
