from __future__ import annotations

from typing import Any, Dict, Mapping

try:
    from main.engine.process.models.v21_live.v21_live_validator import normalize_src_row
except Exception:  # pragma: no cover - standalone fallback
    from process.models.v21_live.v21_live_validator import normalize_src_row


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _f(value: Any, default: float | None = 0.0) -> float | None:
    if value is None and default is None:
        return None
    try:
        return float(value if value is not None else default)
    except Exception:
        return default


def _i(value: Any, default: int = 0) -> int:
    try:
        return int(value if value is not None else default)
    except Exception:
        return int(default)


def _flat_pick(src_row: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in src_row and src_row[key] is not None:
            return src_row[key]
    return default


def _encode_state(value: Any, mapping: Dict[str, float], default: float = 0.0) -> float:
    if value is None:
        return default
    key = str(value).strip().lower()
    return float(mapping.get(key, default))


def _mean(*values: Any) -> float | None:
    xs = [float(v) for v in values if v is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _front_min_positive(*values: Any) -> float | None:
    xs = [float(v) for v in values if v is not None and float(v) > 0]
    if not xs:
        return None
    return float(min(xs))


def map_src_row_to_v21_context(src_row: dict, config: dict | None = None) -> dict:
    """
    Map the canonical src_row into the richer live v21 context object.

    This is intentionally src-row only. Slopes/curvatures still come from
    td_features/gauss because they are native runtime values.
    """
    src = normalize_src_row(_as_dict(src_row))

    context = {
        # --- GCS ---
        "px_mid_8": _f(_flat_pick(src, "src_gcs_px_mid_8"), None),
        "px_mid_23": _f(_flat_pick(src, "src_gcs_px_mid_23"), None),
        "px_mid_38": _f(_flat_pick(src, "src_gcs_px_mid_38"), None),
        "px_mid_53": _f(_flat_pick(src, "src_gcs_px_mid_53"), None),
        "px_mid_68": _f(_flat_pick(src, "src_gcs_px_mid_68"), None),
        "px_mid_83": _f(_flat_pick(src, "src_gcs_px_mid_83"), None),
        "zpos_83": _f(_flat_pick(src, "src_gcs_zpos_83"), None),
        "contracting_count": _i(_flat_pick(src, "src_gcs_contracting_count", default=0), 0),
        "expanding_count": _i(_flat_pick(src, "src_gcs_expanding_count", default=0), 0),
        "contracting_all": bool(_flat_pick(src, "src_gcs_contracting_all", default=0)),
        "transfer_depth": _f(_flat_pick(src, "src_gcs_transfer_depth", "src_msbc_transfer_depth"), None),
        "transfer_dir": _flat_pick(src, "src_gcs_transfer_dir", "src_msbc_transfer_dir"),
        "transfer_state": _flat_pick(src, "src_gcs_transfer_state", "src_msbc_transfer_state"),
        "fast_pos_mean": _f(_flat_pick(src, "src_gcs_fast_pos_mean"), None),
        "slow_pos_mean": _f(_flat_pick(src, "src_gcs_slow_pos_mean"), None),
        "front_back_disagreement": _f(_flat_pick(src, "src_gcs_front_back_disagreement"), None),
        "reclaim_front": _f(_flat_pick(src, "src_gcs_reclaim_front"), None),
        "reclaim_slow": _f(_flat_pick(src, "src_gcs_reclaim_slow"), None),
        # --- CSD / DCSD ---
        "shrink_s8": _f(_flat_pick(src, "src_csd_shrink_s8", "src_gbc_shrink_s8", "src_gbc_s8_shrink"), None),
        "shrink_s23": _f(_flat_pick(src, "src_csd_shrink_s23", "src_gbc_shrink_s23", "src_gbc_s23_shrink"), None),
        "shrink_s38": _f(_flat_pick(src, "src_csd_shrink_s38", "src_gbc_shrink_s38", "src_gbc_s38_shrink"), None),
        "release_score": _f(_flat_pick(src, "src_dcsd_release_score"), None),
        "reversal_stage": _f(_flat_pick(src, "src_dcsd_reversal_stage"), None),
        "same_sign_exhaustion": _f(_flat_pick(src, "src_dcsd_same_sign_exhaustion"), None),
        "csd_spread_now": _f(_flat_pick(src, "src_csd_spread_now"), None),
        "csd_spread_delta": _f(_flat_pick(src, "src_csd_spread_delta"), None),
        # --- MSBC aggregate structure ---
        "msbc_transfer_dir": _flat_pick(src, "src_msbc_transfer_dir"),
        "msbc_transfer_depth": _f(_flat_pick(src, "src_msbc_transfer_depth"), None),
        "msbc_transfer_state": _flat_pick(src, "src_msbc_transfer_state"),
        "msbc_continuity_mean": _f(_flat_pick(src, "src_msbc_continuity_mean"), None),
        "msbc_propagation_agree_ratio": _f(_flat_pick(src, "src_msbc_propagation_agree_ratio"), None),
        "msbc_propagation_disagree_ratio": _f(_flat_pick(src, "src_msbc_propagation_disagree_ratio"), None),
        "msbc_pair_rows": _i(_flat_pick(src, "src_msbc_pair_rows", default=0), 0),
        "msbc_slope_mono": _f(_flat_pick(src, "src_msbc_slope_mono"), None),
        "msbc_tan_mono": _f(_flat_pick(src, "src_msbc_tan_mono"), None),
        "msbc_curve_decay": _f(_flat_pick(src, "src_msbc_curve_decay"), None),
        "msbc_linm_mono": _f(_flat_pick(src, "src_msbc_linm_mono"), None),
        "msbc_override_count": _i(_flat_pick(src, "src_msbc_override_count", default=0), 0),
        "msbc_fast_override_count": _i(_flat_pick(src, "src_msbc_fast_override_count", default=0), 0),
        "msbc_slow_override_count": _i(_flat_pick(src, "src_msbc_slow_override_count", default=0), 0),
        "msbc_disorder_count": _i(_flat_pick(src, "src_msbc_disorder_count", default=0), 0),
        "msbc_age_seconds": _f(_flat_pick(src, "src_msbc_age_seconds"), None),
        "msbc_age_label": _flat_pick(src, "src_msbc_age_label"),
        # --- GBC front diagnostics ---
        "hook_s8": _flat_pick(src, "src_gbc_hook_s8", "src_gbc_s8_hook"),
        "hook_s23": _flat_pick(src, "src_gbc_hook_s23", "src_gbc_s23_hook"),
        "hook_s38": _flat_pick(src, "src_gbc_hook_s38", "src_gbc_s38_hook"),
        "flat_s8": _f(_flat_pick(src, "src_gbc_flat_s8"), None),
        "flat_s23": _f(_flat_pick(src, "src_gbc_flat_s23"), None),
        "flat_s38": _f(_flat_pick(src, "src_gbc_flat_s38"), None),
        "turn_age_s8": _f(_flat_pick(src, "src_gbc_turn_age_s8"), None),
        "turn_age_s23": _f(_flat_pick(src, "src_gbc_turn_age_s23"), None),
        "turn_age_s38": _f(_flat_pick(src, "src_gbc_turn_age_s38"), None),
        "hook_age_s8": _f(_flat_pick(src, "src_gbc_hook_age_s8"), None),
        "hook_age_s23": _f(_flat_pick(src, "src_gbc_hook_age_s23"), None),
        "hook_age_s38": _f(_flat_pick(src, "src_gbc_hook_age_s38"), None),
        "sign_persist_s8": _f(_flat_pick(src, "src_gbc_sign_persist_s8"), None),
        "sign_persist_s23": _f(_flat_pick(src, "src_gbc_sign_persist_s23"), None),
        "sign_persist_s38": _f(_flat_pick(src, "src_gbc_sign_persist_s38"), None),
        "hook_persist_s8": _f(_flat_pick(src, "src_gbc_hook_persist_s8"), None),
        "hook_persist_s23": _f(_flat_pick(src, "src_gbc_hook_persist_s23"), None),
        "hook_persist_s38": _f(_flat_pick(src, "src_gbc_hook_persist_s38"), None),
        "norm_ctx_s8": _f(_flat_pick(src, "src_gbc_norm_ctx_s8"), None),
        "norm_ctx_s23": _f(_flat_pick(src, "src_gbc_norm_ctx_s23"), None),
        "norm_ctx_s38": _f(_flat_pick(src, "src_gbc_norm_ctx_s38"), None),
        "gbc_active_sigma_count": _i(_flat_pick(src, "src_gbc_active_sigma_count", default=0), 0),
        "gbc_series_available": bool(_flat_pick(src, "src_gbc_series_available", default=0)),
        # --- HYST ---
        "hyst_spread_state": _flat_pick(src, "src_hyst_spread_state"),
        "hyst_spread_risk": _flat_pick(src, "src_hyst_spread_risk"),
        "hyst_pressure_state": _flat_pick(src, "src_hyst_pressure_state"),
        "hyst_stack_stability": _f(_flat_pick(src, "src_hyst_stack_stability"), None),
        "hyst_stack_stability_state": _flat_pick(src, "src_hyst_stack_stability_state"),
        "hyst_near_cross_state": _flat_pick(src, "src_hyst_near_cross_state"),
        "hyst_fan_tightness": _f(_flat_pick(src, "src_hyst_fan_tightness"), None),
        "hyst_stack_alignment": _f(_flat_pick(src, "src_hyst_stack_alignment"), None),
        "hyst_probe_flip_watch": _flat_pick(src, "src_hyst_probe_flip_watch"),
        "hyst_probe_fast_collapse": _flat_pick(src, "src_hyst_probe_fast_collapse"),
        "hyst_probe_recovery": _flat_pick(src, "src_hyst_probe_recovery"),
        "hyst_cross_rate": _f(_flat_pick(src, "src_hyst_cross_rate"), None),
        "hyst_order_stability": _f(_flat_pick(src, "src_hyst_order_stability"), None),
        "hyst_ladder_compression": _f(_flat_pick(src, "src_hyst_ladder_compression"), None),
        "hyst_summary_stack": _flat_pick(src, "src_hyst_summary_stack"),
    }

    # Derived front/structure summaries used by the final runner phase.
    context["front_flat_mean"] = _mean(context.get("flat_s8"), context.get("flat_s23"), context.get("flat_s38"))
    context["front_turn_age_min"] = _front_min_positive(
        context.get("turn_age_s8"),
        context.get("turn_age_s23"),
        context.get("turn_age_s38"),
    )
    context["front_hook_persist_max"] = max(
        _f(context.get("hook_persist_s8"), 0.0) or 0.0,
        _f(context.get("hook_persist_s23"), 0.0) or 0.0,
        _f(context.get("hook_persist_s38"), 0.0) or 0.0,
    )
    context["front_sign_persist_max"] = max(
        _f(context.get("sign_persist_s8"), 0.0) or 0.0,
        _f(context.get("sign_persist_s23"), 0.0) or 0.0,
        _f(context.get("sign_persist_s38"), 0.0) or 0.0,
    )
    context["front_norm_ctx_mean"] = _mean(context.get("norm_ctx_s8"), context.get("norm_ctx_s23"), context.get("norm_ctx_s38"))
    context["front_hook_active"] = bool(context.get("hook_s8") or context.get("hook_s23") or context.get("hook_s38"))

    missing = [key for key, value in context.items() if value is None]
    return {
        "context": context,
        "missing": missing,
        "used_aliases": list(src.get("src_debug_alias_applied") or []),
    }


def map_src_row_to_numeric_vector(src_row: dict, config: dict | None = None) -> dict:
    mapped = map_src_row_to_v21_context(src_row, config=config)
    ctx = mapped["context"]

    vector = {
        "px_mid_8": _f(ctx.get("px_mid_8"), 0.0),
        "px_mid_23": _f(ctx.get("px_mid_23"), 0.0),
        "px_mid_38": _f(ctx.get("px_mid_38"), 0.0),
        "px_mid_53": _f(ctx.get("px_mid_53"), 0.0),
        "px_mid_68": _f(ctx.get("px_mid_68"), 0.0),
        "px_mid_83": _f(ctx.get("px_mid_83"), 0.0),
        "zpos_83": _f(ctx.get("zpos_83"), 0.0),
        "contracting_count": float(ctx.get("contracting_count") or 0.0),
        "expanding_count": float(ctx.get("expanding_count") or 0.0),
        "transfer_depth": _f(ctx.get("transfer_depth"), 0.0),
        "front_back_disagreement": _f(ctx.get("front_back_disagreement"), 0.0),
        "reclaim_front": _f(ctx.get("reclaim_front"), 0.0),
        "reclaim_slow": _f(ctx.get("reclaim_slow"), 0.0),
        "shrink_s8": _f(ctx.get("shrink_s8"), 0.0),
        "shrink_s23": _f(ctx.get("shrink_s23"), 0.0),
        "shrink_s38": _f(ctx.get("shrink_s38"), 0.0),
        "release_score": _f(ctx.get("release_score"), 0.0),
        "reversal_stage": _f(ctx.get("reversal_stage"), 0.0),
        "same_sign_exhaustion": _f(ctx.get("same_sign_exhaustion"), 0.0),
        "csd_spread_now": _f(ctx.get("csd_spread_now"), 0.0),
        "csd_spread_delta": _f(ctx.get("csd_spread_delta"), 0.0),
        "msbc_transfer_depth": _f(ctx.get("msbc_transfer_depth"), 0.0),
        "msbc_continuity_mean": _f(ctx.get("msbc_continuity_mean"), 0.0),
        "msbc_propagation_agree_ratio": _f(ctx.get("msbc_propagation_agree_ratio"), 0.0),
        "msbc_propagation_disagree_ratio": _f(ctx.get("msbc_propagation_disagree_ratio"), 0.0),
        "msbc_override_count": float(ctx.get("msbc_override_count") or 0.0),
        "msbc_fast_override_count": float(ctx.get("msbc_fast_override_count") or 0.0),
        "msbc_slow_override_count": float(ctx.get("msbc_slow_override_count") or 0.0),
        "msbc_disorder_count": float(ctx.get("msbc_disorder_count") or 0.0),
        "msbc_slope_mono": _f(ctx.get("msbc_slope_mono"), 0.0),
        "msbc_tan_mono": _f(ctx.get("msbc_tan_mono"), 0.0),
        "msbc_curve_decay": _f(ctx.get("msbc_curve_decay"), 0.0),
        "msbc_linm_mono": _f(ctx.get("msbc_linm_mono"), 0.0),
        "gbc_active_sigma_count": float(ctx.get("gbc_active_sigma_count") or 0.0),
        "gbc_series_available": 1.0 if ctx.get("gbc_series_available") else 0.0,
        "hook_s8": 1.0 if ctx.get("hook_s8") else 0.0,
        "hook_s23": 1.0 if ctx.get("hook_s23") else 0.0,
        "hook_s38": 1.0 if ctx.get("hook_s38") else 0.0,
        "flat_s8": _f(ctx.get("flat_s8"), 0.0),
        "flat_s23": _f(ctx.get("flat_s23"), 0.0),
        "flat_s38": _f(ctx.get("flat_s38"), 0.0),
        "turn_age_s8": _f(ctx.get("turn_age_s8"), 0.0),
        "turn_age_s23": _f(ctx.get("turn_age_s23"), 0.0),
        "turn_age_s38": _f(ctx.get("turn_age_s38"), 0.0),
        "hook_persist_s8": _f(ctx.get("hook_persist_s8"), 0.0),
        "hook_persist_s23": _f(ctx.get("hook_persist_s23"), 0.0),
        "hook_persist_s38": _f(ctx.get("hook_persist_s38"), 0.0),
        "front_flat_mean": _f(ctx.get("front_flat_mean"), 0.0),
        "front_turn_age_min": _f(ctx.get("front_turn_age_min"), 0.0),
        "front_hook_persist_max": _f(ctx.get("front_hook_persist_max"), 0.0),
        "front_sign_persist_max": _f(ctx.get("front_sign_persist_max"), 0.0),
        "front_norm_ctx_mean": _f(ctx.get("front_norm_ctx_mean"), 0.0),
        "front_hook_active": 1.0 if ctx.get("front_hook_active") else 0.0,
        "hyst_stack_stability": _f(ctx.get("hyst_stack_stability"), 0.0),
        "hyst_fan_tightness": _f(ctx.get("hyst_fan_tightness"), 0.0),
        "hyst_stack_alignment": _f(ctx.get("hyst_stack_alignment"), 0.0),
        "hyst_cross_rate": _f(ctx.get("hyst_cross_rate"), 0.0),
        "hyst_order_stability": _f(ctx.get("hyst_order_stability"), 0.0),
        "hyst_ladder_compression": _f(ctx.get("hyst_ladder_compression"), 0.0),
        "hyst_probe_flip_watch": 1.0 if ctx.get("hyst_probe_flip_watch") else 0.0,
        "hyst_probe_fast_collapse": 1.0 if ctx.get("hyst_probe_fast_collapse") else 0.0,
        "hyst_probe_recovery": 1.0 if ctx.get("hyst_probe_recovery") else 0.0,
        "transfer_dir_code": _encode_state(
            ctx.get("transfer_dir"),
            {"bull": 1.0, "up": 1.0, "bear": -1.0, "down": -1.0, "none": 0.0},
        ),
        "transfer_state_code": _encode_state(
            ctx.get("transfer_state"),
            {"front_to_back": 1.0, "back_to_front": -1.0, "stable": 0.0, "none": 0.0, "shallow": 0.25, "partial": 0.5, "deep": 0.75, "full": 1.0},
        ),
        "msbc_transfer_dir_code": _encode_state(
            ctx.get("msbc_transfer_dir"),
            {"bull": 1.0, "up": 1.0, "bear": -1.0, "down": -1.0, "none": 0.0},
        ),
        "msbc_transfer_state_code": _encode_state(
            ctx.get("msbc_transfer_state"),
            {"none": 0.0, "shallow": 0.25, "partial": 0.5, "deep": 0.75, "full": 1.0},
        ),
        "hyst_spread_state_code": _encode_state(
            ctx.get("hyst_spread_state"),
            {"widening": 1.0, "narrowing": -1.0, "frozen": 0.0},
        ),
        "hyst_spread_risk_code": _encode_state(
            ctx.get("hyst_spread_risk"),
            {"continuation_friendly": 0.0, "reversal_watch": 0.5, "whipsaw_risk": 0.75, "collapse_watch": 1.0},
        ),
        "hyst_pressure_state_code": _encode_state(
            ctx.get("hyst_pressure_state"),
            {"calm": 0.0, "hovering": 0.25, "pressuring": 0.5, "approaching": 1.0},
        ),
        "hyst_near_cross_state_code": _encode_state(
            ctx.get("hyst_near_cross_state"),
            {"calm": 0.0, "hovering": 0.25, "pressuring": 0.5, "approaching": 1.0},
        ),
    }
    return {
        "vector": vector,
        "missing": mapped["missing"],
        "used_aliases": mapped["used_aliases"],
        "context": ctx,
    }


def build_v21_feature_bundle(td_features: dict, config: dict | None = None) -> dict:
    td_features = _as_dict(td_features)
    gauss = _as_dict(td_features.get("gauss"))
    src_row = _as_dict(td_features.get("src_row") or td_features.get("live_src_row"))
    slopes = _as_dict(gauss.get("slopes"))
    curvature = _as_dict(gauss.get("curvature"))

    mapped = map_src_row_to_numeric_vector(src_row, config=config)
    src_norm = normalize_src_row(src_row)
    vector = {
        "slope_s8": _f(slopes.get("s8"), 0.0),
        "slope_s23": _f(slopes.get("s23"), 0.0),
        "slope_s38": _f(slopes.get("s38"), 0.0),
        "slope_s53": _f(slopes.get("s53"), 0.0),
        "slope_s68": _f(slopes.get("s68"), 0.0),
        "slope_s83": _f(slopes.get("s83"), 0.0),
        "curvature_s23": _f(curvature.get("s23"), _f(src_norm.get("src_msbc_l2_quad_curv_s23"), 0.0)),
        "curvature_s53": _f(curvature.get("s53"), _f(src_norm.get("src_msbc_l2_quad_curv_s53"), 0.0)),
    }
    vector.update(mapped["vector"])

    return {
        "vector": vector,
        "context": mapped["context"],
        "missing": mapped["missing"],
        "used_aliases": mapped["used_aliases"],
    }


def build_v21_feature_vector(td_features: dict) -> dict:
    return build_v21_feature_bundle(td_features).get("vector", {})


def mapped_context_is_active(mapped: Mapping[str, Any]) -> bool:
    slope_keys = {
        "slope_s8",
        "slope_s23",
        "slope_s38",
        "slope_s53",
        "slope_s68",
        "slope_s83",
        "curvature_s23",
        "curvature_s53",
    }
    for key, value in dict(mapped or {}).items():
        if key in slope_keys:
            continue
        try:
            if float(value) != 0.0:
                return True
        except Exception:
            if value not in (None, False, "", [], {}):
                return True
    return False
