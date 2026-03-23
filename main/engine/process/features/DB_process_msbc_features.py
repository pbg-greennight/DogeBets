from __future__ import annotations

from typing import Any, Dict, Optional

from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL, SIGMAS_FAST, SIGMAS_SLOW


def _extract_sigma_leg_metrics(bell: dict, sigma: int) -> dict:
    """TODO: adapt to actual bell payload structure from build_bell_out()."""
    return {
        "l1_slope": None,
        "l2_slope": None,
        "l1_curve": None,
        "l2_curve": None,
        "l1_lin_r2": None,
        "l2_lin_r2": None,
        "l1_quad_tangent": None,
        "l2_quad_tangent": None,
        "l1_quad_curv": None,
        "l2_quad_curv": None,
        "l1_z": None,
        "l2_z": None,
        "l1_run_score": None,
        "l2_run_score": None,
        "l2_tag": None,
        "override": None,
    }


def _compute_transfer_summary(bell: dict) -> dict:
    """TODO: lift transfer logic out of DB_process_printing_msbc.py."""
    return {
        "transfer_dir": None,
        "transfer_depth": None,
        "transfer_state": None,
    }


def _compute_propagation_summary(bell: dict) -> dict:
    """TODO: lift propagation/stack summary logic out of DB_process_printing_msbc.py."""
    return {
        "propagation_agree_ratio": None,
        "propagation_disagree_ratio": None,
        "pair_rows": None,
        "slope_mono": None,
        "tan_mono": None,
        "curve_decay": None,
        "linm_mono": None,
    }


def _compute_consistency_summary(per_sigma: dict) -> dict:
    overrides = {s: per_sigma.get(s, {}).get("override") for s in SIGMAS_ALL}
    return {
        "override_count": sum(1 for v in overrides.values() if bool(v)),
        "fast_override_count": sum(1 for s in SIGMAS_FAST if bool(overrides.get(s))),
        "slow_override_count": sum(1 for s in SIGMAS_SLOW if bool(overrides.get(s))),
        "overrides": overrides,
    }


def _compute_age_summary(bell_curve_series: Optional[dict]) -> dict:
    """TODO: compute or pass through age from bell_curve_series."""
    return {
        "age_seconds": None,
        "age_label": None,
    }


def build_msbc_feature_payload(
    bell: dict,
    bell_curve_series: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    per_sigma: Dict[int, Dict[str, Any]] = {}
    for sigma in SIGMAS_ALL:
        per_sigma[sigma] = _extract_sigma_leg_metrics(bell, sigma)

    transfer = _compute_transfer_summary(bell)
    propagation = _compute_propagation_summary(bell)
    consistency = _compute_consistency_summary(per_sigma)
    age = _compute_age_summary(bell_curve_series)

    payload = {
        "per_sigma": per_sigma,
        "transfer": transfer,
        "propagation": propagation,
        "consistency": consistency,
        "age": age,
        "meta": {
            "disorder_count": None,
        },
    }
    return payload


def flatten_msbc_to_src(
    msbc_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}

    per_sigma = msbc_obj.get("per_sigma", {})
    for sigma, fields in per_sigma.items():
        out[f"src_msbc_l1_slope_s{sigma}"] = fields.get("l1_slope")
        out[f"src_msbc_l2_slope_s{sigma}"] = fields.get("l2_slope")
        out[f"src_msbc_l1_curve_s{sigma}"] = fields.get("l1_curve")
        out[f"src_msbc_l2_curve_s{sigma}"] = fields.get("l2_curve")
        out[f"src_msbc_l1_lin_r2_s{sigma}"] = fields.get("l1_lin_r2")
        out[f"src_msbc_l2_lin_r2_s{sigma}"] = fields.get("l2_lin_r2")
        out[f"src_msbc_l1_quad_tangent_s{sigma}"] = fields.get("l1_quad_tangent")
        out[f"src_msbc_l2_quad_tangent_s{sigma}"] = fields.get("l2_quad_tangent")
        out[f"src_msbc_l1_quad_curv_s{sigma}"] = fields.get("l1_quad_curv")
        out[f"src_msbc_l2_quad_curv_s{sigma}"] = fields.get("l2_quad_curv")
        out[f"src_msbc_l1_z_s{sigma}"] = fields.get("l1_z")
        out[f"src_msbc_l2_z_s{sigma}"] = fields.get("l2_z")
        out[f"src_msbc_l1_run_score_s{sigma}"] = fields.get("l1_run_score")
        out[f"src_msbc_l2_run_score_s{sigma}"] = fields.get("l2_run_score")
        out[f"src_msbc_l2_tag_s{sigma}"] = fields.get("l2_tag")
        out[f"src_msbc_override_s{sigma}"] = fields.get("override")

    transfer = msbc_obj.get("transfer", {})
    out["src_msbc_transfer_dir"] = transfer.get("transfer_dir")
    out["src_msbc_transfer_depth"] = transfer.get("transfer_depth")
    out["src_msbc_transfer_state"] = transfer.get("transfer_state")

    propagation = msbc_obj.get("propagation", {})
    out["src_msbc_propagation_agree_ratio"] = propagation.get("propagation_agree_ratio")
    out["src_msbc_propagation_disagree_ratio"] = propagation.get("propagation_disagree_ratio")
    out["src_msbc_pair_rows"] = propagation.get("pair_rows")
    out["src_msbc_slope_mono"] = propagation.get("slope_mono")
    out["src_msbc_tan_mono"] = propagation.get("tan_mono")
    out["src_msbc_curve_decay"] = propagation.get("curve_decay")
    out["src_msbc_linm_mono"] = propagation.get("linm_mono")

    consistency = msbc_obj.get("consistency", {})
    out["src_msbc_override_count"] = consistency.get("override_count")
    out["src_msbc_fast_override_count"] = consistency.get("fast_override_count")
    out["src_msbc_slow_override_count"] = consistency.get("slow_override_count")

    meta = msbc_obj.get("meta", {})
    out["src_msbc_disorder_count"] = meta.get("disorder_count")

    age = msbc_obj.get("age", {})
    out["src_msbc_age_seconds"] = age.get("age_seconds")
    out["src_msbc_age_label"] = age.get("age_label")

    return out
