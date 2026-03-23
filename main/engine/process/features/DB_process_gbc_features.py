from __future__ import annotations

from typing import Any, Dict, Optional
from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL


def _extract_diag_for_sigma(bell: dict, sigma: int) -> dict:
    """Extract per-sigma bell diagnostics from bell['diagnostics']['per_sigma']."""
    diag = (((bell or {}).get("diagnostics") or {}).get("per_sigma") or {}).get(sigma, {})
    return {
        "shrink": diag.get("shrink"),
        "flat": diag.get("flat"),
        "hook": diag.get("hook"),
        "prev_abs": diag.get("prev_abs"),
        "last_abs": diag.get("last_abs"),
        "sign_from": diag.get("sign_from"),
        "sign_to": diag.get("sign_to"),
        "eps": diag.get("eps"),
        "turn_age": diag.get("turn_age"),
        "hook_age": diag.get("hook_age"),
        "sign_persist": diag.get("sign_persist"),
        "hook_persist": diag.get("hook_persist"),
        "norm_ctx": diag.get("norm_ctx"),
    }


def build_gbc_feature_payload(
    bell: dict,
    bell_curve_series: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    per_sigma: Dict[int, Dict[str, Any]] = {}
    for sigma in SIGMAS_ALL:
        per_sigma[sigma] = _extract_diag_for_sigma(bell, sigma)

    return {
        "per_sigma": per_sigma,
        "meta": {
            "series_available": bell_curve_series is not None,
        },
    }


def flatten_gbc_to_src(
    gbc_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}
    per_sigma = gbc_obj.get("per_sigma", {})
    for sigma, fields in per_sigma.items():
        out[f"src_gbc_shrink_s{sigma}"] = fields.get("shrink")
        out[f"src_gbc_flat_s{sigma}"] = fields.get("flat")
        out[f"src_gbc_hook_s{sigma}"] = fields.get("hook")
        out[f"src_gbc_prev_abs_s{sigma}"] = fields.get("prev_abs")
        out[f"src_gbc_last_abs_s{sigma}"] = fields.get("last_abs")
        out[f"src_gbc_sign_from_s{sigma}"] = fields.get("sign_from")
        out[f"src_gbc_sign_to_s{sigma}"] = fields.get("sign_to")
        out[f"src_gbc_eps_s{sigma}"] = fields.get("eps")
        out[f"src_gbc_turn_age_s{sigma}"] = fields.get("turn_age")
        out[f"src_gbc_hook_age_s{sigma}"] = fields.get("hook_age")
        out[f"src_gbc_sign_persist_s{sigma}"] = fields.get("sign_persist")
        out[f"src_gbc_hook_persist_s{sigma}"] = fields.get("hook_persist")
        out[f"src_gbc_norm_ctx_s{sigma}"] = fields.get("norm_ctx")
    return out
