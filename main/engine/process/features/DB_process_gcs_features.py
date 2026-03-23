from __future__ import annotations

from typing import Any, Dict, Optional

from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL


def _extract_sigma_snapshot(snapshot: dict, sigma: int) -> dict:
    """TODO: adapt to actual channel snapshot schema."""
    return {
        "regime": None,
        "pos": None,
        "width": None,
        "mid_slope": None,
        "width_change": None,
    }


def _compute_regime_counts(per_sigma: dict) -> dict:
    regimes = [str(per_sigma.get(s, {}).get("regime")).lower() for s in SIGMAS_ALL]
    contract_count = sum(1 for r in regimes if "contract" in r)
    expand_count = sum(1 for r in regimes if "expand" in r)
    flat_count = sum(1 for r in regimes if r in {"flat", "stable"})
    return {
        "regime_contract_count": contract_count,
        "regime_expand_count": expand_count,
        "regime_flat_count": flat_count,
        "regime_contract_all": int(contract_count == len(SIGMAS_ALL)),
    }


def _compute_spacing_summary(per_sigma: dict) -> dict:
    """TODO: lift spacing/fan logic from DB_process_printing_gcs.py."""
    return {
        "spacing_state": None,
        "fan_state": None,
    }


def _compute_transfer_summary(per_sigma: dict, snapshot: dict) -> dict:
    """TODO: lift transfer logic from DB_process_printing_gcs.py."""
    return {
        "transfer_dir": None,
        "transfer_depth": None,
        "transfer_state": None,
    }


def _compute_position_summary(per_sigma: dict) -> dict:
    fast = [per_sigma.get(s, {}).get("pos") for s in [8, 23]]
    slow = [per_sigma.get(s, {}).get("pos") for s in [68, 83]]

    def _mean_clean(vals):
        xs = [float(v) for v in vals if v is not None]
        return sum(xs) / len(xs) if xs else None

    fast_mean = _mean_clean(fast)
    slow_mean = _mean_clean(slow)

    gap = None
    if fast_mean is not None and slow_mean is not None:
        gap = fast_mean - slow_mean

    return {
        "fast_pos_mean": fast_mean,
        "slow_pos_mean": slow_mean,
        "fast_slow_pos_gap": gap,
    }


def build_gcs_feature_payload(
    channel_snapshot: dict,
    per_sigma_full: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    per_sigma: Dict[int, Dict[str, Any]] = {}
    for sigma in SIGMAS_ALL:
        per_sigma[sigma] = _extract_sigma_snapshot(channel_snapshot, sigma)

    regime_counts = _compute_regime_counts(per_sigma)
    spacing = _compute_spacing_summary(per_sigma)
    transfer = _compute_transfer_summary(per_sigma, channel_snapshot)
    position = _compute_position_summary(per_sigma)

    return {
        "per_sigma": per_sigma,
        "regime_counts": regime_counts,
        "spacing": spacing,
        "transfer": transfer,
        "position": position,
    }


def flatten_gcs_to_src(
    gcs_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}

    per_sigma = gcs_obj.get("per_sigma", {})
    for sigma, fields in per_sigma.items():
        out[f"src_gcs_regime_s{sigma}"] = fields.get("regime")
        out[f"src_gcs_pos_s{sigma}"] = fields.get("pos")
        out[f"src_gcs_width_s{sigma}"] = fields.get("width")
        out[f"src_gcs_mid_slope_s{sigma}"] = fields.get("mid_slope")
        out[f"src_gcs_width_change_s{sigma}"] = fields.get("width_change")

    rc = gcs_obj.get("regime_counts", {})
    out["src_gcs_regime_contract_count"] = rc.get("regime_contract_count")
    out["src_gcs_regime_expand_count"] = rc.get("regime_expand_count")
    out["src_gcs_regime_flat_count"] = rc.get("regime_flat_count")
    out["src_gcs_regime_contract_all"] = rc.get("regime_contract_all")

    spacing = gcs_obj.get("spacing", {})
    out["src_gcs_spacing_state"] = spacing.get("spacing_state")
    out["src_gcs_fan_state"] = spacing.get("fan_state")

    transfer = gcs_obj.get("transfer", {})
    out["src_gcs_transfer_dir"] = transfer.get("transfer_dir")
    out["src_gcs_transfer_depth"] = transfer.get("transfer_depth")
    out["src_gcs_transfer_state"] = transfer.get("transfer_state")

    position = gcs_obj.get("position", {})
    out["src_gcs_fast_pos_mean"] = position.get("fast_pos_mean")
    out["src_gcs_slow_pos_mean"] = position.get("slow_pos_mean")
    out["src_gcs_fast_slow_pos_gap"] = position.get("fast_slow_pos_gap")

    return out
