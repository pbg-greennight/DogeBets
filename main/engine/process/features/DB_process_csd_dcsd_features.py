from __future__ import annotations

from typing import Any, Dict, Optional
from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL, SIGMAS_FAST, SIGMAS_SLOW


def _count_series_sign_flips(series) -> Optional[int]:
    if not isinstance(series, (list, tuple)) or len(series) < 2:
        return None
    cleaned = []
    for v in series:
        try:
            fv = float(v)
        except Exception:
            continue
        if fv > 0:
            cleaned.append(1)
        elif fv < 0:
            cleaned.append(-1)
        else:
            cleaned.append(0)
    if len(cleaned) < 2:
        return None
    flips = 0
    prev = cleaned[0]
    for cur in cleaned[1:]:
        if cur != 0 and prev != 0 and cur != prev:
            flips += 1
        if cur != 0:
            prev = cur
    return flips


def _extract_leg_sigma_metrics(leg_payload: dict, sigma: int) -> dict:
    """TODO: adapt to actual pv_tail schema."""
    d_mid_series = None
    d_width_series = None
    return {
        "mid_mean": None,
        "width_mean": None,
        "dmid_mean": None,
        "dwidth_mean": None,
        "dmid_last": None,
        "dwidth_last": None,
        "dmid_flip_count": _count_series_sign_flips(d_mid_series),
        "dwidth_flip_count": _count_series_sign_flips(d_width_series),
    }


def _summarize_group(per_sigma: dict, sigmas: list[int], prefix: str) -> dict:
    def vals(key):
        return [per_sigma.get(s, {}).get(key) for s in sigmas]

    def clean_mean(xs):
        ys = []
        for x in xs:
            try:
                ys.append(float(x))
            except Exception:
                pass
        return sum(ys) / len(ys) if ys else None

    return {
        f"{prefix}_mid_mean": clean_mean(vals("mid_mean")),
        f"{prefix}_width_mean": clean_mean(vals("width_mean")),
        f"{prefix}_dmid_mean": clean_mean(vals("dmid_mean")),
        f"{prefix}_dwidth_mean": clean_mean(vals("dwidth_mean")),
        f"{prefix}_dmid_flip_density": clean_mean(vals("dmid_flip_count")),
        f"{prefix}_dwidth_flip_density": clean_mean(vals("dwidth_flip_count")),
    }


def build_csd_dcsd_feature_payload(
    pv_tail: dict,
    config: Optional[dict] = None,
) -> dict:
    leg1_raw = pv_tail.get("per_sigma_leg1") or {}
    leg2_raw = pv_tail.get("per_sigma_leg2") or {}

    leg1: Dict[int, Dict[str, Any]] = {}
    leg2: Dict[int, Dict[str, Any]] = {}

    for sigma in SIGMAS_ALL:
        leg1[sigma] = _extract_leg_sigma_metrics(leg1_raw, sigma)
        leg2[sigma] = _extract_leg_sigma_metrics(leg2_raw, sigma)

    return {
        "leg1": leg1,
        "leg2": leg2,
        "summary": {
            "leg2_fast": _summarize_group(leg2, SIGMAS_FAST, "l2_fast"),
            "leg2_slow": _summarize_group(leg2, SIGMAS_SLOW, "l2_slow"),
        },
    }


def flatten_csd_dcsd_to_src(
    csd_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}

    for leg_name in ["leg1", "leg2"]:
        leg = csd_obj.get(leg_name, {})
        leg_prefix = "l1" if leg_name == "leg1" else "l2"
        for sigma, fields in leg.items():
            out[f"src_csd_{leg_prefix}_mid_mean_s{sigma}"] = fields.get("mid_mean")
            out[f"src_csd_{leg_prefix}_width_mean_s{sigma}"] = fields.get("width_mean")
            out[f"src_dcsd_{leg_prefix}_dmid_mean_s{sigma}"] = fields.get("dmid_mean")
            out[f"src_dcsd_{leg_prefix}_dwidth_mean_s{sigma}"] = fields.get("dwidth_mean")
            out[f"src_dcsd_{leg_prefix}_dmid_last_s{sigma}"] = fields.get("dmid_last")
            out[f"src_dcsd_{leg_prefix}_dwidth_last_s{sigma}"] = fields.get("dwidth_last")
            out[f"src_dcsd_{leg_prefix}_dmid_flip_count_s{sigma}"] = fields.get("dmid_flip_count")
            out[f"src_dcsd_{leg_prefix}_dwidth_flip_count_s{sigma}"] = fields.get("dwidth_flip_count")

    summary = csd_obj.get("summary", {})
    for section_name, section in summary.items():
        for k, v in section.items():
            out[f"src_{k}"] = v

    return out
