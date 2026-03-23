from __future__ import annotations

from typing import Any, Dict, Optional

from main.engine.process.features.DB_process_hysteresis import compute_hysteresis_features, flatten_hysteresis_to_src
from main.engine.process.features.DB_process_msbc_features import build_msbc_feature_payload, flatten_msbc_to_src
from main.engine.process.features.DB_process_gcs_features import build_gcs_feature_payload, flatten_gcs_to_src
from main.engine.process.features.DB_process_csd_dcsd_features import build_csd_dcsd_feature_payload, flatten_csd_dcsd_to_src
from main.engine.process.features.DB_process_gbc_features import build_gbc_feature_payload, flatten_gbc_to_src


def build_v21_source_payloads(
    timing: dict,
    per_sigma_full: dict,
    per_sigma_hist: Optional[dict],
    catalog: Any,
    config: Optional[dict] = None,
) -> dict:
    calc_out = catalog.ensure_calc(per_sigma_full, timing=timing)

    bell = (calc_out or {}).get("bell") or {}
    bell_curve_series = (calc_out or {}).get("bell_curve_series") or {}
    channels = (calc_out or {}).get("channels") or {}
    channel_snapshot = channels.get("snapshot") or {}
    pv_tail = channels.get("pv_tail") or {}

    hyst_obj = compute_hysteresis_features(per_sigma_full=per_sigma_full, timing=timing, config=config)
    msbc_obj = build_msbc_feature_payload(bell=bell, bell_curve_series=bell_curve_series, config=config)
    gcs_obj = build_gcs_feature_payload(channel_snapshot=channel_snapshot, per_sigma_full=per_sigma_full, config=config)
    csd_obj = build_csd_dcsd_feature_payload(pv_tail=pv_tail, config=config)
    gbc_obj = build_gbc_feature_payload(bell=bell, bell_curve_series=bell_curve_series, config=config)

    src_row = {}
    src_row.update(flatten_hysteresis_to_src(hyst_obj, config=config))
    src_row.update(flatten_msbc_to_src(msbc_obj, config=config))
    src_row.update(flatten_gcs_to_src(gcs_obj, config=config))
    src_row.update(flatten_csd_dcsd_to_src(csd_obj, config=config))
    src_row.update(flatten_gbc_to_src(gbc_obj, config=config))

    return {
        "hyst_obj": hyst_obj,
        "msbc_obj": msbc_obj,
        "gcs_obj": gcs_obj,
        "csd_obj": csd_obj,
        "gbc_obj": gbc_obj,
        "src_row": src_row,
    }
