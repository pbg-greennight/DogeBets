from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from main.engine.process.features.DB_process_hysteresis import compute_hysteresis_features, flatten_hysteresis_to_src
from main.engine.process.features.DB_process_msbc_features import build_msbc_feature_payload, flatten_msbc_to_src
from main.engine.process.features.DB_process_gcs_features import build_gcs_feature_payload, flatten_gcs_to_src
from main.engine.process.features.DB_process_csd_dcsd_features import build_csd_dcsd_feature_payload, flatten_csd_dcsd_to_src
from main.engine.process.features.DB_process_gbc_features import build_gbc_feature_payload, flatten_gbc_to_src


def build_method_v21_src_row(
    timing: dict,
    per_sigma_full: dict,
    per_sigma_hist: Optional[dict],
    catalog: Any,
    config: Optional[dict] = None,
) -> dict:
    """
    Build one canonical source row for trend_method_v2_1 from engine truth objects.
    """
    row: Dict[str, Any] = {}

    row["src_meta_epoch"] = timing.get("epoch")
    row["src_meta_decision_ts"] = timing.get("decision_ts")
    row["src_meta_next_epoch"] = timing.get("next_epoch")
    row["src_meta_next_epoch_time"] = timing.get("next_epoch_time")
    row["src_meta_btc_close"] = timing.get("btc_close")
    row["src_meta_start_price"] = timing.get("start_price")
    row["src_meta_end_price"] = timing.get("end_price")
    row["src_meta_price_diff"] = timing.get("price_diff")

    calc_out = catalog.ensure_calc(per_sigma_full, timing=timing)

    bell = (calc_out or {}).get("bell") or {}
    bell_curve_series = (calc_out or {}).get("bell_curve_series") or {}

    channels = (calc_out or {}).get("channels") or {}
    channel_snapshot = channels.get("snapshot") or {}
    pv_tail = channels.get("pv_tail") or {}

    hyst_obj = compute_hysteresis_features(
        per_sigma_full=per_sigma_full,
        timing=timing,
        config=config,
    )

    msbc_obj = build_msbc_feature_payload(
        bell=bell,
        bell_curve_series=bell_curve_series,
        config=config,
    )

    gcs_obj = build_gcs_feature_payload(
        channel_snapshot=channel_snapshot,
        per_sigma_full=per_sigma_full,
        config=config,
    )

    csd_obj = build_csd_dcsd_feature_payload(
        pv_tail=pv_tail,
        config=config,
    )

    gbc_obj = build_gbc_feature_payload(
        bell=bell,
        bell_curve_series=bell_curve_series,
        config=config,
    )

    row.update(flatten_hysteresis_to_src(hyst_obj, config=config))
    row.update(flatten_msbc_to_src(msbc_obj, config=config))
    row.update(flatten_gcs_to_src(gcs_obj, config=config))
    row.update(flatten_csd_dcsd_to_src(csd_obj, config=config))
    row.update(flatten_gbc_to_src(gbc_obj, config=config))

    row["src_meta_last_sample_age"] = ((bell or {}).get("last_sample_age"))
    row["src_meta_lookback_minutes"] = ((bell or {}).get("lookback_minutes"))
    row["src_meta_pv_ref_sigma"] = ((bell or {}).get("pv_ref_sigma"))
    row["src_meta_pv_pair_ref"] = ((bell or {}).get("pv_pair_ref"))

    return row


def export_method_v21_dataset(
    rows: list[dict],
    config: Optional[dict] = None,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    rows should contain already-built per-epoch engine context dicts, each with:
    - timing
    - per_sigma_full
    - per_sigma_hist
    - catalog
    """
    out_rows = []
    for item in rows:
        row = build_method_v21_src_row(
            timing=item["timing"],
            per_sigma_full=item["per_sigma_full"],
            per_sigma_hist=item.get("per_sigma_hist"),
            catalog=item["catalog"],
            config=config,
        )
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    if save_path:
        df.to_csv(save_path, index=False)
    return df
