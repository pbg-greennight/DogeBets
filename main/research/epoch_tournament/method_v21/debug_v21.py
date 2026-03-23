from __future__ import annotations

from typing import Any

import pandas as pd


def explain_method_v21_row(row: pd.Series, result: dict, config: dict) -> dict[str, Any]:
    _ = config
    return {
        "summary": {
            "trend": result["trend"],
            "confidence": result["confidence"],
            "reason": result["reason"],
        },
        "raw_features": {
            "msbc_transfer_dir": row.get("v21_msbc_transfer_dir_num"),
            "msbc_transfer_depth": row.get("v21_msbc_transfer_depth_norm"),
            "msbc_disorder": row.get("v21_msbc_disorder_norm"),
            "msbc_override_count": row.get("v21_msbc_override_count_norm"),
            "msbc_fast_slow_conflict": row.get("v21_msbc_fast_slow_conflict"),
            "hyst_age": row.get("v21_hyst_episode_age_norm"),
            "hyst_flip_watch": row.get("v21_hyst_probe_flip_watch"),
            "hyst_fast_collapse": row.get("v21_hyst_fast_collapse"),
            "hyst_frozen_exhaustion": row.get("v21_hyst_frozen_exhaustion"),
            "gcs_contract_count": row.get("v21_gcs_regime_contract_count_norm"),
            "gcs_transfer_dir": row.get("v21_gcs_transfer_dir_num"),
            "gcs_transfer_depth": row.get("v21_gcs_transfer_depth_norm"),
            "gcs_pos_gap": row.get("v21_gcs_fast_slow_pos_gap"),
            "motion_fast_osc": row.get("v21_channel_fast_oscillation"),
            "motion_compress": row.get("v21_channel_compression_persistence"),
            "motion_reexpand_fail": row.get("v21_channel_reexpansion_failure"),
            "gbc_fast_hook_instab": row.get("v21_gbc_fast_hook_instability"),
            "gbc_slow_flatten": row.get("v21_gbc_slow_flattening"),
        },
        "scores": result["scores"],
        "debug": result["debug"],
    }


def format_method_v21_debug_block(row: pd.Series, result: dict, config: dict) -> str:
    payload = explain_method_v21_row(row, result, config)
    rf = payload["raw_features"]
    sc = payload["scores"]
    return (
        "[v2_1]\n"
        f"MSBC: dir={rf['msbc_transfer_dir']} depth={rf['msbc_transfer_depth']:.3f} disorder={rf['msbc_disorder']:.3f} override={rf['msbc_override_count']:.3f} fast_slow_conflict={rf['msbc_fast_slow_conflict']:.3f}\n"
        f"HYST: age={rf['hyst_age']:.3f} flip_watch={rf['hyst_flip_watch']:.3f} fast_collapse={rf['hyst_fast_collapse']:.3f} frozen_exhaustion={rf['hyst_frozen_exhaustion']:.3f}\n"
        f"GCS : contract_count={rf['gcs_contract_count']:.3f} dir={rf['gcs_transfer_dir']} depth={rf['gcs_transfer_depth']:.3f} pos_gap={rf['gcs_pos_gap']:.3f}\n"
        f"MOTION: fast_osc={rf['motion_fast_osc']:.3f} compress_persist={rf['motion_compress']:.3f} reexpand_fail={rf['motion_reexpand_fail']:.3f}\n"
        f"GBC : fast_hook_instab={rf['gbc_fast_hook_instab']:.3f} slow_flatten={rf['gbc_slow_flatten']:.3f}\n"
        "[scores]\n"
        f"bull_cont={sc['bull_cont']:.3f} bear_cont={sc['bear_cont']:.3f} bull_exhaust={sc['bull_exhaust']:.3f} bear_exhaust={sc['bear_exhaust']:.3f} conflict={sc['conflict']:.3f} trap={sc['trap']:.3f}\n"
        f"bull_raw={sc['bull_raw']:.3f} bear_raw={sc['bear_raw']:.3f} neutral_raw={sc['neutral_raw']:.3f} sep={sc['separation']:.3f}\n"
        "[decision]\n"
        f"trend={payload['summary']['trend']} confidence={payload['summary']['confidence']:.3f} reason={payload['summary']['reason']}"
    )
