from __future__ import annotations

"""
v21_live_schema.py

Canonical source-row contract for the live v21 integration.

Phase 1 established the meta/GCS/CSD/DCSD contract and the alias/default
normalization layer.

Phase 2 expanded the contract to include persisted hysteresis state so the live
runner can consume the same stack/spread/probe context that the printer was
previously computing only for logs.

Phase 3 adds the remaining bell-derived feature families (MSBC/GBC) so live can
carry the same multi-leg and bell-diagnostic context that the research-side v21
stack expects.
"""

from typing import Dict, List

SIGMAS_ALL: List[int] = [8, 23, 38, 53, 68, 83]
SIGMAS_FAST: List[int] = [8, 23, 38]
SIGMAS_SLOW: List[int] = [53, 68, 83]

SRC_CONTRACT_VERSION = "v21_live_src_v3"


SRC_META_FIELDS: List[str] = [
    "src_meta_epoch",
    "src_meta_next_epoch",
    "src_meta_prev_epoch",
    "src_meta_decision_time",
    "src_meta_window_start",
    "src_meta_window_end",
    "src_meta_close_now",
    "src_meta_hist_points",
]


SRC_MSBC_FIELDS: List[str] = []
for _sigma in SIGMAS_ALL:
    SRC_MSBC_FIELDS.extend(
        [
            f"src_msbc_l1_slope_s{_sigma}",
            f"src_msbc_l2_slope_s{_sigma}",
            f"src_msbc_l1_curve_s{_sigma}",
            f"src_msbc_l2_curve_s{_sigma}",
            f"src_msbc_l1_lin_r2_s{_sigma}",
            f"src_msbc_l2_lin_r2_s{_sigma}",
            f"src_msbc_l1_quad_tangent_s{_sigma}",
            f"src_msbc_l2_quad_tangent_s{_sigma}",
            f"src_msbc_l1_quad_curv_s{_sigma}",
            f"src_msbc_l2_quad_curv_s{_sigma}",
            f"src_msbc_l1_z_s{_sigma}",
            f"src_msbc_l2_z_s{_sigma}",
            f"src_msbc_l1_run_score_s{_sigma}",
            f"src_msbc_l2_run_score_s{_sigma}",
            f"src_msbc_l2_tag_s{_sigma}",
            f"src_msbc_override_s{_sigma}",
            f"src_msbc_sign_agree_s{_sigma}",
            f"src_msbc_accel_agree_s{_sigma}",
            f"src_msbc_slope_ratio_s{_sigma}",
            f"src_msbc_continuity_s{_sigma}",
        ]
    )
SRC_MSBC_FIELDS.extend(
    [
        "src_msbc_transfer_dir",
        "src_msbc_transfer_depth",
        "src_msbc_transfer_state",
        "src_msbc_propagation_agree_ratio",
        "src_msbc_propagation_disagree_ratio",
        "src_msbc_pair_rows",
        "src_msbc_slope_mono",
        "src_msbc_tan_mono",
        "src_msbc_curve_decay",
        "src_msbc_linm_mono",
        "src_msbc_override_count",
        "src_msbc_fast_override_count",
        "src_msbc_slow_override_count",
        "src_msbc_disorder_count",
        "src_msbc_age_seconds",
        "src_msbc_age_label",
        "src_msbc_continuity_mean",
    ]
)


SRC_GCS_FIELDS: List[str] = []
for _sigma in SIGMAS_ALL:
    SRC_GCS_FIELDS.extend(
        [
            f"src_gcs_regime_s{_sigma}",
            f"src_gcs_px_mid_{_sigma}",
            f"src_gcs_zpos_{_sigma}",
            f"src_gcs_width_{_sigma}",
            f"src_gcs_mid_slope_{_sigma}",
            f"src_gcs_width_change_{_sigma}",
            f"src_gcs_width_accel_{_sigma}",
            f"src_gcs_persist_{_sigma}",
        ]
    )
SRC_GCS_FIELDS.extend(
    [
        "src_gcs_contracting_count",
        "src_gcs_expanding_count",
        "src_gcs_flat_count",
        "src_gcs_contracting_all",
        "src_gcs_spacing_state",
        "src_gcs_fan_state",
        "src_gcs_transfer_dir",
        "src_gcs_transfer_depth",
        "src_gcs_transfer_state",
        "src_gcs_fast_pos_mean",
        "src_gcs_slow_pos_mean",
        "src_gcs_fast_slow_pos_gap",
        "src_gcs_front_back_disagreement",
        "src_gcs_reclaim_front",
        "src_gcs_reclaim_slow",
    ]
)


SRC_CSD_FIELDS: List[str] = []
for _sigma in SIGMAS_ALL:
    SRC_CSD_FIELDS.extend(
        [
            f"src_csd_l1_mid_mean_s{_sigma}",
            f"src_csd_l1_width_mean_s{_sigma}",
            f"src_csd_l1_width_last_s{_sigma}",
            f"src_csd_l1_width_delta_s{_sigma}",
            f"src_csd_l1_mid_delta_s{_sigma}",
            f"src_csd_l1_samples_s{_sigma}",
            f"src_csd_l2_mid_mean_s{_sigma}",
            f"src_csd_l2_width_mean_s{_sigma}",
            f"src_csd_l2_width_last_s{_sigma}",
            f"src_csd_l2_width_delta_s{_sigma}",
            f"src_csd_l2_mid_delta_s{_sigma}",
            f"src_csd_l2_samples_s{_sigma}",
            f"src_csd_shrink_s{_sigma}",
        ]
    )
SRC_CSD_FIELDS.extend(
    [
        "src_csd_spread_now",
        "src_csd_spread_delta",
        "src_csd_l2_fast_width_mean",
        "src_csd_l2_slow_width_mean",
    ]
)


SRC_DCSD_FIELDS: List[str] = []
for _sigma in SIGMAS_ALL:
    SRC_DCSD_FIELDS.extend(
        [
            f"src_dcsd_l1_dmid_mean_s{_sigma}",
            f"src_dcsd_l1_dwidth_mean_s{_sigma}",
            f"src_dcsd_l1_dmid_last_s{_sigma}",
            f"src_dcsd_l1_dwidth_last_s{_sigma}",
            f"src_dcsd_l1_dmid_flip_count_s{_sigma}",
            f"src_dcsd_l1_dwidth_flip_count_s{_sigma}",
            f"src_dcsd_l2_dmid_mean_s{_sigma}",
            f"src_dcsd_l2_dwidth_mean_s{_sigma}",
            f"src_dcsd_l2_dmid_last_s{_sigma}",
            f"src_dcsd_l2_dwidth_last_s{_sigma}",
            f"src_dcsd_l2_dmid_flip_count_s{_sigma}",
            f"src_dcsd_l2_dwidth_flip_count_s{_sigma}",
        ]
    )
SRC_DCSD_FIELDS.extend(
    [
        "src_dcsd_release_score",
        "src_dcsd_reversal_stage",
        "src_dcsd_same_sign_exhaustion",
    ]
)


SRC_GBC_FIELDS: List[str] = []
for _sigma in SIGMAS_ALL:
    SRC_GBC_FIELDS.extend(
        [
            f"src_gbc_shrink_s{_sigma}",
            f"src_gbc_flat_s{_sigma}",
            f"src_gbc_hook_s{_sigma}",
            f"src_gbc_prev_abs_s{_sigma}",
            f"src_gbc_last_abs_s{_sigma}",
            f"src_gbc_sign_from_s{_sigma}",
            f"src_gbc_sign_to_s{_sigma}",
            f"src_gbc_eps_s{_sigma}",
            f"src_gbc_turn_age_s{_sigma}",
            f"src_gbc_hook_age_s{_sigma}",
            f"src_gbc_sign_persist_s{_sigma}",
            f"src_gbc_hook_persist_s{_sigma}",
            f"src_gbc_norm_ctx_s{_sigma}",
        ]
    )
SRC_GBC_FIELDS.extend(
    [
        "src_gbc_series_available",
        "src_gbc_active_sigma_count",
    ]
)


SRC_HYST_FIELDS: List[str] = [
    "src_hyst_primary_sign",
    "src_hyst_episode_age_sec",
    "src_hyst_last_cross_age_sec",
    "src_hyst_probe_sign",
    "src_hyst_probe_flip_watch",
    "src_hyst_probe_fast_collapse",
    "src_hyst_probe_recovery",
    "src_hyst_eta_to_end_seconds",
    "src_hyst_summary_stack",
    "src_hyst_spread_state",
    "src_hyst_spread_risk",
    "src_hyst_pressure_state",
    "src_hyst_stack_stability",
    "src_hyst_stack_stability_state",
    "src_hyst_near_cross_state",
    "src_hyst_fan_tightness",
    "src_hyst_stack_alignment",
    "src_hyst_leader_max_sigma",
    "src_hyst_leader_min_sigma",
    "src_hyst_cross_rate",
    "src_hyst_order_stability",
    "src_hyst_ladder_monotonic",
    "src_hyst_ladder_compression",
]
for _stack in ("s0", "s1", "s2", "s3"):
    _base = f"src_hyst_{_stack}"
    SRC_HYST_FIELDS.extend(
        [
            f"{_base}_state",
            f"{_base}_pressure",
            f"{_base}_risk",
            f"{_base}_stability",
            f"{_base}_stability_state",
            f"{_base}_near_cross",
            f"{_base}_near_cross_state",
            f"{_base}_spread_state",
            f"{_base}_spread_momentum",
            f"{_base}_spread_slope",
            f"{_base}_spread_accel",
            f"{_base}_break_risk",
            f"{_base}_fan_tightness",
            f"{_base}_stack_alignment",
            f"{_base}_leader_age",
            f"{_base}_switch_count",
        ]
    )


SRC_DEBUG_FIELDS: List[str] = [
    "src_contract_version",
    "src_live_has_bell",
    "src_live_has_channel_snapshot",
    "src_live_has_pv_tail",
    "src_live_has_hyst",
    "src_live_sigma_count",
    "src_debug_missing_required_count",
    "src_debug_missing_optional_count",
    "src_debug_cov_meta",
    "src_debug_cov_msbc",
    "src_debug_cov_gcs",
    "src_debug_cov_csd",
    "src_debug_cov_dcsd",
    "src_debug_cov_gbc",
    "src_debug_cov_hyst",
    "src_debug_cov_debug",
    "src_debug_cov_overall",
]


SRC_FAMILY_FIELDS: Dict[str, List[str]] = {
    "meta": SRC_META_FIELDS,
    "msbc": SRC_MSBC_FIELDS,
    "gcs": SRC_GCS_FIELDS,
    "csd": SRC_CSD_FIELDS,
    "dcsd": SRC_DCSD_FIELDS,
    "gbc": SRC_GBC_FIELDS,
    "hyst": SRC_HYST_FIELDS,
    "debug": SRC_DEBUG_FIELDS,
}


SRC_REQUIRED_FIELDS: List[str] = [
    "src_meta_epoch",
    "src_meta_next_epoch",
    "src_meta_decision_time",
    "src_live_has_channel_snapshot",
    "src_live_has_pv_tail",
    "src_live_sigma_count",
]

SRC_OPTIONAL_FIELDS: List[str] = [
    field
    for fields in SRC_FAMILY_FIELDS.values()
    for field in fields
    if field not in SRC_REQUIRED_FIELDS
]


SRC_ALIASES: Dict[str, str] = {
    # GCS historical names -> canonical names
    "src_gcs_regime_contract_count": "src_gcs_contracting_count",
    "src_gcs_regime_expand_count": "src_gcs_expanding_count",
    "src_gcs_regime_flat_count": "src_gcs_flat_count",
    "src_gcs_regime_contract_all": "src_gcs_contracting_all",
    # GBC historical names -> canonical names
    "src_gbc_s8_shrink": "src_gbc_shrink_s8",
    "src_gbc_s23_shrink": "src_gbc_shrink_s23",
    "src_gbc_s38_shrink": "src_gbc_shrink_s38",
    "src_gbc_s53_shrink": "src_gbc_shrink_s53",
    "src_gbc_s68_shrink": "src_gbc_shrink_s68",
    "src_gbc_s83_shrink": "src_gbc_shrink_s83",
    "src_gbc_s8_hook": "src_gbc_hook_s8",
    "src_gbc_s23_hook": "src_gbc_hook_s23",
    "src_gbc_s38_hook": "src_gbc_hook_s38",
    "src_gbc_s53_hook": "src_gbc_hook_s53",
    "src_gbc_s68_hook": "src_gbc_hook_s68",
    "src_gbc_s83_hook": "src_gbc_hook_s83",
    # Hysteresis historical names -> canonical names
    "src_hyst_fast_collapse": "src_hyst_probe_fast_collapse",
}
for _sigma in SIGMAS_ALL:
    SRC_ALIASES[f"src_gcs_pos_s{_sigma}"] = f"src_gcs_px_mid_{_sigma}"
    SRC_ALIASES[f"src_gcs_width_s{_sigma}"] = f"src_gcs_width_{_sigma}"
    SRC_ALIASES[f"src_gcs_mid_slope_s{_sigma}"] = f"src_gcs_mid_slope_{_sigma}"
    SRC_ALIASES[f"src_gcs_width_change_s{_sigma}"] = f"src_gcs_width_change_{_sigma}"
    SRC_ALIASES[f"src_gcs_pricepos_s{_sigma}_px_mid"] = f"src_gcs_px_mid_{_sigma}"
    SRC_ALIASES[f"src_gcs_pricepos_s{_sigma}_zpos"] = f"src_gcs_zpos_{_sigma}"
    SRC_ALIASES[f"src_gcs_s{_sigma}_px_mid"] = f"src_gcs_px_mid_{_sigma}"
    SRC_ALIASES[f"src_gcs_s{_sigma}_zpos"] = f"src_gcs_zpos_{_sigma}"


SRC_DEFAULTS: Dict[str, object] = {
    "src_contract_version": SRC_CONTRACT_VERSION,
    "src_meta_hist_points": 0,
    "src_live_has_bell": 0.0,
    "src_live_has_channel_snapshot": 0.0,
    "src_live_has_pv_tail": 0.0,
    "src_live_has_hyst": 0.0,
    "src_live_sigma_count": 0,
    "src_msbc_transfer_dir": "none",
    "src_msbc_transfer_depth": 0.0,
    "src_msbc_transfer_state": "none",
    "src_msbc_propagation_agree_ratio": 0.0,
    "src_msbc_propagation_disagree_ratio": 0.0,
    "src_msbc_pair_rows": 0,
    "src_msbc_slope_mono": 0.0,
    "src_msbc_tan_mono": 0.0,
    "src_msbc_curve_decay": 0.0,
    "src_msbc_linm_mono": 0.0,
    "src_msbc_override_count": 0,
    "src_msbc_fast_override_count": 0,
    "src_msbc_slow_override_count": 0,
    "src_msbc_disorder_count": 0,
    "src_msbc_continuity_mean": 0.0,
    "src_gcs_contracting_count": 0,
    "src_gcs_expanding_count": 0,
    "src_gcs_flat_count": 0,
    "src_gcs_contracting_all": 0,
    "src_gcs_transfer_depth": 0.0,
    "src_gcs_transfer_dir": "none",
    "src_gcs_transfer_state": "none",
    "src_gcs_front_back_disagreement": 0.0,
    "src_gcs_reclaim_front": 0.0,
    "src_gcs_reclaim_slow": 0.0,
    "src_csd_spread_delta": 0.0,
    "src_dcsd_release_score": 0.0,
    "src_dcsd_reversal_stage": 0.0,
    "src_dcsd_same_sign_exhaustion": 0.0,
    "src_gbc_series_available": 0.0,
    "src_gbc_active_sigma_count": 0,
    "src_debug_missing_required_count": 0,
    "src_debug_missing_optional_count": 0,
    "src_debug_cov_meta": 0.0,
    "src_debug_cov_msbc": 0.0,
    "src_debug_cov_gcs": 0.0,
    "src_debug_cov_csd": 0.0,
    "src_debug_cov_dcsd": 0.0,
    "src_debug_cov_gbc": 0.0,
    "src_debug_cov_hyst": 0.0,
    "src_debug_cov_debug": 0.0,
    "src_debug_cov_overall": 0.0,
}


ALL_SRC_FIELDS: List[str] = []
for _fields in SRC_FAMILY_FIELDS.values():
    ALL_SRC_FIELDS.extend(_fields)


__all__ = [
    "ALL_SRC_FIELDS",
    "SIGMAS_ALL",
    "SIGMAS_FAST",
    "SIGMAS_SLOW",
    "SRC_ALIASES",
    "SRC_CONTRACT_VERSION",
    "SRC_CSD_FIELDS",
    "SRC_DCSD_FIELDS",
    "SRC_DEBUG_FIELDS",
    "SRC_DEFAULTS",
    "SRC_FAMILY_FIELDS",
    "SRC_GBC_FIELDS",
    "SRC_GCS_FIELDS",
    "SRC_HYST_FIELDS",
    "SRC_META_FIELDS",
    "SRC_MSBC_FIELDS",
    "SRC_OPTIONAL_FIELDS",
    "SRC_REQUIRED_FIELDS",
]
