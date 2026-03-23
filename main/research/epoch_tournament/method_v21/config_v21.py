from __future__ import annotations


def get_v21_config() -> dict:
    """Return the default config for trend_method_v2_1."""
    return {
        "research": {
            "default_input": r"E:/Trading_Bot_V1.0/DogeBets/main/data/epoch_model_table_v6/epoch_model_table_v21_src.parquet",
            "default_output_dir": r"E:/Trading_Bot_V1.0/DogeBets/main/research/epoch_tournament/method_v21/outputs",
            "default_limit": None,
        },
        "debug": {
            "ENABLE_METHOD_V21_DEBUG": True,
        },
        "msbc": {
            "transfer_depth_max": 5.0,
            "disorder_max": 6.0,
        },
        "hyst": {
            "age_cap_sec": 180.0,
            "spread_slope_cap": 1.0,
            "spread_accel_cap": 1.0,
        },
        "gcs": {
            "fast_only_pos_thresh": 0.35,
            "slow_confirm_pos_thresh": 0.20,
        },
        "csd_dcsd": {
            "flip_count_cap": 6.0,
            "mid_slope_cap": 1.0,
            "mid_slope_gap_cap": 1.0,
            "width_support_min": -0.05,
            "fast_expand_thresh": 0.03,
            "slow_compress_thresh": -0.03,
            "slow_mid_confirm_thresh": 0.05,
            "expand_flip_thresh": 0.45,
            "width_compress_cap": 0.20,
        },
        "gbc": {
            "hook_age_cap": 6.0,
            "turn_age_cap": 12.0,
        },
        "v21": {
            "direction_min": 0.30,
            "separation_min": 0.02,
            "conflict_neutral": 0.78,
            "conflict_block": 0.74,
            "trap_neutral": 0.78,
            "trap_block": 0.72,
            "continuation_min_soft": 0.28,
            "continuation_min_hard": 0.34,
            "exhaustion_override": 0.62,
            "reversal_promotion": 0.70,
            "fast_slow_warn": 0.52,
            "neutral_close_call": 0.08,
            "mixed_dir_margin": 0.08,
        },
    }
