from __future__ import annotations

import pandas as pd

from .common_v21 import clip01, mean_safe, safe_div


def _ensure(df: pd.DataFrame, col: str, default=float("nan")) -> None:
    if col not in df.columns:
        df[col] = default


def add_gbc_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df["v21_src_missing_gbc"] = 0.0

    needed = []
    for sigma in [8, 23, 53, 68, 83]:
        needed += [f"src_gbc_hook_s{sigma}", f"src_gbc_flat_s{sigma}", f"src_gbc_shrink_s{sigma}", f"src_gbc_last_abs_s{sigma}", f"src_gbc_turn_age_s{sigma}"]
    for sigma in [8, 23]:
        needed += [f"src_gbc_sign_persist_s{sigma}", f"src_gbc_hook_age_s{sigma}"]
    for c in needed:
        _ensure(df, c)

    def _fast_hook_instability(r: pd.Series) -> float:
        fast_hook_active = mean_safe([float(bool(r["src_gbc_hook_s8"])), float(bool(r["src_gbc_hook_s23"]))])
        low_sign_persist = 1.0 - mean_safe([clip01(r["src_gbc_sign_persist_s8"]), clip01(r["src_gbc_sign_persist_s23"])])
        short_hook_age = 1.0 - mean_safe([
            clip01(safe_div(r["src_gbc_hook_age_s8"], config["gbc"]["hook_age_cap"])),
            clip01(safe_div(r["src_gbc_hook_age_s23"], config["gbc"]["hook_age_cap"])),
        ])
        fast_abs = mean_safe([abs(r["src_gbc_last_abs_s8"]), abs(r["src_gbc_last_abs_s23"])])
        slow_abs = mean_safe([abs(r["src_gbc_last_abs_s53"]), abs(r["src_gbc_last_abs_s68"]), abs(r["src_gbc_last_abs_s83"])])
        fast_abs_vs_slow = clip01(safe_div(fast_abs, slow_abs + 1e-9))
        fast_turn_density = mean_safe([
            1.0 - clip01(safe_div(r["src_gbc_turn_age_s8"], config["gbc"]["turn_age_cap"])),
            1.0 - clip01(safe_div(r["src_gbc_turn_age_s23"], config["gbc"]["turn_age_cap"])),
        ])
        return clip01(0.30 * fast_hook_active + 0.20 * low_sign_persist + 0.20 * short_hook_age + 0.15 * fast_abs_vs_slow + 0.15 * fast_turn_density)

    def _slow_flattening(r: pd.Series) -> float:
        slow_flat = mean_safe([clip01(r["src_gbc_flat_s53"]), clip01(r["src_gbc_flat_s68"]), clip01(r["src_gbc_flat_s83"])])
        slow_turn_age = mean_safe([
            clip01(safe_div(r["src_gbc_turn_age_s53"], config["gbc"]["turn_age_cap"])),
            clip01(safe_div(r["src_gbc_turn_age_s68"], config["gbc"]["turn_age_cap"])),
            clip01(safe_div(r["src_gbc_turn_age_s83"], config["gbc"]["turn_age_cap"])),
        ])
        slow_shrink_decay = 1.0 - mean_safe([clip01(r["src_gbc_shrink_s53"]), clip01(r["src_gbc_shrink_s68"]), clip01(r["src_gbc_shrink_s83"])])
        slow_abs_fade = mean_safe([
            clip01(1.0 - abs(r["src_gbc_last_abs_s53"])),
            clip01(1.0 - abs(r["src_gbc_last_abs_s68"])),
            clip01(1.0 - abs(r["src_gbc_last_abs_s83"])),
        ])
        return clip01(0.35 * slow_flat + 0.25 * slow_turn_age + 0.20 * slow_shrink_decay + 0.20 * slow_abs_fade)

    df["v21_gbc_fast_hook_instability"] = df.apply(_fast_hook_instability, axis=1)
    df["v21_gbc_slow_flattening"] = df.apply(_slow_flattening, axis=1)
    return df
