import math
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).with_name("gaussian_fan_dataset_v2_2.csv")
LOOKBACK = 30
WINDOW_RADIUS = 2
CHUNK_COUNT = 4

TRUTH_CANDIDATES = ["truth", "true_direction", "actual", "label", "target"]
BASE_PRED_CANDIDATES = ["pred_v22_base", "prediction", "pred", "trend", "model_prediction", "predicted_trend"]
EPOCH_CANDIDATES = ["epoch", "round", "round_id", "epoch_id", "id"]

RULEB_SLOW_THRESHOLD = 0.95
RULEB_HINGE_THRESHOLD = 0.15
RULEB_TRANSITION_MIN = 1
WINNER_HINGE = 0.18
TIGHT_HINGE = 0.22
SOFT_HINGE = 0.20


# -------------------------------
# Generic helpers
# -------------------------------
def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label} column. Tried: {candidates}\nAvailable columns:\n{list(df.columns)}")


def try_pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sign_series(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError(f"sign_series expected 1-D input, got columns={list(series.columns)}")
        series = series.iloc[:, 0]
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return ((s > 0).astype(int) - (s < 0).astype(int)).astype(int)


def compute_stats(df: pd.DataFrame, pred_col: str, truth_col: str) -> dict:
    total_rows = len(df)
    directional_mask = df[pred_col].isin(["Bull", "Bear"])
    directional = df[directional_mask]
    directional_called = int(len(directional))
    neutral_called = int((df[pred_col] == "Neutral").sum())
    directional_accuracy = float((directional[pred_col] == directional[truth_col]).mean()) if directional_called else 0.0
    coverage = directional_called / total_rows if total_rows else 0.0
    neutral_rate = neutral_called / total_rows if total_rows else 0.0
    bull_pred = directional[directional[pred_col] == "Bull"]
    bear_pred = directional[directional[pred_col] == "Bear"]
    bull_precision = float((bull_pred[truth_col] == "Bull").mean()) if len(bull_pred) else 0.0
    bear_precision = float((bear_pred[truth_col] == "Bear").mean()) if len(bear_pred) else 0.0
    truth_bull = df[truth_col] == "Bull"
    truth_bear = df[truth_col] == "Bear"
    bull_recall = float((((df[pred_col] == "Bull") & truth_bull).sum()) / truth_bull.sum()) if truth_bull.sum() else 0.0
    bear_recall = float((((df[pred_col] == "Bear") & truth_bear).sum()) / truth_bear.sum()) if truth_bear.sum() else 0.0
    return {
        "rows": total_rows,
        "directional_called": directional_called,
        "neutral_called": neutral_called,
        "directional_accuracy": directional_accuracy,
        "directional_coverage": coverage,
        "neutral_rate": neutral_rate,
        "bull_precision": bull_precision,
        "bear_precision": bear_precision,
        "bull_recall": bull_recall,
        "bear_recall": bear_recall,
    }


def print_stats_block(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Rows: {stats['rows']}")
    print(f"Directional Called: {stats['directional_called']}")
    print(f"Neutral Called: {stats['neutral_called']}")
    print(f"Directional Accuracy: {stats['directional_accuracy'] * 100:.2f}%")
    print(f"Directional Coverage: {stats['directional_coverage'] * 100:.2f}%")
    print(f"Neutral Rate: {stats['neutral_rate'] * 100:.2f}%")
    print(f"Bull Precision: {stats['bull_precision'] * 100:.2f}%")
    print(f"Bear Precision: {stats['bear_precision'] * 100:.2f}%")
    print(f"Bull Recall: {stats['bull_recall'] * 100:.2f}%")
    print(f"Bear Recall: {stats['bear_recall'] * 100:.2f}%")


# -------------------------------
# Feature engineering
# -------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [
        "g8", "g23", "g38", "g53", "g68", "g83",
        "slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83",
        "hinge_torsion", "fan_width_velocity", "outer_fan_width",
        "phase_disagreement", "fan_polarity_inversion", "slow_retention",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    new_cols: dict[str, pd.Series] = {}
    if "fan_energy_total" not in df.columns:
        new_cols["fan_energy_total"] = df["outer_fan_width"].abs() * (1.0 + df["hinge_torsion"].abs())
        fan_energy_total = new_cols["fan_energy_total"]
    else:
        fan_energy_total = pd.to_numeric(df["fan_energy_total"], errors="coerce")

    new_cols["hinge_torsion_spike"] = (df["hinge_torsion"].abs() > 0.5).astype(int)
    new_cols["transition_score_v23"] = (
        (df["phase_disagreement"] == 1).astype(int)
        + new_cols["hinge_torsion_spike"]
        + (df["fan_polarity_inversion"] == 1).astype(int)
    )
    new_cols["fan_energy_t0"] = fan_energy_total
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_ruleb_warning(df: pd.DataFrame) -> pd.Series:
    return (
        (df["slow_retention"] > RULEB_SLOW_THRESHOLD)
        & (df["hinge_torsion"].abs() > RULEB_HINGE_THRESHOLD)
        & (df["transition_score_v23"] >= RULEB_TRANSITION_MIN)
    ).astype(int)


def rolling_support(flag: pd.Series) -> pd.Series:
    s = flag.fillna(0).astype(int)
    return ((s == 1) | (s.shift(1).fillna(0).astype(int) == 1) | (s.shift(-1).fillna(0).astype(int) == 1)).astype(int)


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    cols["fast_sign_now"] = sign_series(df["slope_g8"] + df["slope_g23"])
    cols["slow_sign_now"] = sign_series(df["slope_g68"] + df["slope_g83"])
    cols["fast_alt_3epoch"] = (
        (cols["fast_sign_now"] != 0)
        & (cols["fast_sign_now"].shift(1).fillna(0).astype(int) != 0)
        & (cols["fast_sign_now"].shift(2).fillna(0).astype(int) != 0)
        & (cols["fast_sign_now"] == cols["fast_sign_now"].shift(2).fillna(0).astype(int))
        & (cols["fast_sign_now"] != cols["fast_sign_now"].shift(1).fillna(0).astype(int))
    ).astype(int)
    cols["fs_disagree_now"] = (
        (cols["fast_sign_now"] != 0)
        & (cols["slow_sign_now"] != 0)
        & (cols["fast_sign_now"] != cols["slow_sign_now"])
    ).astype(int)
    cols["fs_disagree_count_3"] = cols["fs_disagree_now"].rolling(3, min_periods=1).sum().astype(int)
    cols["phase_disagree_count_3"] = df["phase_disagreement"].fillna(0).rolling(3, min_periods=1).sum().astype(int)
    cols["osc_core_phase"] = (cols["phase_disagree_count_3"] >= 2).astype(int)
    cols["osc_core_fs"] = (cols["fs_disagree_count_3"] >= 2).astype(int)
    cols["osc_core_fastalt"] = cols["fast_alt_3epoch"].astype(int)
    cols["osc_core_count"] = (cols["osc_core_phase"] + cols["osc_core_fs"] + cols["osc_core_fastalt"]).astype(int)
    cols["osc_core_a"] = (cols["osc_core_count"] >= 2).astype(int)
    cols["center_any_support"] = (
        (cols["phase_disagree_count_3"] >= 1)
        | (cols["fs_disagree_count_3"] >= 1)
        | (cols["fast_alt_3epoch"] == 1)
    ).astype(int)
    cols["center_phase_or_fs"] = (
        (cols["phase_disagree_count_3"] >= 1)
        | (cols["fs_disagree_count_3"] >= 1)
    ).astype(int)
    cols["phase_support_pm1"] = rolling_support(cols["osc_core_phase"])
    cols["fs_support_pm1"] = rolling_support(cols["osc_core_fs"])
    cols["fastalt_support_pm1"] = rolling_support(cols["osc_core_fastalt"])
    cols["core_window_pm1"] = (
        (cols["fs_support_pm1"] == 1)
        & ((cols["phase_support_pm1"] == 1) | (cols["fastalt_support_pm1"] == 1))
    ).astype(int)
    cols["pm1_centered_soft"] = (
        (cols["core_window_pm1"] == 1)
        & ((cols["center_phase_or_fs"] == 1) | (cols["osc_core_fastalt"] == 1))
        & (df["hinge_torsion"].abs() >= SOFT_HINGE)
    ).astype(int)
    cols["pm1_hinge_018"] = (
        (cols["core_window_pm1"] == 1)
        & (cols["center_any_support"] == 1)
        & (df["hinge_torsion"].abs() >= WINNER_HINGE)
    ).astype(int)
    cols["pm1_hinge_022"] = (
        (cols["core_window_pm1"] == 1)
        & (cols["center_any_support"] == 1)
        & (df["hinge_torsion"].abs() >= TIGHT_HINGE)
    ).astype(int)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


# -------------------------------
# Static policy predictions
# -------------------------------
def apply_bear_neutralizer(base_pred: pd.Series, trigger_mask: pd.Series) -> pd.Series:
    return base_pred.where(~((base_pred == "Bear") & trigger_mask.fillna(False)), "Neutral")


def add_static_policy(df: pd.DataFrame, name: str, trigger_mask: pd.Series, truth_col: str, base_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    pred = apply_bear_neutralizer(df[base_col], trigger_mask)
    trigger = ((df[base_col] == "Bear") & trigger_mask.fillna(False)).astype(int)
    out[f"pred_{name}"] = pred
    out[f"trigger_{name}"] = trigger
    override = pd.Series("NO_EFFECT", index=df.index, dtype="object")
    good = (trigger == 1) & (df[base_col] != df[truth_col])
    bad = (trigger == 1) & (df[base_col] == df[truth_col])
    override.loc[good] = "GOOD_NEUTRALIZE_WRONG_BASE"
    override.loc[bad] = "BAD_NEUTRALIZE_RIGHT_BASE"
    out[f"override_help_class_{name}"] = override
    return out


# -------------------------------
# Adaptive regime controller
# -------------------------------
def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def tail_bear_streak(window: pd.DataFrame, base_col: str, truth_col: str, want_correct: bool) -> int:
    streak = 0
    for _, row in window.iloc[::-1].iterrows():
        if row[base_col] != "Bear":
            continue
        is_correct = row[truth_col] == "Bear"
        if is_correct == want_correct:
            streak += 1
        else:
            break
    return streak


def policy_window_stats(window: pd.DataFrame, policy_name: str) -> dict:
    trig_col = f"trigger_{policy_name}"
    help_col = f"override_help_class_{policy_name}"
    trig = int(window[trig_col].sum())
    good = int((window[help_col] == "GOOD_NEUTRALIZE_WRONG_BASE").sum())
    bad = int((window[help_col] == "BAD_NEUTRALIZE_RIGHT_BASE").sum())
    return {
        "triggers": trig,
        "good": good,
        "bad": bad,
        "good_rate": safe_rate(good, trig),
        "bad_rate": safe_rate(bad, trig),
    }


def choose_policy_for_row(df: pd.DataFrame, idx: int, truth_col: str, base_col: str) -> tuple[str, str, dict]:
    current = df.iloc[idx]
    if idx < LOOKBACK:
        return "base", "warmup_lt_30", {}
    if current[base_col] != "Bear":
        return "base", f"base_pred={current[base_col]}", {}
    if int(current["ruleb_raw_warning"]) == 0:
        return "base", "ruleb_off", {}

    window = df.iloc[idx - LOOKBACK: idx]
    base_bear = window[window[base_col] == "Bear"]
    wrong_bear_rate = safe_rate(int((base_bear[truth_col] == "Bull").sum()), len(base_bear))
    disagreement_density = float(((window["phase_disagree_count_3"] >= 1) | (window["fs_disagree_count_3"] >= 1) | (window["fast_alt_3epoch"] == 1)).mean())
    transition_density = float((window["transition_score_v23"] >= 1).mean())
    neutral_rate = float((window[base_col] == "Neutral").mean())
    abs_hinge_mean = float(window["hinge_torsion"].abs().mean())
    abs_hinge_last10 = float(window["hinge_torsion"].abs().tail(10).mean()) if len(window) >= 10 else abs_hinge_mean
    abs_hinge_prev20 = float(window["hinge_torsion"].abs().head(max(len(window) - 10, 1)).mean()) if len(window) > 10 else abs_hinge_mean
    vol_expansion = abs_hinge_last10 - abs_hinge_prev20
    mean_phase = float(window["phase_disagree_count_3"].mean())
    mean_fs = float(window["fs_disagree_count_3"].mean())
    fast_alt_rate = float(window["fast_alt_3epoch"].mean())
    wrong_bear_streak = tail_bear_streak(window, base_col, truth_col, want_correct=False)
    right_bear_streak = tail_bear_streak(window, base_col, truth_col, want_correct=True)

    stats_corea = policy_window_stats(window, "bear_core_a")
    stats_soft = policy_window_stats(window, "bear_pm1_soft")
    stats_win = policy_window_stats(window, "bear_pm1_h018")
    stats_tight = policy_window_stats(window, "bear_pm1_h022")

    regime = {
        "wrong_bear_rate": wrong_bear_rate,
        "disagreement_density": disagreement_density,
        "transition_density": transition_density,
        "neutral_rate": neutral_rate,
        "abs_hinge_mean": abs_hinge_mean,
        "vol_expansion": vol_expansion,
        "mean_phase": mean_phase,
        "mean_fs": mean_fs,
        "fast_alt_rate": fast_alt_rate,
        "wrong_bear_streak": wrong_bear_streak,
        "right_bear_streak": right_bear_streak,
        "winner_good_rate": stats_win["good_rate"],
        "winner_bad_rate": stats_win["bad_rate"],
        "fallback_good_rate": stats_soft["good_rate"],
        "fallback_bad_rate": stats_soft["bad_rate"],
        "corea_good_rate": stats_corea["good_rate"],
        "corea_bad_rate": stats_corea["bad_rate"],
        "tight_good_rate": stats_tight["good_rate"],
        "tight_bad_rate": stats_tight["bad_rate"],
    }

    quiet_regime = wrong_bear_rate < 0.12 and disagreement_density < 0.45 and transition_density < 0.40
    risk_on = wrong_bear_rate >= 0.22 and disagreement_density >= 0.55
    prefer_fallback = (
        (stats_soft["triggers"] >= 1 and stats_soft["good_rate"] >= 0.75 and stats_soft["bad_rate"] <= 0.25)
        or (stats_win["triggers"] >= 2 and stats_win["bad_rate"] > 0.25 and stats_soft["bad_rate"] < stats_win["bad_rate"])
    )
    prefer_tight = (
        stats_tight["triggers"] >= 1
        and stats_tight["good_rate"] >= 0.60
        and stats_tight["bad_rate"] <= stats_win["bad_rate"]
        and (abs_hinge_mean < 0.30 or vol_expansion < -0.02)
    )

    if quiet_regime:
        return "base", "quiet_regime", regime

    if int(current["pm1_hinge_018"]) == 1:
        if prefer_fallback and int(current["pm1_centered_soft"]) == 1 and not risk_on:
            return "fallback", "current_h018_but_recent_soft_cleaner", regime
        return "winner", "current_h018_and_not_soft_bias", regime

    if int(current["pm1_centered_soft"]) == 1:
        return "fallback", "current_soft_support", regime

    if int(current["pm1_hinge_022"]) == 1 and prefer_tight:
        return "winner_tight", "tight_policy_recently_cleaner", regime

    if int(current["osc_core_a"]) == 1:
        if stats_corea["triggers"] >= 1 and stats_corea["good_rate"] >= 0.90:
            return "corea", "corea_recently_ultra_clean", regime
        if wrong_bear_streak >= 2 and mean_fs >= 0.9:
            return "corea", "wrong_bear_streak_with_core_support", regime

    if risk_on and int(current["core_window_pm1"]) == 1 and abs(float(current["hinge_torsion"])) >= TIGHT_HINGE:
        return "winner_tight", "risk_on_window_tight", regime

    return "base", "no_adaptive_edge", regime


def apply_adaptive_controller(df: pd.DataFrame, truth_col: str, base_col: str) -> pd.DataFrame:
    chosen_policy = []
    chosen_reason = []
    adaptive_pred = []
    adaptive_trigger = []
    regime_rows = []

    pred_map = {
        "base": df[base_col],
        "corea": df["pred_bear_core_a"],
        "fallback": df["pred_bear_pm1_soft"],
        "winner": df["pred_bear_pm1_h018"],
        "winner_tight": df["pred_bear_pm1_h022"],
    }

    for i in range(len(df)):
        policy, reason, regime = choose_policy_for_row(df, i, truth_col, base_col)
        pred = pred_map[policy].iloc[i]
        base_pred = df[base_col].iloc[i]
        trig = int((base_pred == "Bear") and (pred == "Neutral"))
        chosen_policy.append(policy)
        chosen_reason.append(reason)
        adaptive_pred.append(pred)
        adaptive_trigger.append(trig)
        regime_rows.append(regime)

    out = pd.DataFrame(index=df.index)
    out["adaptive_policy"] = pd.Series(chosen_policy, index=df.index, dtype="object")
    out["adaptive_reason"] = pd.Series(chosen_reason, index=df.index, dtype="object")
    out["pred_adaptive_v1"] = pd.Series(adaptive_pred, index=df.index, dtype="object")
    out["trigger_adaptive_v1"] = pd.Series(adaptive_trigger, index=df.index, dtype="int64")
    override = pd.Series("NO_EFFECT", index=df.index, dtype="object")
    good = (out["trigger_adaptive_v1"] == 1) & (df[base_col] != df[truth_col])
    bad = (out["trigger_adaptive_v1"] == 1) & (df[base_col] == df[truth_col])
    override.loc[good] = "GOOD_NEUTRALIZE_WRONG_BASE"
    override.loc[bad] = "BAD_NEUTRALIZE_RIGHT_BASE"
    out["override_help_class_adaptive_v1"] = override
    regime_df = pd.DataFrame(regime_rows, index=df.index)
    return pd.concat([out, regime_df], axis=1)


# -------------------------------
# Diagnostics / reports
# -------------------------------
def summarize_trigger_quality(df: pd.DataFrame, policy_name: str) -> dict:
    trig_col = f"trigger_{policy_name}"
    help_col = f"override_help_class_{policy_name}"
    direction_mask = df[f"pred_{policy_name}"].isin(["Bull", "Bear"]) if policy_name != "adaptive_v1" else df["pred_adaptive_v1"].isin(["Bull", "Bear"])
    wrong_inside = int((df[help_col] == "GOOD_NEUTRALIZE_WRONG_BASE").sum())
    bad_inside = int((df[help_col] == "BAD_NEUTRALIZE_RIGHT_BASE").sum())
    trig = int(df[trig_col].sum())
    dir_inside_err = safe_rate(wrong_inside, trig)
    outside = df[df[trig_col] == 0]
    base_bear_outside = outside[outside["pred_v22_base"] == "Bear"]
    dir_outside_err = safe_rate(int((base_bear_outside["truth"] == "Bull").sum()), len(base_bear_outside))
    return {
        "policy": policy_name,
        "triggers": trig,
        "good": wrong_inside,
        "bad": bad_inside,
        "good_rate": safe_rate(wrong_inside, trig),
        "dir_inside_err": dir_inside_err,
        "dir_outside_err": dir_outside_err,
    }


def build_policy_usage_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(df)
    bear_rows = int((df["pred_v22_base"] == "Bear").sum())
    for policy, group in df.groupby("adaptive_policy", dropna=False):
        rows.append({
            "policy": policy,
            "rows": len(group),
            "share_all": len(group) / total if total else 0.0,
            "base_bear_rows": int((group["pred_v22_base"] == "Bear").sum()),
            "share_of_bear_rows": int((group["pred_v22_base"] == "Bear").sum()) / bear_rows if bear_rows else 0.0,
            "adaptive_triggers": int(group["trigger_adaptive_v1"].sum()),
            "good": int((group["override_help_class_adaptive_v1"] == "GOOD_NEUTRALIZE_WRONG_BASE").sum()),
            "bad": int((group["override_help_class_adaptive_v1"] == "BAD_NEUTRALIZE_RIGHT_BASE").sum()),
            "mean_wrong_bear_rate": float(group["wrong_bear_rate"].fillna(0.0).mean()) if "wrong_bear_rate" in group else 0.0,
            "mean_disagreement_density": float(group["disagreement_density"].fillna(0.0).mean()) if "disagreement_density" in group else 0.0,
            "mean_transition_density": float(group["transition_density"].fillna(0.0).mean()) if "transition_density" in group else 0.0,
            "mean_abs_hinge_mean": float(group["abs_hinge_mean"].fillna(0.0).mean()) if "abs_hinge_mean" in group else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["base_bear_rows", "rows"], ascending=[False, False])


def build_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for reason, group in df.groupby("adaptive_reason", dropna=False):
        rows.append({
            "reason": reason,
            "rows": len(group),
            "adaptive_triggers": int(group["trigger_adaptive_v1"].sum()),
            "good": int((group["override_help_class_adaptive_v1"] == "GOOD_NEUTRALIZE_WRONG_BASE").sum()),
            "bad": int((group["override_help_class_adaptive_v1"] == "BAD_NEUTRALIZE_RIGHT_BASE").sum()),
            "mean_wrong_bear_rate": float(group["wrong_bear_rate"].fillna(0.0).mean()) if "wrong_bear_rate" in group else 0.0,
            "mean_disagreement_density": float(group["disagreement_density"].fillna(0.0).mean()) if "disagreement_density" in group else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["adaptive_triggers", "rows"], ascending=[False, False])


def build_chunk_stability(df: pd.DataFrame, truth_col: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    chunk_size = math.ceil(n / CHUNK_COUNT)
    model_map = {
        "base": "pred_v22_base",
        "winner": "pred_bear_pm1_h018",
        "fallback": "pred_bear_pm1_soft",
        "adaptive_v1": "pred_adaptive_v1",
    }
    trig_map = {
        "base": None,
        "winner": "trigger_bear_pm1_h018",
        "fallback": "trigger_bear_pm1_soft",
        "adaptive_v1": "trigger_adaptive_v1",
    }
    for chunk_idx in range(CHUNK_COUNT):
        start = chunk_idx * chunk_size
        end = min(n, start + chunk_size)
        if start >= end:
            continue
        chunk = df.iloc[start:end]
        for model_name, pred_col in model_map.items():
            stats = compute_stats(chunk, pred_col, truth_col)
            trig_col = trig_map[model_name]
            rows.append({
                "chunk": chunk_idx + 1,
                "model": model_name,
                "rows": len(chunk),
                "triggers": int(chunk[trig_col].sum()) if trig_col else 0,
                "acc": stats["directional_accuracy"],
                "cov": stats["directional_coverage"],
                "bear": stats["bear_precision"],
            })
    return pd.DataFrame(rows)


def build_overlap_report(df: pd.DataFrame) -> pd.DataFrame:
    winner = df["trigger_bear_pm1_h018"] == 1
    fallback = df["trigger_bear_pm1_soft"] == 1
    adaptive = df["trigger_adaptive_v1"] == 1
    groups = {
        "winner_only": winner & ~fallback,
        "fallback_only": fallback & ~winner,
        "winner_and_fallback": winner & fallback,
        "adaptive_only": adaptive & ~winner & ~fallback,
        "adaptive_and_winner": adaptive & winner,
        "adaptive_and_fallback": adaptive & fallback,
        "adaptive_any": adaptive,
    }
    rows = []
    for label, mask in groups.items():
        sub = df[mask]
        rows.append({
            "group": label,
            "rows": len(sub),
            "wrong_bear": int(((sub["pred_v22_base"] == "Bear") & (sub["truth"] == "Bull")).sum()),
            "right_bear": int(((sub["pred_v22_base"] == "Bear") & (sub["truth"] == "Bear")).sum()),
            "mean_slow": float(sub["slow_retention"].mean()) if len(sub) else 0.0,
            "mean_abs_hinge": float(sub["hinge_torsion"].abs().mean()) if len(sub) else 0.0,
            "mean_phase3": float(sub["phase_disagree_count_3"].mean()) if len(sub) else 0.0,
            "mean_fs3": float(sub["fs_disagree_count_3"].mean()) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def build_remaining_false_negative_audit(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["pred_v22_base"] == "Bear")
        & (df["truth"] == "Bull")
        & (df["ruleb_raw_warning"] == 1)
        & (df["trigger_adaptive_v1"] == 0)
    )
    cols = [
        "epoch", "pred_v22_base", "truth", "slow_retention", "hinge_torsion", "transition_score_v23",
        "phase_disagree_count_3", "fs_disagree_count_3", "fast_alt_3epoch", "osc_core_count",
        "core_window_pm1", "pm1_centered_soft", "pm1_hinge_018", "adaptive_policy", "adaptive_reason",
    ]
    cols = [c for c in cols if c in df.columns]
    return df.loc[mask, cols].copy()


def build_event_windows(df: pd.DataFrame, epoch_col: str) -> pd.DataFrame:
    centers = []
    centers.extend([(idx, "ADAPTIVE_TRIGGER") for idx in df.index[df["trigger_adaptive_v1"] == 1][:6]])
    remaining_mask = (
        (df["pred_v22_base"] == "Bear")
        & (df["truth"] == "Bull")
        & (df["ruleb_raw_warning"] == 1)
        & (df["trigger_adaptive_v1"] == 0)
    )
    centers.extend([(idx, "REMAINING_FALSE_NEGATIVE") for idx in df.index[remaining_mask][:6]])

    rows = []
    for center_idx, label in centers:
        center_pos = df.index.get_loc(center_idx)
        for rel in range(-WINDOW_RADIUS, WINDOW_RADIUS + 1):
            pos = center_pos + rel
            if pos < 0 or pos >= len(df):
                continue
            row = df.iloc[pos]
            rows.append({
                "window_type": label,
                "center_index": int(center_pos),
                "center_epoch": row[epoch_col] if rel == 0 else df.iloc[center_pos][epoch_col],
                "rel": rel,
                "epoch": row[epoch_col],
                "truth": row["truth"],
                "base": row["pred_v22_base"],
                "pred": row["pred_adaptive_v1"],
                "adaptive_policy": row["adaptive_policy"],
                "trig": int(row["trigger_adaptive_v1"]),
                "ruleb": int(row["ruleb_raw_warning"]),
                "pm1": int(row["core_window_pm1"]),
                "soft": int(row["pm1_centered_soft"]),
                "h018": int(row["pm1_hinge_018"]),
                "phase3": int(row["phase_disagree_count_3"]),
                "fs3": int(row["fs_disagree_count_3"]),
                "fast_alt": int(row["fast_alt_3epoch"]),
                "slow": float(row["slow_retention"]),
                "abs_hinge": abs(float(row["hinge_torsion"])),
                "transition": int(row["transition_score_v23"]),
            })
    return pd.DataFrame(rows)


def pretty_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# -------------------------------
# Main
# -------------------------------
def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    truth_col = pick_column(df, TRUTH_CANDIDATES, "truth")
    base_col = pick_column(df, BASE_PRED_CANDIDATES, "base prediction")
    epoch_col = try_pick_column(df, EPOCH_CANDIDATES) or truth_col

    df = df.rename(columns={truth_col: "truth", base_col: "pred_v22_base", epoch_col: "epoch"})
    df = build_features(df)
    df["ruleb_raw_warning"] = build_ruleb_warning(df)
    df = build_sequence_features(df)

    static_parts = []
    static_parts.append(add_static_policy(df, "bear_core_a", (df["ruleb_raw_warning"] == 1) & (df["osc_core_a"] == 1), "truth", "pred_v22_base"))
    static_parts.append(add_static_policy(df, "bear_pm1", (df["ruleb_raw_warning"] == 1) & (df["core_window_pm1"] == 1), "truth", "pred_v22_base"))
    static_parts.append(add_static_policy(df, "bear_pm1_soft", (df["ruleb_raw_warning"] == 1) & (df["pm1_centered_soft"] == 1), "truth", "pred_v22_base"))
    static_parts.append(add_static_policy(df, "bear_pm1_h018", (df["ruleb_raw_warning"] == 1) & (df["pm1_hinge_018"] == 1), "truth", "pred_v22_base"))
    static_parts.append(add_static_policy(df, "bear_pm1_h022", (df["ruleb_raw_warning"] == 1) & (df["pm1_hinge_022"] == 1), "truth", "pred_v22_base"))
    df = pd.concat([df] + static_parts, axis=1)

    adaptive = apply_adaptive_controller(df, "truth", "pred_v22_base")
    df = pd.concat([df, adaptive], axis=1)

    print("\n=== CURRENT EPOCH FEATURE SNAPSHOT (LAST ROW) ===")
    last = df.iloc[-1]
    print(f"Base v2_2 prediction: {last['pred_v22_base']}")
    print(f"slow_retention: {float(last['slow_retention']):.4f}")
    print(f"|hinge_torsion|: {abs(float(last['hinge_torsion'])):.4f}")
    print(f"transition_score_v23: {int(last['transition_score_v23'])}")
    print(f"ruleb_raw_warning: {int(last['ruleb_raw_warning'])}")
    print(f"phase_disagree_count_3: {int(last['phase_disagree_count_3'])}")
    print(f"fs_disagree_count_3: {int(last['fs_disagree_count_3'])}")
    print(f"fast_alt_3epoch: {int(last['fast_alt_3epoch'])}")
    print(f"pm1_centered_soft: {int(last['pm1_centered_soft'])}")
    print(f"pm1_hinge_018: {int(last['pm1_hinge_018'])}")
    print(f"Adaptive chosen policy: {last['adaptive_policy']}")
    print(f"Adaptive reason: {last['adaptive_reason']}")

    model_map = {
        "v2_2 (base)": "pred_v22_base",
        "Safe reference: Bear-only RuleB + Core A": "pred_bear_core_a",
        "Fallback: Bear-only RuleB + Center Soft": "pred_bear_pm1_soft",
        "Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18": "pred_bear_pm1_h018",
        "Adaptive v1: Rolling 30-epoch controller": "pred_adaptive_v1",
    }
    for title, pred_col in model_map.items():
        print_stats_block(title, compute_stats(df, pred_col, "truth"))

    rollout_rows = []
    order = [
        ("adaptive_v1", "Adaptive v1: Rolling 30-epoch controller"),
        ("bear_pm1_h018", "Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18"),
        ("bear_pm1_soft", "Fallback: Bear-only RuleB + Center Soft"),
        ("bear_core_a", "Safe reference: Bear-only RuleB + Core A"),
        ("pred_v22_base", "v2_2 (base)"),
    ]
    for key, label in order:
        pred_col = key if key == "pred_v22_base" else f"pred_{key}"
        stats = compute_stats(df, pred_col, "truth")
        if key == "pred_v22_base":
            trig = good = bad = capture = 0
        else:
            q = summarize_trigger_quality(df, key)
            trig, good, bad = q["triggers"], q["good"], q["bad"]
            ruleb_wrong_bear = int(((df["pred_v22_base"] == "Bear") & (df["truth"] == "Bull") & (df["ruleb_raw_warning"] == 1)).sum())
            capture = safe_rate(good, ruleb_wrong_bear)
        rollout_rows.append({
            "rank_label": label,
            "model_key": key,
            "directional_accuracy": stats["directional_accuracy"],
            "directional_coverage": stats["directional_coverage"],
            "bear_precision": stats["bear_precision"],
            "triggers": trig,
            "good": good,
            "bad": bad,
            "capture_of_ruleb_wrong_bear": capture,
        })
    rollout_summary = pd.DataFrame(rollout_rows)

    print("\n=== V274 ROLLING ADAPTIVE LEADERBOARD ===")
    for _, row in rollout_summary.iterrows():
        print(
            f"{row['model_key']} | {row['rank_label']} | acc={pretty_pct(row['directional_accuracy'])} | "
            f"cov={pretty_pct(row['directional_coverage'])} | bear={pretty_pct(row['bear_precision'])} | "
            f"trig={int(row['triggers'])} | good={int(row['good'])} | bad={int(row['bad'])} | "
            f"capture={pretty_pct(row['capture_of_ruleb_wrong_bear'])}"
        )

    trig_quality = pd.DataFrame([
        summarize_trigger_quality(df, "bear_core_a"),
        summarize_trigger_quality(df, "bear_pm1_soft"),
        summarize_trigger_quality(df, "bear_pm1_h018"),
        summarize_trigger_quality(df, "adaptive_v1"),
    ])
    print("\n=== V274 TRIGGER QUALITY SUMMARY ===")
    for _, row in trig_quality.iterrows():
        print(
            f"{row['policy']} | triggers={int(row['triggers'])} | good={int(row['good'])} | bad={int(row['bad'])} | "
            f"good_rate={pretty_pct(row['good_rate'])} | dir_inside_err={pretty_pct(row['dir_inside_err'])} | "
            f"dir_outside_err={pretty_pct(row['dir_outside_err'])}"
        )

    policy_usage = build_policy_usage_summary(df)
    print("\n=== ADAPTIVE POLICY USAGE SUMMARY ===")
    for _, row in policy_usage.iterrows():
        print(
            f"policy={row['policy']} | rows={int(row['rows'])} | bear_rows={int(row['base_bear_rows'])} | "
            f"triggers={int(row['adaptive_triggers'])} | good={int(row['good'])} | bad={int(row['bad'])} | "
            f"mean_wrong_bear_rate={row['mean_wrong_bear_rate']:.3f} | mean_disagree={row['mean_disagreement_density']:.3f}"
        )

    reason_summary = build_reason_summary(df)
    print("\n=== ADAPTIVE REASON SUMMARY ===")
    for _, row in reason_summary.head(10).iterrows():
        print(
            f"reason={row['reason']} | rows={int(row['rows'])} | triggers={int(row['adaptive_triggers'])} | "
            f"good={int(row['good'])} | bad={int(row['bad'])} | mean_wrong_bear_rate={row['mean_wrong_bear_rate']:.3f}"
        )

    chunk_stability = build_chunk_stability(df, "truth")
    print("\n=== ADAPTIVE VS STATIC CHUNK STABILITY ===")
    for _, row in chunk_stability.iterrows():
        print(
            f"chunk={int(row['chunk'])} | model={row['model']} | rows={int(row['rows'])} | trig={int(row['triggers'])} | "
            f"acc={pretty_pct(row['acc'])} | cov={pretty_pct(row['cov'])} | bear={pretty_pct(row['bear'])}"
        )

    overlap_report = build_overlap_report(df)
    print("\n=== WINNER/FALLBACK/ADAPTIVE OVERLAP SUMMARY ===")
    for _, row in overlap_report.iterrows():
        print(
            f"{row['group']} | rows={int(row['rows'])} | wrong_bear={int(row['wrong_bear'])} | "
            f"right_bear={int(row['right_bear'])} | mean_slow={row['mean_slow']:.3f} | mean_|hinge|={row['mean_abs_hinge']:.3f}"
        )

    remaining_false_negative = build_remaining_false_negative_audit(df)
    print("\n=== REMAINING FALSE NEGATIVE AUDIT (MISSED BY ADAPTIVE) ===")
    for _, row in remaining_false_negative.head(12).iterrows():
        print(
            f"epoch={row['epoch']} | base={row['pred_v22_base']} | truth={row['truth']} | slow={float(row['slow_retention']):.3f} | "
            f"|hinge|={abs(float(row['hinge_torsion'])):.3f} | trans={int(row['transition_score_v23'])} | "
            f"phase3={int(row['phase_disagree_count_3'])} | fs3={int(row['fs_disagree_count_3'])} | fast_alt={int(row['fast_alt_3epoch'])} | "
            f"core_count={int(row['osc_core_count'])} | policy={row['adaptive_policy']} | reason={row['adaptive_reason']}"
        )

    event_windows = build_event_windows(df, "epoch")
    print("\n=== SAMPLE 5-ROW WINDOWS (ADAPTIVE TRIGGERS + REMAINING MISSES) ===")
    for (wtype, center_epoch), group in event_windows.groupby(["window_type", "center_epoch"], dropna=False):
        print(f"\nwindow={wtype} | center_epoch={center_epoch}")
        for _, row in group.iterrows():
            print(
                f"  rel={int(row['rel']):+d} | epoch={row['epoch']} | truth={row['truth']} | base={row['base']} | pred={row['pred']} | "
                f"policy={row['adaptive_policy']} | trig={int(row['trig'])} | ruleb={int(row['ruleb'])} | pm1={int(row['pm1'])} | "
                f"soft={int(row['soft'])} | h018={int(row['h018'])} | phase3={int(row['phase3'])} | fs3={int(row['fs3'])} | "
                f"fast_alt={int(row['fast_alt'])} | slow={row['slow']:.3f} | |hinge|={row['abs_hinge']:.3f} | trans={int(row['transition'])}"
            )

    out_dataset = CSV_PATH.with_name("gaussian_fan_dataset_v2_2_with_v274_adaptive_diagnostics.csv")
    out_dataset_df = df.copy()
    out_dataset_df.to_csv(out_dataset, index=False)

    decision_audit = df[[
        "epoch", "truth", "pred_v22_base", "pred_bear_core_a", "pred_bear_pm1_soft", "pred_bear_pm1_h018", "pred_adaptive_v1",
        "trigger_bear_core_a", "trigger_bear_pm1_soft", "trigger_bear_pm1_h018", "trigger_adaptive_v1",
        "override_help_class_bear_core_a", "override_help_class_bear_pm1_soft", "override_help_class_bear_pm1_h018",
        "override_help_class_adaptive_v1", "adaptive_policy", "adaptive_reason",
        "wrong_bear_rate", "disagreement_density", "transition_density", "abs_hinge_mean", "vol_expansion",
        "winner_good_rate", "winner_bad_rate", "fallback_good_rate", "fallback_bad_rate",
        "slow_retention", "hinge_torsion", "transition_score_v23", "phase_disagree_count_3", "fs_disagree_count_3", "fast_alt_3epoch",
        "osc_core_count", "ruleb_raw_warning", "core_window_pm1", "pm1_centered_soft", "pm1_hinge_018"
    ]].copy()

    decision_path = CSV_PATH.with_name("offline_v274_decision_audit.csv")
    trig_quality_path = CSV_PATH.with_name("offline_v274_trigger_quality_summary.csv")
    rollout_path = CSV_PATH.with_name("offline_v274_rollout_summary.csv")
    usage_path = CSV_PATH.with_name("offline_v274_policy_usage_summary.csv")
    reason_path = CSV_PATH.with_name("offline_v274_reason_summary.csv")
    chunk_path = CSV_PATH.with_name("offline_v274_chunk_stability.csv")
    overlap_path = CSV_PATH.with_name("offline_v274_overlap_report.csv")
    remain_path = CSV_PATH.with_name("offline_v274_remaining_false_negative_audit.csv")
    windows_path = CSV_PATH.with_name("offline_v274_event_windows.csv")
    model_path = CSV_PATH.with_name("offline_v274_model_comparison.csv")

    decision_audit.to_csv(decision_path, index=False)
    trig_quality.to_csv(trig_quality_path, index=False)
    rollout_summary.to_csv(rollout_path, index=False)
    policy_usage.to_csv(usage_path, index=False)
    reason_summary.to_csv(reason_path, index=False)
    chunk_stability.to_csv(chunk_path, index=False)
    overlap_report.to_csv(overlap_path, index=False)
    remaining_false_negative.to_csv(remain_path, index=False)
    event_windows.to_csv(windows_path, index=False)
    pd.DataFrame([
        {"model": title, **compute_stats(df, pred_col, "truth")} for title, pred_col in model_map.items()
    ]).to_csv(model_path, index=False)

    print(f"\nSaved dataset: {out_dataset}")
    print(f"Saved decision audit: {decision_path}")
    print(f"Saved trigger quality summary: {trig_quality_path}")
    print(f"Saved rollout summary: {rollout_path}")
    print(f"Saved policy usage summary: {usage_path}")
    print(f"Saved reason summary: {reason_path}")
    print(f"Saved chunk stability: {chunk_path}")
    print(f"Saved overlap report: {overlap_path}")
    print(f"Saved remaining false negative audit: {remain_path}")
    print(f"Saved event windows: {windows_path}")
    print(f"Saved model comparison: {model_path}")


if __name__ == "__main__":
    main()
