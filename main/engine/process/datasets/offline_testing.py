import math
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).with_name("gaussian_fan_dataset_v2_2.csv")
WINDOW_RADIUS = 2
CHUNK_COUNT = 4

TRUTH_CANDIDATES = ["truth", "true_direction", "actual", "label", "target"]
PRED_CANDIDATES = ["prediction", "pred", "trend", "model_prediction", "predicted_trend"]
EPOCH_CANDIDATES = ["epoch", "round", "round_id", "epoch_id", "id"]

RULEB_SLOW_THRESHOLD = 0.95
RULEB_HINGE_THRESHOLD = 0.15
RULEB_TRANSITION_MIN = 1
WINNER_HINGE = 0.18


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
        & (df["hinge_torsion"].abs() >= 0.20)
    ).astype(int)
    cols["pm1_hinge_018"] = (
        (cols["core_window_pm1"] == 1)
        & (cols["center_any_support"] == 1)
        & (df["hinge_torsion"].abs() >= WINNER_HINGE)
    ).astype(int)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def reason_string(df: pd.DataFrame, idx, label: str) -> str:
    return "|".join([
        label,
        f"base={df.at[idx, 'pred_v22_base']}",
        f"truth={df.at[idx, 'truth']}",
        f"slow={float(df.at[idx, 'slow_retention']):.4f}",
        f"abs_hinge={abs(float(df.at[idx, 'hinge_torsion'])):.4f}",
        f"transition={int(df.at[idx, 'transition_score_v23'])}",
        f"phase3={int(df.at[idx, 'phase_disagree_count_3'])}",
        f"fs3={int(df.at[idx, 'fs_disagree_count_3'])}",
        f"fast_alt={int(df.at[idx, 'fast_alt_3epoch'])}",
        f"core_count={int(df.at[idx, 'osc_core_count'])}",
        f"pm1={int(df.at[idx, 'core_window_pm1'])}",
        f"soft={int(df.at[idx, 'pm1_centered_soft'])}",
        f"h018={int(df.at[idx, 'pm1_hinge_018'])}",
    ])


def apply_variant(df: pd.DataFrame, base_col: str, variant_name: str, truth_col: str, trigger_mask: pd.Series, label: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    base_pred = df[base_col].copy()
    eligible = base_pred == "Bear"
    trigger = eligible & trigger_mask.fillna(False)
    pred = base_pred.where(~trigger, "Neutral")
    out[f"pred_{variant_name}"] = pred
    out[f"trigger_{variant_name}"] = trigger.astype(int)
    out[f"reason_{variant_name}"] = pd.Series([
        reason_string(df, idx, label) if bool(trigger.loc[idx]) else "" for idx in df.index
    ], index=df.index, dtype="object")
    override_class = pd.Series("NO_EFFECT", index=df.index, dtype="object")
    base_correct = df[base_col] == df[truth_col]
    override_class.loc[trigger & (~base_correct)] = "GOOD_NEUTRALIZE_WRONG_BASE"
    override_class.loc[trigger & base_correct] = "BAD_NEUTRALIZE_RIGHT_BASE"
    out[f"override_help_class_{variant_name}"] = override_class
    side = pd.Series("", index=df.index, dtype="object")
    side.loc[trigger & (~base_correct)] = "GOOD_SUPPRESS_WRONG_BEAR"
    side.loc[trigger & base_correct] = "BAD_SUPPRESS_RIGHT_BEAR"
    out[f"override_side_class_{variant_name}"] = side
    return out


def compute_override_summary(df: pd.DataFrame, variant_name: str, truth_col: str, base_col: str) -> dict:
    trigger_mask = df[f"trigger_{variant_name}"] == 1
    directional_mask = df[base_col].isin(["Bull", "Bear"])
    good = int((df[f"override_help_class_{variant_name}"] == "GOOD_NEUTRALIZE_WRONG_BASE").sum())
    bad = int((df[f"override_help_class_{variant_name}"] == "BAD_NEUTRALIZE_RIGHT_BASE").sum())
    total = int(trigger_mask.sum())
    inside_dir_mask = trigger_mask & directional_mask
    outside_dir_mask = (~trigger_mask) & directional_mask
    inside_dir = float((df.loc[inside_dir_mask, base_col] != df.loc[inside_dir_mask, truth_col]).mean()) if inside_dir_mask.sum() else 0.0
    outside_dir = float((df.loc[outside_dir_mask, base_col] != df.loc[outside_dir_mask, truth_col]).mean()) if outside_dir_mask.sum() else 0.0
    stats = compute_stats(df, f"pred_{variant_name}", truth_col)
    wrong_bear_ruleb = (df[base_col] == "Bear") & (df[truth_col] == "Bull") & (df["ruleb_raw_warning"] == 1)
    return {
        "model": variant_name,
        "triggers": total,
        "good_neutralizations": good,
        "bad_neutralizations": bad,
        "override_precision": (good / total) if total else 0.0,
        "dir_inside_err": inside_dir,
        "dir_outside_err": outside_dir,
        "good_wrong_bear": int((df[f"override_side_class_{variant_name}"] == "GOOD_SUPPRESS_WRONG_BEAR").sum()),
        "bad_right_bear": int((df[f"override_side_class_{variant_name}"] == "BAD_SUPPRESS_RIGHT_BEAR").sum()),
        "capture_of_ruleb_wrong_bear": float((trigger_mask & wrong_bear_ruleb).sum()) / float(wrong_bear_ruleb.sum()) if wrong_bear_ruleb.sum() else 0.0,
        **stats,
    }


def build_chunk_stability(df: pd.DataFrame, variants: list[str], truth_col: str) -> pd.DataFrame:
    rows = []
    n = len(df)
    chunk_size = math.ceil(n / CHUNK_COUNT)
    for i in range(CHUNK_COUNT):
        start = i * chunk_size
        end = min(n, (i + 1) * chunk_size)
        chunk = df.iloc[start:end].copy()
        for variant in variants:
            pred_col = "pred_v22_base" if variant == "base" else f"pred_{variant}"
            stats = compute_stats(chunk, pred_col, truth_col)
            trig = 0 if variant == "base" else int(chunk[f"trigger_{variant}"].sum())
            rows.append({"chunk_id": i + 1, "row_start": start, "row_end_exclusive": end, "model": variant, "triggers": trig, **stats})
    return pd.DataFrame(rows)


def build_exact_trigger_list(df: pd.DataFrame, variant_name: str, epoch_col: str | None) -> pd.DataFrame:
    mask = df[f"trigger_{variant_name}"] == 1
    rows = []
    for idx, row in df[mask].iterrows():
        rows.append({
            "row_index": int(idx),
            "epoch": row[epoch_col] if epoch_col else idx,
            "base": row["pred_v22_base"],
            "truth": row["truth"],
            "pred": row[f"pred_{variant_name}"],
            "override_help_class": row[f"override_help_class_{variant_name}"],
            "override_side_class": row[f"override_side_class_{variant_name}"],
            "slow_retention": row["slow_retention"],
            "abs_hinge": abs(row["hinge_torsion"]),
            "transition_score_v23": row["transition_score_v23"],
            "phase3": row["phase_disagree_count_3"],
            "fs3": row["fs_disagree_count_3"],
            "fast_alt": row["fast_alt_3epoch"],
            "osc_core_count": row["osc_core_count"],
            "core_window_pm1": row["core_window_pm1"],
            "pm1_centered_soft": row["pm1_centered_soft"],
            "pm1_hinge_018": row["pm1_hinge_018"],
            "reason": row[f"reason_{variant_name}"],
        })
    return pd.DataFrame(rows)


def build_overlap_epoch_report(df: pd.DataFrame, epoch_col: str | None) -> pd.DataFrame:
    a = df["trigger_bear_pm1_h018"] == 1
    b = df["trigger_bear_pm1_soft"] == 1
    groups = {
        "both": a & b,
        "winner_only": a & (~b),
        "fallback_only": b & (~a),
        "either": a | b,
    }
    rows = []
    for label, mask in groups.items():
        for idx, row in df[mask].iterrows():
            rows.append({
                "group": label,
                "row_index": int(idx),
                "epoch": row[epoch_col] if epoch_col else idx,
                "base": row["pred_v22_base"],
                "truth": row["truth"],
                "winner_trigger": int(a.loc[idx]),
                "fallback_trigger": int(b.loc[idx]),
                "slow_retention": row["slow_retention"],
                "abs_hinge": abs(row["hinge_torsion"]),
                "transition_score_v23": row["transition_score_v23"],
                "phase3": row["phase_disagree_count_3"],
                "fs3": row["fs_disagree_count_3"],
                "fast_alt": row["fast_alt_3epoch"],
            })
    return pd.DataFrame(rows)


def build_rollout_summary(df: pd.DataFrame, truth_col: str, base_col: str) -> pd.DataFrame:
    models = [
        ("v2_2_base", None),
        ("bear_core_a", "Bear-only RuleB + Core A"),
        ("bear_pm1", "Bear-only RuleB + Window +/-1"),
        ("bear_pm1_soft", "Fallback: Bear-only RuleB + Center Soft"),
        ("bear_pm1_h018", "Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18"),
    ]
    rows = []
    for key, note in models:
        if key == "v2_2_base":
            stats = compute_stats(df, base_col, truth_col)
            rows.append({
                "rank_hint": 5,
                "model": key,
                "display_name": "v2_2 (base)",
                "candidate_role": "baseline",
                "notes": "Reference model with no Bear-side rescue override.",
                "triggers": 0,
                "good_wrong_bear": 0,
                "bad_right_bear": 0,
                "capture_of_ruleb_wrong_bear": 0.0,
                **stats,
            })
        else:
            summ = compute_override_summary(df, key, truth_col, base_col)
            role = "candidate"
            rank_hint = 3
            if key == "bear_pm1_h018":
                role = "winner"
                rank_hint = 1
            elif key == "bear_pm1_soft":
                role = "fallback"
                rank_hint = 2
            elif key == "bear_core_a":
                role = "safe_reference"
                rank_hint = 4
            rows.append({
                "rank_hint": rank_hint,
                "model": key,
                "display_name": note,
                "candidate_role": role,
                "notes": note,
                **summ,
            })
    out = pd.DataFrame(rows).sort_values(["rank_hint", "directional_accuracy", "bear_precision"], ascending=[True, False, False])
    return out


def build_remaining_false_negative_audit(df: pd.DataFrame, epoch_col: str | None) -> pd.DataFrame:
    wrong_bear_ruleb = (df["pred_v22_base"] == "Bear") & (df["truth"] == "Bull") & (df["ruleb_raw_warning"] == 1)
    missed = wrong_bear_ruleb & (df["trigger_bear_pm1_h018"] == 0) & (df["trigger_bear_pm1_soft"] == 0)
    rows = []
    for idx, row in df[missed].iterrows():
        rows.append({
            "row_index": int(idx),
            "epoch": row[epoch_col] if epoch_col else idx,
            "base": row["pred_v22_base"],
            "truth": row["truth"],
            "slow_retention": row["slow_retention"],
            "abs_hinge": abs(row["hinge_torsion"]),
            "transition_score_v23": row["transition_score_v23"],
            "phase3": row["phase_disagree_count_3"],
            "fs3": row["fs_disagree_count_3"],
            "fast_alt": row["fast_alt_3epoch"],
            "osc_core_count": row["osc_core_count"],
            "core_window_pm1": row["core_window_pm1"],
            "pm1_centered_soft": row["pm1_centered_soft"],
            "pm1_hinge_018": row["pm1_hinge_018"],
        })
    out = pd.DataFrame(rows)
    return out.sort_values(by=["slow_retention", "abs_hinge"], ascending=[False, False]) if not out.empty else out


def build_event_windows(df: pd.DataFrame, epoch_col: str | None, variant_name: str, truth_col: str, base_col: str, radius: int = 2) -> pd.DataFrame:
    rows = []
    centers = []
    for idx in df.index[df[f"trigger_{variant_name}"] == 1].tolist():
        centers.append((idx, "WINNER_TRIGGER"))
    missed_mask = (df[base_col] == "Bear") & (df[truth_col] == "Bull") & (df["ruleb_raw_warning"] == 1) & (df["trigger_bear_pm1_h018"] == 0) & (df["trigger_bear_pm1_soft"] == 0)
    for idx in df.index[missed_mask].tolist()[:20]:
        centers.append((idx, "REMAINING_FALSE_NEGATIVE"))
    seen = set()
    for center_idx, center_type in centers:
        if (center_idx, center_type) in seen:
            continue
        seen.add((center_idx, center_type))
        start = max(int(df.index.min()), int(center_idx) - radius)
        end = min(int(df.index.max()), int(center_idx) + radius)
        for idx, row in df.loc[start:end].iterrows():
            rows.append({
                "window_center_index": int(center_idx),
                "window_type": center_type,
                "relative_pos": int(idx - center_idx),
                "window_center_epoch": df.at[center_idx, epoch_col] if epoch_col else center_idx,
                "epoch": row[epoch_col] if epoch_col else idx,
                "truth": row[truth_col],
                "base": row[base_col],
                f"pred_{variant_name}": row[f"pred_{variant_name}"],
                f"trigger_{variant_name}": row[f"trigger_{variant_name}"],
                "ruleb_raw_warning": row["ruleb_raw_warning"],
                "core_window_pm1": row["core_window_pm1"],
                "pm1_centered_soft": row["pm1_centered_soft"],
                "pm1_hinge_018": row["pm1_hinge_018"],
                "phase3": row["phase_disagree_count_3"],
                "fs3": row["fs_disagree_count_3"],
                "fast_alt": row["fast_alt_3epoch"],
                "slow_retention": row["slow_retention"],
                "abs_hinge": abs(row["hinge_torsion"]),
                "transition_score_v23": row["transition_score_v23"],
            })
    return pd.DataFrame(rows)


def print_windows(df_windows: pd.DataFrame, variant_name: str) -> None:
    print("\n=== SAMPLE 5-ROW WINDOWS (WINNER + REMAINING MISSES) ===")
    if df_windows.empty:
        print("No windows generated.")
        return
    shown = 0
    for (center_idx, window_type), group in df_windows.groupby(["window_center_index", "window_type"]):
        print(f"\nwindow={window_type} | center_index={center_idx} | center_epoch={group.iloc[0]['window_center_epoch']}")
        for _, row in group.iterrows():
            print(
                f"  rel={int(row['relative_pos']):+d} | epoch={row['epoch']} | truth={row['truth']} | base={row['base']} | "
                f"pred={row[f'pred_{variant_name}']} | trig={int(row[f'trigger_{variant_name}'])} | ruleb={int(row['ruleb_raw_warning'])} | "
                f"pm1={int(row['core_window_pm1'])} | soft={int(row['pm1_centered_soft'])} | h018={int(row['pm1_hinge_018'])} | "
                f"phase3={int(row['phase3'])} | fs3={int(row['fs3'])} | fast_alt={int(row['fast_alt'])} | "
                f"slow={float(row['slow_retention']):.3f} | |hinge|={float(row['abs_hinge']):.3f} | trans={int(row['transition_score_v23'])}"
            )
        shown += 1
        if shown >= 8:
            break


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    truth_col = pick_column(df, TRUTH_CANDIDATES, "truth")
    base_pred_col = pick_column(df, PRED_CANDIDATES, "prediction")
    epoch_col = try_pick_column(df, EPOCH_CANDIDATES)

    df = build_features(df)
    df = df.rename(columns={truth_col: "truth"})
    if epoch_col and epoch_col != "epoch":
        df = df.rename(columns={epoch_col: "epoch"})
        epoch_col = "epoch"
    df["pred_v22_base"] = df[base_pred_col].copy()
    df["ruleb_raw_warning"] = build_ruleb_warning(df)
    df = build_sequence_features(df)

    latest = df.iloc[-1]
    print("\n=== CURRENT EPOCH FEATURE SNAPSHOT (LAST ROW) ===")
    for name in [
        "pred_v22_base", "slow_retention", "hinge_torsion", "transition_score_v23", "ruleb_raw_warning",
        "phase_disagree_count_3", "fs_disagree_count_3", "fast_alt_3epoch", "core_window_pm1", "pm1_centered_soft", "pm1_hinge_018"
    ]:
        val = latest[name]
        if name == "pred_v22_base":
            print(f"Base v2_2 prediction: {val}")
        elif name == "hinge_torsion":
            print(f"|hinge_torsion|: {abs(float(val)):.4f}")
        elif isinstance(val, (int, float)):
            if "slow" in name:
                print(f"{name}: {float(val):.4f}")
            else:
                print(f"{name}: {int(val) if float(val).is_integer() else float(val):}")

    base_stats = compute_stats(df, "pred_v22_base", "truth")
    print_stats_block("v2_2 (base)", base_stats)

    masks = {
        "bear_core_a": (df["ruleb_raw_warning"] == 1) & (df["osc_core_a"] == 1),
        "bear_pm1": (df["ruleb_raw_warning"] == 1) & (df["core_window_pm1"] == 1),
        "bear_pm1_soft": (df["ruleb_raw_warning"] == 1) & (df["pm1_centered_soft"] == 1),
        "bear_pm1_h018": (df["ruleb_raw_warning"] == 1) & (df["pm1_hinge_018"] == 1),
    }
    labels = {
        "bear_core_a": "Bear-only RuleB + Core A",
        "bear_pm1": "Bear-only RuleB + Window +/-1",
        "bear_pm1_soft": "Fallback: Bear-only RuleB + Center Soft",
        "bear_pm1_h018": "Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18",
    }

    variant_frames = [apply_variant(df, "pred_v22_base", name, "truth", mask, labels[name]) for name, mask in masks.items()]
    df = pd.concat([df] + variant_frames, axis=1)

    for name in ["bear_core_a", "bear_pm1", "bear_pm1_soft", "bear_pm1_h018"]:
        print_stats_block(labels[name], compute_stats(df, f"pred_{name}", "truth"))

    rollout = build_rollout_summary(df, "truth", "pred_v22_base")
    print("\n=== V273 FINAL LEADERBOARD / ROLLOUT SUMMARY ===")
    for _, row in rollout.iterrows():
        print(
            f"{row['candidate_role']} | {row['display_name']} | acc={row['directional_accuracy'] * 100:.2f}% | "
            f"cov={row['directional_coverage'] * 100:.2f}% | bear={row['bear_precision'] * 100:.2f}% | "
            f"trig={int(row['triggers'])} | good={int(row['good_wrong_bear'])} | bad={int(row['bad_right_bear'])} | "
            f"capture={row['capture_of_ruleb_wrong_bear'] * 100:.2f}%"
        )

    trigger_quality = pd.DataFrame([compute_override_summary(df, name, "truth", "pred_v22_base") for name in ["bear_core_a", "bear_pm1", "bear_pm1_soft", "bear_pm1_h018"]])
    print("\n=== V273 TRIGGER QUALITY SUMMARY ===")
    for _, row in trigger_quality.iterrows():
        print(
            f"{row['model']} | triggers={int(row['triggers'])} | good={int(row['good_neutralizations'])} | bad={int(row['bad_neutralizations'])} | "
            f"good_rate={row['override_precision'] * 100:.2f}% | dir_inside_err={row['dir_inside_err'] * 100:.2f}% | dir_outside_err={row['dir_outside_err'] * 100:.2f}%"
        )

    chunk_stability = build_chunk_stability(df, ["base", "bear_pm1_soft", "bear_pm1_h018"], "truth")
    print("\n=== WINNER VS FALLBACK CHUNK STABILITY ===")
    for _, row in chunk_stability.iterrows():
        print(
            f"chunk={int(row['chunk_id'])} | model={row['model']} | rows={int(row['rows'])} | trig={int(row['triggers'])} | "
            f"acc={row['directional_accuracy'] * 100:.2f}% | cov={row['directional_coverage'] * 100:.2f}% | bear={row['bear_precision'] * 100:.2f}%"
        )

    overlap_epoch = build_overlap_epoch_report(df, epoch_col)
    overlap_summary = overlap_epoch.groupby('group').agg(
        rows=('group', 'size'),
        wrong_bear=('truth', lambda s: int((overlap_epoch.loc[s.index, 'base'].eq('Bear') & overlap_epoch.loc[s.index, 'truth'].eq('Bull')).sum())),
        right_bear=('truth', lambda s: int((overlap_epoch.loc[s.index, 'base'].eq('Bear') & overlap_epoch.loc[s.index, 'truth'].eq('Bear')).sum())),
        mean_slow=('slow_retention', 'mean'),
        mean_abs_hinge=('abs_hinge', 'mean'),
    ).reset_index()
    print("\n=== WINNER VS FALLBACK OVERLAP SUMMARY ===")
    for _, row in overlap_summary.iterrows():
        print(
            f"{row['group']} | rows={int(row['rows'])} | wrong_bear={int(row['wrong_bear'])} | right_bear={int(row['right_bear'])} | "
            f"mean_slow={row['mean_slow']:.3f} | mean_|hinge|={row['mean_abs_hinge']:.3f}"
        )

    remaining_false_neg = build_remaining_false_negative_audit(df, epoch_col)
    print("\n=== REMAINING FALSE NEGATIVE AUDIT (MISSED BY WINNER AND FALLBACK) ===")
    if remaining_false_neg.empty:
        print("No remaining false negatives inside RuleB context.")
    else:
        for _, row in remaining_false_neg.head(12).iterrows():
            print(
                f"epoch={row['epoch']} | base={row['base']} | truth={row['truth']} | slow={row['slow_retention']:.3f} | "
                f"|hinge|={row['abs_hinge']:.3f} | trans={int(row['transition_score_v23'])} | phase3={int(row['phase3'])} | "
                f"fs3={int(row['fs3'])} | fast_alt={int(row['fast_alt'])} | core_count={int(row['osc_core_count'])}"
            )

    winner_trigger_list = build_exact_trigger_list(df, "bear_pm1_h018", epoch_col)
    fallback_trigger_list = build_exact_trigger_list(df, "bear_pm1_soft", epoch_col)
    winner_only = overlap_epoch[overlap_epoch['group'] == 'winner_only'].copy()
    fallback_only = overlap_epoch[overlap_epoch['group'] == 'fallback_only'].copy()
    event_windows = build_event_windows(df, epoch_col, "bear_pm1_h018", "truth", "pred_v22_base", radius=WINDOW_RADIUS)
    print_windows(event_windows, "bear_pm1_h018")

    out_prefix = CSV_PATH.parent
    diag_path = out_prefix / "gaussian_fan_dataset_v2_2_with_v273_rollout_diagnostics.csv"
    decision_audit_path = out_prefix / "offline_v273_decision_audit.csv"
    trigger_quality_path = out_prefix / "offline_v273_trigger_quality_summary.csv"
    rollout_path = out_prefix / "offline_v273_rollout_summary.csv"
    chunk_path = out_prefix / "offline_v273_chunk_stability.csv"
    winner_trigger_path = out_prefix / "offline_v273_winner_trigger_list.csv"
    fallback_trigger_path = out_prefix / "offline_v273_fallback_trigger_list.csv"
    overlap_epoch_path = out_prefix / "offline_v273_overlap_epoch_report.csv"
    winner_only_path = out_prefix / "offline_v273_winner_only_rows.csv"
    fallback_only_path = out_prefix / "offline_v273_fallback_only_rows.csv"
    remaining_path = out_prefix / "offline_v273_remaining_false_negative_audit.csv"
    event_windows_path = out_prefix / "offline_v273_event_windows.csv"
    model_comp_path = out_prefix / "offline_v273_model_comparison.csv"

    df.to_csv(diag_path, index=False)
    df.to_csv(decision_audit_path, index=False)
    trigger_quality.to_csv(trigger_quality_path, index=False)
    rollout.to_csv(rollout_path, index=False)
    chunk_stability.to_csv(chunk_path, index=False)
    winner_trigger_list.to_csv(winner_trigger_path, index=False)
    fallback_trigger_list.to_csv(fallback_trigger_path, index=False)
    overlap_epoch.to_csv(overlap_epoch_path, index=False)
    winner_only.to_csv(winner_only_path, index=False)
    fallback_only.to_csv(fallback_only_path, index=False)
    remaining_false_neg.to_csv(remaining_path, index=False)
    event_windows.to_csv(event_windows_path, index=False)
    rollout.to_csv(model_comp_path, index=False)

    print(f"\nSaved dataset: {diag_path}")
    print(f"Saved decision audit: {decision_audit_path}")
    print(f"Saved trigger quality summary: {trigger_quality_path}")
    print(f"Saved rollout summary: {rollout_path}")
    print(f"Saved chunk stability: {chunk_path}")
    print(f"Saved winner trigger list: {winner_trigger_path}")
    print(f"Saved fallback trigger list: {fallback_trigger_path}")
    print(f"Saved overlap epoch report: {overlap_epoch_path}")
    print(f"Saved winner-only rows: {winner_only_path}")
    print(f"Saved fallback-only rows: {fallback_only_path}")
    print(f"Saved remaining false negative audit: {remaining_path}")
    print(f"Saved event windows: {event_windows_path}")
    print(f"Saved model comparison: {model_comp_path}")


if __name__ == "__main__":
    main()
