import pandas as pd
from pathlib import Path


CSV_PATH = Path(__file__).with_name("gaussian_fan_dataset_v2_2.csv")


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find {label} column. Tried: {candidates}\n"
        f"Available columns:\n{list(df.columns)}"
    )


def compute_stats(df: pd.DataFrame, pred_col: str, truth_col: str) -> dict:
    total_rows = len(df)

    directional_mask = df[pred_col].isin(["Bull", "Bear"])
    directional = df[directional_mask]

    directional_called = len(directional)
    neutral_called = (df[pred_col] == "Neutral").sum()

    directional_accuracy = (
        (directional[pred_col] == directional[truth_col]).mean()
        if directional_called > 0 else 0.0
    )
    coverage = directional_called / total_rows if total_rows else 0.0
    neutral_rate = neutral_called / total_rows if total_rows else 0.0

    bull_pred = directional[directional[pred_col] == "Bull"]
    bear_pred = directional[directional[pred_col] == "Bear"]

    bull_precision = (
        (bull_pred[truth_col] == "Bull").mean()
        if len(bull_pred) > 0 else 0.0
    )
    bear_precision = (
        (bear_pred[truth_col] == "Bear").mean()
        if len(bear_pred) > 0 else 0.0
    )

    return {
        "rows": total_rows,
        "directional_called": directional_called,
        "neutral_called": neutral_called,
        "directional_accuracy": directional_accuracy,
        "directional_coverage": coverage,
        "neutral_rate": neutral_rate,
        "bull_precision": bull_precision,
        "bear_precision": bear_precision,
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


def flip_direction(pred: str) -> str:
    if pred == "Bull":
        return "Bear"
    if pred == "Bear":
        return "Bull"
    return pred


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    truth_col = pick_column(
        df,
        ["truth", "true_direction", "actual", "label", "target"],
        "truth",
    )

    base_pred_col = pick_column(
        df,
        ["prediction", "pred", "trend", "model_prediction", "predicted_trend"],
        "prediction",
    )

    required = [
        "g8", "g23", "g38", "g53", "g68", "g83",
        "slope_g8", "slope_g23", "slope_g38", "slope_g53", "slope_g68", "slope_g83",
        "hinge_torsion",
        "fan_width_velocity",
        "outer_fan_width",
        "phase_disagreement",
        "fan_polarity_inversion",
        "slow_retention",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    if "fan_energy_total" not in df.columns:
        df["fan_energy_total"] = (
            df["outer_fan_width"].abs() * (1.0 + df["hinge_torsion"].abs())
        )

    # -------------------------------------------------
    # Base model preserved for comparison
    # -------------------------------------------------
    df["pred_v22_base"] = df[base_pred_col].copy()

    # -------------------------------------------------
    # Derived geometry / feature stack
    # -------------------------------------------------
    df["outer_fan_width_prev"] = df["outer_fan_width"].shift(1)
    df["hinge_torsion_spike"] = (df["hinge_torsion"].abs() > 0.5).astype(int)

    df["transition_score_v23"] = (
        (df["phase_disagreement"] == 1).astype(int)
        + (df["hinge_torsion_spike"] == 1).astype(int)
        + (df["fan_polarity_inversion"] == 1).astype(int)
    )
    df["transition_gate_v23"] = (df["transition_score_v23"] >= 2).astype(int)

    # Collapse swept winner
    df["fan_collapse_v23_swept"] = (
        (df["fan_width_velocity"] < -15.0)
        & (df["hinge_torsion"].abs() > 0.40)
        & (df["phase_disagreement"] == 1)
        & (df["outer_fan_width"] < df["outer_fan_width_prev"])
    ).astype(int)

    # Inertia swept winner
    df["inertia_wall_active_swept"] = (
        (df["slow_retention"] > 0.75)
        & (df["hinge_torsion"].abs() > 0.25)
        & (df["phase_disagreement"] == 1)
    ).astype(int)

    # Torsion spring
    df["ts_d1"] = df["slope_g8"] - df["slope_g23"]
    df["ts_d2"] = df["slope_g23"] - df["slope_g38"]
    df["ts_d3"] = df["slope_g38"] - df["slope_g53"]
    df["ts_d4"] = df["slope_g53"] - df["slope_g68"]
    df["ts_d5"] = df["slope_g68"] - df["slope_g83"]

    df["torsion_spring_score_v23"] = (
        1.0 * (df["ts_d1"] - df["ts_d2"]).abs()
        + 1.5 * (df["ts_d2"] - df["ts_d3"]).abs()
        + 1.5 * (df["ts_d3"] - df["ts_d4"]).abs()
        + 1.0 * (df["ts_d4"] - df["ts_d5"]).abs()
    )

    df["torsion_spring_strong_swept"] = (
        (df["torsion_spring_score_v23"] > 0.25)
        & (df["hinge_torsion"].abs() > 0.40)
        & (df["phase_disagreement"] == 1)
    ).astype(int)

    # Fan energy breathing cycle
    df["fan_energy_t3"] = df["fan_energy_total"].shift(3)
    df["fan_energy_t2"] = df["fan_energy_total"].shift(2)
    df["fan_energy_t1"] = df["fan_energy_total"].shift(1)
    df["fan_energy_t0"] = df["fan_energy_total"]

    df["fan_energy_cycle_v23"] = (
        (df["fan_energy_t3"] > df["fan_energy_t2"])
        & (df["fan_energy_t1"] > df["fan_energy_t2"])
        & (df["fan_energy_t0"] < df["fan_energy_t1"])
    ).astype(int)

    # Torque imbalance
    df["torque_fast_motion"] = df["slope_g8"].abs() + df["slope_g23"].abs()
    df["torque_slow_motion"] = df["slope_g68"].abs() + df["slope_g83"].abs()
    df["torque_imbalance_score"] = df["torque_fast_motion"] - df["torque_slow_motion"]
    df["torque_imbalance_ratio"] = (
        df["torque_fast_motion"] / (df["torque_slow_motion"] + 1e-6)
    )
    df["torque_fast_signed"] = df["slope_g8"] + df["slope_g23"]
    df["torque_slow_signed"] = df["slope_g68"] + df["slope_g83"]
    df["torque_opposition"] = (
        (df["torque_fast_signed"] * df["torque_slow_signed"]) < 0
    ).astype(int)

    df["torque_imbalance_active_swept"] = (
        (df["torque_imbalance_ratio"] > 1.75)
        & (df["torque_imbalance_score"] > 0.00)
        & (df["torque_opposition"] == 1)
    ).astype(int)

    # -------------------------------------------------
    # High-value combo features
    # -------------------------------------------------
    df["collapse_plus_inertia_swept"] = (
        (df["fan_collapse_v23_swept"] == 1)
        & (df["inertia_wall_active_swept"] == 1)
    ).astype(int)

    df["collapse_plus_torsion_swept"] = (
        (df["fan_collapse_v23_swept"] == 1)
        & (df["torsion_spring_strong_swept"] == 1)
    ).astype(int)

    df["torsion_plus_transition_swept"] = (
        (df["torsion_spring_strong_swept"] == 1)
        & (df["transition_gate_v23"] == 1)
    ).astype(int)

    df["torsion_plus_inertia_swept"] = (
        (df["torsion_spring_strong_swept"] == 1)
        & (df["inertia_wall_active_swept"] == 1)
    ).astype(int)

    df["torque_plus_inertia_swept"] = (
        (df["torque_imbalance_ratio"] > 1.75)
        & (df["torque_imbalance_score"] > 0.00)
        & (df["slow_retention"] > 0.75)
        & (df["hinge_torsion"].abs() > 0.60)
        & (df["phase_disagreement"] == 1)
        & (df["torque_opposition"] == 1)
    ).astype(int)

    df["torque_plus_transition_swept"] = (
        (df["torque_imbalance_active_swept"] == 1)
        & (df["transition_gate_v23"] == 1)
    ).astype(int)

    df["collapse_plus_torque_swept"] = (
        (df["torque_imbalance_ratio"] > 1.25)
        & (df["torque_imbalance_score"] > 0.20)
        & (df["fan_width_velocity"] < -15.0)
        & (df["hinge_torsion"].abs() > 0.60)
        & (df["phase_disagreement"] == 1)
        & (df["outer_fan_width"] < df["outer_fan_width_prev"])
        & (df["torque_opposition"] == 1)
    ).astype(int)

    # -------------------------------------------------
    # Current epoch value snapshot for quick comparison
    # -------------------------------------------------
    latest = df.iloc[-1]
    print("\n=== CURRENT EPOCH FEATURE SNAPSHOT (LAST ROW) ===")
    print(f"Base v2_2 prediction: {latest['pred_v22_base']}")
    print(f"fan_collapse_v23_swept: {int(latest['fan_collapse_v23_swept'])}")
    print(f"inertia_wall_active_swept: {int(latest['inertia_wall_active_swept'])}")
    print(f"torsion_spring_strong_swept: {int(latest['torsion_spring_strong_swept'])}")
    print(f"fan_energy_cycle_v23: {int(latest['fan_energy_cycle_v23'])}")
    print(f"torque_imbalance_active_swept: {int(latest['torque_imbalance_active_swept'])}")
    print(f"collapse_plus_inertia_swept: {int(latest['collapse_plus_inertia_swept'])}")
    print(f"collapse_plus_torsion_swept: {int(latest['collapse_plus_torsion_swept'])}")
    print(f"torsion_plus_transition_swept: {int(latest['torsion_plus_transition_swept'])}")
    print(f"torsion_plus_inertia_swept: {int(latest['torsion_plus_inertia_swept'])}")
    print(f"torque_plus_transition_swept: {int(latest['torque_plus_transition_swept'])}")
    print(f"torque_plus_inertia_swept: {int(latest['torque_plus_inertia_swept'])}")
    print(f"collapse_plus_torque_swept: {int(latest['collapse_plus_torque_swept'])}")

    # -------------------------------------------------
    # Keep earlier replays for leaderboard comparison
    # -------------------------------------------------
    # Proto v2.3 original replay
    pred_original = df["pred_v22_base"].copy()
    for i, row in df.iterrows():
        current_pred = pred_original.iloc[i]
        if row["inertia_wall_active_swept"] == 1 and row["fan_collapse_v23_swept"] == 0:
            if current_pred in ("Bull", "Bear"):
                pred_original.iloc[i] = "Neutral"
                continue
        if row["fan_collapse_v23_swept"] == 1:
            pred_original.iloc[i] = flip_direction(current_pred)
    df["pred_proto_v23_original_replay"] = pred_original

    # Lean proto replay
    pred_lean = df["pred_v22_base"].copy()
    for i, row in df.iterrows():
        current_pred = pred_lean.iloc[i]
        if row["torsion_plus_transition_swept"] == 1 and row["fan_collapse_v23_swept"] == 0:
            if current_pred in ("Bull", "Bear"):
                pred_lean.iloc[i] = "Neutral"
                continue
        if row["fan_collapse_v23_swept"] == 1:
            pred_lean.iloc[i] = flip_direction(current_pred)
    df["pred_proto_v23_lean_replay"] = pred_lean

    # Swept proto replay
    pred_swept = df["pred_v22_base"].copy()
    for i, row in df.iterrows():
        current_pred = pred_swept.iloc[i]
        if row["torque_plus_transition_swept"] == 1 and row["fan_collapse_v23_swept"] == 0:
            if current_pred in ("Bull", "Bear"):
                pred_swept.iloc[i] = "Neutral"
                continue
        if row["torsion_plus_transition_swept"] == 1 and row["fan_collapse_v23_swept"] == 0:
            if current_pred in ("Bull", "Bear"):
                pred_swept.iloc[i] = "Neutral"
                continue
        if row["fan_collapse_v23_swept"] == 1:
            pred_swept.iloc[i] = flip_direction(current_pred)
    df["pred_proto_v23_swept"] = pred_swept

    # -------------------------------------------------
    # NEW: Hybrid overlay model
    # -------------------------------------------------
    pred_hybrid = df["pred_v22_base"].copy()
    hybrid_action = []

    high_conf_flip_count = 0
    medium_conf_flip_count = 0
    pressure_neutral_count = 0
    watch_neutral_count = 0
    continuation_block_count = 0
    no_change_count = 0

    for i, row in df.iterrows():
        current_pred = pred_hybrid.iloc[i]
        action = "KEEP"

        continuation_protection = (row["fan_energy_cycle_v23"] == 1)

        high_conf_release = (
            row["collapse_plus_torque_swept"] == 1
            or row["collapse_plus_inertia_swept"] == 1
        )

        medium_conf_release = (
            row["collapse_plus_torsion_swept"] == 1
        )

        pressure_state = (
            row["torque_plus_inertia_swept"] == 1
            or row["torsion_plus_inertia_swept"] == 1
        )

        watch_state = (
            row["torque_plus_transition_swept"] == 1
            or row["torsion_plus_transition_swept"] == 1
        )

        # 1) Highest-confidence flips override continuation protection
        if high_conf_release:
            new_pred = flip_direction(current_pred)
            if new_pred != current_pred:
                pred_hybrid.iloc[i] = new_pred
                high_conf_flip_count += 1
                action = "HIGH_CONF_FLIP"
            hybrid_action.append(action)
            continue

        # 2) Medium-confidence flip only if not protected by breathing cycle
        if medium_conf_release:
            if not continuation_protection:
                new_pred = flip_direction(current_pred)
                if new_pred != current_pred:
                    pred_hybrid.iloc[i] = new_pred
                    medium_conf_flip_count += 1
                    action = "MEDIUM_CONF_FLIP"
                hybrid_action.append(action)
                continue
            else:
                continuation_block_count += 1
                action = "BLOCK_MEDIUM_FLIP"
                hybrid_action.append(action)
                continue

        # 3) Pressure states: neutralize risky calls, do not flip
        if pressure_state:
            if continuation_protection:
                continuation_block_count += 1
                action = "BLOCK_PRESSURE_NEUTRAL"
            else:
                if current_pred in ("Bull", "Bear"):
                    pred_hybrid.iloc[i] = "Neutral"
                    pressure_neutral_count += 1
                    action = "PRESSURE_NEUTRAL"
            hybrid_action.append(action)
            continue

        # 4) Watch states: neutralize weak directional calls only if not protected
        if watch_state:
            if continuation_protection:
                continuation_block_count += 1
                action = "BLOCK_WATCH_NEUTRAL"
            else:
                if current_pred in ("Bull", "Bear"):
                    pred_hybrid.iloc[i] = "Neutral"
                    watch_neutral_count += 1
                    action = "WATCH_NEUTRAL"
            hybrid_action.append(action)
            continue

        no_change_count += 1
        hybrid_action.append(action)

    df["pred_proto_v23_hybrid_overlay"] = pred_hybrid
    df["hybrid_overlay_action"] = hybrid_action

    # -------------------------------------------------
    # Stats / comparison
    # -------------------------------------------------
    stats_v22 = compute_stats(df, "pred_v22_base", truth_col)
    stats_original = compute_stats(df, "pred_proto_v23_original_replay", truth_col)
    stats_lean = compute_stats(df, "pred_proto_v23_lean_replay", truth_col)
    stats_swept = compute_stats(df, "pred_proto_v23_swept", truth_col)
    stats_hybrid = compute_stats(df, "pred_proto_v23_hybrid_overlay", truth_col)

    print_stats_block("v2_2 (base)", stats_v22)
    print_stats_block("Proto v2.3 (original replay)", stats_original)
    print_stats_block("Lean Proto v2.3 (replay)", stats_lean)
    print_stats_block("Swept Proto v2.3", stats_swept)
    print_stats_block("Hybrid Overlay Proto v2.3", stats_hybrid)

    print("\n=== HYBRID ACTION COUNTS ===")
    print(f"high_conf_flip_count: {high_conf_flip_count}")
    print(f"medium_conf_flip_count: {medium_conf_flip_count}")
    print(f"pressure_neutral_count: {pressure_neutral_count}")
    print(f"watch_neutral_count: {watch_neutral_count}")
    print(f"continuation_block_count: {continuation_block_count}")
    print(f"no_change_count: {no_change_count}")

    print("\n=== HYBRID ACTION DISTRIBUTION ===")
    print(df["hybrid_overlay_action"].value_counts().to_string())

    leaderboard = pd.DataFrame([
        {"model": "v2_2 (base)", **stats_v22},
        {"model": "Proto v2.3 (original replay)", **stats_original},
        {"model": "Lean Proto v2.3 (replay)", **stats_lean},
        {"model": "Swept Proto v2.3", **stats_swept},
        {"model": "Hybrid Overlay Proto v2.3", **stats_hybrid},
    ])

    leaderboard = leaderboard.sort_values(
        by=["directional_accuracy", "bear_precision", "bull_precision", "directional_coverage"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    print("\n=== LEADERBOARD ===")
    for i, row in leaderboard.iterrows():
        print(
            f"{i + 1}. {row['model']} | "
            f"acc={row['directional_accuracy'] * 100:.2f}% | "
            f"cov={row['directional_coverage'] * 100:.2f}% | "
            f"neu={row['neutral_rate'] * 100:.2f}% | "
            f"bull={row['bull_precision'] * 100:.2f}% | "
            f"bear={row['bear_precision'] * 100:.2f}%"
        )

    out_df = CSV_PATH.with_name("gaussian_fan_dataset_v2_2_with_proto_v23_hybrid_overlay.csv")
    out_lb = CSV_PATH.with_name("offline_proto_v23_hybrid_overlay_leaderboard.csv")

    df.to_csv(out_df, index=False)
    leaderboard.to_csv(out_lb, index=False)

    print(f"\nSaved dataset: {out_df}")
    print(f"Saved leaderboard: {out_lb}")


if __name__ == "__main__":
    main()