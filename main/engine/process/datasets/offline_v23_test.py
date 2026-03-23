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
    neutral_called = int((df[pred_col] == "Neutral").sum())

    directional_accuracy = (
        float((directional[pred_col] == directional[truth_col]).mean())
        if directional_called > 0 else 0.0
    )
    coverage = directional_called / total_rows if total_rows else 0.0
    neutral_rate = neutral_called / total_rows if total_rows else 0.0

    bull_pred = directional[directional[pred_col] == "Bull"]
    bear_pred = directional[directional[pred_col] == "Bear"]

    bull_precision = (
        float((bull_pred[truth_col] == "Bull").mean())
        if len(bull_pred) > 0 else 0.0
    )
    bear_precision = (
        float((bear_pred[truth_col] == "Bear").mean())
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


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else 0.0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

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

    df["outer_fan_width_prev"] = df["outer_fan_width"].shift(1)
    df["hinge_torsion_spike"] = (df["hinge_torsion"].abs() > 0.5).astype(int)

    df["transition_score_v23"] = (
        (df["phase_disagreement"] == 1).astype(int)
        + (df["hinge_torsion_spike"] == 1).astype(int)
        + (df["fan_polarity_inversion"] == 1).astype(int)
    )
    df["transition_gate_v23"] = (df["transition_score_v23"] >= 2).astype(int)

    df["fan_collapse_v23_swept"] = (
        (df["fan_width_velocity"] < -15.0)
        & (df["hinge_torsion"].abs() > 0.40)
        & (df["phase_disagreement"] == 1)
        & (df["outer_fan_width"] < df["outer_fan_width_prev"])
    ).astype(int)

    df["inertia_wall_active_swept"] = (
        (df["slow_retention"] > 0.75)
        & (df["hinge_torsion"].abs() > 0.25)
        & (df["phase_disagreement"] == 1)
    ).astype(int)

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

    df["fan_energy_t3"] = df["fan_energy_total"].shift(3)
    df["fan_energy_t2"] = df["fan_energy_total"].shift(2)
    df["fan_energy_t1"] = df["fan_energy_total"].shift(1)
    df["fan_energy_t0"] = df["fan_energy_total"]

    df["fan_energy_cycle_v23"] = (
        (df["fan_energy_t3"] > df["fan_energy_t2"])
        & (df["fan_energy_t1"] > df["fan_energy_t2"])
        & (df["fan_energy_t0"] < df["fan_energy_t1"])
    ).astype(int)

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

    # warning features
    df["collapse_warning_plus_inertia_v23"] = (
        (df["fan_width_velocity"] < -8.0)
        & (df["hinge_torsion"].abs() > 0.35)
        & (df["slow_retention"] > 0.90)
        & (df["transition_score_v23"] >= 1)
    ).astype(int)

    df["torsion_transition_warning_v23"] = (
        (df["torsion_spring_score_v23"] > 0.18)
        & (df["hinge_torsion"].abs() > 0.30)
        & (df["transition_score_v23"] >= 1)
    ).astype(int)

    df["inertia_warning_v23"] = (
        (df["slow_retention"] > 0.90)
        & (df["hinge_torsion"].abs() > 0.20)
        & (df["transition_score_v23"] >= 1)
    ).astype(int)

    df["energy_cycle_warning_v23"] = (
        (df["fan_energy_t3"] > df["fan_energy_t2"])
        & (df["fan_energy_t1"] >= df["fan_energy_t2"])
    ).astype(int)

    return df


def apply_rule_b_custom(
    df: pd.DataFrame,
    base_col: str,
    slow_retention_threshold: float,
    hinge_threshold: float,
    transition_min: int,
    bear_only: bool,
):
    pred = df[base_col].copy()
    actions = []

    count = 0
    no_change = 0

    for i, row in df.iterrows():
        current_pred = pred.iloc[i]
        action = "KEEP"

        if current_pred not in ("Bull", "Bear"):
            no_change += 1
            actions.append(action)
            continue

        side_ok = (current_pred == "Bear") if bear_only else (current_pred in ("Bull", "Bear"))

        rule_b = (
            side_ok
            and (row["slow_retention"] > slow_retention_threshold)
            and (abs(row["hinge_torsion"]) > hinge_threshold)
            and (row["transition_score_v23"] >= transition_min)
        )

        if rule_b:
            pred.iloc[i] = "Neutral"
            count += 1
            action = "RULE_B_CUSTOM_NEUTRAL"
        else:
            no_change += 1

        actions.append(action)

    return pred, pd.Series(actions, index=df.index), {
        "count": count,
        "no_change": no_change,
    }


def print_leaderboard(rows: list[dict], title: str = "=== LEADERBOARD ===") -> pd.DataFrame:
    leaderboard = pd.DataFrame(rows).sort_values(
        by=["directional_accuracy", "bear_precision", "bull_precision", "directional_coverage"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    print(f"\n{title}")
    for i, row in leaderboard.iterrows():
        print(
            f"{i + 1}. {row['model']} | "
            f"acc={row['directional_accuracy'] * 100:.2f}% | "
            f"cov={row['directional_coverage'] * 100:.2f}% | "
            f"neu={row['neutral_rate'] * 100:.2f}% | "
            f"bull={row['bull_precision'] * 100:.2f}% | "
            f"bear={row['bear_precision'] * 100:.2f}%"
        )
    return leaderboard


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

    df = build_features(df)
    df["pred_v22_base"] = df[base_pred_col].copy()

    latest = df.iloc[-1]
    print("\n=== CURRENT EPOCH FEATURE SNAPSHOT (LAST ROW) ===")
    print(f"Base v2_2 prediction: {latest['pred_v22_base']}")
    print(f"inertia_warning_v23: {int(latest['inertia_warning_v23'])}")
    print(f"slow_retention: {float(latest['slow_retention']):.4f}")
    print(f"|hinge_torsion|: {abs(float(latest['hinge_torsion'])):.4f}")
    print(f"transition_score_v23: {int(latest['transition_score_v23'])}")

    # Baseline
    stats_v22 = compute_stats(df, "pred_v22_base", truth_col)
    print_stats_block("v2_2 (base)", stats_v22)

    # Current working RuleB comparison point from previous result
    pred_ruleb_current, action_ruleb_current, counters_ruleb_current = apply_rule_b_custom(
        df=df,
        base_col="pred_v22_base",
        slow_retention_threshold=0.90,
        hinge_threshold=0.20,
        transition_min=1,
        bear_only=False,
    )
    df["pred_ruleb_current"] = pred_ruleb_current
    stats_ruleb_current = compute_stats(df, "pred_ruleb_current", truth_col)
    print_stats_block("Current RuleB reference (Bull+Bear, 0.90 / 0.20 / t>=1)", stats_ruleb_current)

    # Sweep
    slow_grid = [0.85, 0.90, 0.95, 1.00]
    hinge_grid = [0.15, 0.20, 0.25, 0.30]
    transition_grid = [1, 2]
    side_grid = [False, True]  # False = Bull+Bear, True = Bear-only

    sweep_rows = []

    for slow_th in slow_grid:
        for hinge_th in hinge_grid:
            for tmin in transition_grid:
                for bear_only in side_grid:
                    pred_col_name = (
                        f"pred_ruleb_s{str(slow_th).replace('.', '_')}"
                        f"_h{str(hinge_th).replace('.', '_')}"
                        f"_t{tmin}"
                        f"_{'bear' if bear_only else 'both'}"
                    )

                    pred, action, counters = apply_rule_b_custom(
                        df=df,
                        base_col="pred_v22_base",
                        slow_retention_threshold=slow_th,
                        hinge_threshold=hinge_th,
                        transition_min=tmin,
                        bear_only=bear_only,
                    )

                    df[pred_col_name] = pred
                    stats = compute_stats(df, pred_col_name, truth_col)

                    sweep_rows.append({
                        "model": pred_col_name,
                        "slow_retention_threshold": slow_th,
                        "hinge_threshold": hinge_th,
                        "transition_min": tmin,
                        "bear_only": bear_only,
                        "trigger_count": counters["count"],
                        **stats,
                    })

    sweep_df = pd.DataFrame(sweep_rows).sort_values(
        by=["directional_accuracy", "bear_precision", "bull_precision", "directional_coverage"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    print("\n=== TOP 15 RULEB SWEEP RESULTS ===")
    for _, row in sweep_df.head(15).iterrows():
        side = "Bear-only" if row["bear_only"] else "Bull+Bear"
        print(
            f"{side} | slow>{row['slow_retention_threshold']:.2f} | "
            f"|hinge|>{row['hinge_threshold']:.2f} | "
            f"t>={int(row['transition_min'])} | "
            f"triggers={int(row['trigger_count'])} | "
            f"acc={row['directional_accuracy'] * 100:.2f}% | "
            f"cov={row['directional_coverage'] * 100:.2f}% | "
            f"neu={row['neutral_rate'] * 100:.2f}% | "
            f"bull={row['bull_precision'] * 100:.2f}% | "
            f"bear={row['bear_precision'] * 100:.2f}%"
        )

    best = sweep_df.iloc[0]
    best_side = "Bear-only" if best["bear_only"] else "Bull+Bear"

    print("\n=== BEST RULEB SWEEP CONFIG ===")
    print(f"side: {best_side}")
    print(f"slow_retention_threshold: {best['slow_retention_threshold']:.2f}")
    print(f"hinge_threshold: {best['hinge_threshold']:.2f}")
    print(f"transition_min: {int(best['transition_min'])}")
    print(f"trigger_count: {int(best['trigger_count'])}")
    print(f"directional_accuracy: {best['directional_accuracy'] * 100:.2f}%")
    print(f"directional_coverage: {best['directional_coverage'] * 100:.2f}%")
    print(f"neutral_rate: {best['neutral_rate'] * 100:.2f}%")
    print(f"bull_precision: {best['bull_precision'] * 100:.2f}%")
    print(f"bear_precision: {best['bear_precision'] * 100:.2f}%")

    leaderboard = print_leaderboard([
        {"model": "v2_2 (base)", **stats_v22},
        {"model": "Current RuleB reference", **stats_ruleb_current},
        {
            "model": f"Best RuleB sweep ({best_side}, s>{best['slow_retention_threshold']:.2f}, "
                     f"h>{best['hinge_threshold']:.2f}, t>={int(best['transition_min'])})",
            "rows": int(best["rows"]),
            "directional_called": int(best["directional_called"]),
            "neutral_called": int(best["neutral_called"]),
            "directional_accuracy": float(best["directional_accuracy"]),
            "directional_coverage": float(best["directional_coverage"]),
            "neutral_rate": float(best["neutral_rate"]),
            "bull_precision": float(best["bull_precision"]),
            "bear_precision": float(best["bear_precision"]),
        },
    ])

    out_dataset = CSV_PATH.with_name("gaussian_fan_dataset_v2_2_with_ruleb_sweep.csv")
    out_sweep = CSV_PATH.with_name("offline_ruleb_threshold_sweep.csv")
    out_leaderboard = CSV_PATH.with_name("offline_ruleb_threshold_sweep_leaderboard.csv")

    df.to_csv(out_dataset, index=False)
    sweep_df.to_csv(out_sweep, index=False)
    leaderboard.to_csv(out_leaderboard, index=False)

    print(f"\nSaved dataset: {out_dataset}")
    print(f"Saved sweep table: {out_sweep}")
    print(f"Saved leaderboard: {out_leaderboard}")


if __name__ == "__main__":
    main()