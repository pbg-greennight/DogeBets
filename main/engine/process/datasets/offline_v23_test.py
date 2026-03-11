

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


def main() -> None:
    df = pd.read_csv(CSV_PATH)

    # -----------------------------
    # Detect columns safely
    # -----------------------------
    truth_col = pick_column(
        df,
        ["truth", "true_direction", "actual", "label", "target"],
        "truth",
    )

    pred_col = pick_column(
        df,
        ["prediction", "pred", "trend", "model_prediction", "predicted_trend"],
        "prediction",
    )

    required = [
        "hinge_torsion",
        "fan_width_velocity",
        "outer_fan_width",
        "phase_disagreement",
        "fan_polarity_inversion",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}\n"
            f"Available columns:\n{list(df.columns)}"
        )

    # Optional columns
    has_inertia = "inertia_wall_score" in df.columns

    # -----------------------------
    # Derived features
    # -----------------------------
    df["hinge_torsion_spike"] = df["hinge_torsion"].abs() > 0.5

    df["outer_fan_width_prev"] = df["outer_fan_width"].shift(1)

    df["fan_collapse_v23"] = (
        (df["fan_width_velocity"] < -15)
        & (df["outer_fan_width"] < df["outer_fan_width_prev"])
        & (df["phase_disagreement"] == 1)
        & (df["hinge_torsion"].abs() > 0.5)
    )

    df["compression_trap_v23"] = (
        (df["fan_width_velocity"] < -10)
        & (df["outer_fan_width"] < df["outer_fan_width_prev"])
        & (df["phase_disagreement"] == 0)
        & (df["hinge_torsion"].abs() <= 0.5)
    )

    df["transition_score_v23"] = (
        (df["phase_disagreement"] == 1).astype(int)
        + (df["hinge_torsion_spike"]).astype(int)
        + (df["fan_polarity_inversion"] == 1).astype(int)
    )

    # -----------------------------
    # Proto v2.3 simulation
    # Start from existing prediction
    # -----------------------------
    proto_pred = df[pred_col].copy()

    transition_triggers = 0
    collapse_promotions = 0
    trap_blocks = 0
    inertia_neutrals = 0

    for i, row in df.iterrows():
        current_pred = proto_pred.iloc[i]

        # trap blocks reversal-like situations
        if row["compression_trap_v23"]:
            trap_blocks += 1
            continue

        # collapse promotion: flip directional calls only
        if row["fan_collapse_v23"]:
            if current_pred == "Bull":
                proto_pred.iloc[i] = "Bear"
                collapse_promotions += 1
                continue
            elif current_pred == "Bear":
                proto_pred.iloc[i] = "Bull"
                collapse_promotions += 1
                continue

        # transition score gate
        if row["transition_score_v23"] >= 2:
            transition_triggers += 1

            # inertia wall delays reversal into Neutral
            if has_inertia and pd.notna(row["inertia_wall_score"]) and row["inertia_wall_score"] > 5:
                if current_pred in ("Bull", "Bear"):
                    proto_pred.iloc[i] = "Neutral"
                    inertia_neutrals += 1
                    continue

            # otherwise allow velocity-based reversal
            if row["fan_width_velocity"] <= -15 and current_pred == "Bull":
                proto_pred.iloc[i] = "Bear"
            elif row["fan_width_velocity"] >= 15 and current_pred == "Bear":
                proto_pred.iloc[i] = "Bull"

    df["proto_v23_prediction"] = proto_pred

    # -----------------------------
    # Stats
    # -----------------------------
    directional_mask = df["proto_v23_prediction"].isin(["Bull", "Bear"])
    directional = df[directional_mask]

    total_rows = len(df)
    directional_called = len(directional)
    neutral_called = (df["proto_v23_prediction"] == "Neutral").sum()

    directional_accuracy = (
        (directional["proto_v23_prediction"] == directional[truth_col]).mean()
        if directional_called > 0 else 0.0
    )
    coverage = directional_called / total_rows if total_rows else 0.0
    neutral_rate = neutral_called / total_rows if total_rows else 0.0

    bull_pred = directional[directional["proto_v23_prediction"] == "Bull"]
    bear_pred = directional[directional["proto_v23_prediction"] == "Bear"]

    bull_precision = (
        (bull_pred[truth_col] == "Bull").mean()
        if len(bull_pred) > 0 else 0.0
    )
    bear_precision = (
        (bear_pred[truth_col] == "Bear").mean()
        if len(bear_pred) > 0 else 0.0
    )

    print("\n=== PROTO v2.3 OFFLINE TEST ===")
    print(f"Rows: {total_rows}")
    print(f"Directional Called: {directional_called}")
    print(f"Neutral Called: {neutral_called}")
    print(f"Directional Accuracy: {directional_accuracy * 100:.2f}%")
    print(f"Directional Coverage: {coverage * 100:.2f}%")
    print(f"Neutral Rate: {neutral_rate * 100:.2f}%")
    print(f"Bull Precision: {bull_precision * 100:.2f}%")
    print(f"Bear Precision: {bear_precision * 100:.2f}%")

    print("\n=== PROTO v2.3 FEATURE COUNTS ===")
    print(f"transition_score>=2 rows: {(df['transition_score_v23'] >= 2).sum()}")
    print(f"fan_collapse_v23 rows: {df['fan_collapse_v23'].sum()}")
    print(f"compression_trap_v23 rows: {df['compression_trap_v23'].sum()}")
    print(f"transition triggers used: {transition_triggers}")
    print(f"collapse promotions: {collapse_promotions}")
    print(f"trap blocks: {trap_blocks}")
    print(f"inertia neutralizations: {inertia_neutrals}")

    out_path = CSV_PATH.with_name("gaussian_fan_dataset_v2_2_with_proto_v23.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()