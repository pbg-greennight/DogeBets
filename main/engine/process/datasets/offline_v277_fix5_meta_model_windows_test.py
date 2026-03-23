import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

try:
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False
    DecisionTreeClassifier = None

CSV_PATH = Path(__file__).with_name("gaussian_fan_dataset_v2_2.csv")
LOOKBACK_WINDOWS = [30, 50, 70, 90]

TRUTH_CANDIDATES = ["truth", "true_direction", "actual", "label", "target"]
PRED_CANDIDATES = ["pred_v22_base", "prediction", "pred", "trend", "model_prediction", "predicted_trend"]
EPOCH_CANDIDATES = ["epoch", "round", "round_id", "epoch_id", "id"]

ACTIONS = ["base", "corea", "fallback", "winner"]
ACTION_PREFERENCE = {"base": 0, "corea": 1, "fallback": 2, "winner": 3}
META_FEATURES = [
    "slow_retention",
    "abs_hinge",
    "transition_score_v23",
    "phase_disagree_count_3",
    "fs_disagree_count_3",
    "fast_alt_3epoch",
    "osc_core_a",
    "core_window_pm1",
    "pm1_centered_soft",
    "pm1_hinge_018",
]


def pick_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label} column. Tried {candidates}. Available: {list(df.columns)}")


def try_pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def sign_series(series: pd.Series) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] != 1:
            raise ValueError("sign_series expected 1-D input")
        series = series.iloc[:, 0]
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return ((s > 0).astype(int) - (s < 0).astype(int)).astype(int)



def get_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def get_bool_mask(df: pd.DataFrame, col: str, predicate) -> pd.Series:
    s = get_series(df, col)
    return predicate(s)

def ensure_unique_prediction_truth(df: pd.DataFrame, pred_name: str = "pred_v22_base", truth_name: str = "truth") -> pd.DataFrame:
    df = df.copy()
    if pred_name in df.columns:
        obj = df[pred_name]
        if isinstance(obj, pd.DataFrame):
            first = obj.iloc[:, 0].copy()
            df = df.drop(columns=[pred_name])
            df[pred_name] = first
    if truth_name in df.columns:
        obj = df[truth_name]
        if isinstance(obj, pd.DataFrame):
            first = obj.iloc[:, 0].copy()
            df = df.drop(columns=[truth_name])
            df[truth_name] = first
    return df

def compute_stats(df: pd.DataFrame, pred_col: str, truth_col: str) -> dict:
    total_rows = len(df)
    pred = get_series(df, pred_col)
    truth = get_series(df, truth_col)
    directional_mask = pred.isin(["Bull", "Bear"])
    directional_pred = pred[directional_mask]
    directional_truth = truth[directional_mask]
    directional_called = int(len(directional_pred))
    neutral_called = int((pred == "Neutral").sum())
    directional_accuracy = float((directional_pred == directional_truth).mean()) if directional_called else 0.0
    coverage = directional_called / total_rows if total_rows else 0.0
    neutral_rate = neutral_called / total_rows if total_rows else 0.0
    bull_mask = (directional_pred == "Bull")
    bear_mask = (directional_pred == "Bear")
    bull_precision = float((directional_truth[bull_mask] == "Bull").mean()) if bull_mask.sum() else 0.0
    bear_precision = float((directional_truth[bear_mask] == "Bear").mean()) if bear_mask.sum() else 0.0
    truth_bull = (truth == "Bull")
    truth_bear = (truth == "Bear")
    bull_recall = float(((pred == "Bull") & truth_bull).sum() / truth_bull.sum()) if truth_bull.sum() else 0.0
    bear_recall = float(((pred == "Bear") & truth_bear).sum() / truth_bear.sum()) if truth_bear.sum() else 0.0
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
        "hinge_torsion", "fan_width_velocity", "outer_fan_width", "phase_disagreement",
        "fan_polarity_inversion", "slow_retention",
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
    new_cols["outer_fan_width_prev"] = df["outer_fan_width"].shift(1)
    new_cols["hinge_torsion_spike"] = (df["hinge_torsion"].abs() > 0.5).astype(int)
    new_cols["transition_score_v23"] = (
        (df["phase_disagreement"] == 1).astype(int)
        + new_cols["hinge_torsion_spike"]
        + (df["fan_polarity_inversion"] == 1).astype(int)
    )
    new_cols["fan_collapse_v23_swept"] = (
        (df["fan_width_velocity"] < -15.0)
        & (df["hinge_torsion"].abs() > 0.40)
        & (df["phase_disagreement"] == 1)
        & (df["outer_fan_width"] < new_cols["outer_fan_width_prev"])
    ).astype(int)
    new_cols["inertia_wall_active_swept"] = (
        (df["slow_retention"] > 0.75)
        & (df["hinge_torsion"].abs() > 0.25)
        & (df["phase_disagreement"] == 1)
    ).astype(int)
    new_cols["ts_d1"] = df["slope_g8"] - df["slope_g23"]
    new_cols["ts_d2"] = df["slope_g23"] - df["slope_g38"]
    new_cols["ts_d3"] = df["slope_g38"] - df["slope_g53"]
    new_cols["ts_d4"] = df["slope_g53"] - df["slope_g68"]
    new_cols["ts_d5"] = df["slope_g68"] - df["slope_g83"]
    new_cols["torsion_spring_score_v23"] = (
        1.0 * (new_cols["ts_d1"] - new_cols["ts_d2"]).abs()
        + 1.5 * (new_cols["ts_d2"] - new_cols["ts_d3"]).abs()
        + 1.5 * (new_cols["ts_d3"] - new_cols["ts_d4"]).abs()
        + 1.0 * (new_cols["ts_d4"] - new_cols["ts_d5"]).abs()
    )
    new_cols["fan_energy_t3"] = fan_energy_total.shift(3)
    new_cols["fan_energy_t2"] = fan_energy_total.shift(2)
    new_cols["fan_energy_t1"] = fan_energy_total.shift(1)
    new_cols["fan_energy_t0"] = fan_energy_total
    new_cols["torque_fast_motion"] = df["slope_g8"].abs() + df["slope_g23"].abs()
    new_cols["torque_slow_motion"] = df["slope_g68"].abs() + df["slope_g83"].abs()
    new_cols["torque_imbalance_ratio"] = new_cols["torque_fast_motion"] / (new_cols["torque_slow_motion"] + 1e-6)
    new_cols["torque_fast_signed"] = df["slope_g8"] + df["slope_g23"]
    new_cols["torque_slow_signed"] = df["slope_g68"] + df["slope_g83"]
    new_cols["torque_opposition"] = ((new_cols["torque_fast_signed"] * new_cols["torque_slow_signed"]) < 0).astype(int)
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def build_ruleb_warning(df: pd.DataFrame, slow_threshold: float = 0.95, hinge_threshold: float = 0.15, transition_min: int = 1) -> pd.Series:
    return (
        (df["slow_retention"] > slow_threshold)
        & (df["hinge_torsion"].abs() > hinge_threshold)
        & (df["transition_score_v23"] >= transition_min)
    ).astype(int)


def build_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    cols: dict[str, pd.Series] = {}
    fast_signed = df["slope_g8"] + df["slope_g23"]
    slow_signed = df["slope_g68"] + df["slope_g83"]
    cols["fast_sign_now"] = sign_series(fast_signed)
    cols["fast_sign_t1"] = cols["fast_sign_now"].shift(1).fillna(0).astype(int)
    cols["fast_sign_t2"] = cols["fast_sign_now"].shift(2).fillna(0).astype(int)
    cols["slow_sign_now"] = sign_series(slow_signed)
    cols["fast_alt_3epoch"] = (
        (cols["fast_sign_now"] != 0)
        & (cols["fast_sign_t1"] != 0)
        & (cols["fast_sign_t2"] != 0)
        & (cols["fast_sign_now"] == cols["fast_sign_t2"])
        & (cols["fast_sign_now"] != cols["fast_sign_t1"])
    ).astype(int)
    cols["fs_disagree_now"] = ((cols["fast_sign_now"] != 0) & (sign_series(slow_signed) != 0) & (cols["fast_sign_now"] != sign_series(slow_signed))).astype(int)
    cols["fs_disagree_count_3"] = cols["fs_disagree_now"].rolling(3, min_periods=1).sum().astype(int)
    cols["phase_disagree_count_3"] = df["phase_disagreement"].fillna(0).rolling(3, min_periods=1).sum().astype(int)
    cols["abs_hinge"] = df["hinge_torsion"].abs()
    osc_core_count = (
        (cols["phase_disagree_count_3"] >= 2).astype(int)
        + (cols["fs_disagree_count_3"] >= 2).astype(int)
        + cols["fast_alt_3epoch"].astype(int)
    )
    cols["osc_core_count"] = osc_core_count.astype(int)
    cols["osc_core_a"] = (osc_core_count >= 2).astype(int)
    osc = cols["osc_core_a"].fillna(0).astype(int)
    cols["core_window_pm1"] = (
        osc | osc.shift(1).fillna(0).astype(int) | osc.shift(-1).fillna(0).astype(int)
    ).astype(int)
    center_pf = ((cols["phase_disagree_count_3"] >= 1) | (cols["fs_disagree_count_3"] >= 1)).astype(int)
    cols["center_pf"] = center_pf
    cols["pm1_hinge_018"] = (
        (cols["core_window_pm1"] == 1)
        & (center_pf == 1)
        & (cols["abs_hinge"] >= 0.18)
    ).astype(int)
    cols["pm1_centered_soft"] = (
        (cols["core_window_pm1"] == 1)
        & (center_pf == 1)
        & ((cols["abs_hinge"] >= 0.22) | (cols["fast_alt_3epoch"] == 1))
    ).astype(int)
    return pd.concat([df, pd.DataFrame(cols, index=df.index)], axis=1)


def build_locked_predictions(df: pd.DataFrame, base_col: str, truth_col: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    bear_ruleb = (get_series(df, base_col) == "Bear") & (pd.to_numeric(get_series(df, "ruleb_raw_warning"), errors="coerce").fillna(0).astype(int) == 1)
    masks = {
        "corea": bear_ruleb & (df["osc_core_a"] == 1),
        "fallback": bear_ruleb & (df["pm1_centered_soft"] == 1),
        "winner": bear_ruleb & (df["pm1_hinge_018"] == 1),
    }
    out[base_col] = df[base_col]
    for name, mask in masks.items():
        out[f"trigger_{name}"] = mask.astype(int)
        out[f"pred_{name}"] = df[base_col].where(~mask, "Neutral")
    return out




def get_row_scalar(row: pd.Series, key: str):
    val = row[key]
    if isinstance(val, pd.Series):
        return val.iloc[0]
    return val

def action_prediction(row: pd.Series, action: str) -> str:
    if action == "base":
        return get_row_scalar(row, "pred_v22_base")
    return get_row_scalar(row, f"pred_{action}")


def action_trigger(row: pd.Series, action: str) -> int:
    if action == "base":
        return 0
    return int(get_row_scalar(row, f"trigger_{action}"))


def action_utility(row: pd.Series, action: str) -> float:
    pred = action_prediction(row, action)
    truth = get_row_scalar(row, "truth")
    base_pred = get_row_scalar(row, "pred_v22_base")
    if pred in ("Bull", "Bear") and pred == truth:
        return 2.0
    if pred == "Neutral":
        return 1.0 if (base_pred in ("Bull", "Bear") and base_pred != truth) else 0.0
    if pred in ("Bull", "Bear") and pred != truth:
        return -1.0
    return 0.0


def best_action_label(row: pd.Series) -> str:
    scores = {a: action_utility(row, a) for a in ACTIONS}
    best_score = max(scores.values())
    best_actions = [a for a, s in scores.items() if s == best_score]
    # conservative tiebreak among overrides; base first if equally good
    best_actions.sort(key=lambda a: ACTION_PREFERENCE[a])
    return best_actions[0]


def fit_and_predict_action(hist: pd.DataFrame, current: pd.Series, lookback: int) -> tuple[str, str]:
    pred_bear = get_series(hist, "pred_v22_base") == "Bear"
    ruleb_on = pd.to_numeric(get_series(hist, "ruleb_raw_warning"), errors="coerce").fillna(0).astype(int) == 1
    eligible_hist = hist[pred_bear & ruleb_on].copy()
    if len(eligible_hist) < 8:
        return "base", "meta_not_enough_neighbors"

    eligible_hist["label_action"] = eligible_hist.apply(best_action_label, axis=1)
    label_counts = eligible_hist["label_action"].value_counts()
    if label_counts.empty:
        return "base", "meta_empty_labels"

    X_train = eligible_hist[META_FEATURES].copy()
    X_train = X_train.fillna(0.0)
    y_train = eligible_hist["label_action"].copy()

    available = ["base"]
    for a in ["corea", "fallback", "winner"]:
        if action_trigger(current, a) == 1:
            available.append(a)

    if len(set(y_train)) == 1:
        chosen = y_train.iloc[0]
        if chosen not in available:
            chosen = available[0]
        return chosen, f"single_label_{chosen}|lb={lookback}"

    x_cur = current[META_FEATURES].fillna(0.0).to_frame().T

    if SKLEARN_OK:
        clf = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=max(2, min(6, len(X_train) // 6)),
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_cur)[0]
            class_probs = dict(zip(clf.classes_, probs))
            avail_probs = {a: class_probs.get(a, -1.0) for a in available}
            chosen = max(avail_probs.items(), key=lambda kv: kv[1])[0]
            return chosen, f"tree_choice_{chosen}|lb={lookback}"
        chosen = clf.predict(x_cur)[0]
        if chosen not in available:
            chosen = available[0]
        return str(chosen), f"tree_pred_{chosen}|lb={lookback}"

    # Fallback: choose most common label among available actions.
    counts = Counter(a for a in y_train if a in available)
    if not counts:
        return "base", f"fallback_base|lb={lookback}"
    chosen = counts.most_common(1)[0][0]
    return chosen, f"freq_choice_{chosen}|lb={lookback}"


def build_meta_predictions(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    pred = []
    policy = []
    reason = []
    for i, row in df.iterrows():
        base_pred = get_row_scalar(row, "pred_v22_base")
        if i < lookback:
            pred.append(base_pred)
            policy.append("base")
            reason.append(f"warmup_lt_{lookback}")
            continue
        if base_pred == "Neutral":
            pred.append(base_pred)
            policy.append("base")
            reason.append("base_pred=Neutral")
            continue
        if base_pred == "Bull":
            pred.append(base_pred)
            policy.append("base")
            reason.append("base_pred=Bull")
            continue
        if int(row["ruleb_raw_warning"]) != 1:
            pred.append(base_pred)
            policy.append("base")
            reason.append("ruleb_off")
            continue
        hist = df.iloc[max(0, i - lookback): i].copy()
        chosen, why = fit_and_predict_action(hist, row, lookback)
        pred.append(action_prediction(row, chosen))
        policy.append(chosen)
        reason.append(why)
    return pd.DataFrame({
        f"pred_meta_{lookback}": pred,
        f"policy_meta_{lookback}": policy,
        f"reason_meta_{lookback}": reason,
    }, index=df.index)


def trigger_quality(df: pd.DataFrame, pred_col: str) -> dict:
    pred = get_series(df, pred_col)
    base = get_series(df, "pred_v22_base")
    truth = get_series(df, "truth")
    trigger = (pred == "Neutral") & (base == "Bear")
    base_wrong = (base == "Bear") & (truth != "Bear")
    base_right = (base == "Bear") & (truth == "Bear")
    return {
        "triggers": int(trigger.sum()),
        "good": int((trigger & base_wrong).sum()),
        "bad": int((trigger & base_right).sum()),
    }


def capture_wrong_bear(df: pd.DataFrame, pred_col: str) -> float:
    base = get_series(df, "pred_v22_base")
    ruleb = pd.to_numeric(get_series(df, "ruleb_raw_warning"), errors="coerce").fillna(0).astype(int)
    truth = get_series(df, "truth")
    universe = ((base == "Bear") & (ruleb == 1) & (truth == "Bull"))
    if int(universe.sum()) == 0:
        return 0.0
    pred_s = get_series(df, pred_col).astype(str)
    capture = ((pred_s == "Neutral") & universe).sum()
    return float(capture / universe.sum())


def print_leaderboard(rows: list[dict]) -> pd.DataFrame:
    board = pd.DataFrame(rows).sort_values(
        by=["directional_accuracy", "bear_precision", "directional_coverage"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    print("\n=== V277 META-MODEL LEADERBOARD ===")
    for _, row in board.iterrows():
        print(
            f"{row['model']} | {row['label']} | acc={row['directional_accuracy'] * 100:.2f}% | "
            f"cov={row['directional_coverage'] * 100:.2f}% | bear={row['bear_precision'] * 100:.2f}% | "
            f"trig={int(row['triggers'])} | good={int(row['good'])} | bad={int(row['bad'])} | "
            f"capture={row['capture'] * 100:.2f}%"
        )
    return board


def chunk_table(df: pd.DataFrame, pred_cols: dict[str, str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    chunk_size = math.ceil(n / 4)
    for chunk_idx in range(4):
        s = chunk_idx * chunk_size
        e = min(n, (chunk_idx + 1) * chunk_size)
        if s >= e:
            continue
        chunk = df.iloc[s:e]
        for model, pred_col in pred_cols.items():
            stats = compute_stats(chunk, pred_col, "truth")
            rows.append({
                "chunk": chunk_idx + 1,
                "model": model,
                "rows": len(chunk),
                "trig": int(((chunk[pred_col] == "Neutral") & (chunk["pred_v22_base"] == "Bear")).sum()),
                "acc": stats["directional_accuracy"],
                "cov": stats["directional_coverage"],
                "bear": stats["bear_precision"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    truth_col = pick_column(df, TRUTH_CANDIDATES, "truth")
    base_col = pick_column(df, PRED_CANDIDATES, "prediction")
    epoch_col = try_pick_column(df, EPOCH_CANDIDATES)

    df = build_features(df)
    df = build_sequence_features(df)
    df = df.rename(columns={truth_col: "truth"})
    if base_col != "pred_v22_base":
        df["pred_v22_base"] = get_series(df, base_col).copy()
    df = ensure_unique_prediction_truth(df, "pred_v22_base", "truth")
    df["ruleb_raw_warning"] = build_ruleb_warning(df)

    static = build_locked_predictions(df, "pred_v22_base", "truth")
    df = pd.concat([df, static], axis=1)

    latest = df.iloc[-1]
    print("\n=== CURRENT EPOCH FEATURE SNAPSHOT (LAST ROW) ===")
    base_latest = latest['pred_v22_base']
    if isinstance(base_latest, pd.Series):
        base_latest = base_latest.iloc[0]
    print(f"Base v2_2 prediction: {base_latest}")
    print(f"slow_retention: {float(latest['slow_retention']):.4f}")
    print(f"|hinge_torsion|: {abs(float(latest['hinge_torsion'])):.4f}")
    print(f"transition_score_v23: {int(latest['transition_score_v23'])}")
    print(f"ruleb_raw_warning: {int(latest['ruleb_raw_warning'])}")
    print(f"phase_disagree_count_3: {int(latest['phase_disagree_count_3'])}")
    print(f"fs_disagree_count_3: {int(latest['fs_disagree_count_3'])}")
    print(f"fast_alt_3epoch: {int(latest['fast_alt_3epoch'])}")
    print(f"pm1_centered_soft: {int(latest['pm1_centered_soft'])}")
    print(f"pm1_hinge_018: {int(latest['pm1_hinge_018'])}")

    base_stats = compute_stats(df, "pred_v22_base", "truth")
    corea_stats = compute_stats(df, "pred_corea", "truth")
    fallback_stats = compute_stats(df, "pred_fallback", "truth")
    winner_stats = compute_stats(df, "pred_winner", "truth")
    print_stats_block("v2_2 (base)", base_stats)
    print_stats_block("Safe reference: Bear-only RuleB + Core A", corea_stats)
    print_stats_block("Fallback: Bear-only RuleB + Center Soft", fallback_stats)
    print_stats_block("Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18", winner_stats)

    for lb in LOOKBACK_WINDOWS:
        meta = build_meta_predictions(df, lb)
        df = pd.concat([df, meta], axis=1)
        label = f"Adaptive v277: walk-forward tree meta-model ({lb}-epoch lookback)"
        stats = compute_stats(df, f"pred_meta_{lb}", "truth")
        print_stats_block(label, stats)

    leaderboard_rows = [
        {"model": "pred_v22_base", "label": "v2_2 (base)", **base_stats, **trigger_quality(df, "pred_v22_base"), "capture": capture_wrong_bear(df, "pred_v22_base")},
        {"model": "bear_corea", "label": "Safe reference: Bear-only RuleB + Core A", **corea_stats, **trigger_quality(df, "pred_corea"), "capture": capture_wrong_bear(df, "pred_corea")},
        {"model": "bear_pm1_soft", "label": "Fallback: Bear-only RuleB + Center Soft", **fallback_stats, **trigger_quality(df, "pred_fallback"), "capture": capture_wrong_bear(df, "pred_fallback")},
        {"model": "bear_pm1_h018", "label": "Winner: Bear-only RuleB + Window +/-1 + |hinge|>=0.18", **winner_stats, **trigger_quality(df, "pred_winner"), "capture": capture_wrong_bear(df, "pred_winner")},
    ]
    for lb in LOOKBACK_WINDOWS:
        stats = compute_stats(df, f"pred_meta_{lb}", "truth")
        leaderboard_rows.append({
            "model": f"adaptive_meta_{lb}",
            "label": f"Adaptive v277 ({lb}-epoch lookback)",
            **stats,
            **trigger_quality(df, f"pred_meta_{lb}"),
            "capture": capture_wrong_bear(df, f"pred_meta_{lb}"),
        })
    leaderboard = print_leaderboard(leaderboard_rows)

    print("\n=== V277 TRIGGER QUALITY SUMMARY ===")
    for key in ["pred_corea", "pred_fallback", "pred_winner"] + [f"pred_meta_{lb}" for lb in LOOKBACK_WINDOWS]:
        q = trigger_quality(df, key)
        total = q['good'] + q['bad']
        good_rate = q['good'] / total if total else 0.0
        print(f"{key} | triggers={q['triggers']} | good={q['good']} | bad={q['bad']} | good_rate={good_rate * 100:.2f}%")

    print("\n=== ADAPTIVE POLICY USAGE SUMMARY ===")
    policy_rows = []
    for lb in LOOKBACK_WINDOWS:
        col = f"policy_meta_{lb}"
        grp = df.groupby(col, dropna=False)
        for pol, g in grp:
            policy_rows.append({
                "lookback": lb,
                "policy": pol,
                "rows": len(g),
                "bear_rows": int((get_series(g, 'pred_v22_base') == 'Bear').sum()),
                "triggers": int(((get_series(g, f'pred_meta_{lb}') == 'Neutral') & (get_series(g, 'pred_v22_base') == 'Bear')).sum()),
                "good": int((((get_series(g, f'pred_meta_{lb}') == 'Neutral') & (get_series(g, 'pred_v22_base') == 'Bear') & (get_series(g, 'truth') == 'Bull'))).sum()),
                "bad": int((((get_series(g, f'pred_meta_{lb}') == 'Neutral') & (get_series(g, 'pred_v22_base') == 'Bear') & (get_series(g, 'truth') == 'Bear'))).sum()),
            })
            print(f"lookback={lb} | policy={pol} | rows={len(g)} | bear_rows={int((get_series(g, 'pred_v22_base') == 'Bear').sum())} | triggers={int(((get_series(g, f'pred_meta_{lb}') == 'Neutral') & (get_series(g, 'pred_v22_base') == 'Bear')).sum())}")
    policy_usage = pd.DataFrame(policy_rows)

    print("\n=== ADAPTIVE REASON SUMMARY ===")
    reason_rows = []
    for lb in LOOKBACK_WINDOWS:
        col = f"reason_meta_{lb}"
        grp = df.groupby(col, dropna=False)
        top = grp.size().sort_values(ascending=False).head(8)
        for reason, count in top.items():
            g = grp.get_group(reason)
            reason_rows.append({
                "lookback": lb,
                "reason": reason,
                "rows": len(g),
                "triggers": int(((get_series(g, f'pred_meta_{lb}') == 'Neutral') & (get_series(g, 'pred_v22_base') == 'Bear')).sum()),
            })
            print(f"lookback={lb} | reason={reason} | rows={len(g)} | triggers={int(((get_series(g, f'pred_meta_{lb}') == 'Neutral') & (get_series(g, 'pred_v22_base') == 'Bear')).sum())}")
    reason_usage = pd.DataFrame(reason_rows)

    pred_cols = {
        "base": "pred_v22_base",
        "winner": "pred_winner",
        "fallback": "pred_fallback",
    }
    for lb in LOOKBACK_WINDOWS:
        pred_cols[f"meta_{lb}"] = f"pred_meta_{lb}"
    chunk = chunk_table(df, pred_cols)
    print("\n=== ADAPTIVE VS STATIC CHUNK STABILITY ===")
    for _, row in chunk.iterrows():
        print(f"chunk={int(row['chunk'])} | model={row['model']} | rows={int(row['rows'])} | trig={int(row['trig'])} | acc={row['acc'] * 100:.2f}% | cov={row['cov'] * 100:.2f}% | bear={row['bear'] * 100:.2f}%")

    # Save outputs
    base_out = CSV_PATH.with_name("gaussian_fan_dataset_v2_2_with_v277_meta_windows_diagnostics.csv")
    decision_audit = df.copy()
    leaderboard.to_csv(CSV_PATH.with_name("offline_v277_rollout_summary.csv"), index=False)
    policy_usage.to_csv(CSV_PATH.with_name("offline_v277_policy_usage_summary.csv"), index=False)
    reason_usage.to_csv(CSV_PATH.with_name("offline_v277_reason_summary.csv"), index=False)
    chunk.to_csv(CSV_PATH.with_name("offline_v277_chunk_stability.csv"), index=False)
    decision_audit.to_csv(CSV_PATH.with_name("offline_v277_decision_audit.csv"), index=False)
    pd.DataFrame(leaderboard_rows).to_csv(CSV_PATH.with_name("offline_v277_model_comparison.csv"), index=False)
    base_out.write_text("")
    df.to_csv(base_out, index=False)
    print(f"\nSaved dataset: {base_out}")
    print(f"Saved decision audit: {CSV_PATH.with_name('offline_v277_decision_audit.csv')}")
    print(f"Saved rollout summary: {CSV_PATH.with_name('offline_v277_rollout_summary.csv')}")
    print(f"Saved policy usage summary: {CSV_PATH.with_name('offline_v277_policy_usage_summary.csv')}")
    print(f"Saved reason summary: {CSV_PATH.with_name('offline_v277_reason_summary.csv')}")
    print(f"Saved chunk stability: {CSV_PATH.with_name('offline_v277_chunk_stability.csv')}")
    print(f"Saved model comparison: {CSV_PATH.with_name('offline_v277_model_comparison.csv')}")


if __name__ == "__main__":
    main()
