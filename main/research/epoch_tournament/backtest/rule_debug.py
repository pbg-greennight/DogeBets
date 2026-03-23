from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


def save_rule_debug_report(pred_df: pd.DataFrame, output_dir: Path, model_id: str) -> Path:
    out_path = output_dir / f"rule_debug_{model_id}.csv"

    if pred_df.empty:
        pd.DataFrame([{"section": "note", "metric": "empty_prediction_dataframe", "value": ""}]).to_csv(out_path, index=False)
        return out_path

    df = pred_df.copy()

    if "reason" not in df.columns:
        pd.DataFrame([{"section": "note", "metric": "reason_column_missing", "value": ""}]).to_csv(out_path, index=False)
        return out_path

    df["rule_template"] = df["reason"].astype(str).str.split(":", n=1).str[0]
    df["rule_payload"] = df["reason"].astype(str).str.split(":", n=1).str[1].fillna("")

    df["bull_score_dbg"] = df["rule_payload"].apply(lambda s: _extract_float(s, r"bull_score=([-+]?\d*\.?\d+)"))
    df["bear_score_dbg"] = df["rule_payload"].apply(lambda s: _extract_float(s, r"bear_score=([-+]?\d*\.?\d+)"))
    df["segment_valid_dbg"] = df["rule_payload"].apply(lambda s: _extract_int(s, r"segment_valid=(\d+)"))
    df["segment_len_dbg"] = df["rule_payload"].apply(lambda s: _extract_int(s, r"segment_len=(\d+)"))
    df["tail_len_dbg"] = df["rule_payload"].apply(lambda s: _extract_int(s, r"tail_len=(\d+)"))
    df["tail_r2_dbg"] = df["rule_payload"].apply(lambda s: _extract_float(s, r"tail_r2=([-+]?\d*\.?\d+)"))
    df["tail_plateau_dbg"] = df["rule_payload"].apply(lambda s: _extract_float(s, r"tail_plateau=([-+]?\d*\.?\d+)"))
    df["tail_fan_inv_dbg"] = df["rule_payload"].apply(lambda s: _extract_int(s, r"tail_fan_inv=(\d+)"))
    df["fan_violation_now_dbg"] = df["rule_payload"].apply(lambda s: _extract_int(s, r"fan_violation_now=(\d+)"))

    # correctness
    df["is_right"] = ((df["pred_label"].isin([0, 1])) & (df["pred_label"] == df["actual_label"])).astype(int)
    df["is_wrong"] = ((df["pred_label"].isin([0, 1])) & (df["pred_label"] != df["actual_label"])).astype(int)
    df["is_skip"] = (df["pred_label"] == -1).astype(int)

    rows: list[dict] = []

    # overall
    rows.extend([
        {"section": "overall", "metric": "rows_total", "value": int(len(df))},
        {"section": "overall", "metric": "wager_true", "value": int(df["wager"].sum()) if "wager" in df.columns else 0},
        {"section": "overall", "metric": "bull_rows", "value": int((df["pred_trend"] == "Bull").sum())},
        {"section": "overall", "metric": "bear_rows", "value": int((df["pred_trend"] == "Bear").sum())},
        {"section": "overall", "metric": "skip_rows", "value": int((df["pred_trend"] == "Skip").sum())},
        {"section": "overall", "metric": "avg_bull_score", "value": _mean_or_none(df["bull_score_dbg"])},
        {"section": "overall", "metric": "avg_bear_score", "value": _mean_or_none(df["bear_score_dbg"])},
        {"section": "overall", "metric": "usable_segment_rows", "value": int((df["segment_valid_dbg"] == 1).sum())},
        {"section": "overall", "metric": "usable_segment_pct", "value": float((df["segment_valid_dbg"] == 1).mean() * 100.0) if len(df) else None},
    ])

    # right/wrong stats by prediction side
    for trend_name, pred_val in [("Bull", 1), ("Bear", 0), ("Skip", -1)]:
        sub = df[df["pred_label"] == pred_val].copy()
        rows.append({"section": "by_pred", "metric": f"{trend_name}_count", "value": int(len(sub))})

        if pred_val in (0, 1):
            rows.append({"section": "by_pred", "metric": f"{trend_name}_right", "value": int(sub["is_right"].sum())})
            rows.append({"section": "by_pred", "metric": f"{trend_name}_wrong", "value": int(sub["is_wrong"].sum())})
            rows.append({
                "section": "by_pred",
                "metric": f"{trend_name}_accuracy",
                "value": float(sub["is_right"].mean()) if len(sub) else None,
            })
        else:
            rows.append({"section": "by_pred", "metric": "Skip_count", "value": int(len(sub))})

    # template counts
    for template, count in df["rule_template"].value_counts(dropna=False).items():
        rows.append({"section": "template_counts", "metric": str(template), "value": int(count)})

    # top skip fail reasons
    skip_df = df[df["pred_trend"] == "Skip"].copy()
    if not skip_df.empty:
        for metric_name, pattern in [
            ("bull_core_fails", r"bull_core_fails=([^|]+)"),
            ("bear_core_fails", r"bear_core_fails=([^|]+)"),
        ]:
            extracted = skip_df["rule_payload"].apply(lambda s: _extract_text(s, pattern)).fillna("none")
            vc = extracted.value_counts().head(20)
            for k, count in vc.items():
                rows.append({
                    "section": f"skip_{metric_name}",
                    "metric": str(k),
                    "value": int(count),
                })

    # useful averages by template
    for template in sorted(df["rule_template"].dropna().unique()):
        sub = df[df["rule_template"] == template].copy()
        rows.extend([
            {"section": "template_avgs", "metric": f"{template}|avg_bull_score", "value": _mean_or_none(sub["bull_score_dbg"])},
            {"section": "template_avgs", "metric": f"{template}|avg_bear_score", "value": _mean_or_none(sub["bear_score_dbg"])},
            {"section": "template_avgs", "metric": f"{template}|avg_segment_len", "value": _mean_or_none(sub["segment_len_dbg"])},
            {"section": "template_avgs", "metric": f"{template}|avg_tail_len", "value": _mean_or_none(sub["tail_len_dbg"])},
            {"section": "template_avgs", "metric": f"{template}|avg_tail_r2", "value": _mean_or_none(sub["tail_r2_dbg"])},
            {"section": "template_avgs", "metric": f"{template}|avg_tail_plateau", "value": _mean_or_none(sub["tail_plateau_dbg"])},
            {"section": "template_avgs", "metric": f"{template}|avg_tail_fan_inv", "value": _mean_or_none(sub["tail_fan_inv_dbg"])},
        ])

    debug_df = pd.DataFrame(rows)
    debug_df.to_csv(out_path, index=False)
    return out_path


def _extract_float(text: str, pattern: str):
    if not isinstance(text, str):
        return None
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_int(text: str, pattern: str):
    if not isinstance(text, str):
        return None
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _extract_text(text: str, pattern: str):
    if not isinstance(text, str):
        return None
    m = re.search(pattern, text)
    if not m:
        return None
    return m.group(1)


def _mean_or_none(series: pd.Series):
    try:
        s = pd.to_numeric(series, errors="coerce")
        return float(s.mean()) if s.notna().any() else None
    except Exception:
        return None