from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


DEFAULT_INPUT = r"E:/Trading_Bot_V1.0/DogeBets/main/data/epoch_model_table_v6/epoch_sequences.parquet"
DEFAULT_OUTPUT_DIR = r"E:/Trading_Bot_V1.0/DogeBets/main/data/epoch_model_table_v6"
DEFAULT_WRITE_CSV = False
SIGMAS = [8, 23, 38, 53, 68, 83]


def _print_banner(title: str) -> None:
    print("=" * 100)
    print(title)
    print("=" * 100)


def _load_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input type: {suffix}")


def _parse_series_cell(x: Any) -> List[float]:
    """
    Parse a numeric series cell safely.

    Handles:
    - Python lists / tuples
    - numpy arrays / Arrow arrays via .tolist()
    - JSON-like strings
    - scalar fallbacks

    Returns only values that can be converted to float.
    """
    if x is None:
        return []

    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    if hasattr(x, "tolist"):
        vals = x.tolist()
        if not isinstance(vals, list):
            vals = [vals]
        out = []
        for v in vals:
            try:
                out.append(float(v))
            except Exception:
                pass
        return out

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            arr = json.loads(s)
            if not isinstance(arr, list):
                arr = [arr]
            out = []
            for v in arr:
                try:
                    out.append(float(v))
                except Exception:
                    pass
            return out
        except Exception:
            try:
                return [float(s)]
            except Exception:
                return []

    try:
        return [float(x)]
    except Exception:
        return []


def _parse_ts_series_cell(x: Any) -> List[str]:
    """
    Parse a timestamp/string series cell WITHOUT forcing float conversion.
    """
    if x is None:
        return []

    if isinstance(x, (list, tuple)):
        return [str(v) for v in x if v is not None]

    if hasattr(x, "tolist"):
        vals = x.tolist()
        if not isinstance(vals, list):
            vals = [vals]
        return [str(v) for v in vals if v is not None]

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(v) for v in arr if v is not None]
        except Exception:
            pass
        return [s]

    return [str(x)]


def _find_series_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    lowered = {c: c.lower() for c in columns}
    for cand in candidates:
        for c in columns:
            if lowered[c] == cand:
                return c
        for c in columns:
            if cand in lowered[c]:
                return c
    return None


def _gaussian_weights(radius: int, sigma: float) -> np.ndarray:
    xs = np.arange(-radius, radius + 1, dtype=float)
    w = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / len(w)


def _gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if len(values) == 0:
        return values
    if len(values) == 1:
        return values.copy()
    radius = max(1, min(int(max(3, sigma * 2)), max(1, len(values) // 2)))
    w = _gaussian_weights(radius, sigma=max(1.0, sigma / 6.0))
    padded = np.pad(values, (radius, radius), mode="edge")
    out = np.convolve(padded, w, mode="valid")
    return out.astype(float)


def _lin_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    m, _b = np.polyfit(x, y, 1)
    return float(m)


def _quad_terms(y: np.ndarray) -> tuple[float, float]:
    if len(y) < 3:
        return 0.0, 0.0
    x = np.arange(len(y), dtype=float)
    a, b, _c = np.polyfit(x, y, 2)
    tan = 2 * a * x[-1] + b
    curv = 2 * a
    return float(tan), float(curv)


def _run_score(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    diffs = np.diff(y)
    return float(np.mean(np.abs(diffs)))


def _zpos(close_now: float, mid: float, width: float) -> float:
    if width is None or width == 0:
        return 0.0
    return float((close_now - mid) / (width / 2.0))


def clip01(x: float) -> float:
    try:
        return float(max(0.0, min(1.0, x)))
    except Exception:
        return 0.0


def _label_regime(width_now: float, width_prev: float) -> str:
    if width_prev == 0:
        return "flat"
    delta = width_now - width_prev
    if delta < -1e-9:
        return "contracting"
    if delta > 1e-9:
        return "expanding"
    return "flat"


def _tail_metrics(series: np.ndarray, tail_n: int = 12) -> Dict[str, float]:
    if len(series) == 0:
        return {
            "slope": 0.0,
            "curve": 0.0,
            "lin_r2": 0.0,
            "quad_tangent": 0.0,
            "quad_curv": 0.0,
            "z": 0.0,
            "run_score": 0.0,
            "tag": "flat",
        }
    y = series[-min(len(series), max(3, tail_n)):]
    slope = _lin_slope(y)
    tan, curv = _quad_terms(y)
    curve = curv
    run = _run_score(y)
    std = float(np.std(y)) if len(y) > 1 else 1.0
    z = float((y[-1] - np.mean(y)) / std) if std > 1e-12 else 0.0
    if slope > 0 and curve >= 0:
        tag = "UP/accel"
    elif slope > 0 and curve < 0:
        tag = "UP/decel"
    elif slope < 0 and curve <= 0:
        tag = "DOWN/decel"
    elif slope < 0 and curve > 0:
        tag = "DOWN/accel"
    else:
        tag = "flat"
    return {
        "slope": float(slope),
        "curve": float(curve),
        "lin_r2": 0.0,
        "quad_tangent": float(tan),
        "quad_curv": float(curv),
        "z": float(z),
        "run_score": float(run),
        "tag": tag,
    }


def _build_msbc(close_seq: np.ndarray, smoothed: Dict[int, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    pair_summaries = []
    overrides = []
    slope_signs = []

    for sigma in SIGMAS:
        s = smoothed[sigma]
        n = len(s)
        mid = max(3, n // 2)
        leg1 = s[:mid]
        leg2 = s[mid:]

        m1 = _tail_metrics(leg1, tail_n=min(12, len(leg1)))
        m2 = _tail_metrics(leg2, tail_n=min(12, len(leg2)))

        out[f"src_msbc_l1_slope_s{sigma}"] = m1["slope"]
        out[f"src_msbc_l2_slope_s{sigma}"] = m2["slope"]
        out[f"src_msbc_l1_curve_s{sigma}"] = m1["curve"]
        out[f"src_msbc_l2_curve_s{sigma}"] = m2["curve"]
        out[f"src_msbc_l1_lin_r2_s{sigma}"] = m1["lin_r2"]
        out[f"src_msbc_l2_lin_r2_s{sigma}"] = m2["lin_r2"]
        out[f"src_msbc_l1_quad_tangent_s{sigma}"] = m1["quad_tangent"]
        out[f"src_msbc_l2_quad_tangent_s{sigma}"] = m2["quad_tangent"]
        out[f"src_msbc_l1_quad_curv_s{sigma}"] = m1["quad_curv"]
        out[f"src_msbc_l2_quad_curv_s{sigma}"] = m2["quad_curv"]
        out[f"src_msbc_l1_z_s{sigma}"] = m1["z"]
        out[f"src_msbc_l2_z_s{sigma}"] = m2["z"]
        out[f"src_msbc_l1_run_score_s{sigma}"] = m1["run_score"]
        out[f"src_msbc_l2_run_score_s{sigma}"] = m2["run_score"]
        out[f"src_msbc_l2_tag_s{sigma}"] = m2["tag"]

        sign1 = math.copysign(1, m1["slope"]) if abs(m1["slope"]) > 1e-12 else 0.0
        sign2 = math.copysign(1, m2["slope"]) if abs(m2["slope"]) > 1e-12 else 0.0
        override = float(sign1 != sign2 and sign1 != 0.0 and sign2 != 0.0)
        out[f"src_msbc_override_s{sigma}"] = override
        overrides.append(override)
        slope_signs.append(sign2)

    # Transfer summary using tail slopes across sigmas
    positives = sum(1 for s in slope_signs if s > 0)
    negatives = sum(1 for s in slope_signs if s < 0)
    if positives >= 5:
        transfer_dir = "up"
    elif negatives >= 5:
        transfer_dir = "down"
    else:
        transfer_dir = "mixed"

    transfer_depth = float(max(positives, negatives) - min(positives, negatives))
    if transfer_depth >= 5:
        transfer_state = "full"
    elif transfer_depth >= 3:
        transfer_state = "deep"
    elif transfer_depth >= 1:
        transfer_state = "shallow"
    else:
        transfer_state = "none"

    # Propagation across adjacent sigmas
    agree = 0
    total = 0
    disorder = 0
    for a, b in zip(SIGMAS[:-1], SIGMAS[1:]):
        sa = math.copysign(1, out[f"src_msbc_l2_slope_s{a}"]) if abs(out[f"src_msbc_l2_slope_s{a}"]) > 1e-12 else 0.0
        sb = math.copysign(1, out[f"src_msbc_l2_slope_s{b}"]) if abs(out[f"src_msbc_l2_slope_s{b}"]) > 1e-12 else 0.0
        total += 1
        if sa == sb and sa != 0.0:
            agree += 1
        else:
            disorder += 1
        pair_summaries.append((a, b, sa, sb))

    out["src_msbc_transfer_dir"] = transfer_dir
    out["src_msbc_transfer_depth"] = transfer_depth
    out["src_msbc_transfer_state"] = transfer_state
    out["src_msbc_disorder_count"] = float(disorder)
    out["src_msbc_override_count"] = float(sum(overrides))
    out["src_msbc_propagation_agree_ratio"] = float(agree / total) if total else 0.0
    out["src_msbc_propagation_disagree_ratio"] = float((total - agree) / total) if total else 0.0
    return out


def _build_gcs(close_seq: np.ndarray, smoothed: Dict[int, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    regimes = []
    positions = []

    for sigma in SIGMAS:
        s = smoothed[sigma]
        tail = s[-min(len(s), 20):]
        mid = float(tail[-1]) if len(tail) else float("nan")
        width_now = float(np.std(tail) * 2.0) if len(tail) > 1 else 0.0
        prev_tail = s[-min(len(s), 40):-min(len(s), 20)] if len(s) > 25 else tail
        width_prev = float(np.std(prev_tail) * 2.0) if len(prev_tail) > 1 else width_now
        regime = _label_regime(width_now, width_prev)
        mid_slope = _lin_slope(tail)
        width_change = width_now - width_prev
        pos = _zpos(float(close_seq[-1]), mid, width_now if width_now != 0 else 1.0)

        out[f"src_gcs_regime_s{sigma}"] = regime
        out[f"src_gcs_pos_s{sigma}"] = pos
        out[f"src_gcs_width_s{sigma}"] = width_now
        out[f"src_gcs_mid_slope_s{sigma}"] = mid_slope
        out[f"src_gcs_width_change_s{sigma}"] = width_change

        regimes.append(regime)
        positions.append(pos)

    contract_count = sum(1 for r in regimes if r == "contracting")
    expand_count = sum(1 for r in regimes if r == "expanding")
    flat_count = sum(1 for r in regimes if r == "flat")
    out["src_gcs_regime_contract_count"] = float(contract_count)
    out["src_gcs_regime_expand_count"] = float(expand_count)
    out["src_gcs_regime_flat_count"] = float(flat_count)
    out["src_gcs_regime_contract_all"] = float(contract_count == len(SIGMAS))

    spacing = [smoothed[a][-1] - smoothed[b][-1] for a, b in zip(SIGMAS[:-1], SIGMAS[1:])]
    mono = all(x <= 0 for x in spacing) or all(x >= 0 for x in spacing)
    out["src_gcs_spacing_state"] = "ordered" if mono else "mixed"
    out["src_gcs_fan_state"] = "inverted" if spacing and spacing[0] < 0 else "normal"

    fast_pos_mean = float(np.mean([out["src_gcs_pos_s8"], out["src_gcs_pos_s23"]]))
    slow_pos_mean = float(np.mean([out["src_gcs_pos_s68"], out["src_gcs_pos_s83"]]))
    pos_gap = fast_pos_mean - slow_pos_mean

    if fast_pos_mean > 0.15 and slow_pos_mean > 0.15:
        transfer_dir = "up"
    elif fast_pos_mean < -0.15 and slow_pos_mean < -0.15:
        transfer_dir = "down"
    else:
        transfer_dir = "mixed"

    depth = float(sum(1 for s in SIGMAS[:4] if out[f"src_gcs_pos_s{s}"] > 0)) if transfer_dir == "up" else float(sum(1 for s in SIGMAS[:4] if out[f"src_gcs_pos_s{s}"] < 0))
    if depth >= 3:
        transfer_state = "deep"
    elif depth >= 1:
        transfer_state = "shallow"
    else:
        transfer_state = "none"

    out["src_gcs_transfer_dir"] = transfer_dir
    out["src_gcs_transfer_depth"] = depth
    out["src_gcs_transfer_state"] = transfer_state
    out["src_gcs_fast_pos_mean"] = fast_pos_mean
    out["src_gcs_slow_pos_mean"] = slow_pos_mean
    out["src_gcs_fast_slow_pos_gap"] = pos_gap
    return out


def _build_min_hyst(smoothed: Dict[int, np.ndarray], epoch: Any, ts_seq: List[str]) -> Dict[str, Any]:
    g23 = smoothed[23][-1]
    g38 = smoothed[38][-1]
    g83 = smoothed[83][-1]
    primary_delta = float(g38 - g83)
    probe_delta = float(g23 - g83)
    primary_sign = 1 if primary_delta > 0 else -1 if primary_delta < 0 else 0
    probe_sign = 1 if probe_delta > 0 else -1 if probe_delta < 0 else 0

    elapsed = max(0, len(ts_seq) - 1) * 2.5 if ts_seq else 0.0
    out = {
        "src_hyst_primary_sign": float(primary_sign),
        "src_hyst_episode_age_sec": float(elapsed),
        "src_hyst_last_cross_age_sec": None,
        "src_hyst_probe_sign": float(probe_sign),
        "src_hyst_probe_flip_watch": float(primary_sign != probe_sign and probe_sign != 0),
        "src_hyst_fast_collapse": 0.0,
        "src_hyst_eta_to_end_seconds": None,
        "src_hyst_s0_state": None,
        "src_hyst_s0_pressure": None,
        "src_hyst_s0_risk": None,
        "src_hyst_s0_stability": None,
        "src_hyst_s0_near_cross": None,
        "src_hyst_s0_spread_slope": None,
        "src_hyst_s0_spread_accel": None,
        "src_hyst_s1_state": None,
        "src_hyst_s1_pressure": None,
        "src_hyst_s1_risk": None,
        "src_hyst_s1_stability": None,
        "src_hyst_s1_near_cross": None,
        "src_hyst_s1_spread_slope": None,
        "src_hyst_s1_spread_accel": None,
        "src_hyst_leader_max_sigma": 23 if abs(probe_delta) > abs(primary_delta) else 38,
        "src_hyst_leader_min_sigma": 83,
        "src_hyst_cross_rate": 0.0,
        "src_hyst_order_stability": None,
        "src_hyst_ladder_monotonic": None,
        "src_hyst_ladder_compression": None,
    }
    return out


def _extract_meta(row: pd.Series) -> Dict[str, Any]:
    out = {}

    epoch = row.get("epoch", row.get("round"))
    out["src_meta_epoch"] = epoch
    out["src_meta_decision_ts"] = row.get("epoch_ts_est", row.get("epoch_ts_utc", row.get("ts_end")))
    out["src_meta_next_epoch"] = row.get("next_round", row.get("next_epoch"))
    out["src_meta_next_epoch_time"] = row.get("next_round_time_est", row.get("next_round_time_utc", row.get("ts_end")))

    # ------------------------------------------------------------------
    # Canonical epoch-level truth fields from epoch_sequences.parquet
    # ------------------------------------------------------------------
    start_close = row.get("start_close")
    end_close = row.get("end_close")
    price_diff = row.get("price_diff")
    trend_label = row.get("trend_label")

    # Preserve raw truth under stable aliases for downstream research/eval
    out["actual_trend"] = trend_label
    out["actual_price_diff"] = price_diff

    # Preserve truth under src_meta namespace too
    out["src_meta_actual_trend"] = trend_label
    out["src_meta_actual_price_diff"] = price_diff

    # Preserve epoch start/end prices
    out["src_meta_start_price"] = start_close
    out["src_meta_end_price"] = end_close

    # Use epoch end close as BTC close reference
    out["src_meta_btc_close"] = end_close

    # Recompute/fallback price diff safely from start/end if missing
    try:
        if price_diff is None or (isinstance(price_diff, float) and math.isnan(price_diff)):
            out["src_meta_price_diff"] = float(end_close) - float(start_close)
            out["actual_price_diff"] = out["src_meta_price_diff"]
            out["src_meta_actual_price_diff"] = out["src_meta_price_diff"]
        else:
            out["src_meta_price_diff"] = float(price_diff)
    except Exception:
        out["src_meta_price_diff"] = None

    # If trend label is missing, derive it from price diff
    if out["actual_trend"] is None:
        try:
            pdiff = out["src_meta_price_diff"]
            if pdiff is None or (isinstance(pdiff, float) and math.isnan(pdiff)):
                out["actual_trend"] = "Neutral"
            elif pdiff > 0:
                out["actual_trend"] = "Bull"
            elif pdiff < 0:
                out["actual_trend"] = "Bear"
            else:
                out["actual_trend"] = "Neutral"
            out["src_meta_actual_trend"] = out["actual_trend"]
        except Exception:
            out["actual_trend"] = "Neutral"
            out["src_meta_actual_trend"] = "Neutral"

    out["src_meta_last_sample_age"] = None
    out["src_meta_lookback_minutes"] = None
    out["src_meta_pv_ref_sigma"] = 23
    out["src_meta_pv_pair_ref"] = None

    return out

def _sign_changes(arr: np.ndarray) -> int:
    if len(arr) < 2:
        return 0
    s = np.sign(arr)
    flips = 0
    prev = 0
    for v in s:
        if v == 0:
            continue
        if prev != 0 and v != prev:
            flips += 1
        prev = v
    return int(flips)


def _bars_since_last_sign_change(arr: np.ndarray) -> int:
    if len(arr) < 2:
        return 0
    s = np.sign(arr)
    cleaned = []
    last = 0
    for v in s:
        if v == 0:
            cleaned.append(last)
        else:
            cleaned.append(v)
            last = v
    cleaned = np.asarray(cleaned, dtype=float)
    if len(cleaned) == 0:
        return 0
    current = cleaned[-1]
    if current == 0:
        return 0
    for i in range(len(cleaned) - 2, -1, -1):
        if cleaned[i] != current:
            return len(cleaned) - 1 - i
    return len(cleaned) - 1


def _tail_slope_series(y: np.ndarray, win: int = 5) -> np.ndarray:
    if len(y) < max(3, win):
        return np.array([], dtype=float)
    out = []
    for i in range(win, len(y) + 1):
        seg = y[i - win:i]
        out.append(_lin_slope(seg))
    return np.asarray(out, dtype=float)

def _build_dcsd(close_seq: np.ndarray, smoothed: Dict[int, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    for sigma in SIGMAS:
        s = smoothed[sigma]
        tail = s[-min(len(s), 24):]
        prev = s[-min(len(s), 48):-min(len(s), 24)] if len(s) >= 48 else tail

        slope_tail = _tail_slope_series(tail, win=min(5, max(3, len(tail))))
        slope_prev = _tail_slope_series(prev, win=min(5, max(3, len(prev))))

        dmid_mean = float(np.mean(np.diff(tail))) if len(tail) > 1 else 0.0

        width_tail = float(np.std(tail)) if len(tail) > 1 else 0.0
        width_prev = float(np.std(prev)) if len(prev) > 1 else width_tail
        dwidth_mean = float(width_tail - width_prev)

        dmid_flip_count = _sign_changes(slope_tail) if len(slope_tail) else 0
        dwidth_flip_count = _sign_changes(np.array([dwidth_mean, width_tail - width_prev], dtype=float))

        out[f"src_dcsd_l2_dmid_mean_s{sigma}"] = dmid_mean
        out[f"src_dcsd_l2_dwidth_mean_s{sigma}"] = dwidth_mean
        out[f"src_dcsd_l2_dmid_flip_count_s{sigma}"] = float(dmid_flip_count)
        out[f"src_dcsd_l2_dwidth_flip_count_s{sigma}"] = float(dwidth_flip_count)

    return out

def _build_gbc(close_seq: np.ndarray, smoothed: Dict[int, np.ndarray]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    gbc_sigmas = [8, 23, 53, 68, 83]
    fast_sigmas = [8, 23]

    for sigma in gbc_sigmas:
        s = smoothed[sigma]
        tail = s[-min(len(s), 24):]
        if len(tail) < 4:
            slope_series = np.array([], dtype=float)
            last_slope = 0.0
            prev_abs = 0.0
            hook = 0.0
            flat = 1.0
            shrink = 0.0
            turn_age = 0.0
        else:
            slope_series = _tail_slope_series(tail, win=min(5, max(3, len(tail))))
            last_slope = float(slope_series[-1]) if len(slope_series) else 0.0
            prev_abs = float(abs(slope_series[-2])) if len(slope_series) >= 2 else abs(last_slope)

            hook = float(
                len(slope_series) >= 2
                and np.sign(slope_series[-1]) != 0
                and np.sign(slope_series[-2]) != 0
                and np.sign(slope_series[-1]) != np.sign(slope_series[-2])
            )

            flat = clip01(1.0 - min(1.0, abs(last_slope) / 5.0))
            shrink = float(abs(last_slope) < prev_abs)
            turn_age = float(_bars_since_last_sign_change(slope_series))

        out[f"src_gbc_hook_s{sigma}"] = hook
        out[f"src_gbc_flat_s{sigma}"] = flat
        out[f"src_gbc_shrink_s{sigma}"] = shrink
        out[f"src_gbc_last_abs_s{sigma}"] = float(abs(last_slope))
        out[f"src_gbc_turn_age_s{sigma}"] = turn_age

        if sigma in fast_sigmas:
            if len(slope_series) == 0:
                sign_persist = 0.0
                hook_age = 0.0
            else:
                persist_bars = _bars_since_last_sign_change(slope_series)
                sign_persist = clip01(persist_bars / 10.0)

                if hook == 1.0:
                    hook_age = 0.0
                else:
                    hook_age = float(persist_bars)

            out[f"src_gbc_sign_persist_s{sigma}"] = sign_persist
            out[f"src_gbc_hook_age_s{sigma}"] = hook_age

    return out

def build_v21_src_from_sequences(seq_df: pd.DataFrame) -> pd.DataFrame:
    cols = list(seq_df.columns)

    close_col = _find_series_col(cols, ["close_bar_series", "close_series", "close_seq"])
    open_col = _find_series_col(cols, ["open_bar_series", "open_series"])
    high_col = _find_series_col(cols, ["high_bar_series", "high_series"])
    low_col = _find_series_col(cols, ["low_bar_series", "low_series"])
    vol_col = _find_series_col(cols, ["volume_bar_series", "volume_series", "vol_series"])
    ts_col = _find_series_col(cols, ["ts_bar_series", "timestamp_series", "time_series"])

    if not close_col:
        raise ValueError("Could not find a close price sequence column in epoch_sequences.parquet")

    out_rows: List[Dict[str, Any]] = []
    total = len(seq_df)

    for i, (_idx, r) in enumerate(seq_df.iterrows(), start=1):
        if i == 1 or i % 500 == 0 or i == total:
            print(f"[factory] {i}/{total} epoch={r.get('epoch', r.get('round'))}")

        close_seq = np.asarray(_parse_series_cell(r.get(close_col)), dtype=float)
        if len(close_seq) < 10:
            # too short to build useful features
            continue

        ts_seq = _parse_ts_series_cell(r.get(ts_col)) if ts_col else []
        _ = _parse_series_cell(r.get(open_col)) if open_col else []
        _ = _parse_series_cell(r.get(high_col)) if high_col else []
        _ = _parse_series_cell(r.get(low_col)) if low_col else []
        _ = _parse_series_cell(r.get(vol_col)) if vol_col else []

        smoothed = {sigma: _gaussian_smooth(close_seq, sigma=float(sigma)) for sigma in SIGMAS}

        row_out: Dict[str, Any] = {}
        row_out.update(_extract_meta(r))
        row_out.update(_build_msbc(close_seq, smoothed))
        row_out.update(_build_gcs(close_seq, smoothed))
        row_out.update(_build_dcsd(close_seq, smoothed))
        row_out.update(_build_gbc(close_seq, smoothed))
        row_out.update(_build_min_hyst(smoothed, row_out.get("src_meta_epoch"), ts_seq))

        out_rows.append(row_out)

    return pd.DataFrame(out_rows)


def main() -> None:
    _print_banner("method_v21 Feature Factory Pass1 Starting")
    input_path = Path(DEFAULT_INPUT)
    output_dir = Path(DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] input      = {input_path}")
    print(f"[config] output_dir = {output_dir}")
    print("-" * 100)
    print("[stage] loading epoch_sequences dataset...")

    t0 = time.time()
    seq_df = _load_frame(input_path)
    print(f"[data] loaded rows={len(seq_df):,} cols={len(seq_df.columns):,} in {time.time() - t0:.2f}s")
    print("[stage] building first-pass v21 src table (MSBC + GCS + minimal HYST)...")

    t1 = time.time()
    v21_df = build_v21_src_from_sequences(seq_df)
    print(f"[data] built rows={len(v21_df):,} cols={len(v21_df.columns):,} in {time.time() - t1:.2f}s")

    parquet_path = output_dir / "epoch_model_table_v21_src.parquet"
    v21_df.to_parquet(parquet_path, index=False)

    summary = {
        "input": str(input_path),
        "output_parquet": str(parquet_path),
        "rows": int(len(v21_df)),
        "cols": int(len(v21_df.columns)),
        "columns": list(v21_df.columns),
    }
    summary_path = output_dir / "epoch_model_table_v21_src_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if DEFAULT_WRITE_CSV:
        csv_path = output_dir / "epoch_model_table_v21_src.csv"
        v21_df.to_csv(csv_path, index=False)
        print(f"[save] csv       = {csv_path}")

    print("-" * 100)
    print(f"[save] parquet   = {parquet_path}")
    print(f"[save] summary   = {summary_path}")
    print("=" * 100)
    print("method_v21 Feature Factory Pass1 Finished")
    print("=" * 100)


if __name__ == "__main__":
    main()
