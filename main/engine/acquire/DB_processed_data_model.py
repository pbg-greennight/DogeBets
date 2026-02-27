"""
DB_processed_data_model.py

AuSTD v1.1 model engine:
- Builds epoch_df from DB_epoch_ochlv_merged.csv
- Computes Stage 1 (microstructure) + Stage 2 (TA) features
- Trains / evaluates RandomForest (400 trees, depth=6)
- Exposes helpers for live inference: compute_features_for_live_epoch()
"""

from __future__ import annotations
import json
import joblib
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - INFO - %(message)s",
    datefmt="%I:%M:%S %p",
    force=True,
)

# -----------------------------------------------------------------------------
# Neutral label configuration (3-class: 0=Bear, 1=Neutral, 2=Bull)
# -----------------------------------------------------------------------------
# Mode for neutral band:
#   "pct" -> |priceDiff| <= mid_price * NEUTRAL_EPS_PCT
#   "abs" -> |priceDiff| <= NEUTRAL_EPS_ABS
NEUTRAL_EPS_MODE = "pct"
NEUTRAL_EPS_PCT = 0.85e-4   # % of mid-price
NEUTRAL_EPS_ABS = 0.0    # only used if mode == "abs"

# Public label name used by training + live baseline logic
DIRECTION_LABEL_COL = "direction_3"

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
# Minimum number of OCHLV ticks per epoch block to accept it
MIN_TICKS_PER_EPOCH = 85

# ---------------------------------------------------------------------
# Tail evaluation config & baseline configs
# ---------------------------------------------------------------------
# Which engine did best on the last EPOCH_TAIL_EVAL epochs?
#   "model", "baseline_A", "baseline_B", "baseline_C"
CURRENT_TAIL_ENGINE: str | None = None

# Last tail-eval scoreboard (for debugging / logging)
CURRENT_TAIL_STATS: dict | None = None

EPOCH_TAIL_EVAL = 50  # how many last epochs to use for tail sanity

BASELINE_A_CFG = {
    # No thresholds needed for now; using pure MACD + EMA slope logic.
}

BASELINE_B_CFG = {
    "rsi_bull": 55.0,
    "rsi_bear": 45.0,
    "strength_min": 0.30,  # |priceDiff|/ATR below this -> Neutral
}

BASELINE_C_CFG = {
    "bb_width_small": 0.002,   # narrow bands -> consolidation
    "body_frac_small": 0.25,   # small body vs range => consolidation
    "body_frac_large": 0.60,   # big body vs range => impulse candle
    "pctB_bull": 0.80,         # top band region
    "pctB_bear": 0.20,         # bottom band region
}

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

# Epoch-level merged CSV (tick + epoch summary)
EPOCH_OCHLV_CSV = BASE_DIR / "csv" / "DB_epoch_ochlv_merged.csv"

# Where to save the trained model + metadata
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "AuSTD_v1_1.pkl"
METRICS_PATH = MODEL_DIR / "AuSTD_v1_1_metrics.json"

# Stage 1 / Stage 2 feature names (must match what we compute)
STAGE1_FEATS = [
    "body_frac",
    "realized_vol_block",
    "max_runup_norm",
    "max_drawdown_norm",
    "vol_z_20",
    "range_norm_end",
    "volume_epoch",
]

STAGE2_FEATS = [
    "C_over_ema5",
    "C_over_ema10",
    "slope_C_5",
    "slope_C_10",
    "rsi_14",
    "macd",
    "macd_hist",
    "pctB_20",
    "ret_1",
    "ret_3",
    "diff_mean_5",
    "diff_std_5",
    # --- new flip-aware / exhaustion features ---
    "bb_width_20",
    "atr_14",
    "upper_wick_frac",
    "lower_wick_frac",
    "poly2_a_7",
    "poly2_b_7",
    "g1_large_20",
    "g2_large_20",
    "run_length_signed",
]

# RandomForest config for AuSTD v1.1
RF_CFG = dict(
    n_estimators=400,
    max_depth=6,
    random_state=42,
)

# GradientBoosting config (example)
GB_CFG = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    random_state=42,
)


def make_classifier(kind: str = "rf"):
    """
    Factory for different classifier types.

    kind:
      - "rf"  : RandomForest (AuSTD v1.1 baseline)
      - "et"  : ExtraTrees
      - "gb"  : GradientBoosting
      - "lr"  : LogisticRegression
    """
    kind = kind.lower()
    if kind == "rf":
        return RandomForestClassifier(**RF_CFG)
    elif kind == "gb":
        return GradientBoostingClassifier(**GB_CFG)
    else:
        raise ValueError(f"Unknown classifier kind: {kind}")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_float(x: Any) -> Optional[float]:
    """Parse floats from numbers or currency-like strings.

    Handles things like:
        86012.21
        "86012.21"
        "$86012.21"
        "$86,012.21"
    """
    if x is None:
        return None
    if isinstance(x, str):
        # Strip dollar signs, commas, and surrounding spaces
        cleaned = x.replace("$", "").replace(",", "").strip()
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except Exception:
            return None

    try:
        return float(x)
    except Exception:
        return None

def _pick(cols, *candidates) -> Optional[str]:
    """Pick the first matching column name ignoring case/underscores."""
    norm = {c.lower().replace("_", ""): c for c in cols}
    for cand in candidates:
        key = cand.lower().replace("_", "")
        if key in norm:
            return norm[key]
    return None

def _baseline_A_predict(df_tail: pd.DataFrame) -> np.ndarray:
    """
    Baseline A: EMA + MACD combo.
    Uses:
        - macd_hist (Stage 2)
        - slope_C_10 (Stage 2): slope of close over ~10 epochs

    Rules (per epoch):
        If macd_hist > 0 and slope_C_10 > 0 -> Bull (2)
        If macd_hist < 0 and slope_C_10 < 0 -> Bear (0)
        Else -> Neutral (1)
    """
    required = ["macd_hist", "slope_C_10"]
    for col in required:
        if col not in df_tail.columns:
            logging.warning(
                "⚠️ Baseline A: missing column '%s'; returning all Neutral.",
                col,
            )
            return np.ones(len(df_tail), dtype=int)

    macd_hist = df_tail["macd_hist"].to_numpy(dtype=float)
    slope10 = df_tail["slope_C_10"].to_numpy(dtype=float)

    y_pred = np.full(len(df_tail), 1, dtype=int)  # default Neutral
    mask_bull = (macd_hist > 0.0) & (slope10 > 0.0)
    mask_bear = (macd_hist < 0.0) & (slope10 < 0.0)

    y_pred[mask_bull] = 2
    y_pred[mask_bear] = 0

    return y_pred


def _baseline_B_predict(df_tail: pd.DataFrame) -> np.ndarray:
    """
    Baseline B: RSI + ATR neutral-band logic.

    Uses:
        - rsi_14 (Stage 2)
        - atr_14 (Stage 2)
        - Close (epoch-level)

    Steps:
        1) priceDiff_k = Close_k - Close_{k-1}
        2) trend_strength = |priceDiff_k| / max(ATR_14_k, tiny)
        3) If trend_strength < strength_min -> Neutral (1)
           Else:
             RSI > rsi_bull -> Bull (2)
             RSI < rsi_bear -> Bear (0)
             else -> Neutral (1)
    """
    required = ["rsi_14", "atr_14", "Close"]
    for col in required:
        if col not in df_tail.columns:
            logging.warning(
                "⚠️ Baseline B: missing column '%s'; returning all Neutral.",
                col,
            )
            return np.ones(len(df_tail), dtype=int)

    rsi = df_tail["rsi_14"].to_numpy(dtype=float)
    atr = df_tail["atr_14"].to_numpy(dtype=float)
    close = df_tail["Close"].to_numpy(dtype=float)

    # priceDiff: close diff vs previous epoch inside the tail
    price_diff = np.diff(close, prepend=close[0])
    trend_strength = np.abs(price_diff) / np.maximum(atr, 1e-9)

    rsi_bull = BASELINE_B_CFG["rsi_bull"]
    rsi_bear = BASELINE_B_CFG["rsi_bear"]
    strength_min = BASELINE_B_CFG["strength_min"]

    y_pred = np.full(len(df_tail), 1, dtype=int)  # default Neutral

    # First gate: weak moves -> Neutral
    weak = trend_strength < strength_min
    y_pred[weak] = 1

    strong = ~weak
    # Among strong moves, use RSI band
    bull_mask = (strong) & (rsi > rsi_bull)
    bear_mask = (strong) & (rsi < rsi_bear)

    y_pred[bull_mask] = 2
    y_pred[bear_mask] = 0

    return y_pred


def _baseline_C_predict(df_tail: pd.DataFrame) -> np.ndarray:
    """
    Baseline C: Candlestick geometry + Bollinger %B.

    Uses:
        - Open, High, Low, Close
        - pctB_20      (Stage 2)
        - bb_width_20  (Stage 2)

    Rules (per epoch):
        body = |Close - Open|
        range = High - Low
        body_frac = body / range

        If bb_width_20 < bb_width_small and body_frac < body_frac_small:
            -> Neutral
        Else if body_frac > body_frac_large and bullish candle and %B > pctB_bull:
            -> Bull
        Else if body_frac > body_frac_large and bearish candle and %B < pctB_bear:
            -> Bear
        Else:
            -> Neutral
    """
    required = ["Open", "High", "Low", "Close", "pctB_20", "bb_width_20"]
    for col in required:
        if col not in df_tail.columns:
            logging.warning(
                "⚠️ Baseline C: missing column '%s'; returning all Neutral.",
                col,
            )
            return np.ones(len(df_tail), dtype=int)

    open_ = df_tail["Open"].to_numpy(dtype=float)
    high = df_tail["High"].to_numpy(dtype=float)
    low = df_tail["Low"].to_numpy(dtype=float)
    close = df_tail["Close"].to_numpy(dtype=float)
    pctB = df_tail["pctB_20"].to_numpy(dtype=float)
    bb_width = df_tail["bb_width_20"].to_numpy(dtype=float)

    body = np.abs(close - open_)
    range_ = np.maximum(high - low, 1e-9)
    body_frac = body / range_

    cfg = BASELINE_C_CFG
    bb_width_small = cfg["bb_width_small"]
    body_small = cfg["body_frac_small"]
    body_large = cfg["body_frac_large"]
    pctB_bull = cfg["pctB_bull"]
    pctB_bear = cfg["pctB_bear"]

    y_pred = np.full(len(df_tail), 1, dtype=int)  # default Neutral

    # Consolidation: narrow bands + small body
    consolidation = (bb_width < bb_width_small) & (body_frac < body_small)
    y_pred[consolidation] = 1

    # Impulse candles
    bullish_candle = close > open_
    bearish_candle = close < open_

    strong_body = body_frac > body_large

    bull_impulse = strong_body & bullish_candle & (pctB > pctB_bull)
    bear_impulse = strong_body & bearish_candle & (pctB < pctB_bear)

    y_pred[bull_impulse] = 2
    y_pred[bear_impulse] = 0

    return y_pred

def evaluate_tail_engines(
    epoch_df: pd.DataFrame,
    epoch_tail_eval: int = EPOCH_TAIL_EVAL,
    label_col: str = "direction_3",
) -> Dict[str, Any]:
    """
    Evaluate:
        - current saved AuSTD model
        - Baseline A/B/C

    on the last `epoch_tail_eval` labeled epochs of `epoch_df`.

    Returns a dict with accuracies and the winning engine name.
    """
    if epoch_df is None or epoch_df.empty:
        logging.warning("⚠️ evaluate_tail_engines: empty epoch_df; skipping.")
        return {}

    if label_col not in epoch_df.columns:
        logging.warning(
            "⚠️ evaluate_tail_engines: missing label_col='%s'; skipping.", label_col
        )
        return {}

    # Drop rows with NaN labels
    df = epoch_df.dropna(subset=[label_col]).copy()

    if "timestamp_epoch" in df.columns:
        df = df.sort_values("timestamp_epoch").reset_index(drop=True)
    elif "Current Epoch" in df.columns:
        df = df.sort_values("Current Epoch").reset_index(drop=True)

    n_all = len(df)
    if n_all == 0:
        logging.warning("⚠️ evaluate_tail_engines: no labeled rows after filtering.")
        return {}

    tail_n = min(epoch_tail_eval, n_all)
    df_tail = df.iloc[-tail_n:].copy()
    y_true = df_tail[label_col].astype(int).to_numpy()

    # --------------------------------------------------------------
    # Model accuracy on tail
    # --------------------------------------------------------------
    acc_model = float("nan")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logging.warning(
            "⚠️ evaluate_tail_engines: could not load model from %s: %s",
            MODEL_PATH,
            e,
        )
        model = None

    if model is not None:
        # Feature set: same as training (intersection of STAGE1+STAGE2)
        base_feature_cols = list(STAGE1_FEATS) + list(STAGE2_FEATS)
        feature_cols = [c for c in base_feature_cols if c in df_tail.columns]

        if not feature_cols:
            logging.warning(
                "⚠️ evaluate_tail_engines: no feature columns present in tail; "
                "cannot evaluate model."
            )
        else:
            X_tail = df_tail[feature_cols].to_numpy(dtype=float)
            y_pred_model = model.predict(X_tail)
            acc_model = float(accuracy_score(y_true, y_pred_model))

    # --------------------------------------------------------------
    # Baseline accuracies
    # --------------------------------------------------------------
    yA = _baseline_A_predict(df_tail)
    yB = _baseline_B_predict(df_tail)
    yC = _baseline_C_predict(df_tail)

    acc_A = float(accuracy_score(y_true, yA))
    acc_B = float(accuracy_score(y_true, yB))
    acc_C = float(accuracy_score(y_true, yC))

    scores = {
        "model": acc_model,
        "baseline_A": acc_A,
        "baseline_B": acc_B,
        "baseline_C": acc_C,
    }

    # Choose winner (ignore NaNs)
    def _score(name: str, val: float) -> float:
        return val if np.isfinite(val) else float("-inf")

    winner = max(scores.items(), key=lambda kv: _score(*kv))[0]

    logging.info(
        "🧪 Tail sanity (last %d epochs): model=%.4f, A=%.4f, B=%.4f, C=%.4f -> winner=%s",
        tail_n,
        acc_model,
        acc_A,
        acc_B,
        acc_C,
        winner,
    )

    return {
        "tail_size": int(tail_n),
        "acc_model": acc_model,
        "acc_A": acc_A,
        "acc_B": acc_B,
        "acc_C": acc_C,
        "winner": winner,
    }

def load_trained_model(model_path: Optional[str] = None):
    """
    Small helper used by the live runtime (DB_processed_data.py) to load
    the current AuSTD v1.1 model from disk.

    Parameters
    ----------
    model_path : str or None
        Optional override path. If None, defaults to MODEL_PATH.

    Returns
    -------
    model : sklearn estimator
        The deserialized model object.
    """
    path = model_path or MODEL_PATH
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        logging.error("❌ load_trained_model: model file not found at %s", path)
        raise
    except Exception as e:
        logging.error("❌ load_trained_model: failed to load model from %s: %s", path, e)
        raise

# -----------------------------------------------------------------------------
# Epoch builder from DB_epoch_ochlv_merged.csv
# -----------------------------------------------------------------------------
def load_merged_csv(path: Path = EPOCH_OCHLV_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Epoch OCHLV CSV not found: {path}")
    df = pd.read_csv(path)
    return df

def build_epoch_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build epoch_df from merged tick+epoch CSV using anchors on 'Current Epoch'.

    Assumptions about the merged CSV:

    - Most rows are tick-level OCHLV with:
        Current Epoch = NaN
    - Anchor / summary rows (one per epoch) have:
        Current Epoch = <epoch_id> (non-NaN)
        startPrice, endPrice, priceDifference filled for that epoch.

    We treat each non-NaN 'Current Epoch' row as an anchor and build the
    epoch block from the previous anchor (or start of file) up to *this*
    anchor row (inclusive).

    Guards:
    - n_ticks_block >= MIN_TICKS_PER_EPOCH
    - startPrice, endPrice, priceDifference parseable
    - basic OHLCV parseable within the block

    Labels:
    - direction      : binary 0/1 (0=Bear, 1=Bull) kept for compatibility / runs
    - direction_3    : tri-class 0=Bear, 1=Neutral, 2=Bull (used for training)
    """
    cols = list(df.columns)
    epoch_col = _pick(cols, "Current Epoch", "Epoch")
    ts_col = _pick(cols, "Timestamp", "time")
    open_col = _pick(cols, "Open")
    high_col = _pick(cols, "High")
    low_col = _pick(cols, "Low")
    close_col = _pick(cols, "Close", "Price")
    vol_col = _pick(cols, "Volume")

    start_col = _pick(cols, "startPrice", "Start Price")
    end_col = _pick(cols, "endPrice", "End Price")
    diff_col = _pick(cols, "priceDifference", "Price Difference")

    approx_diff_col = None
    if "OCHLV_CLOSE_approx_priceDifference" in cols:
        approx_diff_col = "OCHLV_CLOSE_approx_priceDifference"

    required = {
        "epoch_col": epoch_col,
        "ts_col": ts_col,
        "open_col": open_col,
        "high_col": high_col,
        "low_col": low_col,
        "close_col": close_col,
        "vol_col": vol_col,
        "start_col": start_col,
        "end_col": end_col,
        "diff_col": diff_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns in merged CSV: {', '.join(missing)}")

    # Ensure sorted by timestamp
    df = df.sort_values(by=ts_col).reset_index(drop=True)

    n = len(df)
    if n == 0:
        raise ValueError("Merged CSV is empty")

    # ------------------------------------------------------------------
    # NEW: anchor rows are those with non-NaN 'Current Epoch'
    # ------------------------------------------------------------------
    epoch_series = df[epoch_col]
    anchor_indices: List[int] = list(epoch_series[epoch_series.notna()].index)

    if not anchor_indices:
        raise ValueError(f"No non-NaN '{epoch_col}' anchor rows found in merged CSV.")

    logging.info(
        "🔍 Epoch builder: %d total rows, %d anchor rows (non-NaN '%s').",
        n,
        len(anchor_indices),
        epoch_col,
    )

    epoch_rows: List[Dict[str, Any]] = []
    prev_idx = 0

    for j, cur_idx in enumerate(anchor_indices):
        # Block is from previous anchor (or start) to this anchor row inclusive
        block = df.iloc[prev_idx : cur_idx + 1].copy()
        anchor_row = df.iloc[cur_idx]
        prev_idx = cur_idx + 1

        n_ticks = len(block)
        if n_ticks < MIN_TICKS_PER_EPOCH:
            # Guard: insufficient ticks in epoch
            continue

        # Parse summary fields from the anchor row
        epoch_id_raw = anchor_row[epoch_col]
        start_num = _safe_float(anchor_row[start_col])
        end_num = _safe_float(anchor_row[end_col])
        diff_num = _safe_float(anchor_row[diff_col])

        approx_diff_num = None
        if approx_diff_col is not None:
            approx_diff_num = _safe_float(anchor_row[approx_diff_col])

        if start_num is None or end_num is None or diff_num is None:
            continue

        try:
            epoch_id = int(epoch_id_raw)
        except Exception:
            epoch_id = int(j)  # fallback

        # Basic OHLCV within block
        o_epoch = _safe_float(block[open_col].iloc[0])
        c_epoch = _safe_float(block[close_col].iloc[-1])
        h_epoch = _safe_float(block[high_col].max())
        l_epoch = _safe_float(block[low_col].min())
        vol_epoch = _safe_float(block[vol_col].sum())
        ts_epoch = anchor_row[ts_col]

        if any(v is None for v in (o_epoch, c_epoch, h_epoch, l_epoch, vol_epoch)):
            continue

        # Realized vol, run-up/drawdown etc for Stage 1
        C_block = block[close_col].astype(float).reset_index(drop=True)
        P0 = float(C_block.iloc[0])

        log_returns = np.diff(np.log(C_block.values))
        realized_var = float(np.sum(log_returns ** 2))
        realized_vol_block = float(np.sqrt(realized_var))

        delta = C_block - P0
        max_runup = float(delta.max())
        max_drawdown = float(delta.min())
        max_runup_norm = max_runup / P0 if P0 != 0 else 0.0
        max_drawdown_norm = max_drawdown / P0 if P0 != 0 else 0.0

        body = abs(c_epoch - o_epoch)
        range_epoch = h_epoch - l_epoch
        body_frac = body / range_epoch if range_epoch > 0 else 0.0
        range_norm_end = (range_epoch / end_num) if end_num != 0 else 0.0

        # ------------------------------------------------------------------
        # 3-class label: 0=Bear, 1=Neutral, 2=Bull  (learnable Neutral band)
        # ------------------------------------------------------------------
        mid_price = 0.5 * (start_num + end_num)
        if NEUTRAL_EPS_MODE == "pct":
            eps = abs(mid_price) * NEUTRAL_EPS_PCT
        elif NEUTRAL_EPS_MODE == "abs":
            eps = NEUTRAL_EPS_ABS
        else:
            eps = 0.0

        eps = float(max(eps, 0.0))  # safety

        if diff_num <= -eps:
            direction_3 = 0  # Bear
        elif diff_num >= eps:
            direction_3 = 2  # Bull
        else:
            direction_3 = 1  # Neutral

        # Legacy binary direction for run-length feature etc.
        direction = 1 if diff_num > 0 else 0

        row = dict(
            epoch_id=epoch_id,
            timestamp_epoch=ts_epoch,
            startPrice_num=start_num,
            endPrice_num=end_num,
            priceDiff_num=diff_num,
            direction=direction,
            direction_3=direction_3,
            open_epoch=o_epoch,
            high_epoch=h_epoch,
            low_epoch=l_epoch,
            close_epoch=c_epoch,
            volume_epoch=vol_epoch,
            body_frac=body_frac,
            realized_vol_block=realized_vol_block,
            max_runup_norm=max_runup_norm,
            max_drawdown_norm=max_drawdown_norm,
            range_norm_end=range_norm_end,
        )
        if approx_diff_num is not None:
            row["approx_priceDiff_num"] = approx_diff_num

        epoch_rows.append(row)

    epoch_df = pd.DataFrame(epoch_rows)
    if epoch_df.empty:
        raise ValueError("No epochs survived guards; epoch_df is empty.")

    # Compute vol_z_20 for Stage 1 on epoch_df
    epoch_df["volume_epoch"] = epoch_df["volume_epoch"].astype(float)
    vol_mean_20 = epoch_df["volume_epoch"].rolling(20).mean()
    vol_std_20 = epoch_df["volume_epoch"].rolling(20).std()
    epoch_df["vol_z_20"] = (epoch_df["volume_epoch"] - vol_mean_20) / vol_std_20

    logging.info(f"✅ Built epoch_df with {len(epoch_df)} epochs after guards.")
    return epoch_df


# -----------------------------------------------------------------------------
# Stage 2 Feature Computation
# -----------------------------------------------------------------------------
def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear regression slope over last `window` points."""
    arr = series.values
    n = len(arr)
    idx = np.arange(n)
    slopes = np.full(n, np.nan, dtype=float)

    for i in range(window - 1, n):
        x = idx[i - window + 1 : i + 1]
        y = arr[i - window + 1 : i + 1]
        x_mean = x.mean()
        y_mean = y.mean()
        den = np.sum((x - x_mean) ** 2)
        if den > 0:
            num = np.sum((x - x_mean) * (y - y_mean))
            slopes[i] = num / den
        else:
            slopes[i] = 0.0
    return pd.Series(slopes, index=series.index)

def _poly2_coeff_a(values: np.ndarray) -> float:
    """
    Helper for quadratic fit: returns 'a' (curvature) of y ≈ a x^2 + b x + c
    over the given window of values.
    """
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values))
    try:
        coeffs = np.polyfit(x, values, 2)
        return float(coeffs[0])
    except Exception:
        return 0.0


def _poly2_coeff_b(values: np.ndarray) -> float:
    """
    Helper for quadratic fit: returns 'b' (slope) of y ≈ a x^2 + b x + c
    over the given window of values.
    """
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values))
    try:
        coeffs = np.polyfit(x, values, 2)
        return float(coeffs[1])
    except Exception:
        return 0.0

def add_stage2_features(epoch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Stage 2 features on epoch_df (in-place) and return it.

    This uses:
      - endPrice_num history as the primary price series (C)
      - approx_priceDiff_num for diff-based features when available
        (i.e., WAKEUP-margin approximate epoch moves),
        falling back to priceDiff_num only if approx is missing.

    It also builds the generic OHLCV + TA columns needed by
    the baseline engines (A/B/C): Close, Open, High, Low, Volume,
    rsi_14, atr_14, pctB_20, bb_width_20, macd_hist.
    """
    # ---------------------------
    # Base close series
    # ---------------------------
    if "endPrice_num" not in epoch_df.columns:
        raise KeyError("add_stage2_features: 'endPrice_num' column is required")

    C = epoch_df["endPrice_num"].astype(float)

    # ---------------------------
    # Generic OHLCV aliases for baselines
    # ---------------------------
    # Use epoch-level OHLC if available; otherwise fall back to what we have.
    open_series = epoch_df.get("open_epoch", epoch_df.get("OCHLV_OPEN_startPrice", C)).astype(float)
    high_series = epoch_df.get("high_epoch", epoch_df.get("OCHLV_HIGH_max", C)).astype(float)
    low_series = epoch_df.get("low_epoch", epoch_df.get("OCHLV_LOW_min", C)).astype(float)
    volume_series = epoch_df.get("volume_epoch", epoch_df.get("OCHLV_Volume_sum", 0.0)).astype(float)

    epoch_df["Open"] = open_series
    epoch_df["High"] = high_series
    epoch_df["Low"] = low_series
    epoch_df["Close"] = C
    epoch_df["Volume"] = volume_series

    # ---------------------------
    # Preferred diff series
    # ---------------------------
    if "approx_priceDiff_num" in epoch_df.columns:
        diff = epoch_df["approx_priceDiff_num"].astype(float)
    elif "priceDiff_num" in epoch_df.columns:
        diff = epoch_df["priceDiff_num"].astype(float)
    else:
        # Fallback to simple close-to-close change
        diff = C.diff()

    epoch_df["diff_raw"] = diff

    # ---------------------------
    # Simple EMAs + relative distance
    # ---------------------------
    ema_5 = C.ewm(span=5, adjust=False).mean()
    ema_10 = C.ewm(span=10, adjust=False).mean()
    ema_20 = C.ewm(span=20, adjust=False).mean()

    epoch_df["C_over_ema5"] = C / ema_5 - 1.0
    epoch_df["C_over_ema10"] = C / ema_10 - 1.0
    epoch_df["C_over_ema20"] = C / ema_20 - 1.0

    # ---------------------------
    # Rolling slopes of close (flip-aware-ish)
    # ---------------------------
    epoch_df["slope_C_5"] = _rolling_slope(C, 5)
    epoch_df["slope_C_10"] = _rolling_slope(C, 10)

    # ---------------------------
    # Diff statistics (short tails)
    # ---------------------------
    for w in (5, 10):
        tail = diff.rolling(window=w, min_periods=1)
        epoch_df[f"diff_mean_{w}"] = tail.mean()
        epoch_df[f"diff_std_{w}"] = tail.std().fillna(0.0)

    # ---------------------------
    # RSI-14
    # ---------------------------
    delta = C.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    roll_gain = gain.rolling(window=14, min_periods=14).mean()
    roll_loss = loss.rolling(window=14, min_periods=14).mean()

    rs = roll_gain / (roll_loss.replace(0.0, np.nan))
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))
    epoch_df["rsi_14"] = rsi_14.bfill().ffill()

    # ---------------------------
    # MACD (12/26 EMA) + signal 9, histogram
    # ---------------------------
    ema_fast = C.ewm(span=12, adjust=False).mean()
    ema_slow = C.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    epoch_df["macd_hist"] = macd_hist

    # ---------------------------
    # Bollinger Bands 20: %B and width
    # ---------------------------
    mavg_20 = C.rolling(window=20, min_periods=20).mean()
    std_20 = C.rolling(window=20, min_periods=20).std(ddof=0)

    upper_20 = mavg_20 + 2.0 * std_20
    lower_20 = mavg_20 - 2.0 * std_20
    band_width = upper_20 - lower_20

    # %B = (C - lower) / (upper - lower)
    pctB_20 = (C - lower_20) / (band_width.replace(0.0, np.nan))
    epoch_df["pctB_20"] = pctB_20.clip(lower=0.0, upper=1.0)

    # Relative band width
    epoch_df["bb_width_20"] = band_width / mavg_20.replace(0.0, np.nan)

    # ---------------------------
    # ATR-14 (epoch-level)
    # ---------------------------
    # True range using epoch-level OHLC
    prev_close = C.shift(1)
    tr1 = high_series - low_series
    tr2 = (high_series - prev_close).abs()
    tr3 = (low_series - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_14 = true_range.rolling(window=14, min_periods=14).mean()
    epoch_df["atr_14"] = atr_14

    return epoch_df


# -----------------------------------------------------------------------------
# Training / Evaluation
# -----------------------------------------------------------------------------
@dataclass
class AuSTDModelArtifacts:
    model: ClassifierMixin
    feature_names: List[str]
    metrics: Dict[str, Any]


def train_AuSTD_v1_1(
    epoch_df: pd.DataFrame,
    epoch_tail_eval: int = EPOCH_TAIL_EVAL,
) -> Tuple[Optional[BaseEstimator], Dict[str, Any]]:
    """
    Train AuSTD v1.1 on time-ordered epochs and evaluate ONLY on the last
    `epoch_tail_eval` epochs (tail window), to mimic live-style behavior.

    Pipeline:
        1) Build labeled training universe from epoch_df.
        2) Time-sort by timestamp_epoch (or Current Epoch fallback).
        3) Split into:
               - history: all rows except last `epoch_tail_eval`
               - tail   : last `epoch_tail_eval` rows
        4) Train RF/ET/GB on history, evaluate each on tail.
        5) Pick best-by-tail-accuracy as champion.
        6) Refit champion on ALL labeled epochs and save to disk.
        7) Return champion model + metrics dict.

    Notes:
        - Label is 3-class 'direction_3' → {0: Bear, 1: Neutral, 2: Bull}.
        - Feature set is STAGE1_FEATS + STAGE2_FEATS (only columns present).
        - If data is too small to make a meaningful tail window, we fall back
          to a simple 70/30 time-ordered split, but with the SAME logging
          shape so the rest of the pipeline doesn't change.
    """
    if epoch_df is None or epoch_df.empty:
        logging.warning("⚠️ train_AuSTD_v1_1: epoch_df is empty; skipping training.")
        return None, {}

    label_col = "direction_3"

    # ----------------------------------------------------------------------
    # 1) Build labeled training universe
    # ----------------------------------------------------------------------
    # Start from a copy so we don't mutate caller's frame
    df = epoch_df.copy()

    # Ensure label exists
    if label_col not in df.columns:
        logging.error(
            "❌ train_AuSTD_v1_1: label column '%s' not found; columns=%s",
            label_col,
            list(df.columns),
        )
        return None, {}

    # Build feature list from STAGE1 + STAGE2, but keep only existing columns
    base_feature_cols = list(STAGE1_FEATS) + list(STAGE2_FEATS)
    feature_cols = [c for c in base_feature_cols if c in df.columns]

    if not feature_cols:
        logging.error("❌ train_AuSTD_v1_1: no feature columns found in epoch_df.")
        return None, {}

    # Drop rows with missing label
    df = df.dropna(subset=[label_col])
    n_total = len(epoch_df)
    n_labeled = len(df)

    if n_labeled == 0:
        logging.warning(
            "⚠️ train_AuSTD_v1_1: no rows with label '%s'; n_total=%d",
            label_col,
            n_total,
        )
        return None, {}

    logging.info(
        "📦 Epoch inventory: total built=%d, with valid labels=%d/%d",
        n_total,
        n_labeled,
        n_total,
    )

    # ----------------------------------------------------------------------
    # 2) Time-order the training universe
    # ----------------------------------------------------------------------
    if "timestamp_epoch" in df.columns:
        df = df.sort_values("timestamp_epoch").reset_index(drop=True)
    elif "Current Epoch" in df.columns:
        # Fallback ordering if epoch index is all we have
        df = df.sort_values("Current Epoch").reset_index(drop=True)

    # Sanity: drop rows with NaN in any feature we care about
    df = df.dropna(subset=feature_cols)

    if df.empty:
        logging.warning(
            "⚠️ train_AuSTD_v1_1: after dropping NaNs in features, no rows remain."
        )
        return None, {}

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df[label_col].astype(int).to_numpy()

    n_all = len(y_all)

    # ----------------------------------------------------------------------
    # 3) Split into history vs tail (tail eval)
    # ----------------------------------------------------------------------
    # Tail size is min(epoch_tail_eval, n_all - 1) so there's at least 1 row in history
    if n_all <= 1:
        logging.warning(
            "⚠️ train_AuSTD_v1_1: only %d labeled rows; cannot form history/tail split.",
            n_all,
        )
        return None, {}

    tail_n = min(epoch_tail_eval, max(1, n_all - 1))
    hist_n = n_all - tail_n

    if hist_n < 1:
        # Degenerate: history would be empty; fall back to 70/30 style split
        n_train = int(n_all * 0.7)
        n_train = max(1, min(n_all - 1, n_train))
        logging.warning(
            "⚠️ train_AuSTD_v1_1: insufficient rows for tail split (n_all=%d); "
            "falling back to 70/30 time-based split with n_train=%d",
            n_all,
            n_train,
        )
        X_hist, X_tail = X_all[:n_train], X_all[n_train:]
        y_hist, y_tail = y_all[:n_train], y_all[n_train:]
        split_mode = "time_ordered_70_30_fallback"
    else:
        X_hist, X_tail = X_all[:hist_n], X_all[hist_n:]
        y_hist, y_tail = y_all[:hist_n], y_all[hist_n:]
        split_mode = "tail_eval"
        logging.info(
            "⏱ Time-based split (tail eval): train=%d (oldest), tail_eval=%d (newest, last %d)",
            hist_n,
            tail_n,
            tail_n,
        )

    # ----------------------------------------------------------------------
    # 4) Log label distribution
    # ----------------------------------------------------------------------
    n_bear = int((y_all == 0).sum())
    n_neu = int((y_all == 1).sum())
    n_bull = int((y_all == 2).sum())

    logging.info(
        "🔢 Training universe: %d epochs with %d features (Bear=0: %d, Neutral=1: %d, Bull=2: %d)",
        n_all,
        len(feature_cols),
        n_bear,
        n_neu,
        n_bull,
    )

    # ----------------------------------------------------------------------
    # 5) Train multiple classifier kinds on HISTORY, evaluate on TAIL
    # ----------------------------------------------------------------------
    kinds = ("rf", "gb")
    models_hist: Dict[str, Any] = {}
    all_metrics: Dict[str, Dict[str, Any]] = {}

    for kind in kinds:
        logging.info("🚀 Training classifier kind='%s' on history only ...", kind)
        clf = make_classifier(kind)
        clf.fit(X_hist, y_hist)

        if len(y_tail) == 0:
            # No tail to evaluate; just record a placeholder
            acc = float("nan")
            cm = np.zeros((3, 3), dtype=int)
            logging.warning(
                "⚠️ No tail rows for evaluation; metrics for kind='%s' will be NaN.", kind
            )
        else:
            y_pred = clf.predict(X_tail)
            acc = float(accuracy_score(y_tail, y_pred))
            cm = confusion_matrix(y_tail, y_pred, labels=[0, 1, 2])

            logging.info(
                "📊 [%s] accuracy (tail eval): %.4f (rows=true, cols=pred) confusion matrix:\n%s",
                kind,
                acc,
                cm,
            )

        models_hist[kind] = clf
        all_metrics[kind] = {
            "kind": kind,
            "accuracy_tail": acc,
            "confusion_matrix_tail": cm.tolist(),
            "n_hist": int(len(y_hist)),
            "n_tail": int(len(y_tail)),
            "class_dist_hist": {
                "0": int((y_hist == 0).sum()),
                "1": int((y_hist == 1).sum()),
                "2": int((y_hist == 2).sum()),
            },
            "class_dist_tail": {
                "0": int((y_tail == 0).sum()),
                "1": int((y_tail == 1).sum()),
                "2": int((y_tail == 2).sum()),
            },
        }

    # ----------------------------------------------------------------------
    # 6) Pick best model by tail accuracy
    # ----------------------------------------------------------------------
    # Filter out NaN accuracies if any
    def _acc_or_minus_inf(info: Dict[str, Any]) -> float:
        acc = info.get("accuracy_tail", float("nan"))
        return acc if np.isfinite(acc) else float("-inf")

    best_kind, best_info = max(all_metrics.items(), key=lambda kv: _acc_or_minus_inf(kv[1]))
    best_hist_model = models_hist[best_kind]
    logging.info("⭐ Primary AuSTD v1.1 model kind='%s' selected (by tail accuracy).", best_kind)

    # ----------------------------------------------------------------------
    # 7) Refit champion on ALL labeled data and save
    # ----------------------------------------------------------------------
    champion = make_classifier(best_kind)
    champion.fit(X_all, y_all)

    joblib.dump(champion, MODEL_PATH)
    logging.info(
        "💾 Saved primary AuSTD v1.1 model (kind='%s') to %s",
        best_kind,
        MODEL_PATH,
    )

    # Build metrics payload
    metrics_payload: Dict[str, Any] = {
        "n_epochs_total": int(n_total),
        "n_epochs_labeled": int(n_labeled),
        "label_name": label_col,
        "feature_names": feature_cols,
        "primary_kind": best_kind,
        "split": {
            "mode": split_mode,
            "tail_size_requested": int(epoch_tail_eval),
            "train_count": int(len(y_hist)),
            "tail_count": int(len(y_tail)),
        },
        "models": all_metrics,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logging.info("💾 Saved metrics (all models) to %s", METRICS_PATH)
    logging.info("✅ AuSTD v1.1 training run complete.")

    return champion, metrics_payload

# -------------------------------------------------------------------------
# Convenience: one-shot training wrapper for live scheduler
# -------------------------------------------------------------------------
def run_AuSTD_training_once(quiet: bool = False) -> float:
    """One-shot training helper for the live runtime.

    Steps:
        1) Load merged epoch CSV (DB_epoch_ochlv_merged.csv).
        2) Build epoch_df (Stage 1 features).
        3) Add Stage 2 / flip-aware features.
        4) Train AuSTD v1.1 with tail-eval logic (RF/GB).
        5) Run tail sanity check:
               - model vs Baseline A/B/C on last EPOCH_TAIL_EVAL epochs.
               - choose CURRENT_TAIL_ENGINE as winner (baseline-only).
        6) Log compact summary line:
               "Using Training classifier [kind] accuracy: XX.XX%%"

    Parameters
    ----------
    quiet : bool
        If True, temporarily raises the global log level to WARNING for the
        duration of this function, so INFO-level training logs are suppressed.
        WARN/ERROR messages will still be emitted.

    Returns
    -------
    best_acc : float
        Primary model's tail accuracy in [0, 1].
    """
    global CURRENT_TAIL_ENGINE, CURRENT_TAIL_STATS

    # Optional quiet mode: suppress INFO logs during training pass
    root_logger = logging.getLogger()
    old_level = root_logger.level
    if quiet:
        root_logger.setLevel(logging.WARNING)

    try:
        # 1) Load merged CSV
        logging.info(
            "🚀 [Live] Starting AuSTD v1.1 training pass from merged CSV: %s",
            EPOCH_OCHLV_CSV,
        )
        merged = load_merged_csv(EPOCH_OCHLV_CSV)

        # 2) Build epoch_df (Stage 1 features)
        epoch_df = build_epoch_df(merged)

        # 3) Add Stage 2 / flip-aware features
        epoch_df = add_stage2_features(epoch_df)
        logging.info("✅ Built epoch_df with %d epochs after guards.", len(epoch_df))

        # 4) Train AuSTD v1.1 (tail-eval inside)
        model_obj, metrics = train_AuSTD_v1_1(epoch_df)

        primary_kind = metrics.get("primary_kind", "rf")
        models_info = metrics.get("models", {})
        primary_info = models_info.get(primary_kind, {})

        # Prefer tail accuracy if present; fall back to generic "accuracy"
        if "accuracy_tail" in primary_info:
            best_acc = float(primary_info.get("accuracy_tail", 0.0))
        else:
            best_acc = float(primary_info.get("accuracy", 0.0))

        # 5) Tail sanity check: model vs baselines on last EPOCH_TAIL_EVAL epochs
        try:
            tail_stats = _tail_sanity_eval(
                epoch_df,
                epoch_tail_eval=EPOCH_TAIL_EVAL,
                label_col="direction_3",
            )
        except Exception as e:
            logging.warning(
                "⚠️ Tail baseline evaluation failed; keeping CURRENT_TAIL_ENGINE as-is. Error: %s",
                e,
            )
            tail_stats = {}

        if tail_stats:
            CURRENT_TAIL_STATS = tail_stats
            winner = tail_stats.get("winner")
            if winner is not None:
                CURRENT_TAIL_ENGINE = winner

            logging.info(
                "🧪 Tail sanity (last %d epochs): "
                "model=%.4f (diag), A=%.4f, B=%.4f, C=%.4f -> winner=%s",
                tail_stats.get("tail_size", 0),
                tail_stats.get("acc_model", float("nan")),
                tail_stats.get("acc_A", float("nan")),
                tail_stats.get("acc_B", float("nan")),
                tail_stats.get("acc_C", float("nan")),
                winner,
            )

        # 6) Log the compact summary line for the primary ML model
        logging.info(
            "Using Training classifier [%s] accuracy: %.2f%%",
            primary_kind,
            best_acc * 100.0,
        )

        return best_acc

    finally:
        # Restore original log level even if something blows up
        if quiet:
            root_logger.setLevel(old_level)


# -----------------------------------------------------------------------------
# Live Inference Helper
# -----------------------------------------------------------------------------
def compute_features_for_live_epoch(epoch_block: pd.DataFrame,
                                    epoch_history: pd.DataFrame) -> List[float]:
    """
    Compute the full AuSTD feature vector for a LIVE epoch window.

    Inputs:
        epoch_block  -> OCHLV rows between lastEpochTime → now
        epoch_history -> offline-style epoch_df with at least:
            - endPrice_num, priceDiff_num, volume_epoch
            - ideally: approx_priceDiff_num, direction, O/H/L/C

    Output:
        feat_vals -> List[len(STAGE1_FEATS) + len(STAGE2_FEATS)]
                     in the exact order of STAGE1_FEATS + STAGE2_FEATS.

    NOTE: At the end we sanitize all features to ensure no NaN/inf values are
    passed into sklearn (important for GradientBoostingClassifier).
    """
    total_feats = len(STAGE1_FEATS) + len(STAGE2_FEATS)

    # --------------------------------------------------------
    # 0. Basic guards
    # --------------------------------------------------------
    if epoch_block is None or len(epoch_block) < 5:
        logging.error("❌ epoch_block too small to compute features. Using zero vector.")
        return [0.0] * total_feats

    if ("endPrice_num" not in epoch_history.columns) or ("priceDiff_num" not in epoch_history.columns):
        logging.error("❌ epoch_history missing columns needed for Stage 2 features.")
        return [0.0] * total_feats

    # --------------------------------------------------------
    # 1. Extract O, H, L, C, V (convert to float)
    # --------------------------------------------------------
    try:
        opens = epoch_block["Open"].astype(float)
        highs = epoch_block["High"].astype(float)
        lows = epoch_block["Low"].astype(float)
        closes = epoch_block["Close"].astype(float)
        volume = epoch_block["Volume"].astype(float)
    except Exception as e:
        logging.error(f"❌ Failed to convert epoch_block columns to float: {e}")
        return [0.0] * total_feats

    # --------------------------------------------------------
    # 2. Stage 1 — Microstructure features (same as offline)
    # --------------------------------------------------------

    # 2.1 Body fraction (of the last tick's candle inside the block)
    body = closes - opens
    body_frac = float(body.iloc[-1] / (highs.iloc[-1] - lows.iloc[-1] + 1e-9))

    # 2.2 Realized volatility (per 5-min block)
    logret = np.log(closes).diff()
    realized_vol_block = float(np.sqrt(np.nansum(logret ** 2)))

    # 2.3 Max run-up / drawdown (normalized to P0)
    window_close = closes.values
    max_price = np.maximum.accumulate(window_close)
    min_price = np.minimum.accumulate(window_close)
    max_runup_norm = float((window_close[-1] - min_price[-1]) / (min_price[-1] + 1e-9))
    max_drawdown_norm = float((max_price[-1] - window_close[-1]) / (max_price[-1] + 1e-9))

    # 2.4 Volume Z-score (20)
    vol_series = volume
    vol_mean = vol_series.rolling(20).mean().iloc[-1]
    vol_std = vol_series.rolling(20).std().iloc[-1]
    vol_z_20 = float(
        (vol_series.iloc[-1] - vol_mean) / (vol_std + 1e-9)
        if not np.isnan(vol_mean)
        else 0.0
    )

    # 2.5 Range-normalized end-close (within this block)
    range_norm_end = float(
        (closes.iloc[-1] - lows.min()) /
        (highs.max() - lows.min() + 1e-9)
    )

    # 2.6 Volume epoch total
    vol_epoch = float(volume.sum())

    # --------------------------------------------------------
    # 3. Stage 2 — Indicators using endPrice_num history
    # --------------------------------------------------------

    # Build causal history for closes (offline style)
    C_hist = epoch_history["endPrice_num"].astype(float)
    # Append the latest close from this epoch window
    C_live = pd.concat([C_hist, closes.tail(1)], ignore_index=True)
    C = C_live
    lastC = float(C.iloc[-1])

    # --- EMAs ---
    ema_5 = C.ewm(span=5, adjust=False).mean()
    ema_10 = C.ewm(span=10, adjust=False).mean()

    C_over_ema5 = float(lastC / (ema_5.iloc[-1] + 1e-9) - 1.0)
    C_over_ema10 = float(lastC / (ema_10.iloc[-1] + 1e-9) - 1.0)

    # --- slopes (5, 10) ---
    slope_series_5 = _rolling_slope(C, 5)
    slope_series_10 = _rolling_slope(C, 10)
    slope_C_5_val = slope_series_5.iloc[-1]
    slope_C_10_val = slope_series_10.iloc[-1]
    slope_C_5 = float(0.0 if np.isnan(slope_C_5_val) else slope_C_5_val)
    slope_C_10 = float(0.0 if np.isnan(slope_C_10_val) else slope_C_10_val)

    # --- RSI(14) ---
    delta = C.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs_series = roll_up / (roll_down + 1e-9)
    rs_last = rs_series.iloc[-1]
    rsi_14 = float(100.0 - 100.0 / (1.0 + rs_last)) if not np.isnan(rs_last) else 0.0

    # --- MACD (12, 26, 9) ---
    ema_12 = C.ewm(span=12, adjust=False).mean()
    ema_26 = C.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = float(macd_line.iloc[-1])
    macd_hist = float(macd_line.iloc[-1] - macd_signal.iloc[-1])

    # --- Bollinger %B and width (20) ---
    bb_mid_series = C.rolling(20).mean()
    bb_std_series = C.rolling(20).std()
    bb_mid = bb_mid_series.iloc[-1]
    bb_std = bb_std_series.iloc[-1]

    if np.isnan(bb_mid) or bb_std == 0:
        pctB_20 = 0.0
        bb_width_20 = 0.0
    else:
        bb_up = bb_mid + 2 * bb_std
        bb_lo = bb_mid - 2 * bb_std
        pctB_20 = float((lastC - bb_lo) / (bb_up - bb_lo + 1e-9))
        bb_width_20 = float((bb_up - bb_lo) / (abs(lastC) + 1e-9))

    # --- ret_1 / ret_3 (causal) ---
    C_filled = C.ffill()
    if len(C_filled) >= 2:
        ret_1_val = float(C_filled.pct_change(1, fill_method=None).iloc[-1])
    else:
        ret_1_val = 0.0

    if len(C_filled) >= 4:
        ret_3_val = float(C_filled.pct_change(3, fill_method=None).iloc[-1])
    else:
        ret_3_val = 0.0

    # --- diff_mean_5 / diff_std_5 (from OFFLINE epoch_history only!) ---
    if "approx_priceDiff_num" in epoch_history.columns:
        diff_hist = epoch_history["approx_priceDiff_num"].astype(float)
    else:
        diff_hist = epoch_history["priceDiff_num"].astype(float)

    if len(diff_hist) >= 5:
        dm5 = float(diff_hist.rolling(5).mean().iloc[-1])
        ds5 = float(diff_hist.rolling(5).std().iloc[-1])
        diff_mean_5_val = float(dm5 / (lastC + 1e-9))
        diff_std_5_val = float(ds5 / (lastC + 1e-9))
    else:
        diff_mean_5_val = 0.0
        diff_std_5_val = 0.0

    # --- ATR(14) using epoch_history OHLC ---
    if all(c in epoch_history.columns for c in ["high_epoch", "low_epoch"]):
        H_hist = epoch_history["high_epoch"].astype(float)
        L_hist = epoch_history["low_epoch"].astype(float)
        C_hist_for_atr = C_hist

        C_prev = C_hist_for_atr.shift(1)
        tr1 = H_hist - L_hist
        tr2 = (H_hist - C_prev).abs()
        tr3 = (L_hist - C_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14_series = tr.ewm(span=14, adjust=False).mean()
        atr_last = atr_14_series.iloc[-1]
        atr_14_val = float(atr_last / (abs(lastC) + 1e-9)) if not np.isnan(atr_last) else 0.0
    else:
        atr_14_val = 0.0

    # --- Wick fractions for the latest completed epoch (from this block) ---
    try:
        open_epoch = float(opens.iloc[0])
        close_epoch = float(closes.iloc[-1])
        high_epoch = float(highs.max())
        low_epoch = float(lows.min())
        range_epoch = high_epoch - low_epoch

        upper_wick_frac = float(
            (high_epoch - max(open_epoch, close_epoch)) /
            (range_epoch + 1e-9)
        )
        lower_wick_frac = float(
            (min(open_epoch, close_epoch) - low_epoch) /
            (range_epoch + 1e-9)
        )
    except Exception:
        upper_wick_frac = 0.0
        lower_wick_frac = 0.0

    # --- Tail quadratic fit (window=7) on C ---
    tail = C.tail(7).values
    if len(tail) >= 3:
        x_tail = np.arange(len(tail))
        try:
            coeffs = np.polyfit(x_tail, tail, 2)
            poly2_a_7 = float(coeffs[0])
            poly2_b_7 = float(coeffs[1])
        except Exception:
            poly2_a_7 = 0.0
            poly2_b_7 = 0.0
    else:
        poly2_a_7 = 0.0
        poly2_b_7 = 0.0

    # --- Large-scale slope / curvature (K=20 epochs) ---
    g1_large_series = _rolling_slope(C, 20)
    g1_last = g1_large_series.iloc[-1]
    g2_large_series = g1_large_series.diff()
    g2_last = g2_large_series.iloc[-1]

    g1_large_20 = float(0.0 if np.isnan(g1_last) else g1_last)
    g2_large_20 = float(0.0 if np.isnan(g2_last) else g2_last)

    # --- Signed run-length (Bull/Bear runs) from history directions ---
    if "direction" in epoch_history.columns:
        dir_hist = epoch_history["direction"].astype(int)
        signed = 2 * dir_hist - 1
        run_val = 0.0
        prev_sign = 0
        for s in signed:
            if s == prev_sign:
                run_val += s
            else:
                run_val = float(s)
                prev_sign = s
        run_length_signed_val = float(run_val)
    else:
        run_length_signed_val = 0.0

    # --------------------------------------------------------
    # 5. Assemble final feature vector (raw)
    #     MUST match STAGE1_FEATS + STAGE2_FEATS EXACT ORDER
    # --------------------------------------------------------
    feat_vals = [
        # ----- Stage 1 -----
        body_frac,
        realized_vol_block,
        max_runup_norm,
        max_drawdown_norm,
        vol_z_20,
        range_norm_end,
        vol_epoch,

        # ----- Stage 2 -----
        C_over_ema5,
        C_over_ema10,
        slope_C_5,
        slope_C_10,
        rsi_14,
        macd_val,
        macd_hist,
        pctB_20,
        ret_1_val,
        ret_3_val,
        diff_mean_5_val,
        diff_std_5_val,
        bb_width_20,
        atr_14_val,
        upper_wick_frac,
        lower_wick_frac,
        poly2_a_7,
        poly2_b_7,
        g1_large_20,
        g2_large_20,
        run_length_signed_val,
    ]

    # --------------------------------------------------------
    # 6. FINAL SANITIZATION: kill any NaN / inf
    # --------------------------------------------------------
    cleaned_feats: List[float] = []
    for v in feat_vals:
        try:
            fv = float(v)
        except Exception:
            fv = 0.0
        if np.isnan(fv) or np.isinf(fv):
            fv = 0.0
        cleaned_feats.append(fv)

    return cleaned_feats

def _tail_sanity_eval(
    epoch_df: pd.DataFrame,
    epoch_tail_eval: int,
    label_col: str = "direction_3",
) -> Dict[str, Any]:
    """
    Evaluate the current saved AuSTD model and Baseline A/B/C on the last
    `epoch_tail_eval` labeled epochs of `epoch_df`.

    IMPORTANT:
        - The AuSTD model's tail accuracy is logged for diagnostics ONLY.
        - The winner used for epoch (n+1) selection is chosen EXCLUSIVELY
          from Baseline A/B/C (the pure rules engines).

    Returns a dict with:
        {
            "tail_size": int,
            "acc_model": float,
            "acc_A": float,
            "acc_B": float,
            "acc_C": float,
            "winner": str,  # "baseline_A", "baseline_B", or "baseline_C"
        }
    """
    # ------------------------------------------------------------------
    # 1. Basic checks and tail selection
    # ------------------------------------------------------------------
    if epoch_df is None or epoch_df.empty:
        logging.warning("⚠️ _tail_sanity_eval: empty epoch_df; skipping.")
        return {}

    if label_col not in epoch_df.columns:
        logging.warning(
            "⚠️ _tail_sanity_eval: missing label_col='%s'; skipping.", label_col
        )
        return {}

    # Drop rows without labels
    df = epoch_df.dropna(subset=[label_col]).copy()

    # Sort by time if possible (timestamp_epoch preferred, else epoch_id / Current Epoch)
    sort_key = None
    for cand in ("timestamp_epoch", "epoch_id", "Current Epoch"):
        if cand in df.columns:
            sort_key = cand
            break

    if sort_key is not None:
        df = df.sort_values(sort_key).reset_index(drop=True)

    n_all = len(df)
    if n_all == 0:
        logging.warning("⚠️ _tail_sanity_eval: no labeled rows after filtering.")
        return {}

    # Initial tail slice (before NaN clean-up)
    tail_n_raw = min(epoch_tail_eval, n_all)
    df_tail = df.iloc[-tail_n_raw:].copy()

    # Alias epoch OHLC columns so baselines B/C can see them
    if "close_epoch" in df_tail.columns and "Close" not in df_tail.columns:
        df_tail["Close"] = df_tail["close_epoch"].astype(float)

    if "open_epoch" in df_tail.columns and "Open" not in df_tail.columns:
        df_tail["Open"] = df_tail["open_epoch"].astype(float)

    if "high_epoch" in df_tail.columns and "High" not in df_tail.columns:
        df_tail["High"] = df_tail["high_epoch"].astype(float)

    if "low_epoch" in df_tail.columns and "Low" not in df_tail.columns:
        df_tail["Low"] = df_tail["low_epoch"].astype(float)

    # ------------------------------------------------------------------
    # 2. Drop rows with NaNs in any columns we need
    #    (model features + baseline inputs)
    # ------------------------------------------------------------------
    base_feature_cols = list(STAGE1_FEATS) + list(STAGE2_FEATS)
    feature_cols = [c for c in base_feature_cols if c in df_tail.columns]

    # Columns needed by baselines
    baseline_needed = [
        "Close", "Open", "High", "Low",     # OHLC for B/C
        "rsi_14", "atr_14",                 # Baseline B
        "pctB_20", "bb_width_20",           # Baseline C
    ]
    baseline_needed = [c for c in baseline_needed if c in df_tail.columns]

    # Build full "must not be NaN" set
    must_have = set(feature_cols + baseline_needed + [label_col])
    must_have = [c for c in must_have if c in df_tail.columns]

    if must_have:
        df_tail_clean = df_tail.dropna(subset=must_have).copy()
    else:
        df_tail_clean = df_tail.copy()

    if df_tail_clean.empty:
        logging.warning(
            "⚠️ _tail_sanity_eval: tail window has no rows after NaN cleanup; "
            "skipping tail sanity evaluation."
        )
        return {}

    tail_n = len(df_tail_clean)
    y_true = df_tail_clean[label_col].astype(int).to_numpy()

    # ------------------------------------------------------------------
    # 3. Model accuracy on tail (saved AuSTD model) — DIAGNOSTIC ONLY
    # ------------------------------------------------------------------
    acc_model = float("nan")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logging.warning(
            "⚠️ _tail_sanity_eval: could not load model from %s: %s",
            MODEL_PATH,
            e,
        )
        model = None

    if model is not None and feature_cols:
        X_tail = df_tail_clean[feature_cols].to_numpy(dtype=float)

        # As a final guard, zero-out any remaining NaN/inf
        if not np.isfinite(X_tail).all():
            X_tail = np.where(np.isfinite(X_tail), X_tail, 0.0)

        try:
            y_pred_model = model.predict(X_tail)
            acc_model = float(accuracy_score(y_true, y_pred_model))
        except Exception as e:
            logging.warning(
                "⚠️ _tail_sanity_eval: model prediction on tail failed: %s", e
            )
            acc_model = float("nan")

    # ------------------------------------------------------------------
    # 4. Baseline accuracies (A/B/C) on the same cleaned tail window
    # ------------------------------------------------------------------
    yA = _baseline_A_predict(df_tail_clean)
    yB = _baseline_B_predict(df_tail_clean)
    yC = _baseline_C_predict(df_tail_clean)

    acc_A = float(accuracy_score(y_true, yA))
    acc_B = float(accuracy_score(y_true, yB))
    acc_C = float(accuracy_score(y_true, yC))

    # ------------------------------------------------------------------
    # 5. Choose winner from BASELINES ONLY (A/B/C)
    # ------------------------------------------------------------------
    baseline_scores = {
        "baseline_A": acc_A,
        "baseline_B": acc_B,
        "baseline_C": acc_C,
    }

    def _score(val: float) -> float:
        return val if np.isfinite(val) else float("-inf")

    winner = max(baseline_scores.items(), key=lambda kv: _score(kv[1]))[0]

    logging.info(
        "🧪 Tail sanity (last %d epochs after cleanup): "
        "model=%.4f (diag), A=%.4f, B=%.4f, C=%.4f -> winner=%s",
        tail_n,
        acc_model,
        acc_A,
        acc_B,
        acc_C,
        winner,
    )

    return {
        "tail_size": int(tail_n),
        "acc_model": acc_model,   # diagnostic only
        "acc_A": acc_A,
        "acc_B": acc_B,
        "acc_C": acc_C,
        "winner": winner,
    }


# -----------------------------------------------------------------------------
# CLI entrypoint (offline training + tail sanity)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # When run as a script, just reuse the same live-style helper.
    logging.info("🚀 Offline AuSTD v1.1 training + tail sanity check starting...")
    best_acc = run_AuSTD_training_once()
    logging.info("✅ AuSTD v1.1 training run complete. Best tail accuracy: %.2f%%", best_acc * 100.0)

