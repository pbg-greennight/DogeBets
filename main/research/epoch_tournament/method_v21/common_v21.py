from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SIGMAS_ALL = [8, 23, 38, 53, 68, 83]
SIGMAS_FAST = [8, 23, 38]
SIGMAS_SLOW = [53, 68, 83]
SIGMAS_CORE = [23, 38, 53, 68, 83]

STATE_RISK_MAP = {
    "continuation_friendly": 0.00,
    "watch": 0.35,
    "fragile": 0.65,
    "break_risk": 1.00,
}
PRESSURE_MAP = {
    "calm": 0.15,
    "balanced": 0.35,
    "building": 0.60,
    "stressed": 0.85,
    "unstable": 1.00,
}
SPREAD_STATE_MAP = {
    "stable": 0.15,
    "frozen": 0.35,
    "widening": 0.20,
    "shrinking": 0.55,
    "compressing": 0.75,
    "near_cross": 0.90,
    "cross_risk": 1.00,
}
TRANSFER_DIR_MAP = {"up": 1.0, "down": -1.0, "none": 0.0, "mixed": 0.0, "flat": 0.0}
TRANSFER_STATE_MAP = {"none": 0.00, "shallow": 0.30, "deep": 0.65, "full": 1.00, "split": 0.50}
GCS_REGIME_MAP = {"contracting": -1.0, "expanding": 1.0, "flat": 0.0, "stable": 0.0, "mixed": 0.0}


def clip01(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return float("nan")


def clip11(x: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(x)))
    except Exception:
        return float("nan")


def safe_div(a: Any, b: Any, default: float = 0.0) -> float:
    try:
        af = float(a)
        bf = float(b)
        return af / bf if abs(bf) > 1e-12 else default
    except Exception:
        return default


def sign3(x: Any, eps: float = 1e-12) -> int:
    try:
        xf = float(x)
    except Exception:
        return 0
    if xf > eps:
        return 1
    if xf < -eps:
        return -1
    return 0


def bool01(flag: Any) -> float:
    return 1.0 if bool(flag) else 0.0


def mean_safe(values: list[Any]) -> float:
    vals = [float(v) for v in values if pd.notna(v)]
    return float(np.mean(vals)) if vals else float("nan")
