# main/engine/indicators/gaussian/indicators_gauss68.py
"""
IND_gauss_68 (VIS)

Centered (non-causal) Gaussian smoothing of BTC close using sigma=68 (in bars),
with edge-safe truncation + renormalization so the curve renders to the ends.

- Consumes data from indicators.py
- Maintains rolling plot series (ts, g68)
- Logs one row per new bar via indicators_log.log_gauss68_bar()
"""

import threading
import time
from math import exp
from typing import List, Optional, Dict, Any

from main.engine.indicators import indicators_log
from main.engine.indicators import indicators

SIGMA = 68.0
RADIUS_SIGMAS = 4  # truncate kernel at +/- 4σ

_started = False
_lock = threading.Lock()

_plot_lock = threading.Lock()
_plot_state: Dict[str, List[Any]] = {"ts": [], "g68": []}

_last_ts_utc: Optional[str] = None

print("GAUSS68 sees indicators from:", getattr(indicators, "__file__", "<no file>"))


def _gaussian_kernel_centered(sigma: float) -> List[float]:
    r = int(RADIUS_SIGMAS * sigma)
    if r < 1:
        return [1.0]
    w = [exp(-0.5 * (k / sigma) ** 2) for k in range(-r, r + 1)]
    s = sum(w) or 1.0
    return [x / s for x in w]


_KERNEL = _gaussian_kernel_centered(SIGMA)


def _gaussian_smooth_centered_edge_safe(x: List[float], kernel: List[float]) -> List[Optional[float]]:
    """
    Centered Gaussian with EDGE HANDLING (truncate + renormalize).
    Returns list same length as x and fills edges (no gaps).
    """
    n = len(x)
    m = len(kernel)
    r = m // 2

    out: List[Optional[float]] = [None] * n
    if n == 0:
        return out

    for i in range(n):
        left = max(0, i - r)
        right = min(n - 1, i + r)

        k_left = left - (i - r)
        k_right = k_left + (right - left)

        acc = 0.0
        wsum = 0.0
        for xi, kj in zip(range(left, right + 1), range(k_left, k_right + 1)):
            w = kernel[kj]
            acc += x[xi] * w
            wsum += w

        out[i] = float(acc / wsum) if wsum > 0 else None

    return out


def get_plot_series() -> Dict[str, Any]:
    with _plot_lock:
        return {"ts": list(_plot_state["ts"]), "g68": list(_plot_state["g68"])}


def _loop():
    global _last_ts_utc

    while True:
        bar = indicators.get_latest_bar()
        if not bar:
            time.sleep(0.25)
            continue

        ts_utc = bar.get("ts_utc")
        if not ts_utc:
            time.sleep(0.25)
            continue

        if ts_utc == _last_ts_utc:
            time.sleep(0.25)
            continue

        snap, _ = indicators.get_processed_snapshot()
        ts_list = snap.get("timestamp", [])
        close_raw = snap.get("BTC_close", [])

        if not ts_list or not close_raw:
            time.sleep(0.25)
            continue

        try:
            close_list = [float(c) for c in close_raw]
        except Exception:
            time.sleep(0.25)
            continue

        g68_series = _gaussian_smooth_centered_edge_safe(close_list, _KERNEL)

        with _plot_lock:
            _plot_state["ts"] = list(ts_list)
            _plot_state["g68"] = list(g68_series)

        latest_g68 = g68_series[-1] if g68_series else None

        epoch_snap = indicators.get_epoch_snapshot()
        ochlv = {
            "open": bar.get("open"),
            "high": bar.get("high"),
            "low": bar.get("low"),
            "close": bar.get("close"),
            "volume": bar.get("volume"),
        }
        indicators_log.log_gauss68_bar(ts_utc, ochlv, epoch_snap, {"gauss_68": latest_g68})

        _last_ts_utc = ts_utc


def start():
    global _started
    with _lock:
        if _started:
            return
        indicators.start()
        threading.Thread(target=_loop, daemon=True).start()
        _started = True
