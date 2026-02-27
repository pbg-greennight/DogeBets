# main/engine/indicators/gaussian/indicators_gauss23.py
"""
IND_gauss_23 (VIS)

Centered (non-causal) Gaussian smoothing of BTC close using sigma=23 (in bars).
This is intended for VISUALIZATION: the curve will "re-shape" as new data arrives,
because we recompute the full-window centered convolution each update.

- Consumes data from indicators.py (no server_hub fetch)
- Maintains rolling plot series (ts, g23) same length as BTC_close window
- Logs one row per new bar via indicators_log.log_gauss23_bar()
"""

import threading
import time
from math import exp
from typing import List, Optional, Dict, Any
from main.engine.indicators import indicators_log
from main.engine.indicators import indicators

SIGMA = 23.0
RADIUS_SIGMAS = 4  # truncate kernel at +/- 4σ


_started = False
_lock = threading.Lock()

_plot_lock = threading.Lock()

# IMPORTANT: store plot data in one shared mutable object (no rebind bugs)
_plot_state: Dict[str, List[Any]] = {
    "ts": [],   # list[datetime] EST
    "g23": [],  # list[float|None]
}

_last_ts_utc: Optional[str] = None

print("GAUSS23 sees indicators from:", getattr(indicators, "__file__", "<no file>"))


def _gaussian_kernel_centered(sigma: float) -> List[float]:
    r = int(RADIUS_SIGMAS * sigma)
    if r < 1:
        return [1.0]
    w = [exp(-0.5 * (k / sigma) ** 2) for k in range(-r, r + 1)]
    s = sum(w) or 1.0
    return [x / s for x in w]


_KERNEL = _gaussian_kernel_centered(SIGMA)


def _gaussian_smooth_centered(x: List[float], kernel: List[float]) -> List[Optional[float]]:
    """
    Centered Gaussian with EDGE HANDLING:
    - For indices where the full kernel doesn't fit, use the overlapping part only
      and renormalize weights.
    This fills BOTH the left and right edges (no gaps), and still repaints as new
    data arrives.
    """
    n = len(x)
    m = len(kernel)
    r = m // 2

    out: List[Optional[float]] = [None] * n
    if n == 0:
        return out

    for i in range(n):
        # Determine overlap window in x
        left = max(0, i - r)
        right = min(n - 1, i + r)

        # Determine corresponding kernel slice
        k_left = left - (i - r)          # how far into kernel we start
        k_right = k_left + (right - left)

        acc = 0.0
        wsum = 0.0

        # Apply truncated kernel and renormalize
        for xi, kj in zip(range(left, right + 1), range(k_left, k_right + 1)):
            w = kernel[kj]
            acc += x[xi] * w
            wsum += w

        out[i] = float(acc / wsum) if wsum > 0 else None

    return out


def get_plot_series() -> Dict[str, Any]:
    """
    Returns full-window series for plotting:
      ts: list[datetime] (EST)
      g23: list[float|None] same length as ts (None at edges)
    """
    with _plot_lock:
        return {"ts": list(_plot_state["ts"]), "g23": list(_plot_state["g23"])}


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

        # Only recompute when a NEW bar arrives
        if ts_utc == _last_ts_utc:
            time.sleep(0.25)
            continue

        snap, _ = indicators.get_processed_snapshot()
        ts_list = snap.get("timestamp", [])
        close_raw = snap.get("BTC_close", [])

        # If no window yet, just wait
        if not ts_list or not close_raw:
            time.sleep(0.25)
            continue

        try:
            close_list = [float(c) for c in close_raw]
        except Exception:
            time.sleep(0.25)
            continue

        g23_series = _gaussian_smooth_centered(close_list, _KERNEL)

        # ✅ Update shared plot state (NO rebinding)
        with _plot_lock:
            _plot_state["ts"] = list(ts_list)
            _plot_state["g23"] = list(g23_series)

        # Log one value per bar (right-edge of centered filter is usually None)
        latest_g23 = g23_series[-1] if g23_series else None

        epoch_snap = indicators.get_epoch_snapshot()
        ochlv = {
            "open": bar.get("open"),
            "high": bar.get("high"),
            "low": bar.get("low"),
            "close": bar.get("close"),
            "volume": bar.get("volume"),
        }
        indicators_log.log_gauss23_bar(ts_utc, ochlv, epoch_snap, {"gauss_23": latest_g23})

        _last_ts_utc = ts_utc


def start():
    global _started
    with _lock:
        if _started:
            return
        indicators.start()
        threading.Thread(target=_loop, daemon=True).start()
        _started = True
