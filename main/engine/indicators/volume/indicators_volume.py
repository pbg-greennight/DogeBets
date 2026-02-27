# main/engine/indicators/volume/indicators_volume.py

"""
Volume indicator module (Phase 1/2/3).

Consumes bar-aligned data from indicators.indicators (no server_hub fetch).
Logs volume metrics via indicators_log.log_volume_bar() per new bar.

Also maintains a rolling in-memory series for graphing:
- CVD (cumulative volume delta) as IND_Volume
- CVD_body (body-weighted CVD) for better buy/sell pressure proxy
- (optional) vol_delta, vol_z for future toggles
"""

import threading
import time
from collections import deque
from math import sqrt
from typing import Any, Dict, Optional

from main.engine.graphing.indicators import indicators_log
from main.engine.graphing.indicators import indicators

_started = False
_lock = threading.Lock()

# Rolling stats window for volume z-score
VOL_Z_WINDOW = 240  # ~10 minutes at ~2.5s bars (approx). Tune later.

# Plot buffer length (roughly > 30 minutes at ~2.5s bars)
PLOT_MAXLEN = 1500

_vol_hist = deque(maxlen=VOL_Z_WINDOW)

# each item:
# (ts_est_dt, cvd, cvd_body, vol_delta, vol_delta_body, vol_z)
_plot_buf = deque(maxlen=PLOT_MAXLEN)

_plot_lock = threading.Lock()

_cvd = 0.0
_cvd_body = 0.0
_last_ts_utc: Optional[str] = None


def _zscore(x: float, series: deque) -> Optional[float]:
    if x is None:
        return None
    if len(series) < 30:
        return None
    mu = sum(series) / len(series)
    var = sum((v - mu) ** 2 for v in series) / max(len(series) - 1, 1)
    sd = sqrt(var) if var > 0 else 0.0
    if sd == 0.0:
        return 0.0
    return (x - mu) / sd


def compute_volume_payload(bar: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the computed volume features for the given bar.
    Also updates global CVD and CVD_body.
    """
    global _cvd, _cvd_body

    eps = 1e-9
    h = bar.get("high")
    l = bar.get("low")
    o = bar.get("open")
    c = bar.get("close")
    v = bar.get("volume") or 0.0

    # -------------------------
    # CVD_body (better proxy)
    # -------------------------
    vol_delta_body = 0.0
    if None not in (h, l, o, c):
        rng = float(h) - float(l)
        w = (float(c) - float(o)) / (rng + eps)  # approx [-1..1]
        # clamp for safety
        if w > 1.0:
            w = 1.0
        elif w < -1.0:
            w = -1.0
        vol_delta_body = float(v) * w
        _cvd_body += vol_delta_body

    # -------------------------
    # Existing CVD (simple)
    # -------------------------
    vol_up = 0.0
    vol_down = 0.0

    if o is not None and c is not None:
        if c > o:
            vol_up = float(v)
        elif c < o:
            vol_down = float(v)
        else:
            vol_up = float(v) * 0.5
            vol_down = float(v) * 0.5

    vol_delta = vol_up - vol_down
    _cvd += vol_delta

    # -------------------------
    # Volume regime metrics
    # -------------------------
    _vol_hist.append(float(v))
    vol_z = _zscore(float(v), _vol_hist)

    vol_impact = (abs(c - o) * float(v)) if (o is not None and c is not None) else None

    # Phase 3 starter flags
    price_dir = 0
    if o is not None and c is not None:
        price_dir = 1 if c > o else (-1 if c < o else 0)

    vol_dir = 1 if vol_delta > 0 else (-1 if vol_delta < 0 else 0)
    vol_price_agree = 1 if (price_dir != 0 and price_dir == vol_dir) else 0

    vol_divergence_flag = 1 if (price_dir != 0 and vol_dir != 0 and price_dir != vol_dir) else 0
    vol_note = "divergence" if vol_divergence_flag else ""

    return {
        "vol_up": vol_up,
        "vol_down": vol_down,
        "vol_delta": vol_delta,
        "cvd": _cvd,

        # ✅ NEW
        "vol_delta_body": vol_delta_body,
        "cvd_body": _cvd_body,

        "vol_z": vol_z,
        "vol_impact": vol_impact,
        "vol_price_agree": vol_price_agree,
        "vol_divergence_flag": vol_divergence_flag,
        "vol_note": vol_note,
    }


def get_plot_series() -> Dict[str, Any]:
    """
    Returns plot-ready rolling series for graphrounds.

    Output keys:
      - ts: list[datetime] (EST)
      - cvd: list[float]
      - cvd_body: list[float]
      - vol_delta: list[float]
      - vol_delta_body: list[float]
      - vol_z: list[float|None]
    """
    with _plot_lock:
        ts = [x[0] for x in _plot_buf]
        cvd = [x[1] for x in _plot_buf]
        cvd_body = [x[2] for x in _plot_buf]
        vol_delta = [x[3] for x in _plot_buf]
        vol_delta_body = [x[4] for x in _plot_buf]
        vol_z = [x[5] for x in _plot_buf]

    return {
        "ts": ts,
        "cvd": cvd,
        "cvd_body": cvd_body,
        "vol_delta": vol_delta,
        "vol_delta_body": vol_delta_body,
        "vol_z": vol_z,
    }


def _loop():
    global _last_ts_utc

    while True:
        bar = indicators.get_latest_bar()
        if not bar or not bar.get("ts_utc"):
            time.sleep(0.25)
            continue

        ts_utc = bar["ts_utc"]
        if ts_utc == _last_ts_utc:
            time.sleep(0.25)
            continue

        # New bar arrived
        epoch_snap = indicators.get_epoch_snapshot()
        payload = compute_volume_payload(bar)

        # Update in-memory plot buffer
        ts_est_dt = bar.get("ts_est_dt")
        if ts_est_dt is not None:
            with _plot_lock:
                _plot_buf.append(
                    (
                        ts_est_dt,
                        payload["cvd"],
                        payload["cvd_body"],
                        payload["vol_delta"],
                        payload["vol_delta_body"],
                        payload["vol_z"],
                    )
                )

        # Log to CSV
        ochlv = {
            "open": bar.get("open"),
            "high": bar.get("high"),
            "low": bar.get("low"),
            "close": bar.get("close"),
            "volume": bar.get("volume"),
        }
        indicators_log.log_volume_bar(ts_utc, ochlv, epoch_snap, payload)

        _last_ts_utc = ts_utc


def start():
    global _started
    with _lock:
        if _started:
            return
        indicators.start()
        threading.Thread(target=_loop, daemon=True).start()
        _started = True
