"""
volume_RGbar_15s.py

Red/Green 15-second volume bars aligned to epoch timing.

Spec:
- 15s bins
- aligned to next epoch time so the final bin ends at (next_epoch_time - 15s)
- color: green if bin_close >= bin_open else red
- outputs series for graphing and a latest-bar dict for logging
"""

from __future__ import annotations

import threading
import time
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pytz

# We pull data from your indicators spine
from main.engine.indicators import indicators

EST = pytz.timezone("America/New_York")

BAR_SECONDS = 15
HISTORY_MINUTES = 1000  # overkill history, as requested

_lock = threading.Lock()
_started = False
_thread: Optional[threading.Thread] = None

_last_bars: List[Dict[str, Any]] = []
_last_end_est: Optional[dt.datetime] = None


def _to_est(x: Any) -> Optional[dt.datetime]:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        if x.tzinfo is None:
            return EST.localize(x)
        return x.astimezone(EST)
    # ISO string fallback
    try:
        d = dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=dt.timezone.utc)
        return d.astimezone(EST)
    except Exception:
        return None


def _floor_to_15s(ts: dt.datetime) -> dt.datetime:
    # floor to BAR_SECONDS grid
    sec = int(ts.timestamp())
    sec_f = sec - (sec % BAR_SECONDS)
    return dt.datetime.fromtimestamp(sec_f, tz=ts.tzinfo)


def _compute_latest_bar() -> Optional[Dict[str, Any]]:
    """
    Builds the most-recent 15s bar aligned to the epoch:
      final_end = next_epoch_time - 15s
      bins end every 15s relative to final_end
    """
    snap = indicators.get_epoch_snapshot()
    next_round_time_est = snap.get("next_round_time_est")
    next_epoch_time_est = _to_est(next_round_time_est)
    if not next_epoch_time_est:
        return None

    final_end = next_epoch_time_est - dt.timedelta(seconds=15)

    ticks = indicators.get_tick_series()
    if not ticks:
        return None

    # Use only ticks in a reasonable recent window
    # (history is already big; this just keeps compute light)
    now_est = dt.datetime.now(tz=EST)
    cutoff = now_est - dt.timedelta(minutes=HISTORY_MINUTES)
    ticks = [(t, c, v) for (t, c, v) in ticks if isinstance(t, dt.datetime) and t >= cutoff]

    if not ticks:
        return None

    # Determine which 15s bin we’re currently in, relative to final_end
    # Bin end times are ... final_end - 30, final_end - 15, final_end, ...
    # We compute the newest bin end <= current time or <= final_end?
    # We want bars leading up to final_end; if we're past final_end, we still anchor to it.
    anchor_end = final_end

    # The latest bin end should be <= min(now, anchor_end) if before decision time,
    # but once we're past anchor_end, we still keep building forward bins anchored to it.
    ref = min(now_est, anchor_end) if now_est <= anchor_end else now_est

    # how many 15s steps away from anchor_end is ref?
    delta_s = int((ref - anchor_end).total_seconds())
    steps = delta_s // BAR_SECONDS
    latest_end = anchor_end + dt.timedelta(seconds=steps * BAR_SECONDS)

    # bar start/end
    bar_end = latest_end
    bar_start = bar_end - dt.timedelta(seconds=BAR_SECONDS)

    # collect ticks in (bar_start, bar_end]
    in_bin = [(t, c, v) for (t, c, v) in ticks if bar_start < t <= bar_end]
    if not in_bin:
        # No ticks: keep a “zero” bar, neutral-ish color
        return {
            "ts_end_est": bar_end,
            "ts_end_utc": bar_end.astimezone(dt.timezone.utc),
            "bar_start_est": bar_start,
            "bar_start_utc": bar_start.astimezone(dt.timezone.utc),
            "epoch": snap.get("epoch"),
            "next_epoch": snap.get("next_epoch"),
            "next_round_time_est": snap.get("next_round_time_est"),
            "open": None,
            "close": None,
            "volume": 0.0,
            "color": "rgba(120,120,120,0.55)",
            "is_partial": True,
        }

    in_bin.sort(key=lambda x: x[0])
    open_c = float(in_bin[0][1])
    close_c = float(in_bin[-1][1])
    vol = float(sum(float(v) for (_, _, v) in in_bin))

    color = "rgba(0,200,0,0.65)" if close_c >= open_c else "rgba(220,0,0,0.65)"

    # Partial if we're before bar_end by a lot (i.e., still forming)
    is_partial = now_est < bar_end

    return {
        "ts_end_est": bar_end,
        "ts_end_utc": bar_end.astimezone(dt.timezone.utc),
        "bar_start_est": bar_start,
        "bar_start_utc": bar_start.astimezone(dt.timezone.utc),
        "epoch": snap.get("epoch"),
        "next_epoch": snap.get("next_epoch"),
        "next_round_time_est": snap.get("next_round_time_est"),
        "open": open_c,
        "close": close_c,
        "volume": vol,
        "color": color,
        "is_partial": bool(is_partial),
    }


def _loop():
    global _last_end_est, _last_bars
    # target history size (bars)
    max_bars = int((HISTORY_MINUTES * 60) / BAR_SECONDS) + 50

    while True:
        try:
            bar = _compute_latest_bar()
            if bar is not None:
                end_est = bar.get("ts_end_est")
                if isinstance(end_est, dt.datetime):
                    with _lock:
                        # append only when new end time appears
                        if _last_end_est is None or end_est > _last_end_est:
                            _last_end_est = end_est
                            _last_bars.append(bar)
                            if len(_last_bars) > max_bars:
                                _last_bars = _last_bars[-max_bars:]
        except Exception:
            pass

        time.sleep(2.5)


def start():
    global _started, _thread
    with _lock:
        if _started:
            return
        _started = True
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()


def get_latest_bar() -> Optional[Dict[str, Any]]:
    with _lock:
        if not _last_bars:
            return None
        return dict(_last_bars[-1])


def get_plot_series(x_min_est: dt.datetime, x_max_est: dt.datetime) -> Dict[str, Any]:
    """
    Return bars inside [x_min_est, x_max_est] as arrays for plotting.
    """
    x_min_est = _to_est(x_min_est) or x_min_est
    x_max_est = _to_est(x_max_est) or x_max_est

    with _lock:
        bars = list(_last_bars)

    xs: List[dt.datetime] = []
    ys: List[float] = []
    cs: List[str] = []

    for b in bars:
        t = b.get("ts_end_est")
        if not isinstance(t, dt.datetime):
            continue
        if t < x_min_est or t > x_max_est:
            continue
        xs.append(t)
        ys.append(float(b.get("volume") or 0.0))
        cs.append(str(b.get("color") or "rgba(120,120,120,0.55)"))

    return {
        "ts_end_est": xs,
        "volume": ys,
        "color": cs,
    }
