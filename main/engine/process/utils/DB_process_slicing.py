"""main/engine/process/DB_process_slicing.py

All window slicing helpers.

Includes tail-window fallback behavior used by printing and wide CSV.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Tuple


def slice_tail_window_with_fallback(
    ts_list: List[datetime],
    values_list: List[Any],
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[List[datetime], List[Any]]:
    """Tail-window slicer (best-effort, numeric-safe).

    1) Normal behavior: return points where start_dt <= ts <= end_dt,
       BUT only keep points with a valid numeric value.

    2) If that yields 0 points, fallback to the most recent VALID numeric point
       with ts <= end_dt (even if ts < start_dt).

    This prevents 5-second windows from printing NA simply because:
      - the last sample is slightly older than start_dt, OR
      - the most recent sample(s) have value=None while earlier samples are valid.

    Intended ONLY for tail snapshot windows.
    """

    n = min(len(ts_list), len(values_list))
    if n <= 0:
        return [], []

    ts_list = ts_list[:n]
    values_list = values_list[:n]

    def _is_valid_number(v: Any) -> bool:
        try:
            if v is None:
                return False
            fv = float(v)
            # reject NaN
            return fv == fv
        except Exception:
            return False

    # 1) Collect valid numeric points within the requested window
    win_ts: List[datetime] = []
    win_vals: List[Any] = []
    for t, v in zip(ts_list, values_list):
        if start_dt <= t <= end_dt and _is_valid_number(v):
            win_ts.append(t)
            win_vals.append(v)

    if win_ts and win_vals:
        return win_ts, win_vals

    # 2) Fallback: last valid numeric point <= end_dt
    for i in range(n - 1, -1, -1):
        t = ts_list[i]
        v = values_list[i]
        if t <= end_dt and _is_valid_number(v):
            return [t], [v]

    return [], []


def slice_by_window(
    ts_list: List[datetime],
    values_list: List[Any],
    start_dt: datetime,
    end_dt: datetime,
) -> Tuple[List[datetime], List[Any]]:
    """Slice series by start_dt <= ts <= end_dt."""

    n = min(len(ts_list), len(values_list))
    ts_list = ts_list[:n]
    values_list = values_list[:n]

    out_ts: List[datetime] = []
    out_vals: List[Any] = []
    for t, v in zip(ts_list, values_list):
        if start_dt <= t <= end_dt:
            out_ts.append(t)
            out_vals.append(v)
    return out_ts, out_vals
