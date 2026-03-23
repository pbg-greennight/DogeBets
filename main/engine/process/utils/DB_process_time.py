"""main/engine/process/DB_process_time.py

Time parsing and epoch window helpers for DB_DATA_PROCESS refactor.

Hybrid note:
- Preserve the working (fix2) timing semantics.
- Use shared dataclasses from DB_process_types so printing/orchestrator see the
  same EpochTiming/Windows types.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from dateutil import parser as dtparser

from main.engine.DB_round_fetch import fetch_last_epoch_info
from main.engine.process.utils.DB_process_types import EpochTiming, Windows


def _parse_dt_maybe(value: Any) -> Optional[datetime]:
    """Robust parse for datetime or ISO-like strings (including timezone offsets)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    if not s or s == "N/A":
        return None
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return dtparser.parse(s)
        except Exception:
            return None


def _fmt_ts(dt: datetime) -> str:
    """AM/PM time (no military)."""
    return dt.strftime("%I:%M:%S %p")


def _fmt_range(a: datetime, b: datetime) -> str:
    """AM/PM range for readability."""
    return f"{a.strftime('%I:%M:%S %p')} → {b.strftime('%I:%M:%S %p')}"


def get_epoch_timing() -> Optional[EpochTiming]:
    """Fetch epoch timing from fetch_last_epoch_info().

    Returns EpochTiming or None.

    fetch_last_epoch_info returns:
      (prev_epoch, prev_epoch_time, curr_epoch, curr_epoch_time, next_epoch, next_epoch_time)
    """
    prev_epoch, prev_t, curr_epoch, curr_t, next_epoch, next_t = fetch_last_epoch_info()

    if prev_epoch is None or curr_epoch is None or next_epoch is None:
        return None

    dt_prev = _parse_dt_maybe(prev_t)
    dt_curr = _parse_dt_maybe(curr_t)
    dt_next = _parse_dt_maybe(next_t)

    if dt_curr is None or dt_next is None:
        return None

    if dt_prev is None:
        # fallback: assume prev is curr-5min (rare, but safe)
        dt_prev = dt_curr - timedelta(minutes=5)

    return EpochTiming(
        prev_epoch=int(prev_epoch),
        curr_epoch=int(curr_epoch),
        next_epoch=int(next_epoch),
        dt_prev=dt_prev,
        dt_curr=dt_curr,
        dt_next=dt_next,
    )


def compute_windows(dt_curr: datetime, decision_dt: datetime, next_epoch_time: datetime) -> Windows:
    """Compute FULL window (curr_epoch_time → decision_dt)."""
    return Windows(full_start=dt_curr, full_end=decision_dt, next_epoch_time=next_epoch_time)
