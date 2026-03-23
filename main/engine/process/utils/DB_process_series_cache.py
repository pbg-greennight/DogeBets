"""main/engine/process/DB_process_series_cache.py

Option A implementation: in-process long-horizon cache for gaussian plot series.

Why
----
Upstream gaussian indicator modules typically maintain a short rolling plot
buffer (often ~30 minutes). Your bell-curve PV-anchor logic needs a longer
lookback (e.g. 240 minutes) so the PV extrema pair is found reliably.

This module keeps an accumulating cache keyed by sigma, merging each cycle's
returned plot series and trimming to a configured max age. Over time (within a
few minutes) this grows beyond the upstream module's short window, achieving
the requested lookback without changing the indicator modules.

Types
-----
We treat timestamps as Python datetime objects (timezone-aware). Values are
floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


Series = Tuple[List[datetime], List[float]]


@dataclass
class SeriesCache:
    """In-memory rolling cache of time-series per sigma."""

    # sigma -> (timestamps, values)
    data: Dict[int, Series]
    max_minutes: float

    def __init__(self, max_minutes: float = 240.0) -> None:
        # Default matches the bell-curve desired lookback.
        self.data = {}
        self.max_minutes = float(max_minutes)

    def merge(
        self,
        sigma: int,
        ts_new: List[datetime],
        vals_new: List[float],
        *,
        max_minutes: float | None = None,
    ) -> Series:
        """Merge a new (ts, vals) slice into the cache and trim to max_minutes."""

        if not ts_new or not vals_new:
            return self.data.get(sigma, ([], []))
        if len(ts_new) != len(vals_new):
            # Defensive: upstream should always match; if not, truncate to shorter.
            n = min(len(ts_new), len(vals_new))
            ts_new = ts_new[:n]
            vals_new = vals_new[:n]

        # Ensure increasing order (upstream should already be sorted).
        if len(ts_new) >= 2 and ts_new[0] > ts_new[-1]:
            ts_new = list(reversed(ts_new))
            vals_new = list(reversed(vals_new))

        ts_old, vals_old = self.data.get(sigma, ([], []))

        # Fast path: append only
        if ts_old and ts_new[0] > ts_old[-1]:
            ts_comb = ts_old + ts_new
            vals_comb = vals_old + vals_new
        else:
            # Robust path: de-dup by timestamp, keep latest value per timestamp.
            combined = {t: v for t, v in zip(ts_old, vals_old)}
            for t, v in zip(ts_new, vals_new):
                combined[t] = v
            ts_sorted = sorted(combined.keys())
            ts_comb = ts_sorted
            vals_comb = [combined[t] for t in ts_sorted]

        # Trim by age relative to most recent point.
        if max_minutes is None:
            max_minutes = self.max_minutes
        else:
            max_minutes = float(max_minutes)
        newest = ts_comb[-1]
        cutoff = newest - timedelta(minutes=max_minutes)
        # Find first index >= cutoff
        i0 = 0
        # Linear scan is OK at ~10k points, but do a small binary for cleanliness.
        lo, hi = 0, len(ts_comb)
        while lo < hi:
            mid = (lo + hi) // 2
            if ts_comb[mid] < cutoff:
                lo = mid + 1
            else:
                hi = mid
        i0 = lo

        if i0 > 0:
            ts_comb = ts_comb[i0:]
            vals_comb = vals_comb[i0:]

        self.data[sigma] = (ts_comb, vals_comb)
        return self.data[sigma]

    def get(self, sigma: int) -> Series:
        return self.data.get(sigma, ([], []))

    def stats_minutes(self, sigma: int) -> float:
        ts, _ = self.get(sigma)
        if len(ts) < 2:
            return 0.0
        return (ts[-1] - ts[0]).total_seconds() / 60.0
