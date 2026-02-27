"""main/engine/process/DB_process_types.py

Dataclasses shared across the DB_DATA_PROCESS refactor.

Kept intentionally minimal to avoid behavior drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class EpochTiming:
    prev_epoch: int
    curr_epoch: int
    next_epoch: int
    dt_prev: datetime
    dt_curr: datetime
    dt_next: datetime


@dataclass
class Windows:
    full_start: datetime
    full_end: datetime
    next_epoch_time: datetime  # boundary reference


@dataclass
class TrendDecision:
    """Model output container for decision-time direction.

    Notes:
      - Keep fields stable for logging + JSON output.
      - 'metrics' and 'reversal' are optional and may be omitted by simple models.
    """
    trend: str
    confidence: float
    model: str
    notes: str = ""
    metrics: dict | None = None
    reversal: dict | None = None

