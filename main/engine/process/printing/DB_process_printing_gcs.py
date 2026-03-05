from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)

# =====================================================================================================================
# Helpers
# =====================================================================================================================
from main.engine.process.printing.DB_process_printing_utils import (
    _safe_get, _fmt_time, _fmt_float, _line, _fmt_price, _series_preview, _as_mapping, _fmt_iso,
    _arrow, _print_cfg
)
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger

def print_gaussian_channel_snapshot(
    timing: Any = None,
    windows: Any = None,
    per_sigma_full: Any = None,
    config: Any = None,
    catalog: Any = None,
    decision_dt: datetime | None = None,
    **_kwargs,
) -> None:
    """Print Gaussian Channel snapshot in Log_example.txt style.

    Orchestrator currently calls:
        print_gaussian_channel_snapshot(timing, windows, catalog, config)

    We treat the 3rd arg as either per_sigma_full (legacy) OR catalog (new).
    """

    if config is None:
        config = {}

    p = _print_cfg(config)
    slog = get_section_logger(logger, config)
    if not config.get('LOG_GAUSS_CHANNEL_SNAPSHOT', True):
        return
    if not p.get("GCS", {}).get("ENABLED", True):
        return

    # Detect call pattern: (timing, windows, catalog, config)
    if catalog is None and per_sigma_full is not None and hasattr(per_sigma_full, 'ensure_calc'):
        catalog = per_sigma_full
        per_sigma_full = None

    snapshot = None
    k = float(config.get('GAUSS_CHANNEL_K', 2.0))

    if catalog is not None and hasattr(catalog, 'ensure_calc'):
        calc_out = catalog.ensure_calc(decision_dt=decision_dt or _safe_get(windows, 'full_end', default=None), close_series=None)
        snapshot = _safe_get(calc_out, 'channels', 'snapshot', default=None)
        if not snapshot:
            snapshot = catalog.get('channels.snapshot', default=None)

    if snapshot is None and per_sigma_full is not None:
        # legacy fallback
        try:
            from main.engine.process.DB_process_gauss_channel import build_channel_snapshot
        except Exception:
            from main.engine.process.DB_process_gauss_channel import build_channel_snapshot
        snapshot = build_channel_snapshot(per_sigma_full, k=k)

    if not snapshot:
        slog.GCS('Gaussian Channel Snapshot')
        slog.GCS('[channels] (missing)')
        return

    curr_epoch = _safe_get(timing, 'curr_epoch', default=None)
    next_epoch = _safe_get(timing, 'next_epoch', default=None)

    full_start = _safe_get(windows, 'full_start', default=None)
    full_end = _safe_get(windows, 'full_end', default=None)

    slog.GCS(
        f"[gcs]  Gaussian Channel Snapshot (per-sigma dispersion bands) | (Epoch {curr_epoch} used for Epoch {next_epoch})"
    )
    slog.GCS(_line('-', 155))
    slog.GCS(f"Window: {_fmt_time(full_start)} → {_fmt_time(full_end)} | K={k:.2f}")

    # snapshot: {sigma: ChannelStats dataclass OR dict}
    def _fmt_chan(c: Any) -> str:
        if hasattr(c, 'mid_last'):
            mid = c.mid_last
            lo = c.lower
            hi = c.upper
            width = c.width
            rs = c.robust_std
            delta = c.delta
            slope = c.slope
        elif isinstance(c, dict):
            mid = c.get('mid_last') or c.get('mid')
            lo = c.get('lower')
            hi = c.get('upper')
            width = c.get('width')
            rs = c.get('robust_std') or c.get('robustσ')
            delta = c.get('delta')
            slope = c.get('slope')
        else:
            mid = lo = hi = width = rs = delta = slope = None

        try:
            return (
                f"mid={float(mid):,.2f} | band=[{float(lo):,.2f}..{float(hi):,.2f}] | "
                f"width={float(width):.2f} (robustσ={float(rs):.2f}) | Δ={float(delta):.2f} | slope={float(slope):.6f}"
            )
        except Exception:
            return f"mid={_fmt_price(mid)} | band=[{_fmt_price(lo)}..{_fmt_price(hi)}] | width={_fmt_price(width)} (robustσ={_fmt_price(rs)}) | Δ={_fmt_price(delta)} | slope={_fmt_float(slope, nd=6)}"

    for sigma in sorted(snapshot.keys()):
        c = snapshot[sigma]
        slog.GCS(f"G{int(sigma):<3} | {_fmt_chan(c)}")

    slog.GCS(_line('-', 155))
    slog.GCS(_line('*', 155))
    slog.GCS(_line('-', 155))
