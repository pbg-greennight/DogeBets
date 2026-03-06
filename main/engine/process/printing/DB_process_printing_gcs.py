from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# =====================================================================================================================
# Helpers
# =====================================================================================================================
from main.engine.process.printing.DB_process_printing_utils import (
    _safe_get, _fmt_time, _fmt_float, _line, _fmt_price, _series_preview, _as_mapping, _fmt_iso,
    _arrow, _print_cfg
)
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger
from main.engine.process.DB_process_metrics import snapshot_metrics

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
    calc_out = None
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

    def _series_from_pack(pack: Any) -> tuple[list, list]:
        src = _as_mapping(pack)
        vals = src.get('values') or src.get('series') or src.get('mid') or []
        ts = src.get('ts') or src.get('timestamps') or []
        vals = list(vals) if isinstance(vals, list) else []
        ts = list(ts) if isinstance(ts, list) else []
        n = min(len(vals), len(ts)) if ts else len(vals)
        if n <= 0:
            return [], []
        vals = vals[:n]
        ts = ts[:n] if ts else []
        out_vals: List[float] = []
        out_ts: List[datetime] = []
        for i, v in enumerate(vals):
            try:
                fv = float(v)
            except Exception:
                continue
            out_vals.append(fv)
            if ts:
                t = ts[i]
                out_ts.append(t if isinstance(t, datetime) else None)
        if ts:
            paired = [(t, v) for t, v in zip(out_ts, out_vals) if isinstance(t, datetime)]
            if paired:
                out_ts = [p[0] for p in paired]
                out_vals = [p[1] for p in paired]
            else:
                out_ts = []
        return out_vals, out_ts

    def _split_head_tail(vals: list, ts: list) -> tuple[list, list, list, list] | None:
        n = len(vals)
        if n < 4:
            return None
        half = n // 2
        if half < 2 or (n - half) < 2:
            return None
        vals_l1 = vals[:half]
        vals_l2 = vals[half:]
        ts_l1 = ts[:half] if ts and len(ts) == n else []
        ts_l2 = ts[half:] if ts and len(ts) == n else []
        return vals_l1, vals_l2, ts_l1, ts_l2

    def _metrics(vals: list, ts: list) -> Dict[str, Any]:
        if not vals:
            return {}
        if ts and len(ts) == len(vals) and all(isinstance(t, datetime) for t in ts):
            try:
                return snapshot_metrics(vals, ts)
            except Exception:
                pass
        n = len(vals)
        start = float(vals[0])
        last = float(vals[-1])
        delta = last - start
        slope = (delta / (n - 1)) if n > 1 else 0.0
        mid = n // 2
        slope1 = (float(vals[mid]) - start) / max(1, mid)
        slope2 = (last - float(vals[mid])) / max(1, n - 1 - mid)
        curve = slope2 - slope1
        tag = 'flat'
        if slope > 0:
            tag = 'UP/accel' if curve > 0 else 'UP/decel'
        elif slope < 0:
            tag = 'DOWN/accel' if curve > 0 else 'DOWN/decel'
        return {
            'start': start,
            'last': last,
            'delta': delta,
            'slope': slope,
            'curve': curve,
            'tag': tag,
        }

    per_sigma_full_src = {}
    if isinstance(calc_out, dict):
        per_sigma_full_src = _safe_get(calc_out, 'series', 'per_sigma_full', default={}) or {}
    if not per_sigma_full_src and catalog is not None:
        per_sigma_full_src = _safe_get(catalog, 'per_sigma_full', default={}) or {}
    if not per_sigma_full_src and per_sigma_full is not None:
        per_sigma_full_src = per_sigma_full or {}

    for sigma in sorted(snapshot.keys()):
        c = snapshot[sigma]
        slog.GCS(f"G{int(sigma):<3} | {_fmt_chan(c)}")

        sigma_pack = None
        if isinstance(per_sigma_full_src, dict):
            sigma_pack = per_sigma_full_src.get(int(sigma))
            if sigma_pack is None:
                sigma_pack = per_sigma_full_src.get(str(sigma))
        vals, ts = _series_from_pack(sigma_pack)
        split = _split_head_tail(vals, ts)
        if not split:
            continue

        vals_l1, vals_l2, ts_l1, ts_l2 = split
        m1 = _metrics(vals_l1, ts_l1)
        m2 = _metrics(vals_l2, ts_l2)

        if not m1 or not m2:
            continue

        t0_l1 = ts_l1[0] if ts_l1 else None
        t1_l1 = ts_l1[-1] if ts_l1 else None
        t0_l2 = ts_l2[0] if ts_l2 else None
        t1_l2 = ts_l2[-1] if ts_l2 else None

        slog.GCS_Leg1(
            f"σ={int(sigma):>3}  HEAD DIAG   | "
            f"({_fmt_time(t0_l1)} {_arrow()} {_fmt_time(t1_l1)}) ({len(vals_l1)} Head D.P.'s) | "
            f"start={_fmt_price(m1.get('start'))} last={_fmt_price(m1.get('last'))}, "
            f"Δ={_fmt_price(m1.get('delta'))}, slope={_fmt_float(m1.get('slope'), nd=6)}, "
            f"curve={_fmt_float(m1.get('curve'), nd=6)}, tag={m1.get('tag')}"
        )
        slog.GCS_Leg1_series(
            f"       HEAD SERIES | {_series_preview(vals_l1, max_items=9999, nd=2)}"
        )

        slog.GCS_Leg2(
            f"σ={int(sigma):>3}  TAIL DIAG   | "
            f"({_fmt_time(t0_l2)} {_arrow()} {_fmt_time(t1_l2)}) ({len(vals_l1)} Head D.P.'s to {len(vals_l2)} Tail D.P.'s) | "
            f"start={_fmt_price(m2.get('start'))} last={_fmt_price(m2.get('last'))}, "
            f"Δ={_fmt_price(m2.get('delta'))}, slope={_fmt_float(m2.get('slope'), nd=6)}, "
            f"curve={_fmt_float(m2.get('curve'), nd=6)}, tag={m2.get('tag')}"
        )
        slog.GCS_Leg2_series(
            f"       TAIL SERIES | {_series_preview(vals_l2, max_items=9999, nd=2)}"
        )

    slog.GCS(_line('-', 155))
    slog.GCS(_line('*', 155))
    slog.GCS(_line('-', 155))
