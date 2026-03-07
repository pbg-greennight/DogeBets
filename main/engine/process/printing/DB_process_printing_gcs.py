from __future__ import annotations

import logging
import numpy as np
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
    gcs_cfg = p.get("GCS", {}) or {}
    sec_regime = bool((gcs_cfg.get("REGIME", {}) or {}).get("ENABLED", config.get("LOG_GCS_REGIME", True)))
    sec_price = bool((gcs_cfg.get("PRICE_POSITION", {}) or {}).get("ENABLED", config.get("LOG_GCS_PRICE_POSITION", True)))
    sec_spacing = bool((gcs_cfg.get("SPACING", {}) or {}).get("ENABLED", config.get("LOG_GCS_SPACING", True)))
    sec_transfer = bool((gcs_cfg.get("TRANSFER", {}) or {}).get("ENABLED", config.get("LOG_GCS_TRANSFER", True)))

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

    widths: Dict[int, float] = {}
    mids: Dict[int, float] = {}
    slopes_tail: Dict[int, float] = {}
    width_changes: Dict[int, float] = {}
    price_positions: Dict[int, Dict[str, Any]] = {}
    regimes: Dict[int, Dict[str, Any]] = {}

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

        # best-effort diagnostics caches
        ch = _as_mapping(c)
        mid = ch.get('mid_last') or ch.get('mid')
        width = ch.get('width')
        try:
            mids[int(sigma)] = float(mid)
        except Exception:
            pass
        try:
            widths[int(sigma)] = float(width)
        except Exception:
            pass

        if vals_l2:
            slopes_tail[int(sigma)] = float(m2.get('slope') or 0.0)
            width_series = []
            try:
                width_series = [abs(float(v) - float(m)) * 2.0 for v, m in zip(vals_l2, np.linspace(vals_l2[0], vals_l2[-1], len(vals_l2)))]
            except Exception:
                width_series = []
            if len(width_series) >= 3:
                width_changes[int(sigma)] = float(width_series[-1] - width_series[-2])

        px = _safe_get(calc_out, 'close', 'last', default=None)
        if px is None:
            px = _safe_get(calc_out, 'price', 'last', default=None)
        if px is None:
            px = _safe_get(calc_out, 'bell', 'btc_close', 'current', default=None)
        try:
            pxf = float(px)
        except Exception:
            pxf = None
        if pxf is not None and int(sigma) in mids:
            w = abs(widths.get(int(sigma), 0.0))
            pm = pxf - mids[int(sigma)]
            zpos = pm / max(w, 1e-9)
            state = 'centered'
            if zpos > 1.0:
                state = 'stretched_above'
            elif zpos > 0.1:
                state = 'above_mid'
            elif zpos < -1.0:
                state = 'stretched_below'
            elif zpos < -0.1:
                state = 'below_mid'
            price_positions[int(sigma)] = {'pm': pm, 'z': zpos, 'state': state}

        if int(sigma) in widths:
            w_now = widths[int(sigma)]
            w_chg = width_changes.get(int(sigma), 0.0)
            w_acc = w_chg - float(m1.get('slope') or 0.0)
            pct = 50.0
            hist = np.asarray(vals, dtype=float)
            hist = hist[np.isfinite(hist)] if hist.size else hist
            if hist.size > 6:
                dist = np.abs(hist - np.median(hist))
                pct = float(np.mean(dist <= (w_now / 2.0)) * 100.0)
                z = (w_now - float(np.median(dist) * 2.0)) / (float(np.median(np.abs(dist - np.median(dist))) * 2.0) + 1e-12)
            else:
                z = 0.0
            regime = 'compressed'
            if w_chg > 0 and w_acc > 0:
                regime = 'exploded'
            elif w_chg > 0:
                regime = 'expanding'
            elif w_chg < 0:
                regime = 'contracting'
            regimes[int(sigma)] = {'regime': regime, 'pct': pct, 'z': z, 'persist': len(vals_l2), 'dw': w_chg, 'ddw': w_acc}

    ordered = [s for s in [8, 23, 38, 53, 68, 83] if s in mids]
    if sec_regime and regimes:
        bits = [
            f"σ{s} regime={regimes[s]['regime']} pct={_fmt_float(regimes[s]['pct'], nd=0, none='NA')} z={_fmt_float(regimes[s]['z'], nd=2, none='NA')} "
            f"persist={int(regimes[s]['persist'])} dW={_fmt_float(regimes[s]['dw'], nd=4)} ddW={_fmt_float(regimes[s]['ddw'], nd=4)}"
            for s in ordered if s in regimes
        ]
        if bits:
            slog.GCS_REGIME(f"[gcs_regime] {' | '.join(bits)}")
    if sec_price and price_positions:
        bits = [f"σ{s} px-mid={price_positions[s]['pm']:+.2f} zpos={price_positions[s]['z']:+.2f} {price_positions[s]['state']}" for s in ordered if s in price_positions]
        if bits:
            slog.GCS_PRICEPOS(f"[gcs_pricepos] {' | '.join(bits)}")
    if sec_spacing and len(ordered) >= 2:
        gaps = []
        diffs = []
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            g = mids[b] - mids[a]
            gaps.append(f"{a}-{b}={g:+.2f}")
            diffs.append(g)
        mono = float(np.mean(np.diff(diffs) >= -1e-12)) if len(diffs) > 1 else 1.0
        fan = 'mixed'
        if all(g > 0 for g in diffs):
            fan = 'fanning_out'
        elif all(g < 0 for g in diffs):
            fan = 'inverted'
        elif max(abs(g) for g in diffs) < 0.25:
            fan = 'flat_cluster'
        elif all(abs(d) < 1.0 for d in np.diff(diffs)) if len(diffs) > 1 else False:
            fan = 'compressing'
        slog.GCS_SPACING(f"[gcs_spacing] {' | '.join(gaps)} | mono={mono:.2f} fan={fan}")
    if sec_transfer and ordered:
        dir_s = 0
        if 8 in slopes_tail:
            dir_s = 1 if slopes_tail[8] > 0 else (-1 if slopes_tail[8] < 0 else 0)
        tbits = []
        depth = 0.0
        for a, b in [(8, 23), (23, 38), (38, 53)]:
            ok = int(dir_s != 0 and b in slopes_tail and ((slopes_tail[b] > 0) == (dir_s > 0)))
            depth += ok
            tbits.append(f"{a}→{b}={ok}")
        state = 'none' if depth <= 0 else ('partial' if depth < 3 else 'deep')
        dlabel = 'up' if dir_s > 0 else ('down' if dir_s < 0 else 'none')
        slog.GCS_TRANSFER(f"[gcs_transfer] dir={dlabel} depth={depth:.2f} state={state} | {' '.join(tbits)}")

    slog.GCS(_line('-', 155))
    slog.GCS(_line('*', 155))
    slog.GCS(_line('-', 155))
