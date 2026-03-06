import logging
from datetime import datetime
from typing import Any, Dict

logger = logging.getLogger(__name__)
from main.engine.process.printing.DB_process_printing_utils import (
    _safe_get, _fmt_time, _fmt_float, _line, _fmt_price, _series_preview
)
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger

def print_sigma_tailing_snapshots(
    timing: Any = None,
    windows: Any = None,
    decision_dt: datetime | None = None,
    tail_seconds_list: Any = None,
    catalog: Any = None,
    config: Any = None,
    close_series: Any = None,
    **_kwargs,
) -> Dict[int, Dict[str, Any]]:
    """Print bell-curve PV-leg + tail snapshots.

    This must match Log_example.txt and must never silently print "?" when data is available.

    Expected orchestrator call:
        print_sigma_tailing_snapshots(timing, windows, decision_dt, tail_seconds_list, catalog, config, close_series=...)

    Back-compat call:
        print_sigma_tailing_snapshots(catalog, decision_dt)

    Returns:
        bell_curve_series (PV→NOW per sigma) for downstream dumps.
    """

    # Backwards-compat: if called as (catalog, decision_dt)
    if catalog is None and timing is not None and decision_dt is not None and tail_seconds_list is None:
        catalog = timing
        timing = _safe_get(catalog, 'timing', default=None)
        windows = _safe_get(catalog, 'windows', default=None)

    if config is None:
        config = {}

    slog = get_section_logger(logger, config)

    # Master toggle
    if not config.get('LOG_TAILING_SNAPSHOTS', True):
        return {}

    if catalog is None or not hasattr(catalog, 'ensure_calc'):
        slog.TAIL_ANCHOR('[tail_anchor] No FeatureCatalog provided; cannot compute bell snapshots.')
        return {}

    # Compute truth (once) and read from it
    calc_out = catalog.ensure_calc(decision_dt=decision_dt, close_series=close_series)
    bell = (calc_out.get('bell', {}) or {})
    bell_curve_series = (calc_out.get('bell_curve_series', {}) or {})

    curr_epoch = _safe_get(timing, 'curr_epoch', default=None)
    next_epoch = _safe_get(timing, 'next_epoch', default=None)
    prev_epoch = _safe_get(timing, 'prev_epoch', default=None)

    slog.PERF(
        f"Sigma Bell-Curve Tailing Snapshots from (Epoch {curr_epoch}) used for (Epoch {next_epoch}) Trend Determination"
    )

    # Tail anchor
    last_ts = bell.get('last_ts')
    last_age = bell.get('last_sample_age')
    slog.TAIL_ANCHOR(
        f"[tail_anchor] decision_time={_fmt_time(decision_dt)} | last_ts={_fmt_time(last_ts)} | "
        f"last_sample_age={_fmt_float(last_age, nd=2)}s"
    )
    slog.PERF(_line('-', 155))

    pv_ref_sigma = int(bell.get('pv_ref_sigma') or config.get('PV_REF_SIGMA', 23))
    pv_pair_ref = bell.get('pv_pair_ref')

    # If PV pair missing, make it loud + stop (this is the root cause behind empty prints)
    if not pv_pair_ref:
        lookback = bell.get('lookback_minutes', config.get('BELL_CURVE_LOOKBACK_MINUTES', 240))
        slog.PV_REF(
            f"[pv_ref] BTC Close: start=? current=? | sigma={pv_ref_sigma} | extrema_pair=? (no PV pair found) | "
            f"ref_lookback=?m (cfg={lookback}m) | (Epoch {prev_epoch} → {curr_epoch})"
        )
        return bell_curve_series

    # PV ref audit line (match Log_example style)
    t_prev = _safe_get(pv_pair_ref, 'prev', 'ts', default=None)
    t_last = _safe_get(pv_pair_ref, 'last', 'ts', default=None)
    swing = pv_pair_ref.get('swing')

    prev_kind = _safe_get(pv_pair_ref, 'prev', 'kind', default='?')
    last_kind = _safe_get(pv_pair_ref, 'last', 'kind', default='?')
    prev_val = _safe_get(pv_pair_ref, 'prev', 'val', default=None)
    last_val = _safe_get(pv_pair_ref, 'last', 'val', default=None)

    btc = (bell.get('btc_close', {}) or {})
    btc_start = btc.get('start')
    btc_cur = btc.get('current')

    ref_points = 0
    try:
        ref_points = len((bell_curve_series.get(pv_ref_sigma, {}) or {}).get('values', []) or [])
    except Exception:
        ref_points = 0

    ref_lookback = None
    try:
        if t_prev and last_ts:
            ref_lookback = (last_ts - t_prev).total_seconds() / 60.0
    except Exception:
        ref_lookback = None

    cfg_lookback = bell.get('lookback_minutes', config.get('BELL_CURVE_LOOKBACK_MINUTES', 240))

    slog.PV_REF(
        f"[pv_ref]  BTC Close: start={_fmt_price(btc_start)} current={_fmt_price(btc_cur)} | "
        f"sigma={pv_ref_sigma} | extrema_pair={swing}→CURRENT | "
        f"prev={prev_kind}@{_fmt_price(prev_val)}, last={last_kind}@{_fmt_price(last_val)}, current=CURRENT@{_fmt_price(btc_cur)} | "
        f"t_prev={_fmt_time(t_prev)} | t_last={_fmt_time(t_last)} | t_now={_fmt_time(last_ts)} | "
        f"ref_points={ref_points} | ref_lookback={_fmt_float(ref_lookback, nd=1)}m (cfg={cfg_lookback}m) | "
        f"(Epoch {prev_epoch} → {curr_epoch})"
    )

    slog.PERF(_line('-', 155))
    slog.PERF(_line('*', 155))
    slog.PERF(_line('-', 155))

    sec_leg1 = bool((config.get("PRINT", {}).get("MSBC", {}).get("LEG1", {}) or {}).get("ENABLED", True))
    sec_leg2 = bool((config.get("PRINT", {}).get("MSBC", {}).get("LEG2", {}) or {}).get("ENABLED", True))

    # MSBC headers / LEG 1
    if sec_leg1:
        slog.MSBC_Leg1(
            f"[MSBC_Leg1]  Multi Sigma Bell Curve Segments for (Epoch {prev_epoch}) used for (Epoch {next_epoch}) Trend Determination"
        )

        slog.MSBC_Leg1(
            f"LEG 1 ({swing}) ({_fmt_time(t_prev)} → {_fmt_time(t_last)})  |  leg 1 swing segment"
        )

    leg1 = _safe_get(bell, 'leg1', 'sigmas', default={}) or {}
    leg2 = _safe_get(bell, 'leg2', 'sigmas', default={}) or {}

    # Prefer classic sigma ordering
    sigmas = sorted(set([int(s) for s in list(leg1.keys()) + list(leg2.keys()) if str(s).isdigit()]))

    def _fmt_diag(m: Dict[str, Any]) -> str:
        # matches: | lin(m= 0.753639, r2=0.966) | quad(tan= 0.986199, curv= 0.005782, r2=0.973) | z= 1.44 | run= 0.520
        lin_m = m.get('lin_slope')
        lin_r2 = m.get('lin_r2')
        qt = m.get('quad_tangent')
        qc = m.get('quad_curv')
        qr2 = m.get('quad_r2')
        z = m.get('z_last')
        run = m.get('run_score')

        parts = []
        parts.append(f" | lin(m={_fmt_float(lin_m, nd=6, none=' 0.000000')}, r2={_fmt_float(lin_r2, nd=3, none='0.000')})")
        parts.append(
            f" | quad(tan={_fmt_float(qt, nd=6, none=' 0.000000')}, curv={_fmt_float(qc, nd=6, none=' 0.000000')}, r2={_fmt_float(qr2, nd=3, none='0.000')})"
        )
        parts.append(f" | z={_fmt_float(z, nd=2, none='0.00')}")
        parts.append(f" | run={_fmt_float(run, nd=3, none='0.000')}")
        return ''.join(parts)

    # Print LEG 1 rows + optional series dump
    if sec_leg1:
        for sigma in sigmas:
            pack = leg1.get(int(sigma)) or leg1.get(str(sigma)) or {}
            m = (pack.get('metrics', {}) or {})
            if not m or (pack.get('values') is None):
                continue

            slog.MSBC_Leg1(
                f"σ={int(sigma):>3}  PV-leg | "
                f"start={_fmt_price(m.get('start'))} last={_fmt_price(m.get('last'))}, "
                f"Δ={_fmt_price(m.get('delta'))}, slope={_fmt_float(m.get('slope'), nd=6)}, "
                f"curve={_fmt_float(m.get('curve'), nd=6)}, tag={m.get('tag')}" + _fmt_diag(m)
            )

            if bool(config.get('LOG_BELL_CURVE_LEG_SERIES_DUMP', False)):
                vals_l1 = pack.get('values', []) or []
                t0_l1 = pack.get('t0') or t_prev
                t1_l1 = pack.get('t1') or t_last
                slog.MSBC_Leg1(f"       PV-leg ({_fmt_time(t0_l1)} → {_fmt_time(t1_l1)}) ({len(vals_l1)} Data Points) | {_series_preview(vals_l1, max_items=9999, nd=2)}")

        slog.PERF(_line('-', 155))

    # LEG 2 header
    if sec_leg2:
        slog.MSBC_Leg2(f"[MSBC_Leg2]  Multi Sigma Bell Curve Segments for (Epoch {prev_epoch}) used for (Epoch {next_epoch}) Trend Determination")
        slog.MSBC_Leg2(f"LEG 2 ({last_kind}→NOW) ({_fmt_time(t_last)} → {_fmt_time(last_ts)})  |  leg 2 tail continuation")

        for sigma in sigmas:
            pack = leg2.get(int(sigma)) or leg2.get(str(sigma)) or {}
            m = (pack.get('metrics', {}) or {})
            if not m or (pack.get('values') is None):
                continue

            if bool(config.get('LOG_BELL_CURVE_LEG_SERIES_DUMP', False)):
                leg1_pack = leg1.get(int(sigma)) or leg1.get(str(sigma)) or {}
                vals_l1 = leg1_pack.get('values', []) or []
                vals_l2 = pack.get('values', []) or []
                t0_l2 = pack.get('t0') or t_last
                t1_l2 = pack.get('t1') or last_ts
                slog.MSBC_Leg2(
                    f"σ={int(sigma):>3}  TAIL DIAG   | "
                    f"({_fmt_time(t0_l2)} → {_fmt_time(t1_l2)}) ({len(vals_l1)} Head D.P.'s to {len(vals_l2)} Tail D.P.'s) | "
                    f"start={_fmt_price(m.get('start'))} last={_fmt_price(m.get('last'))}, "
                    f"Δ={_fmt_price(m.get('delta'))}, slope={_fmt_float(m.get('slope'), nd=6)}, "
                    f"curve={_fmt_float(m.get('curve'), nd=6)}, tag={m.get('tag')}" + _fmt_diag(m)
                )
                slog.MSBC_Leg2_series(f"       TAIL SERIES | {_series_preview(vals_l2, max_items=9999, nd=2)}")



    slog.PERF(_line('-', 155))
    slog.PERF(_line('*', 155))
    slog.PERF(_line('-', 155))

    return bell_curve_series
