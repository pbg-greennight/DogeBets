import logging
import numpy as np
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
    msbc_cfg = (config.get("PRINT", {}).get("MSBC", {}) or {})
    sec_stack = bool((msbc_cfg.get("STACK_STATE", {}) or {}).get("ENABLED", config.get("LOG_MSBC_STACK_STATE", True)))
    sec_prop = bool((msbc_cfg.get("PROPAGATION", {}) or {}).get("ENABLED", config.get("LOG_MSBC_PROPAGATION", True)))
    sec_age = bool((msbc_cfg.get("AGE", {}) or {}).get("ENABLED", config.get("LOG_MSBC_AGE", True)))
    sec_consist = bool((msbc_cfg.get("CONSISTENCY", {}) or {}).get("ENABLED", config.get("LOG_MSBC_CONSISTENCY", True)))
    sec_norm = bool((msbc_cfg.get("NORMALIZED", {}) or {}).get("ENABLED", config.get("LOG_MSBC_NORMALIZED", False)))

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
    sigmas = [s for s in [8, 23, 38, 53, 68, 83] if (s in leg1 or str(s) in leg1 or s in leg2 or str(s) in leg2)]

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
    leg1_metrics: Dict[int, Dict[str, Any]] = {}
    leg2_metrics: Dict[int, Dict[str, Any]] = {}
    leg1_values: Dict[int, list] = {}
    leg2_values: Dict[int, list] = {}

    if sec_leg2:
        slog.MSBC_Leg2(f"[MSBC_Leg2]  Multi Sigma Bell Curve Segments for (Epoch {prev_epoch}) used for (Epoch {next_epoch}) Trend Determination")
        slog.MSBC_Leg2(f"LEG 2 ({last_kind}→NOW) ({_fmt_time(t_last)} → {_fmt_time(last_ts)})  |  leg 2 tail continuation")

        for sigma in sigmas:
            pack = leg2.get(int(sigma)) or leg2.get(str(sigma)) or {}
            m = (pack.get('metrics', {}) or {})
            if not m or (pack.get('values') is None):
                continue
            leg2_metrics[int(sigma)] = m
            leg2_values[int(sigma)] = (pack.get('values') or [])
            l1_pack = leg1.get(int(sigma)) or leg1.get(str(sigma)) or {}
            leg1_metrics[int(sigma)] = (l1_pack.get('metrics', {}) or {})
            leg1_values[int(sigma)] = (l1_pack.get('values') or [])

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

    def _f(v: Any) -> float | None:
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else None
        except Exception:
            return None

    def _sgn(v: Any, eps: float = 1e-12) -> int:
        fv = _f(v)
        if fv is None:
            return 0
        if fv > eps:
            return 1
        if fv < -eps:
            return -1
        return 0

    def _age_sign(vals: list) -> Any:
        if not vals or len(vals) < 2:
            return 'NA'
        signs = [_sgn(v) for v in vals]
        now = signs[-1]
        if now == 0:
            return 0
        age = 0
        for i in range(len(signs) - 2, -1, -1):
            if signs[i] != 0 and signs[i] != now:
                break
            age += 1
        return age

    def _age_extrema(vals: list) -> Any:
        if not vals or len(vals) < 4:
            return 'NA'
        arr = np.asarray(vals, dtype=float)
        for i in range(len(arr) - 2, 0, -1):
            d0 = arr[i] - arr[i - 1]
            d1 = arr[i + 1] - arr[i]
            if d0 == 0.0 or d1 == 0.0:
                continue
            if d0 * d1 < 0:
                return (len(arr) - 1) - i
        return len(arr) - 1

    def _z_hist(v: Any, vals: list) -> Any:
        fv = _f(v)
        arr = np.asarray(vals or [], dtype=float)
        arr = arr[np.isfinite(arr)] if arr.size else arr
        if fv is None:
            return None
        if arr.size >= 8:
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med)) + 1e-12)
            return (fv - med) / (1.4826 * mad + 1e-12)
        return None

    def _z_stack(v: Any, sig_vals: list[float]) -> Any:
        fv = _f(v)
        if fv is None:
            return None
        arr = np.asarray([x for x in sig_vals if x is not None], dtype=float)
        if arr.size < 2:
            return None
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-12)
        return (fv - mu) / sd

    pair_rows = []
    transfer_flags = []
    transfer_dir = 0
    slope_vals = []
    tan_vals = []
    curve_vals = []
    linm_vals = []
    disorder = 0
    age_rows = []
    consist_rows = []
    norm_rows = []

    for s in sigmas:
        m2 = leg2_metrics.get(s) or (leg2.get(s) or leg2.get(str(s)) or {}).get('metrics', {}) or {}
        if m2:
            slope_vals.append(_f(m2.get('slope')))
            tan_vals.append(_f(m2.get('quad_tangent')))
            curve_vals.append(_f(m2.get('curve')))
            linm_vals.append(_f(m2.get('lin_slope')))

    for i in range(len(sigmas) - 1):
        a, b = sigmas[i], sigmas[i + 1]
        ma = leg2_metrics.get(a) or (leg2.get(a) or leg2.get(str(a)) or {}).get('metrics', {}) or {}
        mb = leg2_metrics.get(b) or (leg2.get(b) or leg2.get(str(b)) or {}).get('metrics', {}) or {}
        if not ma or not mb:
            continue
        ds = (_f(mb.get('slope')) or 0.0) - (_f(ma.get('slope')) or 0.0)
        dt = (_f(mb.get('quad_tangent')) or 0.0) - (_f(ma.get('quad_tangent')) or 0.0)
        dc = (_f(mb.get('curve')) or 0.0) - (_f(ma.get('curve')) or 0.0)
        dl = (_f(mb.get('lin_slope')) or 0.0) - (_f(ma.get('lin_slope')) or 0.0)
        pair_rows.append((a, b, ds, dt, dc, dl,
                          int(_sgn(ma.get('slope')) == _sgn(mb.get('slope')) != 0),
                          int(_sgn(ma.get('quad_tangent')) == _sgn(mb.get('quad_tangent')) != 0),
                          int(_sgn(ma.get('curve')) == _sgn(mb.get('curve')) != 0),
                          int(_sgn(ma.get('lin_slope')) == _sgn(mb.get('lin_slope')) != 0)))

    if sigmas:
        first = leg2_metrics.get(sigmas[0], {})
        transfer_dir = _sgn(first.get('slope')) or _sgn(first.get('quad_tangent'))
        for i in range(len(sigmas) - 1):
            a, b = sigmas[i], sigmas[i + 1]
            mb = leg2_metrics.get(b, {})
            ok = 1 if transfer_dir != 0 and (_sgn(mb.get('slope')) == transfer_dir or _sgn(mb.get('quad_tangent')) == transfer_dir) else 0
            transfer_flags.append((a, b, ok))

    depth_score = float(sum(x[2] for x in transfer_flags))
    if depth_score <= 0:
        transfer_state = 'none'
    elif depth_score <= 1:
        transfer_state = 'shallow'
    elif depth_score <= 2:
        transfer_state = 'partial'
    elif depth_score <= 3:
        transfer_state = 'deep'
    else:
        transfer_state = 'full'

    def _mono(arr: list[float | None]) -> float:
        v = [x for x in arr if x is not None]
        if len(v) < 2:
            return 0.0
        dif = np.diff(np.asarray(v, dtype=float))
        if dif.size == 0:
            return 0.0
        m = float(np.mean(dif >= -1e-12))
        return m

    slope_mono = _mono(slope_vals)
    tan_mono = _mono(tan_vals)
    linm_mono = _mono(linm_vals)
    curve_decay = _mono([-x if x is not None else None for x in curve_vals])
    for _, _, _, _, _, _, sa, ta, ca, la in pair_rows:
        disorder += (1 - sa) + (1 - ta) + (1 - ca) + (1 - la)

    for s in sigmas:
        m2 = leg2_metrics.get(s, {})
        m1 = leg1_metrics.get(s, {})
        age_rows.append(
            f"σ{s} slope_age={_age_sign(leg2_values.get(s) or [])} tan_age={_age_sign((leg2_values.get(s) or [])[-6:])} "
            f"curve_age={_age_sign([m2.get('curve'), m1.get('curve')])} extrema_age={_age_extrema(leg2_values.get(s) or [])}"
        )

        hs = _f(m1.get('slope'))
        ts = _f(m2.get('slope'))
        hc = _f(m1.get('curve'))
        tc = _f(m2.get('curve'))
        ratio = None if hs in (None, 0.0) else (ts / hs)
        sign_agree = int(_sgn(hs) != 0 and _sgn(hs) == _sgn(ts))
        accel_agree = int(_sgn(hc) != 0 and _sgn(hc) == _sgn(tc))
        override = int(_sgn(hs) != 0 and _sgn(ts) != 0 and _sgn(hs) != _sgn(ts))
        consist_rows.append(
            f"σ{s} sign_agree={sign_agree} accel_agree={accel_agree} slope_ratio={_fmt_float(ratio, nd=2, none='NA')} override={override}"
        )

        sz = _z_hist(m2.get('slope'), leg2_values.get(s) or [])
        tz = _z_hist(m2.get('quad_tangent'), [x for x in [m2.get('quad_tangent'), m1.get('quad_tangent')] if x is not None])
        cz = _z_hist(m2.get('curve'), [x for x in [m2.get('curve'), m1.get('curve')] if x is not None])
        lz = _z_hist(m2.get('lin_slope'), [x for x in [m2.get('lin_slope'), m1.get('lin_slope')] if x is not None])
        if sz is None:
            sz = _z_stack(m2.get('slope'), slope_vals)
        if tz is None:
            tz = _z_stack(m2.get('quad_tangent'), tan_vals)
        if cz is None:
            cz = _z_stack(m2.get('curve'), curve_vals)
        if lz is None:
            lz = _z_stack(m2.get('lin_slope'), linm_vals)
        norm_rows.append(
            f"σ{s} slope_z={_fmt_float(sz, nd=2, none='NA')} tan_z={_fmt_float(tz, nd=2, none='NA')} "
            f"curve_z={_fmt_float(cz, nd=2, none='NA')} linm_z={_fmt_float(lz, nd=2, none='NA')}"
        )

    if sec_stack:
        tdir = 'up' if transfer_dir > 0 else ('down' if transfer_dir < 0 else 'none')
        slog.MSBC_STACK(
            f"[MSBC_STACK] transfer={transfer_state}_{tdir} depth={_fmt_float(depth_score, nd=2, none='0.00')} | "
            f"slope_mono={_fmt_float(slope_mono, nd=2)} tan_mono={_fmt_float(tan_mono, nd=2)} "
            f"curve_decay={_fmt_float(curve_decay, nd=2)} linm_mono={_fmt_float(linm_mono, nd=2)} | disorder={int(disorder)}"
        )
    if sec_prop and pair_rows:
        prop_bits = [f"{a}→{b} dSlope={ds:+.6f} dTan={dt:+.6f} dCurve={dc:+.6f} dLinM={dl:+.6f}" for a, b, ds, dt, dc, dl, *_ in pair_rows]
        slog.MSBC_PROP(f"[MSBC_PROP] {' | '.join(prop_bits)}")
        agree_bits = [f"{a}→{b} agree(s,t,c,l)=({sa},{ta},{ca},{la})" for a, b, _, _, _, _, sa, ta, ca, la in pair_rows]
        slog.MSBC_PROP(f"[MSBC_PROP] {' | '.join(agree_bits)}")
    if sec_age and age_rows:
        slog.MSBC_AGE(f"[MSBC_AGE] {' | '.join(age_rows)}")
    if sec_consist and consist_rows:
        slog.MSBC_CONSIST(f"[MSBC_CONSIST] {' | '.join(consist_rows)}")
    if sec_norm and norm_rows:
        slog.MSBC_NORM(f"[MSBC_NORM] {' | '.join(norm_rows)}")

    slog.PERF(_line('-', 155))
    slog.PERF(_line('*', 155))
    slog.PERF(_line('-', 155))

    return bell_curve_series
