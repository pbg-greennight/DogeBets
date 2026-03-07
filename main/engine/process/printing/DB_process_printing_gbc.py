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

def print_gaussian_bell_curve_series_dump(
    timing: Any,
    decision_dt: datetime | None,
    bell_curve_series: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
) -> None:
    """Print PV→NOW bell series dump + diagnostics in Log_example.txt style."""
    slog = get_section_logger(logger, config)
    p = _print_cfg(config)
    gbc_cfg = p.get("GBC", {}) or {}

    if not bool(config.get('LOG_BELL_CURVE_SERIES_DUMP', True)):
        return

    curr_epoch = _safe_get(timing, 'curr_epoch', default=None)
    next_epoch = _safe_get(timing, 'next_epoch', default=None)

    # points(max)
    max_points = 0
    for _s, pack in (bell_curve_series or {}).items():
        max_points = max(max_points, len((pack or {}).get('values', []) or []))

    slog.GBC_SD(
        f"[gbc_sd]  Gaussian Bell-Curve Series Dump | (Epoch {curr_epoch} used for Epoch {next_epoch})"
    )
    slog.GBC_SD(f"Window: (per-sigma PV → NOW) | points(max)={max_points}")

    for sigma in sorted((bell_curve_series or {}).keys()):
        pack = bell_curve_series.get(sigma, {}) or {}
        ts = pack.get('ts', []) or []
        vals = pack.get('values', []) or []
        if ts and vals:
            slog.GBC_SD(
                f"σ={int(sigma):>3} | ({_fmt_time(ts[0])} → {_fmt_time(ts[-1])}) ({len(vals)} Data Points) | "
                f"{_series_preview(vals, max_items=9999, nd=2)}"
            )
        else:
            slog.GBC_SD(f"σ={int(sigma):>3} | ")

    # Diagnostics: prefer catalog->bell diagnostics if present
    diag = None
    try:
        bell = _safe_get(timing, 'catalog', default=None)
        _ = bell
    except Exception:
        pass

    # If orchestrator passed catalog separately, it is not available here;
    # but FeatureCatalog caches are already printed above. We keep diagnostics optional.

    # If config enables diagnostics AND we can find them via a global, skip.
    if not bool(config.get('LOG_BELL_CURVE_DIAGNOSTICS', True)):
        return

    # Try best-effort fetch: if timing has ._catalog attr set by orchestrator (optional)
    cat = getattr(timing, '_catalog', None)
    if cat is not None and hasattr(cat, 'ensure_calc'):
        calc_out = cat.ensure_calc(decision_dt=decision_dt, close_series=None)
        diag = _safe_get(calc_out, 'bell', 'diagnostics', 'per_sigma', default=None)

    if not diag:
        return

    sec_plus = bool((gbc_cfg.get("DIAG_PLUS", {}) or {}).get("ENABLED", config.get("LOG_GBC_DIAG_PLUS", True)))
    sec_age = bool((gbc_cfg.get("AGE", {}) or {}).get("ENABLED", config.get("LOG_GBC_AGE", True)))
    sec_persist = bool((gbc_cfg.get("PERSISTENCE", {}) or {}).get("ENABLED", config.get("LOG_GBC_PERSISTENCE", True)))

    slog.GBC_DIAG(_line('-', 155))
    slog.GBC_DIAG(
        f"[gbc_diag]  Gaussian Bell-Curve Diagnostic Series | (Epoch {curr_epoch} used for Epoch {next_epoch})"
    )

    for sigma in sorted(diag.keys()):
        d = diag[sigma] or {}
        slog.GBC_DIAG(
            f"σ={int(sigma):>3} | shrink={_fmt_float(d.get('shrink'), nd=3, none='0.000')} "
            f"flat={_fmt_float(d.get('flat'), nd=3, none='0.000')} hook={int(d.get('hook', 0) or 0)} "
            f"prev_abs={_fmt_float(d.get('prev_abs'), nd=6, none='0.000000')} "
            f"last_abs={_fmt_float(d.get('last_abs'), nd=6, none='0.000000')} "
            f"eps={_fmt_float(d.get('eps'), nd=6, none='0.000000')} "
            f"sign={int(d.get('sign_from', 0) or 0)}→{int(d.get('sign_to', 0) or 0)}"
        )

        if not sec_plus:
            continue

        age = d.get('bars_since_turn') if sec_age else 'NA'
        hook_age = d.get('hook_age') if sec_age else 'NA'
        sign_persist = d.get('sign_persistence') if sec_persist else 'NA'
        hook_persist = d.get('hook_persistence') if sec_persist else 'NA'
        shrink = d.get('shrink')
        flat = d.get('flat')
        try:
            shrink_n = float(shrink)
            flat_n = float(flat)
            norm_ctx = abs(shrink_n) / (abs(flat_n) + 1e-9)
        except Exception:
            norm_ctx = None
        slog.GBC_DIAG(
            f"σ={int(sigma):>3} | turn_age={age} hook_age={hook_age} "
            f"norm_ctx={_fmt_float(norm_ctx, nd=3, none='NA')} "
            f"hook_persist={hook_persist} sign_persist={sign_persist}"
        )