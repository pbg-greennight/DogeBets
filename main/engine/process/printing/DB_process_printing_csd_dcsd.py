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

def print_gaussian_channel_pv_tail(timing, windows, pv_tail_status, config, per_sigma_full=None, pv_ref=None):
    """
    Prints the PV-tail channel series + derived series (csd_leg1/2 + dcsd_leg1/2)
    in the same style as Log_example.txt:

      [csd_leg1]  CHANNEL SERIES DUMP ...
      σ=  8 | Midline     :  ...
            | Lower Band  :  ...
            | Upper Band  :  ...
            | ChannelWidth:  ...
            | Position    :  ...

      [dcsd_leg1]  DERIVED SERIES ...
      σ=  8 | MidlineSlope(Δ) : ...
            | WidthChange(Δw) : ...
            | LowerSlope(Δlo) : ...
            | UpperSlope(Δhi) : ...

    Key fixes:
      - supports timing/windows as dict OR objects (EpochTiming) via _safe_get
      - NO extra timestamp prefix (logger already prints timestamp)
      - uses per_sigma_leg1/per_sigma_leg2 (fallback to legacy per_sigma)
      - prints comma-separated numeric series (no python list repr)
    """
    import logging
    log = logging.getLogger(__name__)
    slog = get_section_logger(log, config)

    # Respect config toggle (your codebase uses this flag name)
    if not bool(_safe_get(config, "LOG_GAUSS_PV_TAIL", default=True)):
        return "skipped"

    # Robust epoch labels (works whether timing is dict or an object)
    analyze_epoch = _safe_get(timing, "analyze_epoch", default=None)
    used_epoch = _safe_get(timing, "epoch_used", default=None)
    next_epoch = _safe_get(timing, "next_epoch", default=None)

    # Pick best "used epoch" label
    epoch_used = used_epoch if used_epoch is not None else analyze_epoch
    epoch_next = next_epoch

    # PV-tail payload
    pv_tail_status = pv_tail_status or {}
    pv_pair = _safe_get(pv_tail_status, "pv_pair", default=None)

    # --- Determine legs (new keys preferred, fallback to legacy)
    per_leg1 = _safe_get(pv_tail_status, "per_sigma_leg1", default=None)
    per_leg2 = _safe_get(pv_tail_status, "per_sigma_leg2", default=None)
    per_legacy = _safe_get(pv_tail_status, "per_sigma", default=None)

    # If the builder hasn't been updated or catalog didn't store legs, fall back gracefully
    if not isinstance(per_leg1, dict):
        per_leg1 = {}
    if not isinstance(per_leg2, dict):
        # legacy behavior: treat per_sigma as leg2 if present
        per_leg2 = per_legacy if isinstance(per_legacy, dict) else {}

    # If we have nothing to print, bail
    if not per_leg1 and not per_leg2:
        slog.PV_TAIL_CHANNELS("[pv_tail] PV-tail: no leg data (per_sigma_leg1/per_sigma_leg2 missing)")
        return "empty"

    # --- Helpers
    def _fmt_series(vals, decimals=2):
        if not vals:
            return ""
        out = []
        for v in vals:
            try:
                out.append(f"{float(v):.{decimals}f}")
            except Exception:
                out.append(str(v))
        return ", ".join(out)

    def _diff(vals):
        if not vals:
            return []
        out = [0.0]
        for i in range(1, len(vals)):
            try:
                out.append(float(vals[i]) - float(vals[i - 1]))
            except Exception:
                out.append(0.0)
        return out

    def _position(mid, lower, upper):
        # Midline is centered between bands => position ~0.50 always.
        # Keep it explicit and robust if any array is missing.
        n = 0
        for arr in (mid, lower, upper):
            if isinstance(arr, list):
                n = max(n, len(arr))
        return [0.5] * n

    def _leg_label(kind_prev, kind_last, is_tail=False):
        # kind strings may come as "PEAK"/"VALLEY" or None
        a = (kind_prev or "PV").upper()
        b = (kind_last or ("NOW" if is_tail else "PV")).upper()
        arrow = "→"
        if is_tail:
            return f"TAIL ({b}{arrow}NOW)"
        return f"PV-leg ({a}{arrow}{b})"

    # Get extrema kinds for label text (best effort)
    prev_kind = _safe_get(pv_pair, "prev", "kind", default=None) if pv_pair else None
    last_kind = _safe_get(pv_pair, "last", "kind", default=None) if pv_pair else None

    # --- Printer for one leg
    def _print_leg(tag_csd, tag_dcsd, leg_name, per_sigma_dict):
        # Header line like your example
        epoch_str = f"(Epoch {epoch_used} used for Epoch {epoch_next})" if epoch_used and epoch_next else ""
        slog.PV_TAIL_CHANNELS(_line("-", 155))
        if tag_csd == "csd_leg1":
            slog.CSD_DCSD_Leg1_CSD(f"[{tag_csd}]  CHANNEL SERIES DUMP {epoch_str} ({leg_name})")
        else:
            slog.CSD_DCSD_Leg2_CSD(f"[{tag_csd}]  CHANNEL SERIES DUMP {epoch_str} ({leg_name})")

        # Sort sigmas numerically
        for sigma in sorted(per_sigma_dict.keys(), key=lambda x: int(x)):
            pack = per_sigma_dict.get(sigma) or {}

            mid = pack.get("mid", []) or []
            lower = pack.get("lower", []) or []
            upper = pack.get("upper", []) or []
            width = pack.get("width", []) or []
            pos = _position(mid, lower, upper)

            # CHANNEL SERIES block
            if tag_csd == "csd_leg1":
                slog.CSD_DCSD_Leg1_CSD(f"σ={int(sigma):3d} | Midline     : {_fmt_series(mid, 2)}")
                slog.CSD_DCSD_Leg1_CSD(f"       | Lower Band  : {_fmt_series(lower, 2)}")
                slog.CSD_DCSD_Leg1_CSD(f"       | Upper Band  : {_fmt_series(upper, 2)}")
                slog.CSD_DCSD_Leg1_CSD(f"       | ChannelWidth: {_fmt_series(width, 2)}")
                slog.CSD_DCSD_Leg1_CSD(f"       | Position    : {_fmt_series(pos, 2)}")
                slog.CSD_DCSD_Leg1_CSD("")
            else:
                slog.CSD_DCSD_Leg2_CSD(f"σ={int(sigma):3d} | Midline     : {_fmt_series(mid, 2)}")
                slog.CSD_DCSD_Leg2_CSD(f"       | Lower Band  : {_fmt_series(lower, 2)}")
                slog.CSD_DCSD_Leg2_CSD(f"       | Upper Band  : {_fmt_series(upper, 2)}")
                slog.CSD_DCSD_Leg2_CSD(f"       | ChannelWidth: {_fmt_series(width, 2)}")
                slog.CSD_DCSD_Leg2_CSD(f"       | Position    : {_fmt_series(pos, 2)}")
                slog.CSD_DCSD_Leg2_CSD("")

        # DERIVED SERIES block (Δ lines)
        if tag_dcsd == "dcsd_leg1":
            slog.CSD_DCSD_Leg1_DCSD(f"[{tag_dcsd}]  DERIVED SERIES {epoch_str} ({leg_name})")
        else:
            slog.CSD_DCSD_Leg2_DCSD(f"[{tag_dcsd}]  DERIVED SERIES {epoch_str} ({leg_name})")

        for sigma in sorted(per_sigma_dict.keys(), key=lambda x: int(x)):
            pack = per_sigma_dict.get(sigma) or {}

            mid = pack.get("mid", []) or []
            lower = pack.get("lower", []) or []
            upper = pack.get("upper", []) or []
            width = pack.get("width", []) or []

            d_mid = pack.get("d_mid", None)
            d_width = pack.get("d_width", None)

            # If builder didn’t provide these, compute them
            if not isinstance(d_mid, list):
                d_mid = _diff(mid)
            if not isinstance(d_width, list):
                d_width = _diff(width)

            d_lower = _diff(lower)
            d_upper = _diff(upper)

            if tag_dcsd == "dcsd_leg1":
                slog.CSD_DCSD_Leg1_DCSD(f"σ={int(sigma):3d} | MidlineSlope(Δ) : {_fmt_series(d_mid, 2)}")
                slog.CSD_DCSD_Leg1_DCSD(f"       | WidthChange(Δw) : {_fmt_series(d_width, 2)}")
                slog.CSD_DCSD_Leg1_DCSD(f"       | LowerSlope(Δlo) : {_fmt_series(d_lower, 2)}")
                slog.CSD_DCSD_Leg1_DCSD(f"       | UpperSlope(Δhi) : {_fmt_series(d_upper, 2)}")
                slog.CSD_DCSD_Leg1_DCSD("")
            else:
                slog.CSD_DCSD_Leg2_DCSD(f"σ={int(sigma):3d} | MidlineSlope(Δ) : {_fmt_series(d_mid, 2)}")
                slog.CSD_DCSD_Leg2_DCSD(f"       | WidthChange(Δw) : {_fmt_series(d_width, 2)}")
                slog.CSD_DCSD_Leg2_DCSD(f"       | LowerSlope(Δlo) : {_fmt_series(d_lower, 2)}")
                slog.CSD_DCSD_Leg2_DCSD(f"       | UpperSlope(Δhi) : {_fmt_series(d_upper, 2)}")
                slog.CSD_DCSD_Leg2_DCSD("")

    # --- LEG 1 (prev -> last) and LEG 2 (last -> now)
    # If pv_pair is missing, we still print, just use generic labels.
    leg1_label = _leg_label(prev_kind, last_kind, is_tail=False)
    leg2_label = _leg_label(prev_kind, last_kind, is_tail=True)  # shows "...→NOW"

    if per_leg1:
        _print_leg("csd_leg1", "dcsd_leg1", leg1_label, per_leg1)
    else:
        # Don’t spam; just note once
        slog.PV_TAIL_CHANNELS("[pv_tail] no LEG 1 data (per_sigma_leg1 empty)")

    if per_leg2:
        _print_leg("csd_leg2", "dcsd_leg2", leg2_label, per_leg2)
    else:
        slog.PV_TAIL_CHANNELS("[pv_tail] no LEG 2 data (per_sigma_leg2/per_sigma empty)")

    return "printed"

