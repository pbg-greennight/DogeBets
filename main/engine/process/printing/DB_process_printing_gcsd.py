# main/engine/process/DB_process_printing_gcsd.py
from __future__ import annotations

import logging

from main.engine.process.printing.DB_process_printing_utils import (
    _fmt_dt,
    _fmt_val,
)

def _fmt_prices(vals: list[float], nd: int = 2) -> str:
    # full series (no cap) because your MSBC standard prints the whole leg
    out = []
    for v in vals:
        try:
            out.append(f"{float(v):.{nd}f}")
        except Exception:
            out.append("nan")
    return ", ".join(out)

def _mean(vals: list[float]) -> float:
    try:
        return sum(vals) / max(1, len(vals))
    except Exception:
        return 0.0

def _tag_mid_width(mid_slope: float, width_slope: float) -> str:
    dir_tag = "UP" if mid_slope > 0 else ("DOWN" if mid_slope < 0 else "FLAT")
    w_tag = "widen" if width_slope > 0 else ("narrow" if width_slope < 0 else "flat")
    # compress to something easy to scan
    if dir_tag == "FLAT":
        return f"FLAT/{w_tag}"
    return f"{dir_tag}/{w_tag}"

def _leg_label_from_direction(direction: str | None) -> str:
    if not direction:
        return "LEG"
    d = direction.strip().upper()
    if d == "PV":  # peak->valley
        return "LEG 1 (PEAK→VALLEY)"
    if d == "VP":  # valley->peak
        return "LEG 1 (VALLEY→PEAK)"
    return "LEG 1"

def print_gaussian_channel_pv_tail(
    catalog: dict,
    close_proxy: float | None,
    cfg: dict,
    log: logging.Logger
) -> None:
    """
    MSBC-style printer for Gaussian Channel PV-tail, emitting both LEG 1 and LEG 2.

    Expected catalog key:
      catalog["pv_tail_gcsd"] = output of build_channel_pv_tail(...)
    """
    pv_tail = catalog.get("pv_tail_gcsd") or {}
    status = pv_tail.get("status", "missing")

    log.info("-" * 155)
    log.info("****************************************************************" * 3)
    log.info("-" * 155)

    log.info("[pv_tail]  Gaussian Channel PV-tail (channel series + deltas)")
    log.info(f"[pv_tail]  pv_ref_sigma={pv_tail.get('pv_ref_sigma')} | decision_dt={_fmt_dt(catalog.get('decision_dt'))}")

    if close_proxy is not None:
        log.info(f"[pv_tail]  close_proxy={_fmt_val(close_proxy, 2)} | status={status}")
    else:
        log.info(f"[pv_tail]  status={status}")

    if status != "ok":
        log.info(f"PV-TAIL STATUS: {status} | meta={pv_tail.get('meta')}")
        return

    pv_pair = pv_tail.get("pv_pair") or {}
    prev = pv_pair.get("prev") or {}
    curr = pv_pair.get("curr") or pv_pair.get("last") or {}
    direction = pv_pair.get("direction") or pv_pair.get("dir") or ""

    t_prev = prev.get("ts")
    t_curr = curr.get("ts")
    t_last = pv_tail.get("last_ts")

    # ---------- LEG 1 ----------
    leg1_name = _leg_label_from_direction(direction)
    leg1_ts = f"{_fmt_dt(t_prev, short=True)} → {_fmt_dt(t_curr, short=True)}"
    log.info(f"[pv_GCSD_Leg1]  Gaussian Channel Segments (LEG 1) | {leg1_name} ({leg1_ts})  |  leg 1 swing segment")

    csd_leg1 = pv_tail.get("csd_leg1") or {}
    dcsd_leg1 = pv_tail.get("dcsd_leg1") or {}
    if not csd_leg1:
        log.info("LEG 1: (missing) csd_leg1 is empty — check build_channel_pv_tail() slicing around pv_pair prev/curr extrema.")
    else:
        for sigma in sorted(csd_leg1.keys(), key=lambda x: int(x)):
            leg = csd_leg1[sigma]
            mid = leg.get("mid_tail") or []
            width = leg.get("width_tail") or []
            d_pack = dcsd_leg1.get(sigma) or dcsd_leg1.get(int(sigma)) or {}
            d_mid = d_pack.get("d_mid_tail") or []
            d_width = d_pack.get("d_width_tail") or []
            n = len(mid)

            if n < 3:
                continue

            start = float(mid[0])
            last = float(mid[-1])
            delta = last - start
            mid_slope = _mean([float(x) for x in d_mid]) if d_mid else 0.0
            width_slope = _mean([float(x) for x in d_width]) if d_width else 0.0
            tag = _tag_mid_width(mid_slope, width_slope)

            log.info(
                f"σ={int(sigma):>3}  PV-leg | start={start:.2f} last={last:.2f}, Δ={delta:+.2f}, "
                f"mid_slope={mid_slope:+.6f}, width_slope={width_slope:+.6f}, tag={tag} | n={n}"
            )
            log.info(
                f"       PV-leg ({leg1_ts}) ({n} Data Points) | { _fmt_prices([float(v) for v in mid], nd=2) }"
            )

    # ---------- LEG 2 ----------
    leg2_ts = f"{_fmt_dt(t_curr, short=True)} → {_fmt_dt(t_last, short=True)}"
    log.info("-" * 155)
    log.info(f"[pv_GCSD_Leg2]  Gaussian Channel Segments (LEG 2) | LEG 2 (EXTREMA→NOW) ({leg2_ts})  |  leg 2 tail continuation")

    csd_leg2 = pv_tail.get("csd_leg2") or pv_tail.get("per_sigma") or {}
    dcsd_leg2 = pv_tail.get("dcsd_leg2") or {}
    if not csd_leg2:
        log.info("LEG 2: (missing) csd_leg2 is empty.")
    else:
        for sigma in sorted(csd_leg2.keys(), key=lambda x: int(x)):
            leg = csd_leg2[sigma]
            mid = leg.get("mid_tail") or []
            width = leg.get("width_tail") or []
            d_pack = dcsd_leg2.get(sigma) or dcsd_leg2.get(int(sigma)) or {}
            d_mid = d_pack.get("d_mid_tail") or []
            d_width = d_pack.get("d_width_tail") or []
            n = len(mid)

            if n < 3:
                continue

            start = float(mid[0])
            last = float(mid[-1])
            delta = last - start
            mid_slope = _mean([float(x) for x in d_mid]) if d_mid else 0.0
            width_slope = _mean([float(x) for x in d_width]) if d_width else 0.0
            tag = _tag_mid_width(mid_slope, width_slope)

            log.info(
                f"σ={int(sigma):>3}  TAIL   | start={start:.2f} last={last:.2f}, Δ={delta:+.2f}, "
                f"mid_slope={mid_slope:+.6f}, width_slope={width_slope:+.6f}, tag={tag} | n={n}"
            )
            log.info(
                f"       TAIL  ({leg2_ts}) ({n} Data Points) | { _fmt_prices([float(v) for v in mid], nd=2) }"
            )

    log.info(f"PV-TAIL STATUS: ok")