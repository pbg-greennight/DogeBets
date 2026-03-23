# main/engine/process/printing/DB_process_printing_hyst.py

import logging
import numpy as np
from datetime import datetime
from typing import Any, Dict, Optional

from .DB_process_printing_utils import (
    _safe_get,
    _fmt_time,
    _line,
)
from main.engine.process.printing.DB_process_SectionLogger import get_section_logger
from main.engine.process.features.DB_process_hysteresis import (
    compute_hysteresis_features,
    get_hyst_tracker,
)

logger = logging.getLogger(__name__)
_HYST_TRACKER = get_hyst_tracker()


def print_hysteresis_fan_stack(
    timing: Any = None,
    windows: Any = None,
    decision_dt: Optional[datetime] = None,
    catalog: Any = None,
    config: Any = None,
    **_kwargs,
) -> Dict[str, Any]:
    """
    Logging-only printer.
    Returns hyst_obj features for downstream trend/calc code.

    Phase 2 consumes the persisted hysteresis states from hyst_obj instead of
    recomputing separate spread/risk/stability logic inside the printer.
    """

    if not config or not config.get("LOG_HYSTERESIS", True):
        return {}

    slog = get_section_logger(logger, config)

    if decision_dt is None:
        decision_dt = datetime.now()

    per_sigma_hist = _safe_get(catalog, "per_sigma_hist", default={})
    if not per_sigma_hist:
        return {"meta": {"skipped": True, "skip_reason": "no_sigma_data"}}

    hyst = compute_hysteresis_features(
        per_sigma_hist=per_sigma_hist,
        decision_dt=decision_dt,
        lookback_seconds=3600,
        tail_seconds=120,
        align_tol_seconds=3.0,
    )

    if hyst.get("meta", {}).get("skipped"):
        return hyst

    ep = hyst["episode"]
    pr = hyst["probe"]
    eta = hyst["eta"]
    stacks = hyst["stacks"]
    spread = hyst.get("spread_state", {}) or {}
    p = (config.get("PRINT", {}) or {})
    hcfg = (p.get("HYSTERESIS", {}) or {})
    hcfg_alt = (p.get("HYST", {}) or {})

    def _h_on(key: str, legacy: str, default: bool = True) -> bool:
        node = hcfg.get(key, hcfg_alt.get(key, {})) or {}
        return bool(node.get("ENABLED", config.get(legacy, default)))

    sec_stability = _h_on("STABILITY", "LOG_HYST_STABILITY", True)
    sec_pressure = _h_on("PRESSURE", "LOG_HYST_PRESSURE", True)
    sec_spread = _h_on("SPREAD_STATE", "LOG_HYST_SPREAD_STATE", True)
    sec_risk = _h_on("RISK", "LOG_HYST_RISK", True)

    slog.HYST_HEADER(_line("=", 155))
    slog.HYST_HEADER("HYSTERESIS FAN STACK (method_hyster_v1.1)")
    slog.HYST_HEADER(_line("-", 155))

    slog.HYST_EPISODE(
        f"[hyst_episode] decision_time={_fmt_time(decision_dt)} | "
        f"pair=({ep['pair_used'][0]},{ep['pair_used'][1]}) | "
        f"start={_fmt_time(ep['start_ts']) if ep['start_ts'] else 'NONE'} | "
        f"elapsed={(ep['elapsed_seconds'] if ep['elapsed_seconds'] is not None else 0.0):.1f}s | "
        f"stage={ep['stage']}"
    )
    slog.HYST_EPISODE(
        f"[hyst_primary] d{ep['pair_used'][0]}_{ep['pair_used'][1]} now={ep['d_now']:+.6f} | "
        f"sign={ep['sign_now']} | "
        f"last_cross={_fmt_time(ep['last_cross_ts']) if ep['last_cross_ts'] else 'NONE'} | "
        f"lookback=60m | mode={ep['mode']}"
    )

    slog.HYST_PROBE(
        f"[hyst_probe] pair=(23,83) | d23_83 now={pr['d_now']:+.6f} | "
        f"sign={pr['sign_now']} | flip_watch={int(pr['flip_watch'])} | "
        f"fast_collapse={int(pr['fast_collapse'])} | recovery={int(pr.get('recovery', False))} | "
        f"d_abs_slope_tail={pr['d_abs_slope_tail']:+.6f}"
    )

    if eta.get("eta_to_end_seconds") is not None:
        slog.HYST_ETA(
            f"[hyst_eta] eta={eta['eta_to_end_seconds']:.1f}s (~{eta['eta_to_end_seconds'] / 300.0:.2f} epochs) | "
            f"source={eta.get('source_stack')}"
        )

    for sid in ["S0", "S1", "S2", "S3"]:
        lm = stacks.get(sid)
        if not lm:
            continue
        sigs = ",".join(map(str, lm["sigmas"]))
        W_pct = lm.get("W_pct")
        W_pct_s = "nan" if not np.isfinite(W_pct) else f"{int(round(W_pct)):>3d}"
        slog.HYST_LADDER(
            f"  [hyst_{sid}] sigmas={sigs:<14} | "
            f"m_norm={lm['m_norm']:+.3f} | "
            f"W_pct={W_pct_s} | "
            f"dW_norm={lm['dW_norm']:+.3f} | "
            f"eff_order={lm['eff_order']:.2f} | "
            f"cross={lm['cross_rate']:.2f}/m"
        )

        W_now_s = "nan" if not np.isfinite(lm.get("W_now", float("nan"))) else f"{lm['W_now']:.4f}"
        W_z_s = "nan" if not np.isfinite(lm.get("W_z", float("nan"))) else f"{lm['W_z']:+.2f}"
        W_ratio_s = "nan" if not np.isfinite(lm.get("W_ratio", float("nan"))) else f"{lm['W_ratio']:.2f}"
        ddW_s = f"{lm.get('ddW_norm', 0.0):+.3f}"
        dom_s = f"{lm.get('dom', 0.0):.2f}"
        lead_max = lm.get("leader_max_sigma")
        lead_min = lm.get("leader_min_sigma")
        leader_s = f"{lead_max}->{lead_min}" if (lead_max is not None and lead_min is not None) else "NONE"

        cross_pairs = lm.get("cross_pairs") or {}
        parts = []
        for k, v in cross_pairs.items():
            try:
                c = int(v.get("count", 0))
            except Exception:
                c = 0
            if c > 0:
                parts.append(f"{k}:{c}")
        cross_pairs_s = ",".join(parts) if parts else "none"

        slog.HYST_LADDER(
            f"  [hyst_{sid}_GEOM] "
            f"W_now={W_now_s} | W_z={W_z_s} | W_ratio={W_ratio_s} | "
            f"ddW_norm={ddW_s} | m2_norm={lm.get('m2_norm', 0.0):+.3f} | "
            f"dom={dom_s} | leader={leader_s} | crosses={cross_pairs_s}"
        )

        if sec_stability:
            slog.HYST_STABILITY(
                f"[hyst_stability] {sid} stability={lm.get('stability', 0.0):.2f} {lm.get('stability_state', 'unknown')} | "
                f"leader_age={int(lm.get('leader_age', 0))} switchN={int(lm.get('switch_count', 0))} "
                f"leader_stab={float(lm.get('leader_stability', 0.0)):.2f}"
            )
        if sec_pressure:
            min_gap = lm.get("min_gap", float("nan"))
            slog.HYST_PRESSURE(
                f"[hyst_pressure] {sid} near_cross={int(lm.get('near_cross', 0))} min_gap={min_gap if np.isfinite(min_gap) else float('nan'):.3f} "
                f"gap_vel={float(lm.get('gap_vel', 0.0)):+.3f} state={lm.get('pressure', 'unknown')}"
            )
        if sec_spread or sec_risk:
            slog.HYST_SPREAD_STATE(
                f"[hyst_spread_state] {sid} spread={lm.get('spread_state', 'unknown')} "
                f"momentum={lm.get('spread_momentum', 'unknown')} exhaust={int(lm.get('exhaust', 0))} "
                f"order_conf={float(lm.get('order_conf', 0.0)):.2f} break_risk={float(lm.get('break_risk', 0.0)):.2f} "
                f"risk={lm.get('risk', 'unknown')}"
            )

    if spread:
        slog.HYST_DEBUG(
            f"  [hyst_summary] stack={spread.get('stack')} spread={spread.get('spread')} risk={spread.get('risk')} "
            f"pressure={spread.get('pressure')} stability={float(spread.get('stability') or 0.0):.2f} "
            f"fan_tightness={float(spread.get('fan_tightness') or 0.0):.2f} alignment={float(spread.get('stack_alignment') or 0.0):.2f}"
        )

    slog.HYST_DEBUG(
        f"  [hyst_dbg] tail={hyst['meta']['tail_window_seconds']}s | "
        f"baseline={hyst['meta']['baseline_window_seconds']}s | "
        f"align_tol={hyst['meta']['align_tol_seconds']:.1f}s | "
        f"tracker_keys={len((_HYST_TRACKER.get('leader_state') or {}))}"
    )

    slog.HYST_HEADER(_line("-", 155))

    return hyst
