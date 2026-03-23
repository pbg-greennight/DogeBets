# main/engine/process/printing/DB_process_printing_hyst.py

import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
    Returns hyst_obj features for downstream trend/calc code (but does not decide trend).
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

    # ----------------------------------------------------------------------------------
    # LOG BLOCK (matches the layout we designed)
    # ----------------------------------------------------------------------------------
    ep = hyst["episode"]
    pr = hyst["probe"]
    eta = hyst["eta"]
    stacks = hyst["stacks"]
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
    slog.HYST_HEADER("HYSTERESIS FAN STACK (method_hyster_v1.0)")
    slog.HYST_HEADER(_line("-", 155))

    if True:
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

    if True:
        slog.HYST_PROBE(
            f"[hyst_probe] pair=(23,83) | d23_83 now={pr['d_now']:+.6f} | "
            f"sign={pr['sign_now']} | flip_watch={int(pr['flip_watch'])} | "
            f"fast_collapse={int(pr['fast_collapse'])} | d_abs_slope_tail={pr['d_abs_slope_tail']:+.6f}"
        )

    if eta.get("eta_to_end_seconds") is not None:
        slog.HYST_ETA(
            f"[hyst_eta] eta={eta['eta_to_end_seconds']:.1f}s (~{eta['eta_to_end_seconds'] / 300.0:.2f} epochs) | "
            f"source={eta.get('source_stack')}"
        )

    if True:
        for sid in ["S0", "S1", "S2", "S3"]:
            lm = stacks.get(sid)
            if not lm:
                continue
            sigs = ",".join(map(str, lm["sigmas"]))
            W_pct = lm["W_pct"]
            W_pct_s = "nan" if not np.isfinite(W_pct) else f"{int(round(W_pct)):>3d}"
            slog.HYST_LADDER(
                f"  [hyst_{sid}] sigmas={sigs:<14} | "
                f"m_norm={lm['m_norm']:+.3f} | "
                f"W_pct={W_pct_s} | "
                f"dW_norm={lm['dW_norm']:+.3f} | "
                f"eff_order={lm['eff_order']:.2f} | "
                f"cross={lm['cross_rate']:.2f}/m"
            )

            # ---- appended GEOM line (same section: obeys slog.HYST_LADDER toggles)
            W_now_s = "nan" if not np.isfinite(lm.get("W_now", float("nan"))) else f"{lm['W_now']:.4f}"
            W_z_s = "nan" if not np.isfinite(lm.get("W_z", float("nan"))) else f"{lm['W_z']:+.2f}"
            W_ratio_s = "nan" if not np.isfinite(lm.get("W_ratio", float("nan"))) else f"{lm['W_ratio']:.2f}"
            ddW_s = f"{lm.get('ddW_norm', 0.0):+.3f}"
            dom_s = f"{lm.get('dom', 0.0):.2f}"
            lead_max = lm.get("leader_max_sigma")
            lead_min = lm.get("leader_min_sigma")
            leader_s = f"{lead_max}->{lead_min}" if (lead_max is not None and lead_min is not None) else "NONE"

            # cross pair breakdown (only show pairs with crossings)
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

            # ---- added optional diagnostics
            cross_pairs_obj = lm.get("cross_pairs") or {}
            near_count = 0
            min_gap = float("inf")
            for pair_key in cross_pairs_obj.keys():
                a, b = pair_key.split("-")
                try:
                    idx_a = lm["sigmas"].index(int(a))
                    idx_b = lm["sigmas"].index(int(b))
                    vals = _safe_get(per_sigma_hist, int(a), "values", default=[])
                    vals_b = _safe_get(per_sigma_hist, int(b), "values", default=[])
                    if vals and vals_b:
                        g = abs(float(vals[-1]) - float(vals_b[-1]))
                        min_gap = min(min_gap, g)
                        if g <= max(abs(lm.get("W_now", 0.0)) * 0.08, 0.25):
                            near_count += 1
                except Exception:
                    _ = idx_a if 'idx_a' in locals() else None
                    _ = idx_b if 'idx_b' in locals() else None
            if not np.isfinite(min_gap):
                min_gap = float("nan")

            gap_vel = -float(lm.get("dW", 0.0))
            if near_count > 0 and gap_vel > 0:
                near_state = "approaching"
            elif near_count > 0 and gap_vel <= 0:
                near_state = "hovering"
            elif gap_vel > 0:
                near_state = "pressuring"
            else:
                near_state = "calm"

            leader_key = f"{sid}_leader"
            cur_leader = leader_s
            ls = (_HYST_TRACKER.get("leader_state") or {}).get(leader_key, {
                "leader": None,
                "age": 0,
                "switch_ts": [],
            })
            if ls.get("leader") == cur_leader:
                ls["age"] = int(ls.get("age", 0)) + 1
            else:
                ls["leader"] = cur_leader
                ls["age"] = 1
                ts_hist = list(ls.get("switch_ts") or [])
                ts_hist.append(decision_dt)
                ls["switch_ts"] = ts_hist[-20:]
            _HYST_TRACKER.setdefault("leader_state", {})[leader_key] = ls

            switchN = len(ls.get("switch_ts") or [])
            leader_stab = float(ls.get("age", 0) / max(1, ls.get("age", 0) + switchN))

            cross_rate = float(lm.get("cross_rate", 0.0))
            order = float(lm.get("eff_order", 0.0))
            W_stability = float(1.0 / (1.0 + abs(lm.get("ddW_norm", 0.0))))
            cross_stability = float(1.0 / (1.0 + cross_rate))
            near_penalty = min(1.0, near_count / max(1.0, float(len(cross_pairs_obj) or 1)))
            stability_score = max(0.0, min(1.0, 0.35 * cross_stability + 0.30 * order + 0.20 * W_stability + 0.15 * leader_stab - 0.20 * near_penalty))
            if stability_score < 0.25:
                stability_state = "fragile"
            elif stability_score < 0.50:
                stability_state = "unstable"
            elif stability_score < 0.75:
                stability_state = "stable"
            else:
                stability_state = "strong"

            dWn = float(lm.get("dW_norm", 0.0))
            ddWn = float(lm.get("ddW_norm", 0.0))
            if abs(dWn) < 0.05:
                spread_state = "frozen"
            elif dWn > 0:
                spread_state = "widening"
            else:
                spread_state = "narrowing"
            spread_momentum = "flat"
            if ddWn > 0.05:
                spread_momentum = "strengthening"
            elif ddWn < -0.05:
                spread_momentum = "weakening"
            exhaust = int(abs(dWn) < 0.12 and abs(ddWn) > 0.10)

            order_conf = max(0.0, min(1.0, 0.6 * order + 0.4 * cross_stability))
            order_break = max(0.0, min(1.0, 1.0 - order_conf + near_penalty * 0.4))

            risk_state = "continuation_friendly"
            if order_break > 0.70:
                risk_state = "collapse_watch"
            elif near_penalty > 0.5 and cross_rate > 0.6:
                risk_state = "whipsaw_risk"
            elif spread_state == "narrowing" and spread_momentum == "weakening":
                risk_state = "reversal_watch"

            if sec_stability:
                slog.HYST_STABILITY(
                    f"[hyst_stability] {sid} stability={stability_score:.2f} {stability_state} | "
                    f"leader_age={int(ls.get('age', 0))} switchN={switchN} leader_stab={leader_stab:.2f}"
                )
            if sec_pressure:
                slog.HYST_PRESSURE(
                    f"[hyst_pressure] {sid} near_cross={near_count} min_gap={min_gap if np.isfinite(min_gap) else float('nan'):.3f} "
                    f"gap_vel={gap_vel:+.3f} state={near_state}"
                )
            if sec_spread or sec_risk:
                slog.HYST_SPREAD_STATE(
                    f"[hyst_spread_state] {sid} spread={spread_state} momentum={spread_momentum} exhaust={exhaust} "
                    f"order_conf={order_conf:.2f} break_risk={order_break:.2f} risk={risk_state}"
                )

    if True:
        slog.HYST_DEBUG(
            f"  [hyst_dbg] tail={hyst['meta']['tail_window_seconds']}s | "
            f"baseline={hyst['meta']['baseline_window_seconds']}s | "
            f"align_tol={hyst['meta']['align_tol_seconds']:.1f}s"
        )

    slog.HYST_HEADER(_line("-", 155))

    return hyst