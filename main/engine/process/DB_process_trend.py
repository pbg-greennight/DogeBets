# main/engine/process/DB_process_trend.py
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from main.engine.process.DB_process_types import TrendDecision
from main.engine.process.DB_process_calc import load_trend_method_config_v2
from main.engine.process.DB_process_rule_engine import evaluate_regimes
from main.engine.process.printing.DB_process_printing_hyst import compute_hysteresis_features

# -------------------------------------------------------------------------------------------------
# Model path defaults
# -------------------------------------------------------------------------------------------------

DEFAULT_MODEL_FILE = "trend_method_v2.json"
DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent / "models" / DEFAULT_MODEL_FILE)

log = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Small numeric helpers
# -------------------------------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _median_abs_dev(x: list[float]) -> float:
    """MAD around median (robust scale)."""
    if not x:
        return 0.0
    med = statistics.median(x)
    dev = [abs(v - med) for v in x]
    return float(statistics.median(dev))


def _safe_get_values(blob: Any) -> list[float]:
    """
    per_sigma_full[s] might be:
      - {'values': [...], ...}
      - or directly a list/tuple of floats
    """
    if blob is None:
        return []
    if isinstance(blob, dict):
        vals = blob.get("values") or blob.get("vals") or blob.get("series") or []
        return list(vals) if isinstance(vals, (list, tuple)) else []
    if isinstance(blob, (list, tuple)):
        return list(blob)
    return []


def _compute_sign_to(values: list[float], k: int = 21) -> int:
    """Direction sign using a short tail slope (last vs last-k)."""
    if len(values) < 2:
        return 0
    kk = min(max(2, k), len(values))
    a = float(values[-kk])
    b = float(values[-1])
    d = b - a
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0


def _compute_hook(values: list[float], short_k: int = 10, long_k: int = 50) -> int:
    """
    Hook = short-term slope sign disagrees with longer-term slope sign
           AND short-term move is meaningfully strong vs tail noise.
    Returns 1 if hook detected else 0.
    """
    n = len(values)
    if n < 12:
        return 0

    sk = min(max(2, short_k), n - 1)
    lk = min(max(3, long_k), n - 1)

    short_slope = (float(values[-1]) - float(values[-1 - sk])) / float(max(1, sk))
    long_slope = (float(values[-1]) - float(values[-1 - lk])) / float(max(1, lk))

    s_sign = 1 if short_slope > 0 else (-1 if short_slope < 0 else 0)
    l_sign = 1 if long_slope > 0 else (-1 if long_slope < 0 else 0)

    if s_sign == 0 or l_sign == 0:
        return 0
    if s_sign == l_sign:
        return 0

    tail = [float(v) for v in values[-(sk + 1):]]
    diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    mad = _median_abs_dev(diffs) + 1e-9

    strength = abs(short_slope) / mad
    return 1 if strength >= 2.0 else 0


def _compute_flat(values: list[float], k: int = 21) -> float:
    """
    Flatness score in [0..1], higher means flatter.
    Uses |tail slope| relative to robust tail noise (MAD of diffs).
    """
    if len(values) < 3:
        return 1.0

    kk = min(max(2, k), len(values) - 1)
    tail = [float(v) for v in values[-(kk + 1):]]
    diffs = [tail[i + 1] - tail[i] for i in range(len(tail) - 1)]
    mad = _median_abs_dev(diffs) + 1e-9

    slope = (tail[-1] - tail[0]) / float(max(1, kk))

    # big slope vs noise -> flat ~ 0; tiny slope vs noise -> flat ~ 1
    ratio = abs(slope) / (5.0 * mad)
    return _clamp01(1.0 - ratio)


def _build_per_sigma_inputs(
    per_sigma_full: Dict[int, Any],
    wanted_sigmas: list[int],
    diag_chunk: int = 12,
    tail_n: int = 21,
    debug: bool = False,
) -> Dict[int, Dict[str, Any]]:
    """
    Build the exact per-sigma payload used by hysteresis + future engines.

    Keys per sigma:
      - values: list[float]
      - sign_to: int (-1/0/+1)
      - hook: int (0/1)
      - flat: float [0..1]
      - diag: list[float] (small tail) for debugging/printing
    """
    out: Dict[int, Dict[str, Any]] = {}

    if debug:
        log.info("[trend_debug] ---- BUILD PER SIGMA INPUTS START ----")
        try:
            log.info(f"[trend_debug] per_sigma_full keys: {sorted(list(per_sigma_full.keys()))}")
        except Exception:
            log.info("[trend_debug] per_sigma_full keys: <unavailable>")

    for s in wanted_sigmas:
        blob = per_sigma_full.get(int(s), {}) if isinstance(per_sigma_full, dict) else {}
        values = _safe_get_values(blob)

        if not values:
            out[int(s)] = {"values": [], "sign_to": 0, "hook": 0, "flat": 1.0, "diag": []}
            if debug:
                log.info(f"[trend_debug] sigma={s} MISSING/EMPTY series")
            continue

        sign_to = _compute_sign_to(values, k=tail_n)
        hook = _compute_hook(values, short_k=max(8, tail_n // 2), long_k=max(30, tail_n * 2))
        flat = _compute_flat(values, k=tail_n)
        diag = [float(v) for v in (values[-diag_chunk:] if diag_chunk else [])]

        out[int(s)] = {
            "values": [float(v) for v in values],
            "sign_to": int(sign_to),
            "hook": int(hook),
            "flat": float(flat),
            "diag": diag,
        }

        if debug:
            log.info(
                f"[trend_debug] sigma={s} n_vals={len(values)} "
                f"first={float(values[0]):.4f} last={float(values[-1]):.4f} "
                f"sign_to={sign_to} hook={hook} flat={flat:.3f}"
            )

    if debug:
        log.info("[trend_debug] ---- BUILD PER SIGMA INPUTS END ----")

    return out


# -------------------------------------------------------------------------------------------------
# A2: Expression-rule engine decision (trend_method_v2.json)
# -------------------------------------------------------------------------------------------------


def _linear_slope(y: List[float]) -> float:
    """Simple least-squares slope vs index (0..n-1). Returns 0.0 if insufficient data."""
    n = len(y)
    if n < 2:
        return 0.0
    x = list(range(n))
    x_mean = (n - 1) / 2.0
    y_mean = sum(y) / n
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    return float(num / den) if den != 0 else 0.0


def _build_hyst_stacks_from_pairs(
    per_sigma_full: Dict[int, Dict[str, Any]],
    pairs_cfg: Dict[str, Any],
    tail_points: int = 60,
) -> Dict[str, Any]:
    """
    Adapter: create the truth-schema expected by the V2 rule engine:
        hyst.stacks.S1.{a,b,W_now,W_p25,m,m_norm,d_abs_slope_tail,eff_order,sign}
    If an upstream hysteresis engine already provides this, we keep it.
    """
    stacks: Dict[str, Any] = {}
    pairs = pairs_cfg.get("pairs") or {}
    for sname, pair in pairs.items():
        try:
            a = int(pair.get("a"))
            b = int(pair.get("b"))
        except Exception:
            continue

        va = _safe_get_values(per_sigma_full.get(a))
        vb = _safe_get_values(per_sigma_full.get(b))
        n = min(len(va), len(vb))
        if n < 2:
            continue

        # diff series
        d = [float(va[i]) - float(vb[i]) for i in range(n)]
        d_abs = [abs(x) for x in d]

        W_now = float(d[-1])
        # Robust scale: 25th percentile of |d|
        sorted_abs = sorted(d_abs)
        k = int(0.25 * (len(sorted_abs) - 1))
        W_p25 = float(sorted_abs[k]) if sorted_abs else 0.0

        # Additional percentiles/ratios used by some regimes
        k75 = int(0.75 * (len(sorted_abs) - 1))
        W_p75 = float(sorted_abs[k75]) if sorted_abs else 0.0
        # Percentile rank of |W_now| within |d|
        try:
            # count fraction of |d| <= |W_now|
            W_pct = 100.0 * (sum(1 for v in d_abs if v <= abs(W_now)) / float(max(1, len(d_abs))))
        except Exception:
            W_pct = 0.0

        tail = d[-tail_points:] if len(d) >= tail_points else d
        tail_abs = d_abs[-tail_points:] if len(d_abs) >= tail_points else d_abs

        m = _linear_slope(tail)
        d_abs_slope_tail = _linear_slope(tail_abs)
        m_norm = float(m / W_p25) if W_p25 not in (0.0, -0.0) else 0.0

        # dW_norm: normalized change in W over the tail
        try:
            dW = float(tail[-1] - tail[0]) if len(tail) >= 2 else 0.0
            dW_norm = float(dW / W_p25) if W_p25 not in (0.0, -0.0) else 0.0
        except Exception:
            dW_norm = 0.0

        # cross_rate: sign flips per point in tail
        try:
            flips = 0
            last_s = 0
            for x in tail:
                s = 1 if x > 0 else (-1 if x < 0 else 0)
                if last_s != 0 and s != 0 and s != last_s:
                    flips += 1
                if s != 0:
                    last_s = s
            cross_rate = float(flips) / float(max(1, len(tail) - 1))
        except Exception:
            cross_rate = 0.0

        # eff_order: a light-weight proxy for how "separated" the pair is
        # (kept conservative; regimes typically allow <= 3)
        eff_order = 1 if W_p25 == 0.0 else (1 if abs(W_now) <= 2.0 * W_p25 else 2)

        stacks[sname] = {
            "a": a,
            "b": b,
            "W_now": W_now,
            "W_p25": W_p25,
            "W_p75": W_p75,
            "W_pct": float(W_pct),
            "m": m,
            "m_norm": m_norm,
            "dW_norm": float(dW_norm),
            "cross_rate": float(cross_rate),
            "d_abs_slope_tail": float(d_abs_slope_tail),
            "eff_order": int(eff_order),
            "sign": 1 if W_now > 0 else (-1 if W_now < 0 else 0),
        }

    return {"stacks": stacks}

def _resolve_model_path(model_path: Optional[str], config: Dict[str, Any]) -> str:
    mp = (model_path or (config.get("MODEL_PATH") if isinstance(config, dict) else None) or DEFAULT_MODEL_PATH)
    return str(mp)


def _get_model_id(cfg: Dict[str, Any], mp: str) -> str:
    for key in ("model_id", "method_id", "id", "MODEL_ID"):
        v = cfg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    try:
        return Path(mp).name
    except Exception:
        return "trend_method_v2"


def _build_truth_object(
    decision_dt: datetime,
    epoch_id: Optional[int],
    per_sigma_full: Dict[int, Any],
    cfg: Dict[str, Any],
    hyst_obj: Optional[dict] = None,
    bell: Optional[dict] = None,
    channels: Optional[dict] = None,
) -> Dict[str, Any]:
    """
    Build the *truth* dict consumed by the V2 rule-engine.

    IMPORTANT:
    - The JSON regime layer matches on paths like `hyst.stacks.S1.W_now`.
      Your upstream hysteresis builder may log those, but it does not always
      expose them under the exact schema expected by the rule engine.
    - When `hyst_obj` doesn't already contain `stacks`, we build a minimal,
      deterministic `hyst.stacks` from the configured pairs (S1/S2/...) using
      the current per-sigma series (no extra dependencies).
    """

    # --- decision ---
    decision = {
        "dt_iso": decision_dt.isoformat(),
        "epoch_id": int(epoch_id) if epoch_id is not None else None,
    }

    # --- hyst (ensure stacks schema exists) ---
    hyst_dict: Dict[str, Any] = hyst_obj or {}
    if not isinstance(hyst_dict, dict):
        hyst_dict = {"raw": hyst_dict}

    if "stacks" not in hyst_dict or not isinstance(hyst_dict.get("stacks"), dict):
        pairs_cfg = None
        # pairs can live at cfg['pairs'] or cfg['parts']['pairs'] depending on loader
        if isinstance(cfg, dict):
            if isinstance(cfg.get("pairs"), dict):
                pairs_cfg = {"pairs": cfg["pairs"]}
            elif isinstance((cfg.get("parts") or {}).get("pairs"), dict):
                pairs_cfg = {"pairs": (cfg["parts"] or {})["pairs"]}
            elif isinstance(cfg.get("time_inputs_pairs"), dict) and isinstance(cfg["time_inputs_pairs"].get("pairs"), dict):
                pairs_cfg = cfg["time_inputs_pairs"]

        hyst_built = _build_hyst_stacks_from_pairs(per_sigma_full, pairs_cfg or {"pairs": {}}, tail_points=60)
        # Merge: keep any upstream meta, but guarantee .stacks
        meta = hyst_dict.get("meta") if isinstance(hyst_dict.get("meta"), dict) else {}
        hyst_dict = {**hyst_dict, **hyst_built}
        if meta:
            hyst_dict["meta"] = meta

    # --- gbc_diag ---
    # If caller provides a bell/diag blob, use it; otherwise create a minimal per-sigma diag.
    gbc_diag = bell if isinstance(bell, dict) else {}
    if not gbc_diag:
        wanted = sorted([int(k) for k in (per_sigma_full or {}).keys() if isinstance(k, int)])
        sigmas_out: Dict[str, Any] = {}
        for s in wanted:
            obj = per_sigma_full.get(s) or {}
            sigmas_out[str(s)] = {
                "last": float((obj.get("vals") or [0.0])[-1]) if (obj.get("vals") or []) else 0.0,
            }
        gbc_diag = {"sigmas": sigmas_out}

    # --- csd (channels) ---
    csd = channels if isinstance(channels, dict) else {}

    return {
        "decision": decision,
        "gbc_diag": gbc_diag,
        "csd": csd,
        "dcsd": {},  # delta-channel snapshot (optional)
        "msbc": {},  # multi-sigma bell catalog (optional)
        "hyst": hyst_dict,
    }

def calculate_trend(
    curr_epoch: int,
    next_epoch: int,
    windows: Any,
    per_sigma_full: Dict[int, Any],
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    hyst_obj: Optional[Any] = None,
) -> TrendDecision:
    """
    Orchestrator entry point (signature must stay stable).

    This function:
      - Resolves/loads trend_method_v2.json (multi-include)
      - Builds hysteresis truth object (unless provided)
      - Evaluates regimes using A2 expression language
      - Returns TrendDecision + attaches extras for printing/debug
    """
    try:
        debug = bool(config.get("TREND_DEBUG", False)) if isinstance(config, dict) else False
        mp = _resolve_model_path(model_path, config)

        if debug:
            p = Path(mp)
            log.info(f"[trend_debug] model_path resolved: {mp}")
            log.info(f"[trend_debug] model_path exists: {p.exists()}")
            log.info(f"[trend_debug] this_file_dir: {Path(__file__).resolve().parent}")

        # Build per-sigma derived inputs (used by hyst + future engines)
        wanted_sigmas = (config.get("GAUSS_SIGMAS") if isinstance(config, dict) else None) or [8, 23, 38, 53, 68, 83]
        diag_chunk = int(config.get("DUMP_DIAG_CHUNK", 12)) if isinstance(config, dict) else 12
        tail_n = int(config.get("TAIL_FEATURE_POINTS", 21)) if isinstance(config, dict) else 21
        _ = _build_per_sigma_inputs(per_sigma_full, wanted_sigmas, diag_chunk=diag_chunk, tail_n=tail_n, debug=debug)

        # Load merged v2 model config (includes + engine_library injection handled there)
        cfg = load_trend_method_config_v2(mp)
        model_id = _get_model_id(cfg, mp)

        # Build hysteresis features if not supplied
        decision_dt = datetime.now()
        if isinstance(hyst_obj, dict):
            hyst_dict = hyst_obj
        else:
            try:
                hyst_dict = compute_hysteresis_features(
                    per_sigma_hist=per_sigma_full,
                    decision_dt=decision_dt,
                    lookback_seconds=int(config.get("HYST_LOOKBACK_SECONDS", 3600)),
                    tail_seconds=int(config.get("HYST_TAIL_SECONDS", 120)),
                    align_tol_seconds=float(config.get("HYST_ALIGN_TOL_SECONDS", 3.0)),
                    primary_default=tuple(config.get("HYST_PRIMARY_DEFAULT_PAIR", (38, 83))),
                    primary_slow=tuple(config.get("HYST_PRIMARY_SLOW_PAIR", (53, 83))),
                    probe_pair=tuple(config.get("HYST_PROBE_PAIR", (23, 83))),
                )
            except Exception as e:
                hyst_dict = {"meta": {"skipped": True, "skip_reason": "hyst_build_exception", "err": str(e)}}

        truth = _build_truth_object(decision_dt=decision_dt, epoch_id=curr_epoch, per_sigma_full=per_sigma_full, cfg=cfg, hyst_obj=hyst_dict)

        # Evaluate JSON-defined regimes (A2)
        dec = evaluate_regimes(cfg, truth) or {}

        trend = str(dec.get("trend", "Neutral") or "Neutral")
        conf = float(dec.get("confidence", 0.0) or 0.0)

        reason = str(dec.get("reason", "RULE_ENGINE") or "RULE_ENGINE")
        regime = dec.get("regime")
        scores = dec.get("scores", {}) or {}

        # ----------------------------------------------------------
        # ADD DETAILED DISORDER DIAGNOSTICS
        # ----------------------------------------------------------
        if trend == "Neutral" and reason in ("DISORDER", "RULE_ENGINE"):
            try:
                s1 = (((truth.get("hyst") or {}).get("stacks") or {}).get("S1") or {})

                if isinstance(s1, dict) and s1:

                    cross_rate = float(s1.get("cross_rate", 0.0))
                    eff_order = float(s1.get("eff_order", 0.0))
                    W_now = float(s1.get("W_now", 0.0))
                    W_p75 = float(s1.get("W_p75", 0.0))
                    W_pct = float(s1.get("W_pct", 0.0))
                    m_norm = float(s1.get("m_norm", 0.0))

                    reason = (
                        f"DISORDER | "
                        f"cross_rate={cross_rate:.3f} "
                        f"eff_order={eff_order:.2f} "
                        f"W_pct={W_pct:.1f} "
                        f"W_now={W_now:.3f} "
                        f"m_norm={m_norm:.3f}"
                    )

                else:
                    reason = "DISORDER | hyst.S1 missing"

            except Exception as e:
                reason = f"DISORDER | diag_error={str(e)}"

        if debug:
            try:
                s1 = (((truth.get("hyst") or {}).get("stacks") or {}).get("S1") or {})
                if isinstance(s1, dict) and s1:
                    log.info(
                        "[trend_debug] hyst.S1: cross_rate=%.3f eff_order=%.3f order_bull=%.1f order_bear=%.1f W_now=%.3f W_p75=%.3f W_pct=%.1f m_norm=%.3f",
                        float(s1.get("cross_rate", 0.0) or 0.0),
                        float(s1.get("eff_order", 0.0) or 0.0),
                        float(s1.get("order_bull", 0.0) or 0.0),
                        float(s1.get("order_bear", 0.0) or 0.0),
                        float(s1.get("W_now", 0.0) or 0.0),
                        float(s1.get("W_p75", 0.0) or 0.0),
                        float(s1.get("W_pct", 0.0) or 0.0),
                        float(s1.get("m_norm", 0.0) or 0.0),
                    )
                else:
                    log.info("[trend_debug] hyst.S1 missing or empty (stacks keys=%s)", list(((truth.get("hyst") or {}).get("stacks") or {}).keys()))
            except Exception as _e:
                log.info("[trend_debug] hyst.S1 debug dump failed: %s", _e)

            log.info(f"[trend_debug] regime={regime} trend={trend} conf={conf:.3f} reason={reason}")

        td = TrendDecision(
            trend=trend,
            confidence=float(conf),
            model=str(model_id),
            notes=str(reason),
        )

        setattr(td, "extras", {
            "regime": regime,
            "scores": scores,
            "reason": reason,
            "decision_dt": decision_dt.isoformat(),
            "epochs": {"curr": curr_epoch, "next": next_epoch},
            "hyst": hyst_dict,
        })

        return td

    except Exception as e:
        mp = _resolve_model_path(model_path, config if isinstance(config, dict) else {})
        log.exception(f"[calculate_trend] failed: {e}")
        log.error(f"[calculate_trend] model_path attempted: {mp}")
        log.error(f"[calculate_trend] default_model_path: {DEFAULT_MODEL_PATH}")
        log.error(f"[calculate_trend] this_file_dir: {Path(__file__).resolve().parent}")
        try:
            exists = Path(mp).exists()
        except Exception:
            exists = False
        log.error(f"[calculate_trend] model_exists: {exists}")

        td = TrendDecision(
            trend="Neutral",
            confidence=0.0,
            model="trend_method_v2",
            notes="ERROR_FALLBACK",
        )
        setattr(td, "extras", {"error": str(e), "model_path": mp})
        return td


# -------------------------------------------------------------------------------------------------
# Optional engine layer API (kept for backward compatibility)
# -------------------------------------------------------------------------------------------------

DEFAULT_ENGINE_ORDER = ("MODEL_hyster", "MODEL_gaussian")


@dataclass
class EngineResult:
    engine_key: str
    enabled: bool
    model_id: str
    decision: TrendDecision
    diagnostics: Dict[str, Any]


def run_trend_layer(
    per_sigma_full: Dict[int, Any],
    timing: Dict[str, Any],
    windows: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    slog: Any = None,
) -> Tuple[TrendDecision, Dict[str, Any]]:
    """
    Compatibility wrapper. If your orchestrator calls run_trend_layer(), this will:
      - compute a TrendDecision via calculate_trend() using v2 model
      - return diagnostics for printing
    """
    config = config or {}
    windows = windows or {}

    curr_epoch = int(timing.get("epoch_analyze") or timing.get("curr_epoch") or 0)
    next_epoch = int(timing.get("epoch_predict") or timing.get("next_epoch") or 0)

    td = calculate_trend(
        curr_epoch=curr_epoch,
        next_epoch=next_epoch,
        windows=windows,
        per_sigma_full=per_sigma_full,
        config=config,
        model_path=(config.get("MODEL_PATH") if isinstance(config, dict) else None),
        hyst_obj=None,
    )

    diag_out = {
        "engine_layer": {
            "winner_engine": "MODEL_hyster_v2",
            "decision": {
                "trend": getattr(td, "trend", "Neutral"),
                "confidence": float(getattr(td, "confidence", 0.0) or 0.0),
                "model": getattr(td, "model", ""),
                "notes": getattr(td, "notes", ""),
            },
        }
    }

    return td, diag_out