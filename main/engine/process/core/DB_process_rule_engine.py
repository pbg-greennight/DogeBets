# main/engine/process/DB_process_rule_engine.py
"""
A2: Rule engine that evaluates JSON-defined regimes using the expression language.

This is intentionally small + strict:
- Regimes are evaluated in ascending priority.
- Each regime has expr.when[] list; all must be True.
- The first matching regime yields a decision.

Config expectations (trend_method_engine_layer.json):
{
  "regimes": {
    "TIGHT_DRIFT": {
      "priority": 1,
      "override": true,
      "expr": {
        "when": ["..."],
        "trend": "'Bull' if ... else 'Bear'",
        "confidence": "clamp(...,0,1)",
        "scores": {
          "bull": "...", "bear": "...", "neutral": "..."
        },
        "reason": "'TIGHT_DRIFT'"
      }
    }
  }
}
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

from main.engine.process.core.DB_process_expr import eval_expr, resolve_path, clamp, sign


def build_eval_context(cfg: Dict[str, Any], calc_out: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the evaluation context consumed by expressions.
    """
    ctx: Dict[str, Any] = {}

    # truth object
    ctx["bell"] = calc_out.get("bell", {}) or {}
    ctx["channels"] = calc_out.get("channels", {}) or {}
    ctx["hyst"] = calc_out.get("hyst", {}) or {}

    # decision time
    ctx["decision_dt"] = calc_out.get("decision_dt")

    # thresholds: keep as a stable path
    ft = cfg.get("features_thresholds", {}) or {}
    ctx["thresholds"] = ft.get("dynamic_thresholds", {}) or {}

    # convenience aliases
    # (so you can write get("S1.m_norm") etc if desired later)
    stacks = resolve_path(ctx, "hyst.stacks", {}) or {}
    if isinstance(stacks, dict):
        for k, v in stacks.items():
            ctx[k] = v

    return ctx


def _regime_items(engine_layer: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    regimes = engine_layer.get("regimes", {}) or {}
    items = []
    for name, block in regimes.items():
        if isinstance(block, dict):
            items.append((name, block))
    # sort by priority (default 999)
    items.sort(key=lambda kv: int(kv[1].get("priority", 999)))
    return items


def evaluate_regimes(cfg: Dict[str, Any], calc_out: Dict[str, Any]) -> Dict[str, Any]:
    engine_layer = cfg.get("engine_layer", {}) or {}
    ctx = build_eval_context(cfg, calc_out)

    selected = None
    selected_name = None

    for name, block in _regime_items(engine_layer):
        expr = (block.get("expr", {}) or {}) if isinstance(block.get("expr", {}), dict) else {}
        when_list = expr.get("when", []) or []
        if not isinstance(when_list, list):
            continue
        ok = True
        for cond in when_list:
            if not isinstance(cond, str) or not cond.strip():
                continue
            val = eval_expr(cond, ctx, default=False)
            if not bool(val):
                ok = False
                break
        if ok:
            selected = block
            selected_name = name
            break

    if not selected:
        # fallback: neutral
        return {
            "trend": "Neutral",
            "confidence": 0.0,
            "scores": {"neutral": 1.0, "bull": 0.0, "bear": 0.0, "reversal": 0.0},
            "reason": "NO_REGIME_MATCH",
            "regime": None,
        }

    expr = selected.get("expr", {}) or {}
    trend = eval_expr(expr.get("trend", "'Neutral'"), ctx, default="Neutral")
    if trend not in ("Bull", "Bear", "Neutral"):
        trend = "Neutral"

    conf = eval_expr(expr.get("confidence", "0.0"), ctx, default=0.0)
    try:
        conf_f = float(conf)
    except Exception:
        conf_f = 0.0
    conf_f = clamp(conf_f, 0.0, 1.0)

    scores = {"neutral": 0.0, "bull": 0.0, "bear": 0.0, "reversal": 0.0}
    sc_expr = expr.get("scores", {}) or {}
    if isinstance(sc_expr, dict):
        for k in list(scores.keys()):
            if k in sc_expr and isinstance(sc_expr[k], str):
                v = eval_expr(sc_expr[k], ctx, default=0.0)
                try:
                    scores[k] = float(v)
                except Exception:
                    scores[k] = 0.0

    reason = eval_expr(expr.get("reason", f"'{selected_name}'"), ctx, default=selected_name)

    return {
        "trend": trend,
        "confidence": conf_f,
        "scores": scores,
        "reason": str(reason),
        "regime": selected_name,
    }
