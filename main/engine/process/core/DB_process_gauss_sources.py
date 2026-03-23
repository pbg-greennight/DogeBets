"""main/engine/process/DB_process_gauss_sources.py

Gaussian source registry and access helpers.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from main.engine.indicators.gaussian import (
    indicators_gauss8,
    indicators_gauss23,
    indicators_gauss38,
    indicators_gauss53,
    indicators_gauss68,
    indicators_gauss83,
)

from main.engine.process.utils.DB_process_time import _parse_dt_maybe



def get_gauss_registry() -> List[Dict[str, Any]]:
    return [
        {"sigma": 8, "module": indicators_gauss8, "key": "g8"},
        {"sigma": 23, "module": indicators_gauss23, "key": "g23"},
        {"sigma": 38, "module": indicators_gauss38, "key": "g38"},
        {"sigma": 53, "module": indicators_gauss53, "key": "g53"},
        {"sigma": 68, "module": indicators_gauss68, "key": "g68"},
        {"sigma": 83, "module": indicators_gauss83, "key": "g83"},
    ]


def start_gauss_sources_once(registry: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None) -> None:
    """Start the gaussian indicator sources.

    Additive behavior: if the downstream gaussian modules expose a configurable
    warm-start lookback value, we set it here before calling start().

    This is best-effort (no hard dependency on any attribute names) so it won't
    break if a given sigma module doesn't implement it.
    """

    warm_minutes = None
    if isinstance(config, dict):
        try:
            warm_minutes = int(config.get("WARM_START_MINUTES", 240))
        except Exception:
            warm_minutes = 240

    for r in registry:
        mod = r["module"]

        # Best-effort knobs (different modules may use different names)
        if warm_minutes is not None:
            for attr in (
                "WARM_START_MINUTES",
                "WARM_START_LOOKBACK_MINUTES",
                "WARM_START_WINDOW_MINUTES",
                "WARM_START_WINDOW_MINS",
                "WARM_START_MINS",
            ):
                if hasattr(mod, attr):
                    try:
                        setattr(mod, attr, warm_minutes)
                    except Exception:
                        pass

        try:
            if hasattr(mod, "start"):
                mod.start()
        except Exception as e:
            logging.warning(f"⚠️ Failed to start gauss sigma={r['sigma']}: {e}")


def _normalize_ts_list(ts_list: List[Any], tz_fallback: Optional[Any]) -> List[datetime]:
    out: List[datetime] = []
    for t in ts_list:
        dt = _parse_dt_maybe(t)
        if dt is None:
            dt = datetime.now(tz_fallback) if tz_fallback is not None else datetime.now()
        out.append(dt)
    return out


def fetch_plot_series(module: Any, key: str, tz_fallback: Optional[Any]) -> Tuple[List[datetime], List[Any]]:
    """
    Fetch plot series from gaussian module.

    Expected: module.get_plot_series() -> {"ts": [...], <value_key>: [...]}

    We try:
      1) requested `key`
      2) common fallback keys
      3) auto-pick the first non-"ts" list-like field with data
    """
    d = module.get_plot_series() or {}

    ts_raw = d.get("ts", []) or []
    ts = _normalize_ts_list(list(ts_raw), tz_fallback)

    # 1) requested key
    vals = d.get(key, None)
    if isinstance(vals, list) and len(vals) > 0:
        return ts, list(vals)

    # 2) common fallback keys
    for k in ("g", "gauss", "y", "values", "smoothed", "series"):
        vals2 = d.get(k, None)
        if isinstance(vals2, list) and len(vals2) > 0:
            logging.info(f"[gauss_sources] key '{key}' missing -> using '{k}' for module={getattr(module, '__name__', str(module))}")
            return ts, list(vals2)

    # 3) auto-pick any list field that isn't ts and has data
    for k, v in d.items():
        if k == "ts":
            continue
        if isinstance(v, list) and len(v) > 0:
            logging.info(f"[gauss_sources] key '{key}' missing -> auto-selected '{k}' for module={getattr(module, '__name__', str(module))}")
            return ts, list(v)

    # nothing usable yet
    return ts, []
