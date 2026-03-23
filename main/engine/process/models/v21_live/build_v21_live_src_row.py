from __future__ import annotations

"""
build_v21_live_src_row.py

Adapter skeleton for porting the best v21.x research decision layer into the live
process engine. This module does NOT fetch archive/parquet data. It uses the
existing process snapshot objects:

- FeatureCatalog (`catalog`)
- live per-sigma history (`per_sigma_hist`)
- live hysteresis payload (`hyst_obj`)
- timing/windows/config

Goal:
    build one canonical `src_*` row dict whose field names match the research
    model's expected source schema, so the same v21 feature/scoring code can be
    reused live.

Notes:
- This is intentionally an adapter only.
- The process feature-family modules already exist, so we reuse them here.
- Some process feature modules still contain TODO extraction internals; this file
  assumes those builders return the best available live payloads from the current
  snapshot and simply flattens them into research-compatible names.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from main.engine.process.features.DB_process_msbc_features import (
    build_msbc_feature_payload,
    flatten_msbc_to_src,
)
from main.engine.process.features.DB_process_gcs_features import (
    build_gcs_feature_payload,
    flatten_gcs_to_src,
)
from main.engine.process.features.DB_process_csd_dcsd_features import (
    build_csd_dcsd_feature_payload,
    flatten_csd_dcsd_to_src,
)
from main.engine.process.features.DB_process_gbc_features import (
    build_gbc_feature_payload,
    flatten_gbc_to_src,
)
from main.engine.process.features.DB_process_hysteresis import (
    flatten_hysteresis_to_src,
)

SIGMAS_ALL = [8, 23, 38, 53, 68, 83]


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return default


def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _catalog_get(catalog: Any, path: str, default: Any = None) -> Any:
    """
    FeatureCatalog path helper.

    Works with:
    - catalog.get("a.b.c", default)
    - nested dict-like fallback if needed
    """
    try:
        getter = getattr(catalog, "get", None)
        if callable(getter):
            try:
                return getter(path, default)
            except TypeError:
                val = getter(path)
                return default if val is None else val
    except Exception:
        pass

    cur = catalog
    try:
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
            if cur is None:
                return default
        return cur
    except Exception:
        return default


def _extract_close_proxy(per_sigma_hist: Dict[int, Dict[str, Any]], sigma: int = 23) -> Optional[dict]:
    blob = per_sigma_hist.get(int(sigma), {}) if isinstance(per_sigma_hist, dict) else {}
    ts = blob.get("ts") or []
    values = blob.get("values") or []
    if not ts or not values:
        return None
    return {"ts": ts, "values": values}


def _build_meta_src(
    timing: Any,
    windows: Any,
    decision_dt: datetime,
    per_sigma_hist: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build live meta fields that parallel the research src_meta_* contract.
    """
    out: Dict[str, Any] = {}

    out["src_meta_epoch"] = _safe_int(_safe_getattr(timing, "curr_epoch"))
    out["src_meta_next_epoch"] = _safe_int(_safe_getattr(timing, "next_epoch"))
    out["src_meta_prev_epoch"] = _safe_int(_safe_getattr(timing, "prev_epoch"))
    out["src_meta_decision_time"] = decision_dt.isoformat() if decision_dt else None

    full_start = _safe_getattr(windows, "full_start")
    full_end = _safe_getattr(windows, "full_end")
    out["src_meta_window_start"] = full_start.isoformat() if hasattr(full_start, "isoformat") else None
    out["src_meta_window_end"] = full_end.isoformat() if hasattr(full_end, "isoformat") else None

    # Use sigma-23 as BTC close proxy, matching the current process tailing reference convention.
    close_proxy = _extract_close_proxy(per_sigma_hist, sigma=23)
    if close_proxy:
        vals = close_proxy["values"]
        out["src_meta_close_now"] = _safe_float(vals[-1]) if vals else None
        out["src_meta_hist_points"] = len(vals)
    else:
        out["src_meta_close_now"] = None
        out["src_meta_hist_points"] = 0

    return out


@dataclass
class V21LiveSrcInputs:
    timing: Any
    windows: Any
    decision_dt: datetime
    catalog: Any
    per_sigma_hist: Dict[int, Dict[str, Any]]
    hyst_obj: Optional[dict]
    config: Optional[dict] = None


def build_v21_live_src_row(inp: V21LiveSrcInputs) -> Dict[str, Any]:
    """
    Main live adapter.

    Returns a single row-shaped dict with canonical `src_*` field names that
    match the research pipeline as closely as possible.
    """
    timing = inp.timing
    windows = inp.windows
    decision_dt = inp.decision_dt
    catalog = inp.catalog
    per_sigma_hist = inp.per_sigma_hist or {}
    hyst_obj = inp.hyst_obj or {}
    config = inp.config or {}

    row: Dict[str, Any] = {}

    # ---------------------------------------------------------------------------------
    # 1) META
    # ---------------------------------------------------------------------------------
    row.update(_build_meta_src(timing, windows, decision_dt, per_sigma_hist))

    # ---------------------------------------------------------------------------------
    # 2) SOURCE SNAPSHOTS FROM CURRENT LIVE CATALOG
    # ---------------------------------------------------------------------------------
    # These are the live process objects already produced at decision time.
    #
    # Expected current sources:
    # - bell payload / series used by MSBC + GBC
    # - gaussian channel snapshot used by GCS
    # - pv_tail channel payload used by DCSD
    # - hysteresis object used by HYST
    #
    # We intentionally keep these lookups centralized here so any path changes in
    # FeatureCatalog only need one patch.
    bell = _catalog_get(catalog, "bell", {}) or _catalog_get(catalog, "bell.current", {}) or {}
    bell_curve_series = _catalog_get(catalog, "bell.series", None) or _catalog_get(catalog, "bell_curve_series", None)

    channel_snapshot = (
        _catalog_get(catalog, "channels.snapshot", {})
        or _catalog_get(catalog, "channels.current", {})
        or _catalog_get(catalog, "gaussian_channels.snapshot", {})
        or {}
    )

    pv_tail = (
        _catalog_get(catalog, "channels.pv_tail", {})
        or _catalog_get(catalog, "pv_tail", {})
        or {}
    )

    # ---------------------------------------------------------------------------------
    # 3) FAMILY PAYLOADS -> FLATTEN TO RESEARCH src_* CONTRACT
    # ---------------------------------------------------------------------------------
    msbc_obj = build_msbc_feature_payload(
        bell=bell,
        bell_curve_series=bell_curve_series,
        config=config,
    )
    row.update(flatten_msbc_to_src(msbc_obj, config))

    gcs_obj = build_gcs_feature_payload(
        channel_snapshot=channel_snapshot,
        per_sigma_full=per_sigma_hist,   # history snapshot is the correct live source
        config=config,
    )
    row.update(flatten_gcs_to_src(gcs_obj, config))

    dcsd_obj = build_csd_dcsd_feature_payload(
        pv_tail=pv_tail,
        config=config,
    )
    row.update(flatten_csd_dcsd_to_src(dcsd_obj, config))

    gbc_obj = build_gbc_feature_payload(
        bell=bell,
        bell_curve_series=bell_curve_series,
        config=config,
    )
    row.update(flatten_gbc_to_src(gbc_obj, config))

    row.update(flatten_hysteresis_to_src(hyst_obj, config))

    # ---------------------------------------------------------------------------------
    # 4) LIVE SNAPSHOT QUALITY / DEBUG FIELDS
    # ---------------------------------------------------------------------------------
    # These are additive and help us compare live/runtime decisions against research.
    row["src_live_has_bell"] = 1.0 if bell else 0.0
    row["src_live_has_channel_snapshot"] = 1.0 if channel_snapshot else 0.0
    row["src_live_has_pv_tail"] = 1.0 if pv_tail else 0.0
    row["src_live_has_hyst"] = 1.0 if hyst_obj else 0.0
    row["src_live_sigma_count"] = len([s for s in SIGMAS_ALL if per_sigma_hist.get(s, {}).get("values")])

    return row


def build_v21_live_src_dataframe_row(inp: V21LiveSrcInputs):
    """
    Convenience wrapper for feeding the research-style pandas feature builders.
    """
    import pandas as pd

    return pd.DataFrame([build_v21_live_src_row(inp)])


# -------------------------------------------------------------------------------------
# Suggested process integration
# -------------------------------------------------------------------------------------
#
# In DB_process_trend.calculate_trend(...) or the new live model wrapper:
#
#   from main.engine.process.models.v21_live.build_v21_live_src_row import (
#       V21LiveSrcInputs,
#       build_v21_live_src_dataframe_row,
#   )
#
#   src_df = build_v21_live_src_dataframe_row(
#       V21LiveSrcInputs(
#           timing=timing,
#           windows=windows,
#           decision_dt=decision_dt,
#           catalog=catalog,
#           per_sigma_hist=per_sigma_hist,
#           hyst_obj=hyst_obj,
#           config=config,
#       )
#   )
#
#   # then hand src_df to the same v21 feature-builders/scoring logic used in research
#


def build_v21_live_src_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact summary from the canonical src_* live adapter.

    This is intentionally lightweight: it exposes only the linkage / availability
    fields most useful for live-vs-research comparison in diagnostics.
    """
    row = row if isinstance(row, dict) else {}
    keep = [
        "src_meta_epoch",
        "src_meta_next_epoch",
        "src_meta_decision_time",
        "src_meta_close_now",
        "src_meta_hist_points",
        "src_live_has_bell",
        "src_live_has_channel_snapshot",
        "src_live_has_pv_tail",
        "src_live_has_hyst",
        "src_live_sigma_count",
    ]
    return {k: row.get(k) for k in keep if k in row}
