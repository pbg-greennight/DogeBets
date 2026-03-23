from __future__ import annotations

"""
build_v21_live_src_row.py

Adapter for porting the best v21.x research decision layer into the live
process engine. This module does NOT fetch archive/parquet data. It uses the
existing process snapshot objects:

- FeatureCatalog (`catalog`) or calc-out like nested dicts
- live per-sigma history (`per_sigma_hist`)
- live hysteresis payload (`hyst_obj`)
- timing/windows/config

Goal:
    build one canonical `src_*` row dict whose field names match the research
    model's expected source schema, so the same v21 feature/scoring code can be
    reused live.

This first pass focuses on fixing the live source bridge for:
- Gaussian Channel Snapshot (GCS)
- PV-tail compression / decompression (CSD/DCSD)
- schema normalization / validation metadata
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

try:
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
    from main.engine.process.models.v21_live.v21_live_schema import (
        SIGMAS_ALL,
        SRC_CONTRACT_VERSION,
    )
    from main.engine.process.models.v21_live.v21_live_validator import (
        normalize_src_row,
        summarize_src_family_coverage,
        validate_src_row,
    )
except Exception:  # pragma: no cover - local fallback for standalone testing
    from process.features.DB_process_msbc_features import (
        build_msbc_feature_payload,
        flatten_msbc_to_src,
    )
    from process.features.DB_process_gcs_features import (
        build_gcs_feature_payload,
        flatten_gcs_to_src,
    )
    from process.features.DB_process_csd_dcsd_features import (
        build_csd_dcsd_feature_payload,
        flatten_csd_dcsd_to_src,
    )
    from process.features.DB_process_gbc_features import (
        build_gbc_feature_payload,
        flatten_gbc_to_src,
    )
    from process.features.DB_process_hysteresis import (
        flatten_hysteresis_to_src,
    )
    from process.models.v21_live.v21_live_schema import (
        SIGMAS_ALL,
        SRC_CONTRACT_VERSION,
    )
    from process.models.v21_live.v21_live_validator import (
        normalize_src_row,
        summarize_src_family_coverage,
        validate_src_row,
    )


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
    FeatureCatalog / nested-dict path helper.

    Supports:
    - FeatureCatalog-like objects that cache dotted keys directly
    - plain nested dicts that must be traversed segment by segment
    - generic objects with attributes
    """
    if catalog is None:
        return default

    getter = getattr(catalog, "get", None)
    if callable(getter):
        try:
            direct = getter(path, None)
        except TypeError:
            try:
                direct = getter(path)
            except Exception:
                direct = None
        except Exception:
            direct = None
        if direct is not None:
            return direct

    cur = catalog
    try:
        for part in path.split("."):
            if isinstance(cur, dict):
                if part not in cur:
                    return default
                cur = cur[part]
            else:
                getter = getattr(cur, "get", None)
                nxt = None
                if callable(getter):
                    try:
                        nxt = getter(part, None)
                    except TypeError:
                        try:
                            nxt = getter(part)
                        except Exception:
                            nxt = None
                    except Exception:
                        nxt = None
                if nxt is None:
                    nxt = getattr(cur, part, None)
                if nxt is None:
                    return default
                cur = nxt
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
    out: Dict[str, Any] = {}

    out["src_meta_epoch"] = _safe_int(_safe_getattr(timing, "curr_epoch"))
    out["src_meta_next_epoch"] = _safe_int(_safe_getattr(timing, "next_epoch"))
    out["src_meta_prev_epoch"] = _safe_int(_safe_getattr(timing, "prev_epoch"))
    out["src_meta_decision_time"] = decision_dt.isoformat() if decision_dt else None

    full_start = _safe_getattr(windows, "full_start")
    full_end = _safe_getattr(windows, "full_end")
    out["src_meta_window_start"] = full_start.isoformat() if hasattr(full_start, "isoformat") else None
    out["src_meta_window_end"] = full_end.isoformat() if hasattr(full_end, "isoformat") else None

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


def _build_channel_snapshot_payload(
    *,
    channel_snapshot: Any,
    price_now: Optional[float],
    config: Dict[str, Any],
) -> dict:
    if isinstance(channel_snapshot, dict) and "snapshots" in channel_snapshot:
        out = dict(channel_snapshot)
    else:
        out = {"snapshots": channel_snapshot or {}}
    if price_now is not None and out.get("price_now") is None:
        out["price_now"] = price_now
    if out.get("close_now") is None and price_now is not None:
        out["close_now"] = price_now
    if out.get("_price_proxy") is None and price_now is not None:
        out["_price_proxy"] = price_now
    out.setdefault("k", _safe_float(config.get("GAUSS_CHANNEL_K"), 2.0) or 2.0)
    out.setdefault("window_n", int(_safe_float(config.get("PV_TAIL_CHANNEL_WINDOW_N"), 21) or 21))
    return out


def build_v21_live_src_row(inp: V21LiveSrcInputs) -> Dict[str, Any]:
    timing = inp.timing
    windows = inp.windows
    decision_dt = inp.decision_dt
    catalog = inp.catalog
    per_sigma_hist = inp.per_sigma_hist or {}
    hyst_obj = inp.hyst_obj or {}
    config = inp.config or {}

    row: Dict[str, Any] = {}

    row.update(_build_meta_src(timing, windows, decision_dt, per_sigma_hist))

    bell = _catalog_get(catalog, "bell", {}) or _catalog_get(catalog, "bell.current", {}) or {}
    bell_curve_series = _catalog_get(catalog, "bell.series", None) or _catalog_get(catalog, "bell_curve_series", None)

    channel_snapshot_raw = (
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

    price_now = (
        _safe_float(_catalog_get(catalog, "close.last", None))
        or _safe_float(_catalog_get(catalog, "price.last", None))
        or _safe_float(_catalog_get(catalog, "bell.btc_close.current", None))
        or _safe_float(((bell or {}).get("btc_close") or {}).get("current"))
        or _safe_float(row.get("src_meta_close_now"))
    )

    channel_snapshot = _build_channel_snapshot_payload(
        channel_snapshot=channel_snapshot_raw,
        price_now=price_now,
        config=config,
    )

    msbc_obj = build_msbc_feature_payload(
        bell=bell,
        bell_curve_series=bell_curve_series,
        config=config,
    )
    row.update(flatten_msbc_to_src(msbc_obj, config))

    gcs_obj = build_gcs_feature_payload(
        channel_snapshot=channel_snapshot,
        per_sigma_full=per_sigma_hist,
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

    row["src_contract_version"] = SRC_CONTRACT_VERSION
    row["src_live_has_bell"] = 1.0 if bell else 0.0
    row["src_live_has_channel_snapshot"] = 1.0 if channel_snapshot_raw else 0.0
    row["src_live_has_pv_tail"] = 1.0 if pv_tail else 0.0
    row["src_live_has_hyst"] = 1.0 if hyst_obj else 0.0
    row["src_live_sigma_count"] = len([s for s in SIGMAS_ALL if per_sigma_hist.get(s, {}).get("values")])

    row = normalize_src_row(row)
    audit = validate_src_row(row)
    row["src_debug_missing_required_count"] = len(audit.get("missing_required") or [])
    row["src_debug_missing_optional_count"] = len(audit.get("missing_optional") or [])

    coverage = summarize_src_family_coverage(row)
    for family, value in coverage.items():
        row[f"src_debug_cov_{family}"] = value

    return row


def build_v21_live_src_bundle(inp: V21LiveSrcInputs) -> Dict[str, Any]:
    row = build_v21_live_src_row(inp)
    summary = build_v21_live_src_summary(row)
    audit = validate_src_row(row)
    return {
        "src_row": row,
        "src_summary": summary,
        "src_audit": audit,
    }


def build_v21_live_src_dataframe_row(inp: V21LiveSrcInputs):
    import pandas as pd

    return pd.DataFrame([build_v21_live_src_row(inp)])


def build_v21_live_src_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    row = row if isinstance(row, dict) else {}
    keep = [
        "src_contract_version",
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
        "src_debug_missing_required_count",
        "src_debug_missing_optional_count",
        "src_debug_cov_meta",
        "src_debug_cov_msbc",
        "src_debug_cov_gcs",
        "src_debug_cov_csd",
        "src_debug_cov_dcsd",
        "src_debug_cov_gbc",
        "src_debug_cov_hyst",
        "src_debug_cov_overall",
    ]
    return {k: row.get(k) for k in keep if k in row}
