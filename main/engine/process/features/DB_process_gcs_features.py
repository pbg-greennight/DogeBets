from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

try:
    from main.engine.process.features.DB_process_v21_common import SIGMAS_ALL
except Exception:  # pragma: no cover - local fallback for standalone testing
    from process.features.DB_process_v21_common import SIGMAS_ALL


def _safe_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sign3(x: Any, eps: float = 1e-12) -> int:
    v = _safe_float(x, 0.0) or 0.0
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def _mean_clean(values) -> Optional[float]:
    xs = []
    for value in values:
        fv = _safe_float(value)
        if fv is not None:
            xs.append(fv)
    return (sum(xs) / len(xs)) if xs else None


def _series_values(pack: Any) -> list[float]:
    if pack is None:
        return []
    if isinstance(pack, dict):
        vals = pack.get("values") or pack.get("series") or pack.get("mid") or []
    else:
        vals = pack if isinstance(pack, (list, tuple)) else []
    out = []
    for value in vals:
        fv = _safe_float(value)
        if fv is not None:
            out.append(fv)
    return out


def _rolling_width_series(values: list[float], k: float = 2.0, window_n: int = 21) -> list[float]:
    import statistics

    if not values:
        return []
    window_n = max(5, int(window_n))
    out: list[float] = []
    for i in range(len(values)):
        j0 = max(0, i - window_n + 1)
        seg = values[j0 : i + 1]
        if len(seg) < 2:
            out.append(0.0)
            continue
        med = statistics.median(seg)
        mad = statistics.median([abs(v - med) for v in seg]) if seg else 0.0
        robust_std = 1.4826 * mad
        out.append(2.0 * float(k) * robust_std)
    return out


def _tail_slope(values: list[float], frac: float = 0.5) -> Optional[float]:
    if not values or len(values) < 2:
        return None
    half = max(2, int(round(len(values) * frac)))
    tail = values[-half:]
    if len(tail) < 2:
        return None
    return (tail[-1] - tail[0]) / float(len(tail) - 1)


def _normalize_snapshot_container(snapshot: dict | None) -> Tuple[dict, Optional[float], float, int]:
    snapshot = snapshot or {}
    if isinstance(snapshot, dict) and "snapshots" in snapshot and isinstance(snapshot.get("snapshots"), dict):
        snaps = snapshot.get("snapshots") or {}
        price_now = _safe_float(
            snapshot.get("price_now", snapshot.get("close_now", snapshot.get("_price_proxy", snapshot.get("btc_close_now"))))
        )
        k = _safe_float(snapshot.get("k"), 2.0) or 2.0
        window_n = int(_safe_float(snapshot.get("window_n"), 21) or 21)
        return snaps, price_now, k, window_n
    price_now = _safe_float(snapshot.get("price_now", snapshot.get("close_now", snapshot.get("_price_proxy", snapshot.get("btc_close_now"))))) if isinstance(snapshot, dict) else None
    return snapshot if isinstance(snapshot, dict) else {}, price_now, 2.0, 21


def _extract_snapshot_pack(snapshot_pack: Any) -> dict:
    if snapshot_pack is None:
        return {}
    if isinstance(snapshot_pack, dict):
        return snapshot_pack
    out = {}
    for attr in ("mid_last", "upper", "lower", "width", "robust_std", "delta", "slope", "tag"):
        if hasattr(snapshot_pack, attr):
            out[attr] = getattr(snapshot_pack, attr)
    return out


def _extract_sigma_snapshot(
    snapshot: dict,
    sigma: int,
    *,
    price_now: Optional[float] = None,
    series_pack: Any = None,
    k: float = 2.0,
    window_n: int = 21,
) -> dict:
    sigma_pack = snapshot.get(int(sigma)) if isinstance(snapshot, dict) else None
    if sigma_pack is None and isinstance(snapshot, dict):
        sigma_pack = snapshot.get(str(sigma))
    pack = _extract_snapshot_pack(sigma_pack)

    mid_last = _safe_float(pack.get("mid_last", pack.get("mid")))
    lower = _safe_float(pack.get("lower"))
    upper = _safe_float(pack.get("upper"))
    width = _safe_float(pack.get("width"))
    if width is None and lower is not None and upper is not None:
        width = upper - lower

    values = _series_values(series_pack)
    width_series = _rolling_width_series(values, k=k, window_n=window_n) if values else []
    width_change = None
    width_accel = None
    persist = len(values)
    if len(width_series) >= 2:
        width_change = width_series[-1] - width_series[-2]
    if len(width_series) >= 3:
        width_accel = (width_series[-1] - width_series[-2]) - (width_series[-2] - width_series[-3])
    if width is None and width_series:
        width = width_series[-1]

    mid_slope = _safe_float(pack.get("slope"))
    if mid_slope is None:
        mid_slope = _tail_slope(values)

    if width_change is None:
        tag = str(pack.get("tag") or "").lower()
        if "contract" in tag:
            width_change = -1.0
        elif "expand" in tag:
            width_change = 1.0
        elif tag in {"flat", "stable"}:
            width_change = 0.0

    regime = str(pack.get("tag") or "").lower() or None
    if not regime:
        if width_change is not None:
            if width_change < 0:
                regime = "contracting"
            elif width_change > 0:
                regime = "expanding"
            else:
                regime = "flat"
        else:
            regime = None

    px_mid = None
    zpos = None
    if price_now is not None and mid_last is not None:
        px_mid = price_now - mid_last
        denom = max(abs(width or 0.0), 1e-9)
        zpos = px_mid / denom

    return {
        "regime": regime,
        "px_mid": px_mid,
        "zpos": zpos,
        "width": width,
        "mid_slope": mid_slope,
        "width_change": width_change,
        "width_accel": width_accel,
        "persist": persist if persist > 0 else None,
    }


def _compute_regime_counts(per_sigma: dict) -> dict:
    regimes = [str(per_sigma.get(s, {}).get("regime") or "").lower() for s in SIGMAS_ALL]
    contract_count = sum(1 for r in regimes if "contract" in r)
    expand_count = sum(1 for r in regimes if "expand" in r)
    flat_count = sum(1 for r in regimes if r in {"flat", "stable", "compressed"})
    return {
        "contracting_count": contract_count,
        "expanding_count": expand_count,
        "flat_count": flat_count,
        "contracting_all": int(contract_count == len(SIGMAS_ALL) and len(SIGMAS_ALL) > 0),
    }


def _compute_spacing_summary(per_sigma: dict) -> dict:
    mids = {s: _safe_float(per_sigma.get(s, {}).get("px_mid")) for s in SIGMAS_ALL}
    ordered = [s for s in SIGMAS_ALL if mids.get(s) is not None]
    if len(ordered) < 2:
        return {"spacing_state": None, "fan_state": None}

    diffs = []
    for idx in range(len(ordered) - 1):
        a = ordered[idx]
        b = ordered[idx + 1]
        diffs.append((mids[b] or 0.0) - (mids[a] or 0.0))

    fan_state = "mixed"
    if diffs and all(g > 0 for g in diffs):
        fan_state = "fanning_out"
    elif diffs and all(g < 0 for g in diffs):
        fan_state = "inverted"
    elif diffs and max(abs(g) for g in diffs) < 0.25:
        fan_state = "flat_cluster"
    elif len(diffs) > 1 and all(abs(diffs[i] - diffs[i - 1]) < 1.0 for i in range(1, len(diffs))):
        fan_state = "compressing"

    spacing_state = "mixed"
    if diffs and all(abs(g) < 0.25 for g in diffs):
        spacing_state = "tight"
    elif diffs and all(abs(g) < 1.0 for g in diffs):
        spacing_state = "compressed"
    elif diffs and max(abs(g) for g in diffs) > 3.0:
        spacing_state = "wide"

    return {
        "spacing_state": spacing_state,
        "fan_state": fan_state,
    }


def _compute_transfer_summary(per_sigma: dict, snapshot: dict) -> dict:
    slopes = {s: _safe_float(per_sigma.get(s, {}).get("mid_slope"), 0.0) or 0.0 for s in SIGMAS_ALL}
    dir_s = _sign3(slopes.get(8, 0.0))
    depth = 0.0
    checks = [(8, 23), (23, 38), (38, 53)]
    for _, target in checks:
        if dir_s != 0 and _sign3(slopes.get(target, 0.0)) == dir_s:
            depth += 1.0
    if depth <= 0.0:
        state = "none"
    elif depth < float(len(checks)):
        state = "partial"
    elif depth < float(len(checks)) + 0.5:
        state = "deep"
    else:
        state = "full"
    direction = "up" if dir_s > 0 else ("down" if dir_s < 0 else "none")
    return {
        "transfer_dir": direction,
        "transfer_depth": depth,
        "transfer_state": state,
    }


def _compute_position_summary(per_sigma: dict) -> dict:
    fast = [per_sigma.get(s, {}).get("px_mid") for s in [8, 23]]
    slow = [per_sigma.get(s, {}).get("px_mid") for s in [68, 83]]
    fast_mean = _mean_clean(fast)
    slow_mean = _mean_clean(slow)
    gap = None
    if fast_mean is not None and slow_mean is not None:
        gap = fast_mean - slow_mean

    front_back_disagreement = 0.0
    if fast_mean is not None and slow_mean is not None:
        front_back_disagreement = 1.0 if _sign3(fast_mean) != 0 and _sign3(fast_mean) != _sign3(slow_mean) else 0.0

    reclaim_front = 0.0
    if per_sigma.get(8, {}).get("px_mid") is not None and per_sigma.get(23, {}).get("px_mid") is not None:
        reclaim_front = 1.0 if _sign3(per_sigma[8]["px_mid"]) == _sign3(per_sigma[23]["px_mid"]) != 0 else 0.0

    reclaim_slow = 0.0
    if per_sigma.get(53, {}).get("px_mid") is not None and per_sigma.get(83, {}).get("px_mid") is not None:
        reclaim_slow = 1.0 if _sign3(per_sigma[53]["px_mid"]) == _sign3(per_sigma[83]["px_mid"]) != 0 else 0.0

    return {
        "fast_pos_mean": fast_mean,
        "slow_pos_mean": slow_mean,
        "fast_slow_pos_gap": gap,
        "front_back_disagreement": front_back_disagreement,
        "reclaim_front": reclaim_front,
        "reclaim_slow": reclaim_slow,
    }


def build_gcs_feature_payload(
    channel_snapshot: dict,
    per_sigma_full: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    config = config or {}
    snaps, price_now, k, window_n = _normalize_snapshot_container(channel_snapshot)
    per_sigma: Dict[int, Dict[str, Any]] = {}
    for sigma in SIGMAS_ALL:
        series_pack = None
        if isinstance(per_sigma_full, dict):
            series_pack = per_sigma_full.get(int(sigma))
            if series_pack is None:
                series_pack = per_sigma_full.get(str(sigma))
        per_sigma[sigma] = _extract_sigma_snapshot(
            snaps,
            sigma,
            price_now=price_now,
            series_pack=series_pack,
            k=_safe_float(config.get("GAUSS_CHANNEL_K"), k) or k,
            window_n=int(_safe_float(config.get("PV_TAIL_CHANNEL_WINDOW_N"), window_n) or window_n),
        )

    regime_counts = _compute_regime_counts(per_sigma)
    spacing = _compute_spacing_summary(per_sigma)
    transfer = _compute_transfer_summary(per_sigma, channel_snapshot)
    position = _compute_position_summary(per_sigma)

    return {
        "per_sigma": per_sigma,
        "regime_counts": regime_counts,
        "spacing": spacing,
        "transfer": transfer,
        "position": position,
        "meta": {
            "price_now": price_now,
        },
    }


def flatten_gcs_to_src(
    gcs_obj: dict,
    config: Optional[dict] = None,
) -> dict:
    out: Dict[str, Any] = {}

    per_sigma = gcs_obj.get("per_sigma", {})
    for sigma, fields in per_sigma.items():
        out[f"src_gcs_regime_s{sigma}"] = fields.get("regime")
        out[f"src_gcs_px_mid_{sigma}"] = fields.get("px_mid")
        out[f"src_gcs_zpos_{sigma}"] = fields.get("zpos")
        out[f"src_gcs_width_{sigma}"] = fields.get("width")
        out[f"src_gcs_mid_slope_{sigma}"] = fields.get("mid_slope")
        out[f"src_gcs_width_change_{sigma}"] = fields.get("width_change")
        out[f"src_gcs_width_accel_{sigma}"] = fields.get("width_accel")
        out[f"src_gcs_persist_{sigma}"] = fields.get("persist")

        # Back-compat / alias-friendly fields used by some older readers.
        out[f"src_gcs_pos_s{sigma}"] = fields.get("px_mid")
        out[f"src_gcs_width_s{sigma}"] = fields.get("width")
        out[f"src_gcs_mid_slope_s{sigma}"] = fields.get("mid_slope")
        out[f"src_gcs_width_change_s{sigma}"] = fields.get("width_change")
        out[f"src_gcs_pricepos_s{sigma}_px_mid"] = fields.get("px_mid")
        out[f"src_gcs_pricepos_s{sigma}_zpos"] = fields.get("zpos")
        out[f"src_gcs_s{sigma}_px_mid"] = fields.get("px_mid")
        out[f"src_gcs_s{sigma}_zpos"] = fields.get("zpos")

    rc = gcs_obj.get("regime_counts", {})
    out["src_gcs_contracting_count"] = rc.get("contracting_count")
    out["src_gcs_expanding_count"] = rc.get("expanding_count")
    out["src_gcs_flat_count"] = rc.get("flat_count")
    out["src_gcs_contracting_all"] = rc.get("contracting_all")
    # Historical names preserved.
    out["src_gcs_regime_contract_count"] = rc.get("contracting_count")
    out["src_gcs_regime_expand_count"] = rc.get("expanding_count")
    out["src_gcs_regime_flat_count"] = rc.get("flat_count")
    out["src_gcs_regime_contract_all"] = rc.get("contracting_all")

    spacing = gcs_obj.get("spacing", {})
    out["src_gcs_spacing_state"] = spacing.get("spacing_state")
    out["src_gcs_fan_state"] = spacing.get("fan_state")

    transfer = gcs_obj.get("transfer", {})
    out["src_gcs_transfer_dir"] = transfer.get("transfer_dir")
    out["src_gcs_transfer_depth"] = transfer.get("transfer_depth")
    out["src_gcs_transfer_state"] = transfer.get("transfer_state")

    position = gcs_obj.get("position", {})
    out["src_gcs_fast_pos_mean"] = position.get("fast_pos_mean")
    out["src_gcs_slow_pos_mean"] = position.get("slow_pos_mean")
    out["src_gcs_fast_slow_pos_gap"] = position.get("fast_slow_pos_gap")
    out["src_gcs_front_back_disagreement"] = position.get("front_back_disagreement")
    out["src_gcs_reclaim_front"] = position.get("reclaim_front")
    out["src_gcs_reclaim_slow"] = position.get("reclaim_slow")

    return out
