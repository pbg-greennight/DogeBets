from __future__ import annotations

from typing import Any, Dict, Iterable

SIGMAS_ALL = [8, 23, 38, 53, 68, 83]
SIGMAS_FAST = [8, 23, 38]
SIGMAS_SLOW = [53, 68, 83]


def _safe_get(d: Any, *path: Any, default=None):
    cur = d
    for key in path:
        try:
            if isinstance(cur, dict):
                cur = cur.get(key, default)
            else:
                cur = cur[key]
        except Exception:
            return default
        if cur is None:
            return default
    return cur


def _to_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _sign3(x: Any, eps: float = 1e-12) -> int:
    v = _to_float(x, default=0.0)
    if v > eps:
        return 1
    if v < -eps:
        return -1
    return 0


def _mean(vals: Iterable[Any], default: float = float("nan")) -> float:
    xs = []
    for v in vals:
        fv = _to_float(v, default=float("nan"))
        if fv == fv:
            xs.append(fv)
    if not xs:
        return default
    return sum(xs) / len(xs)


def _count_true(vals: Iterable[Any]) -> int:
    return sum(1 for v in vals if bool(v))


def _norm_count(n: Any, denom: float) -> float:
    v = _to_float(n, 0.0)
    if denom <= 0:
        return 0.0
    out = v / denom
    if out < 0:
        return 0.0
    if out > 1:
        return 1.0
    return out


def _flatten_sigma_dict(prefix: str, sigma_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sigma, fields in sigma_map.items():
        for k, v in fields.items():
            out[f"{prefix}_{k}_s{sigma}"] = v
    return out
