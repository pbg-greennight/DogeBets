# main/engine/process/DB_process_printing_utils.py

def _arrow() -> str:
    """Human-friendly arrow used in log lines (kept in utils so printing modules stay lean)."""
    return "→"

import logging


from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

def _safe_get(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for p in path:
        if cur is None:
            return default

        if isinstance(cur, dict):
            if p not in cur:
                return default
            cur = cur[p]
            continue

        get_fn = getattr(cur, "get", None)
        if callable(get_fn):
            try:
                nxt = get_fn(p, None)
            except TypeError:
                try:
                    nxt = get_fn(p)
                except Exception:
                    nxt = None
            except Exception:
                nxt = None
            if nxt is not None:
                cur = nxt
                continue

        if hasattr(cur, p):
            try:
                cur = getattr(cur, p)
            except Exception:
                return default
            continue

        return default

    return cur


def _as_mapping(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return asdict(obj)
    except Exception:
        pass

    out: Dict[str, Any] = {}
    for k in ("trend", "confidence", "model_id", "scores", "features", "reason", "notes", "raw"):
        if hasattr(obj, k):
            try:
                out[k] = getattr(obj, k)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            out = {**obj.__dict__, **out}
        except Exception:
            pass
    return out


def _fmt_time(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return "?"
    return dt.strftime("%I:%M:%S %p")


def _fmt_iso(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return "?"
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _fmt_float(x: Any, nd: int = 4, none: str = "?") -> str:
    if x is None:
        return none
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)



def _fmt_int(x: Any, none: str = "?") -> str:
    """Format an int-ish value for logs."""
    if x is None:
        return none
    try:
        return str(int(x))
    except Exception:
        return str(x)


def _is_enabled(config: dict, key: str, default: bool = False) -> bool:
    """Return True if a logging section/flag is enabled in config.

    Supports both flat config (config[key]) and nested config (config['LOG'][key]).
    """
    try:
        if isinstance(config, dict):
            if key in config:
                return bool(config.get(key, default))
            for parent in ("LOG", "PRINT", "FLAGS", "SECTIONS"):
                sub = config.get(parent)
                if isinstance(sub, dict) and key in sub:
                    return bool(sub.get(key, default))
    except Exception:
        pass
    return bool(default)

def _fmt_price(x: Any, none: str = "?") -> str:
    if x is None:
        return none
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


# -------------------------------------------------------------------------------------------------
# Back-compat aliases
#
# Some printing modules (e.g., older MSBC printers) expect `_fmt_dt` and `_fmt_val`.
# Keep these as thin wrappers so mixed module versions don't break imports.
# -------------------------------------------------------------------------------------------------

def _fmt_dt(dt: Any, *, short: bool = False) -> str:
    """Format a datetime-like value.

    - short=True  -> human time (HH:MM:SS AM/PM)
    - short=False -> ISO-like timestamp
    """
    return _fmt_time(dt) if short else _fmt_iso(dt)


def _fmt_val(x: Any, *, nd: int = 4, price: bool = False) -> str:
    """Generic numeric formatter used by some legacy printers."""
    return _fmt_price(x) if price else _fmt_float(x, nd=nd)


def _line(ch: str = "-", n: int = 155) -> str:
    return ch * n


def _series_preview(values: Iterable[Any], max_items: int = 60, nd: int = 4) -> str:
    vals = list(values)
    if not vals:
        return "(empty)"
    show = vals[:max_items]
    s = ", ".join(_fmt_float(v, nd=nd) for v in show)
    if len(vals) > max_items:
        s += f", ... (+{len(vals) - max_items})"
    return s


def _print_cfg(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        p = config.get("PRINT", {})
        return p if isinstance(p, dict) else {}
    return {}


def _section_on(config: Any, key: str, default: bool = True) -> bool:
    p = _print_cfg(config)
    sec = p.get("SECTIONS", {})
    if isinstance(sec, dict) and key in sec:
        return bool(sec.get(key))
    return default

# ----------------------------------------------------------------------------------------------------------------------
# Extra helpers used by DB_process_printing.py (kept here to avoid circular imports)

_DEFAULT_LOG = logging.getLogger("process")


def _log_line(msg: str, logger: Optional[logging.Logger] = None) -> None:
    """Small wrapper so printing code can call _log_line(...) consistently."""
    (logger or _DEFAULT_LOG).info(msg)


def _fmt_hms(dt: Any) -> str:
    """Return HH:MM:SS from datetime-like objects; fallback to str."""
    if isinstance(dt, datetime):
        return dt.strftime("%H:%M:%S")
    sf = getattr(dt, "strftime", None)
    if callable(sf):
        try:
            return sf("%H:%M:%S")
        except Exception:
            pass
    return str(dt) if dt is not None else "?"


def first_present(container: Any, keys: Iterable[str], default: Any = None) -> Any:
    """Return the first non-empty value for any key in `keys` from a mapping-like container."""
    m = _as_mapping(container)
    for k in keys:
        try:
            v = m.get(k)
        except Exception:
            v = None
        if v is None:
            continue
        if isinstance(v, (list, dict, str)) and len(v) == 0:
            continue
        return v
    return default

def _safe_get(d: Any, *path: str, default: Any = None) -> Any:
    """Safely traverse nested structures.

    Supports:
      - dict (key lookup)
      - objects (attribute lookup)
      - objects implementing .get(key[, default]) (FeatureCatalog-like)
    """
    cur = d
    for p in path:
        if cur is None:
            return default

        if isinstance(cur, dict):
            if p not in cur:
                return default
            cur = cur[p]
            continue

        get_fn = getattr(cur, "get", None)
        if callable(get_fn):
            try:
                nxt = get_fn(p, None)  # dict.get / FeatureCatalog.get
            except TypeError:
                try:
                    nxt = get_fn(p)
                except Exception:
                    nxt = None
            except Exception:
                nxt = None
            if nxt is not None:
                cur = nxt
                continue

        if hasattr(cur, p):
            try:
                cur = getattr(cur, p)
            except Exception:
                return default
            continue

        return default

    return cur


def _as_mapping(obj: Any) -> Dict[str, Any]:
    """Normalize dict-like / dataclass-like / simple objects to a dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    try:
        return asdict(obj)
    except Exception:
        pass
    out: Dict[str, Any] = {}
    for k in ("trend", "confidence", "model_id", "scores", "features", "reason", "notes"):
        if hasattr(obj, k):
            try:
                out[k] = getattr(obj, k)
            except Exception:
                pass
    if hasattr(obj, "__dict__"):
        try:
            out = {**obj.__dict__, **out}
        except Exception:
            pass
    return out


def _fmt_time(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return "?"
    return dt.strftime("%I:%M:%S %p")


def _fmt_iso(dt: Optional[datetime]) -> str:
    if not isinstance(dt, datetime):
        return "?"
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def _fmt_float(x: Any, nd: int = 4, none: str = "?") -> str:
    if x is None:
        return none
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _fmt_price(x: Any, none: str = "?") -> str:
    if x is None:
        return none
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def _arrow() -> str:
    return "→"


def _line(ch: str = "-", n: int = 155) -> str:
    return ch * n


def _series_preview(values: Iterable[Any], max_items: int = 60, nd: int = 4) -> str:
    vals = list(values)
    if not vals:
        return "(empty)"
    show = vals[:max_items]
    s = ", ".join(_fmt_float(v, nd=nd) for v in show)
    if len(vals) > max_items:
        s += f", ... (+{len(vals) - max_items})"
    return s


def _kind_to_extrema_pair(kind: Any) -> str:
    k = str(kind or "").strip().lower()
    if "peak" in k:
        return "PEAK"
    if "val" in k:
        return "VALLEY"
    return str(kind or "?").upper() if kind is not None else "?"


def _print_cfg(config: Any) -> Dict[str, Any]:
    if isinstance(config, dict):
        p = config.get("PRINT", {})
        return p if isinstance(p, dict) else {}
    return {}


def _section_on(config: Any, key: str, default: bool = True) -> bool:
    """Fine-grained bracket-section toggle.

    Controlled via PROCESS_CONFIG["PRINT"]["SECTIONS"][<key>].
    """
    p = _print_cfg(config)
    sec = p.get("SECTIONS", {})
    if isinstance(sec, dict) and key in sec:
        return bool(sec.get(key))
    return default