# main/engine/process/DB_process_catalog.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import math
import statistics

# ----------------------------
# Minimal helpers (kept local to avoid tight coupling)
# ----------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _median_abs_dev(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    med = statistics.median(xs)
    dev = [abs(v - med) for v in xs]
    return statistics.median(dev) if dev else 0.0

def _safe_get_values(blob: Any) -> List[float]:
    if blob is None:
        return []
    if isinstance(blob, dict):
        vals = blob.get("values") or blob.get("vals") or blob.get("series") or blob.get("mid") or []
        return list(vals) if isinstance(vals, (list, tuple)) else []
    if isinstance(blob, (list, tuple)):
        return list(blob)
    return []

def _tail_slope(values: Sequence[float], k: int = 21) -> float:
    if len(values) < 2:
        return 0.0
    kk = max(1, min(int(k), len(values) - 1))
    return (_safe_float(values[-1]) - _safe_float(values[-1 - kk])) / float(kk)

def _compute_sign_to(values: Sequence[float], k: int = 21) -> int:
    d = _tail_slope(values, k=k)
    if d > 0:
        return 1
    if d < 0:
        return -1
    return 0

def _compute_hook(values: Sequence[float], short_k: int = 10, long_k: int = 50) -> int:
    n = len(values)
    if n < 12:
        return 0

    sk = max(1, min(int(short_k), n - 1))
    lk = max(1, min(int(long_k), n - 1))

    short_slope = (_safe_float(values[-1]) - _safe_float(values[-1 - sk])) / float(sk)
    long_slope = (_safe_float(values[-1]) - _safe_float(values[-1 - lk])) / float(lk)

    s_sign = 1 if short_slope > 0 else (-1 if short_slope < 0 else 0)
    l_sign = 1 if long_slope > 0 else (-1 if long_slope < 0 else 0)

    if s_sign == 0 or l_sign == 0:
        return 0
    if s_sign == l_sign:
        return 0

    tail = list(values[-(sk + 1) :])
    diffs = [(_safe_float(tail[i + 1]) - _safe_float(tail[i])) for i in range(len(tail) - 1)]
    mad = _median_abs_dev(diffs) + 1e-9

    strength = abs(short_slope) / mad
    return 1 if strength >= 2.0 else 0

def _compute_flat(values: Sequence[float], k: int = 21) -> float:
    if len(values) < 3:
        return 1.0

    kk = max(1, min(int(k), len(values) - 1))
    tail = list(values[-(kk + 1) :])
    diffs = [(_safe_float(tail[i + 1]) - _safe_float(tail[i])) for i in range(len(tail) - 1)]
    mad = _median_abs_dev(diffs) + 1e-9

    slope = (_safe_float(tail[-1]) - _safe_float(tail[0])) / float(kk)

    ratio = abs(slope) / (5.0 * mad)
    return _clamp01(1.0 - ratio)


# ----------------------------
# Feature specs + catalogue
# ----------------------------

ComputeFn = Callable[["FeatureCatalog"], Any]

@dataclass(frozen=True)
class FeatureSpec:
    key: str
    compute: ComputeFn
    deps: Tuple[str, ...] = ()
    doc: str = ""


class FeatureCatalog:
    """Unified read-only feature store for one decision cycle."""

    def __init__(
        self,
        *,
        timing: Any = None,
        windows: Any = None,
        per_sigma_full: Optional[Dict[int, Any]] = None,
        per_sigma_hist: Optional[Dict[int, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.timing = timing
        self.windows = windows
        self.per_sigma_full = per_sigma_full or {}
        self.per_sigma_hist = per_sigma_hist or {}
        self.config = config or {}

        self._specs: Dict[str, FeatureSpec] = {}
        self._cache: Dict[str, Any] = {}

        # Stage 1: calc truth cache
        self._calc_ready: bool = False
        self._register_builtins()

    # ----------------------------
    # Stage 1: calc truth (single compute, cached)
    # ----------------------------

    def ensure_calc(self, *, decision_dt, close_series=None) -> Dict[str, Any]:
        """Compute & cache calc truth output once per catalog instance."""
        if self._calc_ready and "calc.out" in self._cache:
            return self._cache["calc.out"]

        try:
            from DB_process_calc import build_calc_out
        except Exception:
            from main.engine.process.core.DB_process_calc import build_calc_out

        calc_out = build_calc_out(
            timing=self.timing,
            windows=self.windows,
            decision_dt=decision_dt,
            per_sigma_full=self.per_sigma_full,
            per_sigma_hist=self.per_sigma_hist,
            config=self.config,
            close_series=close_series,
        )
        self._cache["calc.out"] = calc_out
        self._cache["bell"] = calc_out.get("bell", {}) or {}
        self._cache["bell_curve_series"] = calc_out.get("bell_curve_series", {}) or {}
        self._cache["channels.snapshot"] = (calc_out.get("channels", {}) or {}).get("snapshot", {}) or {}
        self._cache["channels.pv_tail"] = (calc_out.get("channels", {}) or {}).get("pv_tail", {}) or {}

        self._calc_ready = True
        return calc_out

    # ----------------------------
    # Public API
    # ----------------------------

    def has(self, key: str) -> bool:
        return key in self._specs or key in self._cache

    def keys(self) -> List[str]:
        return sorted(set(list(self._specs.keys()) + list(self._cache.keys())))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a feature value (lazy + memoized)."""

        # Stage 1: cached calc truth keys
        if key in self._cache:
            return self._cache[key]

        # Allow direct access to calc namespaces even before computed
        if key in ("calc.out", "bell", "bell_curve_series", "channels.snapshot", "channels.pv_tail"):
            return default

        spec = self._specs.get(key)
        if spec is None:
            if default is not None:
                return default
            raise KeyError(f"[FeatureCatalog] unknown key: {key}")

        for dep in spec.deps:
            self.get(dep)

        try:
            val = spec.compute(self)
        except Exception as e:
            logging.warning(f"[FeatureCatalog] compute failed key={key}: {e}")
            val = default

        self._cache[key] = val
        return val

    def build(self, keys: Iterable[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in keys:
            out[k] = self.get(k)
        return out

    def build_per_sigma_inputs(
        self,
        *,
        wanted_sigmas: Sequence[int],
        diag_chunk: int = 12,
        tail_n: int = 21,
        use_hist: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        src = self.per_sigma_hist if use_hist else self.per_sigma_full
        out: Dict[int, Dict[str, Any]] = {}

        for s in wanted_sigmas:
            values = _safe_get_values(src.get(int(s), {}))
            if not values:
                out[int(s)] = {"values": [], "sign_to": 0, "hook": 0, "flat": 1.0, "diag": []}
                continue

            sign_to = _compute_sign_to(values, k=tail_n)
            hook = _compute_hook(values, short_k=max(8, tail_n // 2), long_k=max(30, tail_n * 2))
            flat = _compute_flat(values, k=tail_n)
            diag = list(values[-diag_chunk:]) if diag_chunk else []

            out[int(s)] = {
                "values": list(values),
                "sign_to": int(sign_to),
                "hook": int(hook),
                "flat": float(flat),
                "diag": diag,
            }

        return out

    def build_for_model(
        self,
        *,
        model_path: Optional[str] = None,
        wanted_sigmas: Optional[Sequence[int]] = None,
        diag_chunk: Optional[int] = None,
        tail_n: Optional[int] = None,
        use_hist: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        ws = wanted_sigmas
        if ws is None and model_path:
            try:
                from DB_process_calc import load_model_config
            except Exception:
                from main.engine.process.core.DB_process_calc import load_model_config

            try:
                cfg = load_model_config(str(model_path))
                ws = cfg.get("sigmas", {}).get("wanted")
            except Exception:
                ws = None

        if ws is None:
            ws = self.config.get("GAUSS_SIGMAS") or [8, 23, 38, 53, 68, 83]

        dc = int(diag_chunk if diag_chunk is not None else self.config.get("DUMP_DIAG_CHUNK", 12))
        tn = int(tail_n if tail_n is not None else self.config.get("TAIL_FEATURE_POINTS", 21))

        return self.build_per_sigma_inputs(wanted_sigmas=list(ws), diag_chunk=dc, tail_n=tn, use_hist=use_hist)

    # ----------------------------
    # Registration
    # ----------------------------

    def register(self, spec: FeatureSpec) -> None:
        if spec.key in self._specs:
            logging.debug(f"[FeatureCatalog] overriding key: {spec.key}")
        self._specs[spec.key] = spec

    def _register_builtins(self) -> None:
        self.register(
            FeatureSpec(
                key="epoch.prev",
                compute=lambda c: int(getattr(c.timing, "prev_epoch", 0) or 0),
                doc="Previous epoch id",
            )
        )
        self.register(
            FeatureSpec(
                key="epoch.curr",
                compute=lambda c: int(getattr(c.timing, "curr_epoch", 0) or 0),
                doc="Current epoch id",
            )
        )
        self.register(
            FeatureSpec(
                key="epoch.next",
                compute=lambda c: int(getattr(c.timing, "next_epoch", 0) or 0),
                doc="Next epoch id",
            )
        )

        self.register(
            FeatureSpec(
                key="window.full.start",
                compute=lambda c: getattr(getattr(c.windows, "full_start", None), "isoformat", lambda: None)(),
                doc="Full window start timestamp (isoformat)",
            )
        )
        self.register(
            FeatureSpec(
                key="window.full.end",
                compute=lambda c: getattr(getattr(c.windows, "full_end", None), "isoformat", lambda: None)(),
                doc="Full window end timestamp (isoformat)",
            )
        )

        sigmas = self.config.get("GAUSS_SIGMAS") or [8, 23, 38, 53, 68, 83]
        for s in sigmas:
            s = int(s)
            self._register_gauss_sigma(s)

        self._register_fan_features(sigmas=[int(x) for x in sigmas])
        self._register_channel_features(sigmas=[int(x) for x in sigmas])

    def _register_gauss_sigma(self, sigma: int) -> None:
        s = int(sigma)
        base = f"gauss.s{s}"

        self.register(
            FeatureSpec(
                key=f"{base}.values",
                compute=lambda c, ss=s: _safe_get_values(c.per_sigma_full.get(ss, {})),
                doc=f"Gaussian midline series values (epoch slice) for sigma={s}",
            )
        )
        self.register(
            FeatureSpec(
                key=f"{base}.values_hist",
                compute=lambda c, ss=s: _safe_get_values(c.per_sigma_hist.get(ss, {})),
                doc=f"Gaussian midline series values (history slice) for sigma={s}",
            )
        )
        self.register(
            FeatureSpec(
                key=f"{base}.last",
                deps=(f"{base}.values",),
                compute=lambda c, k=f"{base}.values": _safe_float((c.get(k) or [0.0])[-1], 0.0) if c.get(k) else 0.0,
                doc=f"Last gaussian value (epoch slice) for sigma={s}",
            )
        )
        self.register(
            FeatureSpec(
                key=f"{base}.tail.slope",
                deps=(f"{base}.values",),
                compute=lambda c, ss=s: _tail_slope(c.get(f"gauss.s{ss}.values") or [], k=int(c.config.get("TAIL_FEATURE_POINTS", 21))),
                doc=f"Tail slope (epoch slice) for sigma={s}",
            )
        )
        self.register(
            FeatureSpec(
                key=f"{base}.tail.sign_to",
                deps=(f"{base}.values",),
                compute=lambda c, ss=s: int(_compute_sign_to(c.get(f"gauss.s{ss}.values") or [], k=int(c.config.get("TAIL_FEATURE_POINTS", 21)))),
                doc=f"Tail direction sign (-1/0/+1) for sigma={s}",
            )
        )
        self.register(
            FeatureSpec(
                key=f"{base}.tail.hook",
                deps=(f"{base}.values",),
                compute=lambda c, ss=s: int(
                    _compute_hook(
                        c.get(f"gauss.s{ss}.values") or [],
                        short_k=max(8, int(c.config.get("TAIL_FEATURE_POINTS", 21)) // 2),
                        long_k=max(30, int(c.config.get("TAIL_FEATURE_POINTS", 21)) * 2),
                    )
                ),
                doc=f"Hook flag (0/1) for sigma={s}",
            )
        )
        self.register(
            FeatureSpec(
                key=f"{base}.tail.flat",
                deps=(f"{base}.values",),
                compute=lambda c, ss=s: float(_compute_flat(c.get(f"gauss.s{ss}.values") or [], k=int(c.config.get("TAIL_FEATURE_POINTS", 21)))),
                doc=f"Flatness score [0..1] for sigma={s}",
            )
        )

    def _register_fan_features(self, sigmas: Sequence[int]) -> None:
        sigmas = [int(s) for s in sigmas]

        def _rank_map(c: "FeatureCatalog") -> Dict[int, int]:
            vals: List[Tuple[int, float]] = []
            for s in sigmas:
                vals.append((s, _safe_float(c.get(f"gauss.s{s}.last"), 0.0)))
            ordered = sorted(vals, key=lambda t: t[1], reverse=True)
            return {s: (i + 1) for i, (s, _v) in enumerate(ordered)}

        self.register(
            FeatureSpec(
                key="gauss.rank_last.map",
                compute=_rank_map,
                doc="Mapping sigma -> rank by last gaussian value (1=highest)",
            )
        )

        for s in sigmas:
            self.register(
                FeatureSpec(
                    key=f"gauss.s{s}.rank_last",
                    deps=("gauss.rank_last.map",),
                    compute=lambda c, ss=s: int((c.get("gauss.rank_last.map") or {}).get(ss, 0)),
                    doc=f"Rank of sigma={s} last value within the stack (1=highest)",
                )
            )

        self.register(
            FeatureSpec(
                key="gauss.fan.top_sigma",
                deps=("gauss.rank_last.map",),
                compute=lambda c: int(min((c.get("gauss.rank_last.map") or {}).items(), key=lambda kv: kv[1])[0])
                if (c.get("gauss.rank_last.map") or {})
                else 0,
                doc="Sigma whose gaussian last value is the highest (top of fan)",
            )
        )
        self.register(
            FeatureSpec(
                key="gauss.fan.bottom_sigma",
                deps=("gauss.rank_last.map",),
                compute=lambda c: int(max((c.get("gauss.rank_last.map") or {}).items(), key=lambda kv: kv[1])[0])
                if (c.get("gauss.rank_last.map") or {})
                else 0,
                doc="Sigma whose gaussian last value is the lowest (bottom of fan)",
            )
        )

        micro = int(self.config.get("sigma_micro", 8) or 8)
        macro = int(self.config.get("sigma_macro", 23) or 23)
        self.register(
            FeatureSpec(
                key=f"gauss.fan.micro_gt_macro_last",
                deps=(f"gauss.s{micro}.last", f"gauss.s{macro}.last"),
                compute=lambda c, mi=micro, ma=macro: 1 if _safe_float(c.get(f"gauss.s{mi}.last"), 0.0) > _safe_float(c.get(f"gauss.s{ma}.last"), 0.0) else 0,
                doc=f"1 if g{micro} last > g{macro} last else 0",
            )
        )

    def _register_channel_features(self, sigmas: Sequence[int]) -> None:
        sigmas = [int(s) for s in sigmas]

        # NOTE: Stage 1 truth is calc.out channels.snapshot; these are still useful
        # when calc.out isn't computed, so we keep them as-is.
        def _channel_snapshot(c: "FeatureCatalog") -> Dict[int, Any]:
            try:
                try:
                    from DB_process_gauss_channel import build_channel_snapshot
                except Exception:
                    from main.engine.process.core.DB_process_gauss_channel import build_channel_snapshot
                return build_channel_snapshot(c.per_sigma_full, k=float(c.config.get("GAUSS_CHANNEL_K", c.config.get("k", 2.0) or 2.0)))
            except Exception as e:
                logging.debug(f"[catalog] channel snapshot build failed: {e}")
                return {}

        self.register(
            FeatureSpec(
                key="channel.snapshot",
                compute=_channel_snapshot,
                doc="ChannelStats per sigma from per_sigma_full (mid_last, width, slope, tag, etc.)",
            )
        )

        for s in sigmas:
            base = f"channel.s{s}"

            self.register(
                FeatureSpec(
                    key=f"{base}.mid_last",
                    deps=("channel.snapshot",),
                    compute=lambda c, ss=s: _safe_float(getattr((c.get("channel.snapshot") or {}).get(ss), "mid_last", 0.0), 0.0),
                    doc=f"Channel mid_last for sigma={s}",
                )
            )
            self.register(
                FeatureSpec(
                    key=f"{base}.width",
                    deps=("channel.snapshot",),
                    compute=lambda c, ss=s: _safe_float(getattr((c.get("channel.snapshot") or {}).get(ss), "width", 0.0), 0.0),
                    doc=f"Channel width for sigma={s}",
                )
            )
            self.register(
                FeatureSpec(
                    key=f"{base}.slope",
                    deps=("channel.snapshot",),
                    compute=lambda c, ss=s: _safe_float(getattr((c.get("channel.snapshot") or {}).get(ss), "slope", 0.0), 0.0),
                    doc=f"Channel midline slope for sigma={s}",
                )
            )
            self.register(
                FeatureSpec(
                    key=f"{base}.tag",
                    deps=("channel.snapshot",),
                    compute=lambda c, ss=s: str(getattr((c.get("channel.snapshot") or {}).get(ss), "tag", "")),
                    doc=f"Channel width trend tag for sigma={s} (expanding|contracting|flat)",
                )
            )

        def _nb_like(c: "FeatureCatalog") -> float:
            try:
                try:
                    from DB_process_gauss_channel import build_gauss_channel_snapshot
                except Exception:
                    from main.engine.process.core.DB_process_gauss_channel import build_gauss_channel_snapshot

                payload = build_gauss_channel_snapshot(
                    c.timing,
                    c.windows,
                    c.per_sigma_full,
                    {
                        "k": float(c.config.get("GAUSS_CHANNEL_K", c.config.get("k", 2.0) or 2.0)),
                        "sigma_micro": int(c.config.get("sigma_micro", 8) or 8),
                        "sigma_macro": int(c.config.get("sigma_macro", 23) or 23),
                        **(c.config.get("GAUSS_CHANNEL_CONFIG", {}) if isinstance(c.config.get("GAUSS_CHANNEL_CONFIG"), dict) else {}),
                    },
                    prev_snapshot=None,
                )
                return _safe_float((payload or {}).get("next_epoch_nb_likelihood", 0.0), 0.0)
            except Exception as e:
                logging.debug(f"[catalog] next_epoch_nb_likelihood failed: {e}")
                return 0.0

        self.register(
            FeatureSpec(
                key="channel.next_epoch_nb_likelihood",
                compute=_nb_like,
                doc="Scalar in [0..1] representing Neutral/Bear likelihood for next epoch (channel-based)",
            )
        )