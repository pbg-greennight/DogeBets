"""main/engine/process/DB_process_orchestrator.py

Main loop and trigger-time orchestration for DB_DATA_PROCESS refactor.

This is a mechanical split of DB_DATA_PROCESS.py; behavior is intended to be identical.
"""

from __future__ import annotations

import logging
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
from typing import Any, Dict

from DB_process_catalog import FeatureCatalog
from DB_process_config import cfg, tail_print_seconds
from DB_process_csv import write_long_snapshot, write_wide_snapshot
from DB_process_gauss_sources import fetch_plot_series, get_gauss_registry, start_gauss_sources_once
from DB_process_series_cache import SeriesCache
from main.engine.process.printing.DB_process_printing import (
    print_epoch_dump,
    print_header,
    print_trend_decision,
)
from main.engine.process.printing.DB_process_printing_msbc import print_sigma_tailing_snapshots
from main.engine.process.printing.DB_process_printing_gcs import print_gaussian_channel_snapshot
from main.engine.process.printing.DB_process_printing_csd_dcsd import print_gaussian_channel_pv_tail
from main.engine.process.printing.DB_process_printing_gbc import print_gaussian_bell_curve_series_dump
from main.engine.process.printing.DB_process_printing_hyst import print_hysteresis_fan_stack
from DB_process_slicing import slice_by_window
from DB_process_time import _fmt_ts, compute_windows, get_epoch_timing
from DB_process_trend import calculate_trend
from DB_process_types import EpochTiming, Windows


def _write_trend_out_json(timing: EpochTiming, decision_dt: datetime, trend_obj, config: Dict[str, Any]) -> None:
    """Write the latest trend decision for DB_DATA_TREND to consume."""
    out_path = config.get("TREND_OUT_JSON")
    if not out_path:
        return

    try:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": datetime.now(tz=decision_dt.tzinfo).isoformat(),
            "decision_time": decision_dt.isoformat(),
            "prev_epoch": int(timing.prev_epoch),
            "curr_epoch": int(timing.curr_epoch),
            "next_epoch": int(timing.next_epoch),
            "trend": str(getattr(trend_obj, "trend", "Neutral")),
            "confidence": float(getattr(trend_obj, "confidence", 0.0) or 0.0),
            "model": str(getattr(trend_obj, "model", "")),
            "notes": str(getattr(trend_obj, "notes", "")),
            "extras": getattr(trend_obj, "extras", {}) or {},
        }

        # Atomic write
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        # Never let a write failure crash the live loop.
        return


def process_epoch_window(
    timing: EpochTiming,
    windows: Windows,
    decision_dt: datetime,
    registry: list,
    config: Dict[str, Any],
    series_cache: SeriesCache | None = None,
) -> None:
    """Runs at trigger time.

    Pull + slice series for each sigma, print sections based on toggles, save WIDE+LONG.
    Catalog-first path:
      - build FeatureCatalog here (first point where per_sigma_* exists)
      - use it to build the per-sigma payload for trend (model-driven sigma selection)
    """

    tz_fallback = timing.dt_next.tzinfo

    # Two views of the same series (additive-only):
    # - per_sigma_full: epoch-window slice for the 5-min "Gaussian Series Dump" (existing behavior)
    # - per_sigma_hist: longer history slice for bell-curve PV detection + tailing snapshots
    per_sigma_full: Dict[int, Dict[str, Any]] = {}
    per_sigma_hist: Dict[int, Dict[str, Any]] = {}

    lookback_minutes = int(config.get("BELL_CURVE_LOOKBACK_MINUTES", 240))
    hist_start = decision_dt - timedelta(minutes=lookback_minutes)
    hist_end = decision_dt

    for r in registry:
        sigma = int(r["sigma"])
        mod = r["module"]
        key = r["key"]

        try:
            ts_all, vals_all = fetch_plot_series(mod, key, tz_fallback)

            # Epoch slice (unchanged)
            ts_full, vals_full = slice_by_window(ts_all, vals_all, windows.full_start, windows.full_end)
            per_sigma_full[sigma] = {"ts": ts_full, "values": vals_full}

            # History slice (new, additive)
            ts_hist, vals_hist = slice_by_window(ts_all, vals_all, hist_start, hist_end)
            per_sigma_hist[sigma] = {"ts": ts_hist, "values": vals_hist}
        except Exception as e:
            logging.warning(f"⚠️ Sigma {sigma} fetch/slice failed: {e}")
            per_sigma_full[sigma] = {"ts": [], "values": []}
            per_sigma_hist[sigma] = {"ts": [], "values": []}

    # ------------------------------------------------------------------
    # CATALOG-FIRST: build the unified feature dictionary access layer here
    # ------------------------------------------------------------------
    catalog = FeatureCatalog(
        timing=timing,
        windows=windows,
        per_sigma_full=per_sigma_full,
        per_sigma_hist=per_sigma_hist,
        config=config,
    )

    # Allow printing helpers to fetch diagnostics from catalog if needed
    try:
        setattr(timing, "_catalog", catalog)
    except Exception:
        pass

    # Print blocks
    print_header(timing, windows, decision_dt, config)

    tail_seconds_list = tail_print_seconds(config)

    # Tailing snapshots are meant to be anchored to "now" and can require >epoch history for PV detection.
    # Use per_sigma_hist (>=240min target) while leaving per_sigma_full for the 5-min dump.
    pv_ref_sigma = int(config.get("PV_REF_SIGMA", 23))
    close_proxy = None
    try:
        ref = per_sigma_hist.get(pv_ref_sigma, {})
        if ref.get("ts") and ref.get("values"):
            close_proxy = (ref["ts"], ref["values"])
    except Exception:
        close_proxy = None

    bell_curve_series = print_sigma_tailing_snapshots(
        timing,
        windows,
        decision_dt,
        tail_seconds_list,
        catalog,
        config,
        close_series=close_proxy,
    )

    # --------------------------------------------------------------
    # Gaussian Channels (A + B) — complements PV-leg/Tail snapshots
    # --------------------------------------------------------------
    pv_tail_status = "skipped"
    if config.get("LOG_GAUSS_CHANNELS", True):
        # Gaussian channel snapshot is sourced from FeatureCatalog computations
        print_gaussian_channel_snapshot(timing, windows, catalog, config)

        # ✅ FIX: print_gaussian_channel_pv_tail expects (timing, windows, pv_tail_status_dict, config, ...)
        # NOT (timing, windows, catalog, pv_ref_sigma, config)
        try:
            pv_tail_dict = catalog.get("channels.pv_tail", {})  # will be populated if ensure_calc ran above
            pv_tail_status = print_gaussian_channel_pv_tail(
                timing,
                windows,
                pv_tail_dict,
                config,
            )
        except Exception:
            pv_tail_status = "failed"
            logging.exception("[pv_tail] print_gaussian_channel_pv_tail() failed")

    if bool(config.get("DEBUG_PV_TAIL_STATUS", True)):
        logging.info(f"PV-TAIL STATUS: {pv_tail_status}")

    # Replace the legacy 5-minute dump with the bell-curve PV window dump.
    print_gaussian_bell_curve_series_dump(timing, decision_dt, bell_curve_series, config)

    # ------------------------------------------------------------------
    # HYSTERESIS FAN STACK (LOGGING + FEATURES ONLY)
    # ------------------------------------------------------------------
    hyst_obj: Dict[str, Any] = {}
    if bool(config.get("LOG_HYSTERESIS", True)):
        try:
            hyst_obj = print_hysteresis_fan_stack(
                timing=timing,
                windows=windows,
                decision_dt=decision_dt,
                catalog=catalog,
                config=config,
            ) or {}
        except Exception as e:
            hyst_obj = {"meta": {"skipped": True, "skip_reason": f"exception:{e}"}}
            logging.exception("[hyst] print_hysteresis_fan_stack() failed")

    # Stash on catalog for DB_process_trend / DB_process_calc to use later (Phase 2 will formalize this)
    try:
        setattr(catalog, "hyst_obj", hyst_obj)
    except Exception:
        pass

    # Legacy dump remains available for debugging via LOG_EPOCH_SERIES_DUMP_LEGACY.
    if bool(config.get("LOG_EPOCH_SERIES_DUMP_LEGACY", False)):
        # Keep epoch dump based on per-sigma raw/full payload
        print_epoch_dump(timing, windows, per_sigma_full, config)

    # ------------------------------------------------------------------
    # CATALOG-FIRST: build the per-sigma payload for trend using model config
    # ------------------------------------------------------------------
    model_path = (
        config.get("TREND_MODEL_PATH")
        or config.get("MODEL_PATH")
        or config.get("DEV_MODEL_PATH")
    )

    # If model_path exists and includes: {"sigmas":{"wanted":[...]}}
    # then catalog will use that list. Otherwise it falls back to config["GAUSS_SIGMAS"].
    per_sigma_for_trend = catalog.build_for_model(model_path=model_path, use_hist=False)

    trend = calculate_trend(
        timing.curr_epoch,
        timing.next_epoch,
        windows,
        per_sigma_for_trend,
        config,
        model_path=model_path,
    )
    print_trend_decision(timing, trend, config)

    import main.engine.DB_DATA_TREND as T
    T.LATEST_TREND_DECISION, T.LATEST_EPOCH_TIMING = trend, timing

    # Publish latest decision for DB_DATA_TREND (or any other consumer)
    _write_trend_out_json(timing, decision_dt, trend, config)

    # Save CSVs (silent)
    write_wide_snapshot(config, timing, windows, decision_dt, registry, per_sigma_full, tail_seconds_list, trend)
    write_long_snapshot(config, timing, windows, decision_dt, per_sigma_full)


def run_continuously() -> None:
    config = cfg()

    if config.get("DEBUG_IMPORT_PATHS", True):
        try:
            from main.engine.process.printing import DB_process_printing as _p
            import DB_process_config as _c
            logging.info(
                "[debug_paths] orchestrator=%s | printing=%s | config=%s",
                __file__,
                getattr(_p, "__file__", "?"),
                getattr(_c, "__file__", "?"),
            )
        except Exception:
            logging.exception("[debug_paths] failed to resolve module paths")

    logging.info(
        "[debug_config] LOG_GAUSS_CHANNELS=%s | LOG_GAUSS_CHANNEL_PV_TAIL=%s | LOG_GAUSS_CHANNEL_PV_TAIL_FULLSTACK=%s",
        config.get("LOG_GAUSS_CHANNELS"),
        config.get("LOG_GAUSS_CHANNEL_PV_TAIL", config.get("LOG_GAUSS_CHANNEL_PVTAIL")),
        config.get("LOG_GAUSS_CHANNEL_PV_TAIL_FULLSTACK"),
    )
    registry = get_gauss_registry()
    start_gauss_sources_once(registry, config)

    # Option A: accumulate plot series across cycles for long-horizon bell-curve PV anchoring
    series_cache = SeriesCache(max_minutes=float(config.get("BELL_CURVE_LOOKBACK_MINUTES", 240)))

    WAKE_OFFSET = float(config.get("WAKE_OFFSET_SECONDS", 12))
    POST_COOLDOWN = float(config.get("POST_TRIGGER_COOLDOWN_SECONDS", 45))
    LOG_SLEEP = bool(config.get("LOG_SLEEP_STATUS", True))
    DEBUG_TIMING = bool(config.get("DEBUG_WAKE_TIMING", True))

    if DEBUG_TIMING:
        logging.info(
            f"[run_continuously] WAKE_OFFSET_SECONDS={WAKE_OFFSET:.1f}s | "
            f"POST_TRIGGER_COOLDOWN_SECONDS={POST_COOLDOWN:.1f}s | "
            f"DEBUG_WAKE_TIMING={DEBUG_TIMING}"
        )

    while True:
        # -------- Outer fetch (if timing missing, retry) --------
        t_fetch0 = time.perf_counter()
        timing = get_epoch_timing()
        t_fetch1 = time.perf_counter()

        if timing is None:
            logging.warning("⚠️ Epoch timing unavailable. Retrying in 10s.")
            time.sleep(10)
            continue

        if DEBUG_TIMING:
            logging.info(f"[timing_fetch] outer get_epoch_timing() took {(t_fetch1 - t_fetch0)*1000:.1f}ms")

        # -------- Inner re-check loop to reach wake window --------
        while True:
            t_fetch0 = time.perf_counter()
            timing = get_epoch_timing()
            t_fetch1 = time.perf_counter()

            if timing is None:
                logging.warning("⚠️ Epoch timing unavailable during re-check. Retrying in 10s.")
                time.sleep(10)
                break

            if DEBUG_TIMING:
                logging.info(f"[timing_fetch] inner get_epoch_timing() took {(t_fetch1 - t_fetch0)*1000:.1f}ms")

            dt_next = timing.dt_next
            now = datetime.now(dt_next.tzinfo) if dt_next.tzinfo is not None else datetime.now()

            seconds_until_next = (dt_next - now).total_seconds()
            wake_target_dt = dt_next - timedelta(seconds=WAKE_OFFSET)
            seconds_until_wake = (wake_target_dt - now).total_seconds()
            target_sleep = max(0.0, seconds_until_wake)

            # Long sleeps: re-check periodically
            if target_sleep > 310:
                if LOG_SLEEP:
                    logging.info(f"⏳ Sleep time ({target_sleep:.1f}s) > 310s. Sleeping 25s then rechecking.")
                time.sleep(25)
                continue

            if target_sleep > 0:
                if LOG_SLEEP:
                    logging.info(
                        f"[DB_DATA_PROCESS] prev_epoch={timing.prev_epoch} @ {_fmt_ts(timing.dt_prev)} | "
                        f"curr_epoch={timing.curr_epoch} @ {_fmt_ts(timing.dt_curr)} | "
                        f"next_epoch={timing.next_epoch} @ {_fmt_ts(timing.dt_next)}"
                    )
                    logging.info(
                        f"[DB_DATA_PROCESS] now={now.strftime('%I:%M:%S %p')} | "
                        f"seconds_until_next={seconds_until_next:.3f}s | "
                        f"WAKE_OFFSET={WAKE_OFFSET:.1f}s | "
                        f"wake_target={wake_target_dt.strftime('%I:%M:%S %p')} | "
                        f"seconds_until_wake={seconds_until_wake:.3f}s | "
                        f"target_sleep={target_sleep:.3f}s"
                    )
                    logging.info(f"⏳ Sleeping {target_sleep:.3f}s until wake window.")
                    logging.info("- - " * 75)

                sleep_start = time.perf_counter()
                time.sleep(target_sleep)
                sleep_end = time.perf_counter()

                if DEBUG_TIMING:
                    now_after = datetime.now(dt_next.tzinfo) if dt_next.tzinfo is not None else datetime.now()
                    drift = (sleep_end - sleep_start) - target_sleep
                    late_vs_wake = (now_after - wake_target_dt).total_seconds()
                    remaining_to_epoch = (dt_next - now_after).total_seconds()
                    logging.info(
                        f"[sleep_wake] slept={(sleep_end - sleep_start):.3f}s "
                        f"(target={target_sleep:.3f}s, drift={drift:+.3f}s) | "
                        f"now_after={now_after.strftime('%I:%M:%S %p')} | "
                        f"late_vs_wake={late_vs_wake:+.3f}s | "
                        f"remaining_to_epoch={remaining_to_epoch:.3f}s"
                    )

                break
            else:
                # already within wake window
                break

        # -------- Trigger moment (NO extra get_epoch_timing unless we overslept) --------
        dt_next = timing.dt_next
        decision_dt = datetime.now(dt_next.tzinfo) if dt_next.tzinfo is not None else datetime.now()

        if (dt_next - decision_dt).total_seconds() <= -0.250:
            if DEBUG_TIMING:
                logging.warning("[trigger] Overslept past dt_next; refreshing epoch timing once.")
            timing_ref = get_epoch_timing()
            if timing_ref is None:
                logging.warning("⚠️ Timing lost at trigger. Skipping this cycle.")
                time.sleep(5)
                continue
            timing = timing_ref
            dt_next = timing.dt_next
            decision_dt = datetime.now(dt_next.tzinfo) if dt_next.tzinfo is not None else datetime.now()

        wake_target_dt = dt_next - timedelta(seconds=WAKE_OFFSET)
        late_vs_wake = (decision_dt - wake_target_dt).total_seconds()
        remaining_to_epoch = (dt_next - decision_dt).total_seconds()

        if DEBUG_TIMING:
            logging.info("=" * 139)
            logging.info(
                f"[trigger] decision_time={decision_dt.strftime('%I:%M:%S %p')} | "
                f"next_epoch_time={dt_next.strftime('%I:%M:%S %p')} | "
                f"wake_target={wake_target_dt.strftime('%I:%M:%S %p')} | "
                f"late_vs_wake={late_vs_wake:+.3f}s | "
                f"remaining_to_epoch={remaining_to_epoch:.3f}s"
            )
            logging.info("=" * 139)

        t0 = time.perf_counter()

        windows = compute_windows(timing.dt_curr, decision_dt, timing.dt_next)

        # support dict OR object
        full_end = windows["full_end"] if isinstance(windows, dict) else windows.full_end
        decision_dt = full_end

        t1 = time.perf_counter()

        if DEBUG_TIMING:
            logging.info(f"[perf] compute_windows() took {(t1 - t0)*1000:.1f}ms")

        t0 = time.perf_counter()
        process_epoch_window(timing, windows, decision_dt, registry, config, series_cache=series_cache)
        t1 = time.perf_counter()

        if DEBUG_TIMING:
            now_after = datetime.now(dt_next.tzinfo) if dt_next.tzinfo is not None else datetime.now()
            rem_after = (dt_next - now_after).total_seconds()
            logging.info(
                f"[perf] process_epoch_window() took {(t1 - t0):.3f}s | remaining_to_epoch_after={rem_after:.3f}s"
            )

        time.sleep(POST_COOLDOWN)


def main() -> None:
    run_continuously()
