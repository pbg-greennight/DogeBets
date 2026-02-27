# main/engine/indicators/indicators.py
# Log Dir location: main/engine/indicators/logs/

"""
Fetches data from server_hub.py and performs all processing currently done in graphrounds.py:
- Warm start: load last 3 hours (/block preferred, /tail fallback)
- Live updates: poll /tail every 5s, append only new rows (dedupe by timestamp)
- Keep rolling 30-minute window (save removed rows to removed_data.csv)
- Epoch display: poll /get_round every 1s and build the display string
- Epoch markers:
    - Rebuild history markers on warm start
    - Drop a new marker exactly once when countdown hits 0 (latched boundary)

NEW (for logging + downstream indicators):
- Store ts_utc alongside timestamp (EST datetime)
- get_latest_bar(): returns last bar (ts_utc + OCHLV + ts_est dt)
- get_epoch_snapshot(): cached normalized /get_round payload
- OPTIONAL: bar-aligned logging hooks (2.5s cadence) to indicators_log.py
"""

import datetime
import threading
import time
import logging
import pandas as pd
import requests
import pytz

# -----------------------------
# Config
# -----------------------------
HUB_BASE_URL = "http://127.0.0.1:5001"

TAIL_SECONDS_LIVE = 300
WARM_START_HOURS = 3
WINDOW_MINUTES = 30
UPDATE_CADENCE_SECONDS = 2.5

LOGGING_ENABLED = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

EST = pytz.timezone("America/New_York")

# Logging hooks (safe if file not present)
ENABLE_BASE_CSV_LOGGING = True  # set True to log OCHLV+epoch per bar


def log_info(msg: str):
    if LOGGING_ENABLED:
        logging.info(msg)


def log_error(msg: str):
    logging.error(msg)


# -----------------------------
# Shared state
# -----------------------------
_data_lock = threading.Lock()
_epoch_lock = threading.Lock()

data = {
    "ts_utc": [],        # NEW: list[str] Zulu ISO
    "timestamp": [],     # list[datetime] EST
    "BTC_close": [],
    "BTC_volume": [],
    "BTC_open": [],
    "BTC_high": [],
    "BTC_low": [],
}

epoch_markers = []  # list of dicts: {"epoch": int, "ts": datetime}


# -----------------------------
# Internal latching state for epoch display/marker drops
# -----------------------------
class _EpochLatch:
    latched_boundary_iso: str = ""
    latched_boundary_dt: datetime.datetime | None = None
    latched_next_epoch: int | None = None
    last_dropped_boundary_iso: str = ""


_epoch_latch = _EpochLatch()

_started = False
_start_lock = threading.Lock()


# -----------------------------
# Epoch snapshot cache (avoid hammering /get_round)
# -----------------------------
_epoch_cache_lock = threading.Lock()
_epoch_cache = {
    "ts": 0.0,
    "snap": {
        "epoch": 0,
        "next_epoch": 0,
        "countdown_s": 0,
        "next_round_time_est": "",
    },
}


def _parse_iso_to_est(iso_str: str) -> datetime.datetime | None:
    if not iso_str:
        return None
    try:
        s = iso_str.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(EST).replace(microsecond=0)
    except Exception:
        return None


def _save_removed_data(removed_data: dict):
    try:
        df = pd.DataFrame(removed_data)
        df.to_csv("removed_data.csv", mode="a", header=False, index=False)
    except Exception as e:
        log_error(f"Error saving removed data: {e}")


# -----------------------------
# Fetchers
# -----------------------------
def fetch_ochlv_tail() -> list[dict]:
    try:
        r = requests.get(f"{HUB_BASE_URL}/tail?seconds={TAIL_SECONDS_LIVE}", timeout=3)
        r.raise_for_status()
        payload = r.json() or {}
        return payload.get("rows", []) or []
    except Exception as e:
        log_error(f"Error fetching OCHLV from server: {e}")
        return []


def fetch_round_state() -> dict:
    try:
        r = requests.get(f"{HUB_BASE_URL}/get_round", timeout=3)
        r.raise_for_status()
        return r.json() or {}
    except Exception as e:
        log_error(f"Error fetching epoch data: {e}")
        return {}


def get_epoch_snapshot() -> dict:
    """
    Cached normalized epoch snapshot:
      {epoch, next_epoch, countdown_s, next_round_time_est}
    """
    now = time.time()
    with _epoch_cache_lock:
        if now - _epoch_cache["ts"] < 0.8:
            return dict(_epoch_cache["snap"])

    raw = fetch_round_state()
    snap = {
        "epoch": int(raw.get("epoch", 0) or 0),
        "next_epoch": int(raw.get("next_epoch", 0) or 0),
        "countdown_s": int(raw.get("countdown", 0) or 0),
        "next_round_time_est": (raw.get("next_round_time_est", "") or "").strip(),
    }

    with _epoch_cache_lock:
        _epoch_cache["ts"] = now
        _epoch_cache["snap"] = snap

    return dict(snap)


# -----------------------------
# Warm start (3h) + rebuild history markers
# -----------------------------
def warm_start_load_3hr_block() -> bool:
    def load_rows(rows: list[dict]) -> bool:
        ts_utc_list, ts_list = [], []
        o_list, h_list, l_list, c_list, v_list = [], [], [], [], []

        marker_map: dict[tuple[int, datetime.datetime], bool] = {}

        for row in rows:
            try:
                ts_utc_iso = row["ts_utc"]
                ts_utc = datetime.datetime.fromisoformat(ts_utc_iso.replace("Z", "+00:00"))
                ts_est = ts_utc.astimezone(EST).replace(microsecond=0)

                ts_utc_list.append(ts_utc_iso)
                ts_list.append(ts_est)
                o_list.append(row.get("open"))
                h_list.append(row.get("high"))
                l_list.append(row.get("low"))
                c_list.append(row.get("close"))
                v_list.append(row.get("volume"))

                boundary_iso = row.get("epoch_end_est") or row.get("next_round_time_est") or ""
                boundary_dt = _parse_iso_to_est(boundary_iso)

                ep = row.get("epoch")
                try:
                    ep_int = int(ep) if ep is not None else None
                except Exception:
                    ep_int = None

                if boundary_dt and ep_int is not None:
                    marker_map[(ep_int + 1, boundary_dt)] = True

            except Exception:
                continue

        zipped = list(zip(ts_list, ts_utc_list, o_list, h_list, l_list, c_list, v_list))
        zipped.sort(key=lambda x: x[0])

        with _data_lock:
            data["timestamp"] = [z[0] for z in zipped]
            data["ts_utc"] = [z[1] for z in zipped]
            data["BTC_open"] = [z[2] for z in zipped]
            data["BTC_high"] = [z[3] for z in zipped]
            data["BTC_low"] = [z[4] for z in zipped]
            data["BTC_close"] = [z[5] for z in zipped]
            data["BTC_volume"] = [z[6] for z in zipped]

        history_markers = [{"epoch": ep, "ts": ts} for (ep, ts) in marker_map.keys()]
        history_markers.sort(key=lambda m: m["ts"])

        with _epoch_lock:
            epoch_markers.clear()
            epoch_markers.extend(history_markers)

        if data["timestamp"]:
            log_error(
                f"warm_start loaded {len(data['timestamp'])} rows: "
                f"{data['timestamp'][0]} -> {data['timestamp'][-1]} | "
                f"markers={len(history_markers)}"
            )
        else:
            log_error("warm_start loaded 0 rows.")
        return True

    # Try /block
    try:
        r = requests.get(f"{HUB_BASE_URL}/block?hours={WARM_START_HOURS}", timeout=10)
        if r.status_code == 200:
            payload = r.json() or {}
            rows = payload.get("rows", []) or []
            if rows:
                return load_rows(rows)
    except Exception as e:
        log_error(f"warm_start /block failed: {e}")

    # Fallback /tail
    try:
        r = requests.get(f"{HUB_BASE_URL}/tail?seconds={WARM_START_HOURS * 3600}", timeout=10)
        r.raise_for_status()
        payload = r.json() or {}
        rows = payload.get("rows", []) or []
        if rows:
            return load_rows(rows)
        log_error("warm_start fallback /tail returned 0 rows.")
        return False
    except Exception as e:
        log_error(f"warm_start fallback /tail failed: {e}")
        return False


# -----------------------------
# Public accessor for graph
# -----------------------------
def get_processed_snapshot():
    with _data_lock:
        snap = {
            "timestamp": list(data["timestamp"]),
            "ts_utc": list(data["ts_utc"]),   # <-- ADD THIS LINE
            "BTC_close": list(data["BTC_close"]),
            "BTC_open": list(data["BTC_open"]),
            "BTC_high": list(data["BTC_high"]),
            "BTC_low": list(data["BTC_low"]),
            "BTC_volume": list(data["BTC_volume"]),
        }
    with _epoch_lock:
        markers = list(epoch_markers)
    return snap, markers


def get_latest_bar() -> dict | None:
    """
    Returns the latest bar dict (bar-aligned), or None if empty.
    Keys:
      ts_utc, ts_est_dt, open, high, low, close, volume
    """
    with _data_lock:
        if not data["timestamp"]:
            return None
        i = len(data["timestamp"]) - 1
        return {
            "ts_utc": data["ts_utc"][i] if i < len(data["ts_utc"]) else None,
            "ts_est_dt": data["timestamp"][i],
            "open": data["BTC_open"][i],
            "high": data["BTC_high"][i],
            "low": data["BTC_low"][i],
            "close": data["BTC_close"][i],
            "volume": data["BTC_volume"][i],
        }


# -----------------------------
# Live updater (5s)
# -----------------------------
def _update_data_loop():
    # Optional logger import (lazy)
    ilog = None
    if ENABLE_BASE_CSV_LOGGING:
        try:
            from main.engine.indicators import indicators_log as ilog
        except Exception:
            ilog = None

    last_logged_ts_utc = None

    while True:
        try:
            rows = fetch_ochlv_tail()
            if not rows:
                time.sleep(1)
                continue

            appended = 0
            removed_copy = None

            for r in rows:
                try:
                    ts_utc_iso = r["ts_utc"]
                    ts_utc = datetime.datetime.fromisoformat(ts_utc_iso.replace("Z", "+00:00"))
                    ts_est = ts_utc.astimezone(EST).replace(microsecond=0)

                    with _data_lock:
                        if data["timestamp"] and ts_est <= data["timestamp"][-1]:
                            continue

                        data["timestamp"].append(ts_est)
                        data["ts_utc"].append(ts_utc_iso)
                        data["BTC_close"].append(r.get("close"))
                        data["BTC_volume"].append(r.get("volume"))
                        data["BTC_open"].append(r.get("open"))
                        data["BTC_high"].append(r.get("high"))
                        data["BTC_low"].append(r.get("low"))

                    appended += 1

                    # Bar-aligned base logging (once per new bar)
                    if ilog is not None and ts_utc_iso != last_logged_ts_utc:
                        epoch_snap = get_epoch_snapshot()
                        ochlv = {
                            "open": r.get("open"),
                            "high": r.get("high"),
                            "low": r.get("low"),
                            "close": r.get("close"),
                            "volume": r.get("volume"),
                        }
                        ilog.log_ochlv_bar(ts_utc_iso, ochlv, epoch_snap)
                        # optional: separate epoch stream (not required)
                        # ilog.log_epoch_bar(ts_utc_iso, epoch_snap)
                        last_logged_ts_utc = ts_utc_iso

                except Exception:
                    continue

            if appended:
                log_info(f"Appended {appended} new rows")

            # Trim to rolling 30 minutes (and save removed rows)
            with _data_lock:
                if data["timestamp"]:
                    latest_ts = data["timestamp"][-1]
                    cutoff_ts = latest_ts - datetime.timedelta(minutes=WINDOW_MINUTES)

                    if data["timestamp"][0] < cutoff_ts:
                        removed_data = {
                            "timestamp": [],
                            "ts_utc": [],
                            "BTC_close": [],
                            "BTC_volume": [],
                            "BTC_open": [],
                            "BTC_high": [],
                            "BTC_low": [],
                        }

                        while data["timestamp"] and data["timestamp"][0] < cutoff_ts:
                            removed_data["timestamp"].append(data["timestamp"].pop(0))
                            removed_data["ts_utc"].append(data["ts_utc"].pop(0))
                            removed_data["BTC_close"].append(data["BTC_close"].pop(0))
                            removed_data["BTC_volume"].append(data["BTC_volume"].pop(0))
                            removed_data["BTC_open"].append(data["BTC_open"].pop(0))
                            removed_data["BTC_high"].append(data["BTC_high"].pop(0))
                            removed_data["BTC_low"].append(data["BTC_low"].pop(0))

                        removed_copy = {k: list(v) for k, v in removed_data.items()}

            if removed_copy:
                # keep your original behavior (writes removed_data.csv)
                # (strip ts_utc to keep old schema compatible)
                removed_for_old = {k: v for k, v in removed_copy.items() if k != "ts_utc"}
                _save_removed_data(removed_for_old)

        except Exception as e:
            log_error(f"Error updating data: {e}")

        time.sleep(UPDATE_CADENCE_SECONDS - (time.time() % UPDATE_CADENCE_SECONDS))


# -----------------------------
# Epoch display + marker dropping (1s)
# -----------------------------
def get_epoch_display() -> str:
    epoch_data = fetch_round_state()

    now_est = datetime.datetime.now(EST).replace(microsecond=0)
    current_time = now_est.strftime("%I:%M:%S %p")

    current_epoch = int(epoch_data.get("epoch", 0) or 0)
    next_epoch = int(epoch_data.get("next_epoch", current_epoch + 1) or (current_epoch + 1))

    boundary_iso = (epoch_data.get("next_round_time_est") or epoch_data.get("epoch_end_est") or "").strip()
    boundary_dt = _parse_iso_to_est(boundary_iso) if boundary_iso else None

    ts_iso = (epoch_data.get("epoch_post_ts_est") or epoch_data.get("timestamp") or "").strip()
    est_time = _parse_iso_to_est(ts_iso) if ts_iso else None
    formatted_time = est_time.strftime("%I:%M:%S %p") if est_time else "??:??:?? ??"

    if boundary_iso and boundary_iso != _epoch_latch.latched_boundary_iso:
        _epoch_latch.latched_boundary_iso = boundary_iso
        _epoch_latch.latched_boundary_dt = boundary_dt
        _epoch_latch.latched_next_epoch = next_epoch

    latched_dt = _epoch_latch.latched_boundary_dt
    latched_next_epoch = _epoch_latch.latched_next_epoch if _epoch_latch.latched_next_epoch is not None else next_epoch

    if latched_dt:
        remaining = int((latched_dt - now_est).total_seconds())
        remaining = max(remaining, 0)
    else:
        remaining = 0

    minutes, seconds = divmod(remaining, 60)

    if latched_dt:
        next_epoch_time = latched_dt.strftime("%I:%M:%S %p")
    elif boundary_dt:
        next_epoch_time = boundary_dt.strftime("%I:%M:%S %p")
    else:
        next_epoch_time = (now_est + datetime.timedelta(minutes=5)).strftime("%I:%M:%S %p")

    if latched_dt and remaining == 0 and _epoch_latch.latched_boundary_iso:
        if _epoch_latch.last_dropped_boundary_iso != _epoch_latch.latched_boundary_iso:
            with _epoch_lock:
                epoch_markers.append({"epoch": latched_next_epoch, "ts": latched_dt})
            _epoch_latch.last_dropped_boundary_iso = _epoch_latch.latched_boundary_iso
            log_error(f"[epoch] dropped marker: epoch={latched_next_epoch} at {latched_dt.isoformat()}")

    return (
        f"Current Round: {current_epoch} at {formatted_time} - "
        f"Next Round: {latched_next_epoch} at {next_epoch_time} in {minutes:02d}:{seconds:02d} "
        f"(Current Time: {current_time})"
    )


# -----------------------------
# Startup
# -----------------------------
def start():
    global _started
    with _start_lock:
        if _started:
            return
        warm_start_load_3hr_block()
        threading.Thread(target=_update_data_loop, daemon=True).start()
        _started = True
