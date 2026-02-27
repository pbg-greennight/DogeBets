# main/engine/graphing/indicators/indicators_log.py

from __future__ import annotations
import csv
import os
import datetime as dt
from typing import Dict, Any, Optional
import pytz

EST = pytz.timezone("America/New_York")

# --- Logging root (ABSOLUTE, stable) ---
# Write logs next to this file: main/engine/graphing/indicators/logs/...
LOG_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

OCHLV_DIR = os.path.join(LOG_ROOT, "ochlv")
EPOCH_DIR = os.path.join(LOG_ROOT, "epoch")
VOLUME_DIR = os.path.join(LOG_ROOT, "volume")
MERGED_DIR = os.path.join(LOG_ROOT, "merged")
GAUSS_DIR = os.path.join(LOG_ROOT, "gauss")

print("[indicators_log] LOG_ROOT =", LOG_ROOT)
print("[indicators_log] OCHLV_DIR =", OCHLV_DIR)
print("[indicators_log] EPOCH_DIR =", EPOCH_DIR)
print("[indicators_log] VOLUME_DIR =", VOLUME_DIR)
print("[indicators_log] GAUSS_DIR =", GAUSS_DIR)
print("[indicators_log] MERGED_DIR =", MERGED_DIR)



def _ensure_dirs():
    os.makedirs(OCHLV_DIR, exist_ok=True)
    os.makedirs(EPOCH_DIR, exist_ok=True)
    os.makedirs(VOLUME_DIR, exist_ok=True)
    os.makedirs(MERGED_DIR, exist_ok=True)
    os.makedirs(GAUSS_DIR, exist_ok=True)


def _day_str(ts_est: dt.datetime) -> str:
    return ts_est.strftime("%Y-%m-%d")


def _fmt_ts_est(ts_est: dt.datetime) -> str:
    return ts_est.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_ts_label(ts_est: dt.datetime) -> str:
    # "01-08-26 12:42:37PM"
    return ts_est.strftime("%m-%d-%y %I:%M:%S%p")


def _parse_utc_iso(ts_utc_iso: str) -> dt.datetime:
    s = ts_utc_iso.replace("Z", "+00:00")
    t = dt.datetime.fromisoformat(s)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    return t.astimezone(dt.timezone.utc).replace(microsecond=0)


def build_time_fields(ts_utc_iso: str) -> Dict[str, Any]:
    t_utc = _parse_utc_iso(ts_utc_iso)
    t_est = t_utc.astimezone(EST).replace(microsecond=0)
    return {
        "ts_utc": t_utc.isoformat().replace("+00:00", "Z"),
        "ts_est": _fmt_ts_est(t_est),
        "ts_est_label": _fmt_ts_label(t_est),
        "_ts_est_dt": t_est,
    }


class _DailyWriter:
    def __init__(self, base_dir: str, prefix: str, fieldnames: list[str]):
        self.base_dir = base_dir
        self.prefix = prefix
        self.fieldnames = fieldnames
        self._current_day = ""
        self._fh = None
        self._writer: Optional[csv.DictWriter] = None
        self._path = ""

    def _rollover(self, now_est: dt.datetime):
        _ensure_dirs()
        day = _day_str(now_est)

        if day == self._current_day and self._fh and self._writer:
            return

        # close prior
        if self._fh:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass

        self._current_day = day
        self._path = os.path.join(self.base_dir, f"{self.prefix}_{day}.csv")
        is_new = (not os.path.exists(self._path)) or os.path.getsize(self._path) == 0

        self._fh = open(self._path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fieldnames, extrasaction="ignore")

        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def append(self, row: Dict[str, Any], now_est: dt.datetime):
        self._rollover(now_est)
        assert self._writer is not None
        self._writer.writerow(row)
        self._fh.flush()

    @property
    def path(self) -> str:
        return self._path


# ---- Locked schemas (stable prefixes) ----
OCHLV_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
]

EPOCH_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
]

VOLUME_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "vol_up", "vol_down", "vol_delta", "cvd",
    "vol_delta_body", "cvd_body",
    "vol_z", "vol_impact",
    "vol_price_agree", "vol_divergence_flag", "vol_note",
]

GAUSS23_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "gauss_23",
]
GAUSS8_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "gauss_8",
]

GAUSS38_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "gauss_38",
]
GAUSS53_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "gauss_53",
]
GAUSS68_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "gauss_68",
]
GAUSS83_FIELDS = [
    "ts_utc", "ts_est", "ts_est_label",
    "epoch", "next_epoch", "countdown_s", "next_round_time_est",
    "open", "high", "low", "close", "volume",
    "gauss_83",
]
_gauss8 = _DailyWriter(GAUSS_DIR, "gauss8", GAUSS8_FIELDS)
_gauss23 = _DailyWriter(GAUSS_DIR, "gauss23", GAUSS23_FIELDS)
_gauss38 = _DailyWriter(GAUSS_DIR, "gauss38", GAUSS38_FIELDS)
_gauss53 = _DailyWriter(GAUSS_DIR, "gauss53", GAUSS53_FIELDS)
_gauss68 = _DailyWriter(GAUSS_DIR, "gauss68", GAUSS68_FIELDS)
_gauss83 = _DailyWriter(GAUSS_DIR, "gauss83", GAUSS83_FIELDS)
_ochlv = _DailyWriter(OCHLV_DIR, "ochlv", OCHLV_FIELDS)
_epoch = _DailyWriter(EPOCH_DIR, "epoch", EPOCH_FIELDS)
_volume = _DailyWriter(VOLUME_DIR, "volume", VOLUME_FIELDS)


# ---- Public log API ----
def log_ochlv_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],
        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),
        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),
    }
    _ochlv.append(row, now_est)


def log_epoch_bar(ts_utc_iso: str, epoch_snapshot: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],
        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),
    }
    _epoch.append(row, now_est)


def log_volume_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], vol: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "vol_up": vol.get("vol_up"),
        "vol_down": vol.get("vol_down"),
        "vol_delta": vol.get("vol_delta"),
        "cvd": vol.get("cvd"),
        "vol_delta_body": vol.get("vol_delta_body"),
        "cvd_body": vol.get("cvd_body"),

        "vol_z": vol.get("vol_z"),
        "vol_impact": vol.get("vol_impact"),

        "vol_price_agree": vol.get("vol_price_agree"),
        "vol_divergence_flag": vol.get("vol_divergence_flag"),
        "vol_note": vol.get("vol_note"),
    }
    _volume.append(row, now_est)

def log_gauss8_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], g: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "gauss_8": g.get("gauss_8"),
    }
    _gauss8.append(row, now_est)

def log_gauss23_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], g: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "gauss_23": g.get("gauss_23"),
    }
    _gauss23.append(row, now_est)

def log_gauss38_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], g: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "gauss_38": g.get("gauss_38"),
    }
    _gauss38.append(row, now_est)

def log_gauss53_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], g: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "gauss_53": g.get("gauss_53"),
    }
    _gauss53.append(row, now_est)

def log_gauss68_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], g: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "gauss_68": g.get("gauss_68"),
    }
    _gauss68.append(row, now_est)

def log_gauss83_bar(ts_utc_iso: str, ochlv: Dict[str, Any], epoch_snapshot: Dict[str, Any], g: Dict[str, Any]):
    t = build_time_fields(ts_utc_iso)
    now_est = t["_ts_est_dt"]
    row = {
        "ts_utc": t["ts_utc"],
        "ts_est": t["ts_est"],
        "ts_est_label": t["ts_est_label"],

        "epoch": epoch_snapshot.get("epoch"),
        "next_epoch": epoch_snapshot.get("next_epoch"),
        "countdown_s": epoch_snapshot.get("countdown_s"),
        "next_round_time_est": epoch_snapshot.get("next_round_time_est"),

        "open": ochlv.get("open"),
        "high": ochlv.get("high"),
        "low": ochlv.get("low"),
        "close": ochlv.get("close"),
        "volume": ochlv.get("volume"),

        "gauss_83": g.get("gauss_83"),
    }
    _gauss83.append(row, now_est)
