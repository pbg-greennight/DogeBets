# main/server_hub.py

from flask import Flask, jsonify, request
import datetime
import pytz
import threading
import time
import os
from pathlib import Path
from collections import deque
import ccxt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
logging.getLogger('werkzeug').disabled = True

app = Flask(__name__)
EST = pytz.timezone("America/New_York")

# -----------------------------
# Round state (fed by TS bot)
# -----------------------------
current_round_info = {
    # TS bot posts "next epoch" (contract design)
    "next_epoch": 0,
    # Derived, always: next_epoch - 1
    "current_epoch": 0,
    # When the TS bot posted the epoch timestamp (EST ISO)
    "epoch_post_ts_est": "",
    # Canonical boundary: start of next epoch (EST ISO) (includes your -22s logic)
    "next_round_time_est": "",
    # --- Legacy keys (kept for compatibility with any older clients) ---
    "epoch": 0,             # mirrors next_epoch
    "timestamp": "",        # mirrors epoch_post_ts_est
    "next_round_time": ""   # mirrors next_round_time_est
}


@app.route("/update_round", methods=["POST"])
def update_round():
    """
    TS bot posts: {"epoch": <int>, "timestamp": "<UTC iso>"}
    NOTE: 'epoch' here is the NEXT epoch by contract design.

    Robust parsing: accepts timestamps like:
      - 2026-01-03T22:02:45Z
      - 2026-01-03T22:02:45+00:00
      - 2026-01-03T22:02:45.123Z
    """
    data = request.json or {}
    try:
        posted_next_epoch = int(data["epoch"]) - 1
        ts_raw = str(data["timestamp"]).strip()

        # ✅ Normalize Zulu timestamps for Python
        if ts_raw.endswith("Z"):
            ts_raw = ts_raw[:-1] + "+00:00"

        # Parse to aware datetime
        utc_time = datetime.datetime.fromisoformat(ts_raw)

        # If somehow still naive, force UTC
        if utc_time.tzinfo is None:
            utc_time = utc_time.replace(tzinfo=datetime.timezone.utc)

        est_time = utc_time.astimezone(EST)

        # Canonical boundary model (your 5m - 11s)
        next_round_time = est_time + datetime.timedelta(minutes=5) - datetime.timedelta(seconds=2)

        current_round_info["next_epoch"] = posted_next_epoch
        current_round_info["current_epoch"] = max(posted_next_epoch - 1, 0)
        current_round_info["epoch_post_ts_est"] = est_time.isoformat()
        current_round_info["next_round_time_est"] = next_round_time.isoformat()

        # Legacy mirrors
        current_round_info["epoch"] = posted_next_epoch
        current_round_info["timestamp"] = current_round_info["epoch_post_ts_est"]
        current_round_info["next_round_time"] = current_round_info["next_round_time_est"]

        print(
            f"[hub] update_round OK: next_epoch={posted_next_epoch} "
            f"current_epoch={current_round_info['current_epoch']} "
            f"post_ts_est={current_round_info['epoch_post_ts_est']} "
            f"next_round_time_est={current_round_info['next_round_time_est']}"
        )

        return jsonify({"status": "success"}), 200

    except Exception as e:
        print(f"[hub] update_round ERROR: {e} | payload={data}")
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route("/get_round", methods=["GET"])
def get_round():
    """
    Returns current epoch state for graph/bot.

    IMPORTANT:
      - 'epoch' returned here is CURRENT epoch (not next epoch).
      - 'next_epoch' is also included.
      - boundary timestamps are canonical and should be used by clients (no recompute).
    """
    try:
        now = datetime.datetime.now(EST)

        nrt_iso = current_round_info.get("next_round_time_est", "") or ""
        nrt = datetime.datetime.fromisoformat(nrt_iso).astimezone(EST) if nrt_iso else None

        countdown = int((nrt - now).total_seconds()) if nrt else 0
        countdown = max(countdown, 0)

        # Derive current window start/end using canonical boundary
        if nrt:
            epoch_end = nrt
            epoch_start = nrt - datetime.timedelta(minutes=5)
            epoch_start_iso = epoch_start.isoformat()
            epoch_end_iso = epoch_end.isoformat()
        else:
            epoch_start_iso = ""
            epoch_end_iso = ""

        return jsonify({
            # ✅ current epoch (clients should NOT subtract)
            "epoch": int(current_round_info.get("current_epoch", 0) or 0),
            # ✅ also provide next epoch explicitly
            "next_epoch": int(current_round_info.get("next_epoch", 0) or 0),
            # Canonical boundary = start of next epoch
            "next_round_time_est": nrt_iso,
            # Convenience: epoch window bounds
            "epoch_start_est": epoch_start_iso,
            "epoch_end_est": epoch_end_iso,
            # Keep your posted timestamp too (not used for boundary)
            "epoch_post_ts_est": current_round_info.get("epoch_post_ts_est", ""),
            "countdown": countdown
        })
    except Exception as e:
        return jsonify({
            "epoch": int(current_round_info.get("current_epoch", 0) or 0),
            "next_epoch": int(current_round_info.get("next_epoch", 0) or 0),
            "next_round_time_est": current_round_info.get("next_round_time_est", ""),
            "epoch_start_est": "",
            "epoch_end_est": "",
            "epoch_post_ts_est": current_round_info.get("epoch_post_ts_est", ""),
            "countdown": 0,
            "error": str(e)
        })


# -----------------------------
# 2.5s OCHLV market hub
# -----------------------------
SYMBOL = "BTC/USDT"
TF_SECONDS = 2.5
TF_MS = int(TF_SECONDS * 1000)

# Hot buffer: keep at least ~3 hours of 2.5s bars
# 3 hours = 10800s / 2.5 = 4320 rows. We keep more for safety.
HOT_MAX_ROWS = 12000
hot_bars = deque(maxlen=HOT_MAX_ROWS)
hot_lock = threading.Lock()

# Latest bar snapshot (for /get_state)
latest_bar = {
    "ts_utc": "",
    "open": None, "high": None, "low": None, "close": None, "volume": None,
    "symbol": SYMBOL, "tf": "2p5s",
    "error": None,
    "updated_at_est": ""
}

# Parquet archive settings
BASE_DIR = Path(__file__).resolve().parent
PARQUET_ROOT = BASE_DIR / "data" / "parquet" / "symbol=BTCUSDT" / "tf=2p5s"
PARQUET_ROOT.mkdir(parents=True, exist_ok=True)

# Batch flush to Parquet every N bars
PARQUET_FLUSH_EVERY = 200  # ~8 minutes of 2.5s bars

parquet_batch = []
parquet_lock = threading.Lock()


def three_hour_block(hour: int) -> int:
    return (hour // 3) * 3


def parquet_partition_dir(ts_utc: datetime.datetime) -> Path:
    # Partition by date + 3-hour block (00,03,06,09,12,15,18,21)
    date_str = ts_utc.strftime("%Y-%m-%d")
    block = three_hour_block(ts_utc.hour)
    p = PARQUET_ROOT / f"date={date_str}" / f"block={block:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet_batch(rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows)

    # Ensure stable column order
    cols = [
        "ts_utc", "ts_est",
        "open", "high", "low", "close", "volume",
        "epoch", "epoch_ts_est", "next_round_time_est",
        "symbol", "tf"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    # Partition based on first row timestamp
    ts0 = datetime.datetime.fromisoformat(df.iloc[0]["ts_utc"].replace("Z", "+00:00")).astimezone(datetime.timezone.utc)
    part_dir = parquet_partition_dir(ts0)

    # Write an immutable "part-*.parquet" file
    fname = f"part-{int(time.time())}-{len(df)}.parquet"
    out_path = part_dir / fname

    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, out_path)

def hydrate_hot_from_parquet(max_rows: int = HOT_MAX_ROWS, lookback_days: int = 2):
    """
    Refill hot_bars from Parquet on disk so a hub restart doesn't lose history.

    Strategy:
      - scan last `lookback_days` partition folders (date=YYYY-MM-DD), newest first
      - within each day, scan all block=xx folders, newest first
      - read parquet files newest first
      - concatenate and take last `max_rows` rows by ts_utc
    """
    try:
        # Collect candidate parquet files (newest first by mtime)
        files = []

        # date partition directories look like: PARQUET_ROOT/date=YYYY-MM-DD/block=XX/part-*.parquet
        # We’ll scan a small time window for speed.
        today_utc = datetime.datetime.now(datetime.timezone.utc).date()

        for d_off in range(lookback_days):
            d = today_utc - datetime.timedelta(days=d_off)
            date_dir = PARQUET_ROOT / f"date={d.strftime('%Y-%m-%d')}"
            if not date_dir.exists():
                continue

            # block folders
            block_dirs = [p for p in date_dir.glob("block=*") if p.is_dir()]
            # sort blocks newest-ish: by block number descending (21,18,...)
            def _block_num(p: Path):
                try:
                    return int(p.name.split("=")[1])
                except Exception:
                    return -1
            block_dirs.sort(key=_block_num, reverse=True)

            for bdir in block_dirs:
                parts = list(bdir.glob("part-*.parquet"))
                # sort by filesystem mtime descending
                parts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                files.extend(parts)

        if not files:
            print("[hub] hydrate_hot_from_parquet: no parquet files found.")
            return 0

        # Read enough recent files to accumulate max_rows (avoid reading everything)
        dfs = []
        rows_loaded_est = 0
        for f in files:
            try:
                df = pd.read_parquet(f)
                if df is None or df.empty:
                    continue
                dfs.append(df)
                rows_loaded_est += len(df)
                # heuristic: stop once we have ~2x needed rows (we'll sort+tail)
                if rows_loaded_est >= max_rows * 2:
                    break
            except Exception:
                continue

        if not dfs:
            print("[hub] hydrate_hot_from_parquet: failed reading parquet files.")
            return 0

        df_all = pd.concat(dfs, ignore_index=True)

        # Normalize timestamps and sort
        df_all["ts_dt"] = pd.to_datetime(df_all["ts_utc"], utc=True, errors="coerce")
        df_all = df_all.dropna(subset=["ts_dt"]).sort_values("ts_dt")

        # Keep only last max_rows
        df_all = df_all.tail(max_rows)

        # Convert rows back to dicts, preserving the keys your API serves
        rows = df_all.drop(columns=["ts_dt"], errors="ignore").to_dict(orient="records")

        with hot_lock:
            hot_bars.clear()
            for r in rows:
                # Ensure the hot buffer contains plain JSONable fields
                hot_bars.append({
                    "ts_utc": r.get("ts_utc"),
                    "ts_est": r.get("ts_est"),
                    "open": float(r.get("open")) if r.get("open") is not None else None,
                    "high": float(r.get("high")) if r.get("high") is not None else None,
                    "low": float(r.get("low")) if r.get("low") is not None else None,
                    "close": float(r.get("close")) if r.get("close") is not None else None,
                    "volume": float(r.get("volume")) if r.get("volume") is not None else None,

                    # keep whatever epoch fields exist in parquet rows
                    "epoch": r.get("epoch"),
                    "epoch_start_est": r.get("epoch_start_est", ""),
                    "epoch_end_est": r.get("epoch_end_est", ""),
                    "next_round_time_est": r.get("next_round_time_est", ""),
                    "epoch_post_ts_est": r.get("epoch_post_ts_est", ""),

                    "symbol": r.get("symbol", "BTCUSDT"),
                    "tf": r.get("tf", "2p5s"),
                })

        # Also refresh latest_bar snapshot from the last hot bar
        with hot_lock:
            if hot_bars:
                last = hot_bars[-1]
                latest_bar.update({
                    "ts_utc": last.get("ts_utc", ""),
                    "open": last.get("open"),
                    "high": last.get("high"),
                    "low": last.get("low"),
                    "close": last.get("close"),
                    "volume": last.get("volume"),
                    "symbol": last.get("symbol", SYMBOL),
                    "tf": last.get("tf", "2p5s"),
                    "error": None,
                    "updated_at_est": datetime.datetime.now(EST).isoformat()
                })

        print(f"[hub] hydrate_hot_from_parquet: loaded {len(rows)} rows into hot buffer.")
        return len(rows)

    except Exception as e:
        print(f"[hub] hydrate_hot_from_parquet error: {e}")
        return 0

def append_bar(bar_row: dict):
    # Update hot buffer
    with hot_lock:
        hot_bars.append(bar_row)

    # Update latest snapshot
    latest_bar.update({
        "ts_utc": bar_row["ts_utc"],
        "open": bar_row["open"],
        "high": bar_row["high"],
        "low": bar_row["low"],
        "close": bar_row["close"],
        "volume": bar_row["volume"],
        "symbol": bar_row["symbol"],
        "tf": bar_row["tf"],
        "error": None,
        "updated_at_est": datetime.datetime.now(EST).isoformat()
    })

    # Add to parquet batch and flush sometimes
    with parquet_lock:
        parquet_batch.append(bar_row)
        if len(parquet_batch) >= PARQUET_FLUSH_EVERY:
            batch = parquet_batch[:]
            parquet_batch.clear()
            # write outside lock? safe enough here because batch copied
            write_parquet_batch(batch)


def market_loop_trades_to_2p5s():
    """
    Build 2.5s OHLCV bars from Binance trades using ccxt.fetch_trades.
    Volume here is base-asset amount summed across trades.
    """
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "rateLimit": 200,
        "timeout": 30000
    })

    # ccxt uses ms timestamps
    since = None
    current_bucket = None  # bucket_id (ms) aligned to 2.5s
    bar = None

    while True:
        try:
            trades = exchange.fetch_trades(SYMBOL, since=since, limit=1000)
            if trades:
                # Advance since to last trade + 1ms
                since = trades[-1]["timestamp"] + 1

            for t in trades:
                ts_ms = t["timestamp"]
                price = float(t["price"])
                amount = float(t["amount"]) if t.get("amount") is not None else 0.0

                bucket = (ts_ms // TF_MS) * TF_MS
                if current_bucket is None:
                    current_bucket = bucket
                    bar = {
                        "bucket_ms": bucket,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": amount
                    }
                    continue

                if bucket == current_bucket:
                    # Update current bar
                    bar["high"] = max(bar["high"], price)
                    bar["low"] = min(bar["low"], price)
                    bar["close"] = price
                    bar["volume"] += amount
                else:
                    # Finalize old bar (flush) and start new one
                    ts_utc = datetime.datetime.fromtimestamp(current_bucket / 1000, tz=datetime.timezone.utc)
                    ts_est = ts_utc.astimezone(EST)

                    # --- Canonical epoch tagging by boundary time ---
                    nrt_iso = current_round_info.get("next_round_time_est", "") or ""
                    nrt = datetime.datetime.fromisoformat(nrt_iso).astimezone(EST) if nrt_iso else None

                    cur_epoch = int(current_round_info.get("current_epoch", 0) or 0)
                    next_epoch = int(current_round_info.get("next_epoch", 0) or 0)

                    if nrt and ts_est >= nrt:
                        bar_epoch = next_epoch
                        epoch_start_est = nrt
                        epoch_end_est = nrt + datetime.timedelta(minutes=5)
                    else:
                        bar_epoch = cur_epoch
                        epoch_end_est = nrt if nrt else None
                        epoch_start_est = (nrt - datetime.timedelta(minutes=5)) if nrt else None

                    row = {
                        "ts_utc": ts_utc.isoformat().replace("+00:00", "Z"),
                        "ts_est": ts_est.isoformat(),
                        "open": float(bar["open"]),
                        "high": float(bar["high"]),
                        "low": float(bar["low"]),
                        "close": float(bar["close"]),
                        "volume": float(bar["volume"]),
                        # ✅ epoch aligned to bar timestamp vs boundary
                        "epoch": int(bar_epoch),
                        # ✅ explicit epoch window bounds (helps graph + debugging)
                        "epoch_start_est": epoch_start_est.isoformat() if epoch_start_est else "",
                        "epoch_end_est": epoch_end_est.isoformat() if epoch_end_est else "",
                        # ✅ canonical boundary for this moment (start of next epoch)
                        "next_round_time_est": nrt_iso,
                        # keep this for reference if you want it
                        "epoch_post_ts_est": current_round_info.get("epoch_post_ts_est", ""),

                        "symbol": "BTCUSDT",
                        "tf": "2p5s"
                    }

                    append_bar(row)

                    # Start next bar
                    current_bucket = bucket
                    bar = {
                        "bucket_ms": bucket,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": amount
                    }

            # Idle a bit — keep rate low
            time.sleep(0.25)

        except Exception as e:
            latest_bar["error"] = str(e)
            latest_bar["updated_at_est"] = datetime.datetime.now(EST).isoformat()
            time.sleep(1.0)


# -----------------------------
# API endpoints for dashboards
# -----------------------------
@app.route("/block", methods=["GET"])
def block():
    """
    Returns the most recent 'hours' of bars (default 3).
    Primary: reads from hot buffer if it has enough.
    Fallback: reads from Parquet partitions on disk (current 3-hour block).
    """
    hours = request.args.get("hours", default=3, type=int)
    seconds = int(hours * 3600)

    # 1) Try hot buffer first (fast)
    with hot_lock:
        bars = list(hot_bars)

    if bars:
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)
        hot_out = []
        for r in bars:
            try:
                ts = datetime.datetime.fromisoformat(r["ts_utc"].replace("Z", "+00:00"))
                if ts >= cutoff:
                    hot_out.append(r)
            except Exception:
                continue
        if len(hot_out) > 0:
            return jsonify({"source": "hot", "count": len(hot_out), "rows": hot_out})

    # 2) Fallback to Parquet for current 3-hour block
    try:
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        date_str = now_utc.strftime("%Y-%m-%d")
        blk = three_hour_block(now_utc.hour)
        part_dir = PARQUET_ROOT / f"date={date_str}" / f"block={blk:02d}"

        if not part_dir.exists():
            return jsonify({"source": "parquet", "count": 0, "rows": [], "error": "No parquet partition found"}), 200

        files = sorted([p for p in part_dir.glob("part-*.parquet")])
        if not files:
            return jsonify({"source": "parquet", "count": 0, "rows": [], "error": "No parquet files in partition"}), 200

        # Read all files in this block
        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_parquet(f))
            except Exception:
                continue

        if not dfs:
            return jsonify({"source": "parquet", "count": 0, "rows": [], "error": "Failed reading parquet files"}), 200

        df = pd.concat(dfs, ignore_index=True)

        # Keep last 'hours' hours just in case partition contains extra
        df["ts_dt"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=seconds)
        df = df[df["ts_dt"] >= cutoff].sort_values("ts_dt")

        # Convert to rows with the same keys graphtest expects (keep epoch fields too)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "ts_utc": row.get("ts_utc"),
                "ts_est": row.get("ts_est"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),

                # ✅ include epoch timing fields (needed for restart epoch markers)
                "epoch": row.get("epoch"),
                "epoch_ts_est": row.get("epoch_ts_est"),
                "next_round_time_est": row.get("next_round_time_est"),

                "symbol": row.get("symbol"),
                "tf": row.get("tf"),
            })

        return jsonify({"source": "parquet", "count": len(rows), "rows": rows})
    except Exception as e:
        return jsonify({"source": "parquet", "count": 0, "rows": [], "error": str(e)}), 200

@app.route("/get_state", methods=["GET"])
def get_state():
    # Round countdown
    round_payload = {
        "epoch": current_round_info.get("epoch", 0),
        "timestamp": current_round_info.get("timestamp", ""),
        "countdown": 0
    }
    try:
        if current_round_info.get("next_round_time"):
            nrt = datetime.datetime.fromisoformat(current_round_info["next_round_time"]).astimezone(EST)
            now = datetime.datetime.now(EST)
            round_payload["countdown"] = max(int((nrt - now).total_seconds()), 0)
    except Exception as e:
        round_payload["countdown"] = 0
        round_payload["error"] = str(e)

    return jsonify({
        "round": round_payload,
        "market": latest_bar
    })


@app.route("/tail", methods=["GET"])
def tail():
    """
    Hot tail for graphtest or remote clients.
    Examples:
      /tail?epochs=30   -> last 30 epochs (~9000 seconds)
      /tail?seconds=9000
      /tail?n=2000
    """
    epochs = request.args.get("epochs", default=None, type=int)
    seconds = request.args.get("seconds", default=None, type=int)
    n = request.args.get("n", default=None, type=int)

    if epochs is not None:
        seconds = int(epochs * 300)  # 5min per epoch
    if seconds is None and n is None:
        # default: last 30 epochs
        seconds = 30 * 300

    with hot_lock:
        bars = list(hot_bars)

    if n is not None:
        bars = bars[-n:]
        return jsonify({"count": len(bars), "rows": bars})

    # seconds window
    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)
    out = []
    for r in bars:
        try:
            ts = datetime.datetime.fromisoformat(r["ts_utc"].replace("Z", "+00:00"))
            if ts >= cutoff:
                out.append(r)
        except Exception:
            continue

    return jsonify({"count": len(out), "rows": out})

def hub_status_logger():
    est = EST
    while True:
        try:
            with hot_lock:
                n = len(hot_bars)
                last = hot_bars[-1] if n else None

            # Pull canonical window times
            nrt_iso = current_round_info.get("next_round_time_est", "") or ""
            nrt = datetime.datetime.fromisoformat(nrt_iso).astimezone(est) if nrt_iso else None

            cur_epoch = int(current_round_info.get("current_epoch", 0) or 0)

            if nrt:
                win_start = (nrt - datetime.timedelta(minutes=5)).strftime("%I:%M:%S%p").lstrip("0")
                win_end = nrt.strftime("%I:%M:%S%p").lstrip("0")
                win_str = f"from {win_start} to {win_end}"
            else:
                win_str = "from ? to ?"

            if last:
                # last_ts as EST, show date + time like your example
                try:
                    last_ts_est = datetime.datetime.fromisoformat(last["ts_est"]).astimezone(est)
                    last_ts_str = last_ts_est.strftime("%Y-%m-%d, %I:%M:%S%p").lstrip("0")
                except Exception:
                    last_ts_str = last.get("ts_est")

                o = last.get("open")
                c = last.get("close")
                lo = last.get("low")
                hi = last.get("high")
                v = last.get("volume")

                print(
                    f"[hub] epoch={cur_epoch} {win_str} | hot_rows={n} "
                    f"last_ts={last_ts_str} | open={o} close={c} low={lo} high={hi} volume={v}"
                )
            else:
                print(f"[hub] epoch={cur_epoch} {win_str} | hot_rows=0")

        except Exception as e:
            print(f"[hub] logger error: {e}")

        time.sleep(5)

def start_background_threads():
    hydrate_hot_from_parquet(max_rows=HOT_MAX_ROWS, lookback_days=2)
    threading.Thread(target=market_loop_trades_to_2p5s, daemon=True).start()
    threading.Thread(target=hub_status_logger, daemon=True).start()



if __name__ == "__main__":
    start_background_threads()
    app.run(port=5001, use_reloader=False)

