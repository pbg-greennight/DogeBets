import os
import csv
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from main.engine.acquire.DB_epoch_timing import parse_csv

# --- Logging Config ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - INFO - %(message)s", datefmt="%I:%M:%S %p", force=True)

# --- Constants ---
AXIS_LENGTH = 45  # minutes
CSV_FILE = "../csv/DB_epoch_time.csv"
PROCESSED_DATA_FILE = '../csv/DB_PROCESSED_DATA.csv'

# --- Matplotlib Setup ---
fig, ax = plt.subplots(figsize=(15.0, 7.5))
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.085)

# --- Global (kept to minimize changes in plot()) ---
btc_times = []
btc_closes = []
# We’ll keep containers keyed by sigma for clarity
smoothed_by_sigma = {}   # {10:[...],20:[...],40:[...],60:[...],80:[...]}
peaks_by_sigma    = {}   # {10:[(ts,val),...], ...}
valleys_by_sigma  = {}   # {10:[(ts,val),...], ...}
available_sigmas  = []   # list[int] of present sigmas (e.g., [10,20,40,60,80] or [20,40,60])


# ---------- Helpers ----------
def _detect_schema(fieldnames):
    """
    Return ("new", sigmas=[10,20,40,60,80]) if new 5σ schema is present,
    else ("legacy", sigmas=[20,40,60]) if old 3σ schema is present.
    """
    f = set(fieldnames or [])
    new_ok = all(c in f for c in [
        "smoothed_sigma_10","smoothed_sigma_20","smoothed_sigma_40","smoothed_sigma_60","smoothed_sigma_80",
        "is_peak_10","is_valley_10","is_peak_20","is_valley_20","is_peak_40","is_valley_40",
        "is_peak_60","is_valley_60","is_peak_80","is_valley_80"
    ])
    if new_ok:
        return "new", [10,20,40,60,80]

    legacy_ok = all(c in f for c in [
        "smoothed_sigma_20","smoothed_sigma_40","smoothed_sigma_60",
        "is_peak_20","is_valley_20","is_peak_40","is_valley_40","is_peak_60","is_valley_60"
    ]) or all(c in f for c in [
        # allow the oldest style "1/2/3" just in case:
        "smoothed_sigma_1","smoothed_sigma_2","smoothed_sigma_3",
        "is_peak_1","is_valley_1","is_peak_2","is_valley_2","is_peak_3","is_valley_3"
    ])
    if legacy_ok:
        return "legacy", [20,40,60]

    return "unknown", []


def _col_names_for_sigma(schema, sigma):
    """
    Given schema ('new' or 'legacy') and a sigma (10/20/40/60/80),
    return (smooth_col, peak_col, valley_col) present in the csv row.
    """
    if schema == "new":
        return (f"smoothed_sigma_{sigma}", f"is_peak_{sigma}", f"is_valley_{sigma}")
    # legacy only supports 20/40/60; some old files used 1/2/3 indexes
    if schema == "legacy":
        # Prefer numbered by value if present, else fallback to 1/2/3 mapping
        smooth_val = f"smoothed_sigma_{sigma}"
        peak_val   = f"is_peak_{sigma}"
        val_val    = f"is_valley_{sigma}"
        return (smooth_val, peak_val, val_val)
    return (None, None, None)


# ---------- Data fetch ----------
def fetch_processed_data():
    """
    Reads PROCESSED_DATA_FILE and returns:
      - btc_times: [datetime, ...]
      - btc_closes: [float, ...]
      - smoothed_by_sigma: dict[int, list[float]]
      - peaks_by_sigma:    dict[int, list[(datetime,float)]]
      - valleys_by_sigma:  dict[int, list[(datetime,float)]]
      - available_sigmas:  list[int]
    """
    logging.info("🔍 Entering fetch_processed_data()")

    global smoothed_by_sigma, peaks_by_sigma, valleys_by_sigma, available_sigmas

    btc_times_local = []
    btc_closes_local = []
    smoothed_by_sigma = {}
    peaks_by_sigma = {}
    valleys_by_sigma = {}
    available_sigmas = []

    if not os.path.exists(PROCESSED_DATA_FILE):
        logging.warning(f"⚠️ File not found: {PROCESSED_DATA_FILE}")
        return btc_times_local, btc_closes_local, smoothed_by_sigma, peaks_by_sigma, valleys_by_sigma, available_sigmas

    try:
        with open(PROCESSED_DATA_FILE, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            schema, sigmas = _detect_schema(reader.fieldnames)
            if schema == "unknown":
                logging.warning("⚠️ Unrecognized CSV schema; no smoothed/peak/valley columns detected.")
                return btc_times_local, btc_closes_local, smoothed_by_sigma, peaks_by_sigma, valleys_by_sigma, available_sigmas

            # If legacy uses 1/2/3 names, remap those on the fly to 20/40/60
            legacy_idx_map = {}
            if schema == "legacy":
                # detect whether we have the 1/2/3 style
                fn = set(reader.fieldnames or [])
                if "smoothed_sigma_1" in fn:
                    legacy_idx_map = {20: "1", 40: "2", 60: "3"}  # used for access below

            for s in sigmas:
                smoothed_by_sigma[s] = []
                peaks_by_sigma[s] = []
                valleys_by_sigma[s] = []

            for i, row in enumerate(reader):
                try:
                    ts_str = row.get('Timestamp')
                    close_str = row.get('BTC_Close')
                    if ts_str is None or close_str is None:
                        continue

                    # Timestamp is intraday HH:MM:SS AM/PM; stamp with today's date
                    ts = datetime.strptime(ts_str, "%I:%M:%S %p")
                    now = datetime.now()
                    ts = ts.replace(year=now.year, month=now.month, day=now.day)

                    close = float(close_str)

                    btc_times_local.append(ts)
                    btc_closes_local.append(close)

                    # per-sigma extraction
                    for s in sigmas:
                        smooth_col, peak_col, valley_col = _col_names_for_sigma(schema, s)

                        # legacy 1/2/3 fallback mapping:
                        if schema == "legacy" and legacy_idx_map:
                            idx = legacy_idx_map.get(s)
                            if idx:
                                smooth_col = smooth_col if smooth_col in row else f"smoothed_sigma_{idx}"
                                peak_col   = peak_col   if peak_col   in row else f"is_peak_{idx}"
                                valley_col = valley_col if valley_col in row else f"is_valley_{idx}"

                        # Some rows may be missing (during transitions); guard read
                        s_val = row.get(smooth_col)
                        if s_val is not None and s_val != "":
                            try:
                                smoothed_by_sigma[s].append(float(s_val))
                            except ValueError:
                                smoothed_by_sigma[s].append(float('nan'))
                        else:
                            smoothed_by_sigma[s].append(float('nan'))

                        # Peak/Valley flags
                        if row.get(peak_col) == '1':
                            try:
                                y = float(s_val) if s_val not in (None, "") else float('nan')
                                peaks_by_sigma[s].append((ts, y))
                            except ValueError:
                                peaks_by_sigma[s].append((ts, float('nan')))
                        if row.get(valley_col) == '1':
                            try:
                                y = float(s_val) if s_val not in (None, "") else float('nan')
                                valleys_by_sigma[s].append((ts, y))
                            except ValueError:
                                valleys_by_sigma[s].append((ts, float('nan')))

                except Exception as e:
                    logging.warning(f"⚠️ Error parsing row {i}: {e}")
                    continue

            available_sigmas = [s for s in sigmas if smoothed_by_sigma.get(s)]
            logging.info(f"✅ Loaded {len(btc_times_local)} entries; schema={schema}, sigmas={available_sigmas}")

    except Exception as e:
        logging.error(f"❌ Exception while reading {PROCESSED_DATA_FILE}: {e}", exc_info=True)

    return btc_times_local, btc_closes_local, smoothed_by_sigma, peaks_by_sigma, valleys_by_sigma, available_sigmas


# ---------- Plot ----------
def plot(frame):
    ax.clear()

    # Style setup
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(bottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("BTC_Close Price")
    ax.grid(True)

    now = datetime.now()
    start_time = now - timedelta(minutes=AXIS_LENGTH)

    # Refresh data
    global btc_times, btc_closes, smoothed_by_sigma, peaks_by_sigma, valleys_by_sigma, available_sigmas
    (btc_times, btc_closes, smoothed_by_sigma, peaks_by_sigma, valleys_by_sigma, available_sigmas) = fetch_processed_data()

    # Filter to visible window
    times_filtered = []
    closes_filtered = []
    smoothed_filtered = {s: [] for s in available_sigmas}
    combined = []

    # Build aligned filtered arrays (assume all smoothed series have same length as btc_times)
    for idx in range(len(btc_times)):
        t = btc_times[idx]
        if start_time <= t <= now:
            times_filtered.append(t)
            closes_filtered.append(btc_closes[idx])
            for s in available_sigmas:
                series = smoothed_by_sigma.get(s, [])
                if idx < len(series):
                    smoothed_filtered[s].append(series[idx])

    logging.info(f"📈 Plotting {len(times_filtered)} points in visible window.")

    # Plot BTC Close
    if times_filtered and closes_filtered:
        ax.plot(times_filtered, closes_filtered, label='BTC Close', linewidth=0.75)
        combined += closes_filtered

    # Plot smoothed curves in ascending sigma order
    for s in sorted(available_sigmas):
        y = smoothed_filtered.get(s, [])
        if len(y) == len(times_filtered) and len(y) > 0:
            ax.plot(times_filtered, y, label=f'σ={s}', linewidth=0.65)
            combined += y

    # Plot Peaks and Valleys for each sigma
    for s in sorted(available_sigmas):
        for ts, val in peaks_by_sigma.get(s, []):
            if start_time <= ts <= now:
                ax.plot(ts, val, marker='o', markersize=5)
        for ts, val in valleys_by_sigma.get(s, []):
            if start_time <= ts <= now:
                ax.plot(ts, val, marker='o', markersize=5)

    # Auto-scaling
    if combined:
        y_min = min(combined)
        y_max = max(combined)
        y_range = y_max - y_min
        if y_range == 0:
            y_min -= y_min * 0.05
            y_max += y_max * 0.05
        else:
            y_min -= 0.05 * y_range
            y_max += 0.05 * y_range
        ax.set_ylim(y_min, y_max)

    # Epoch vertical markers
    try:
        timestamps, next_epoch_times, next_epoch_labels = parse_csv(CSV_FILE)
        for i, event_time in enumerate(next_epoch_times):
            if start_time <= event_time <= now:
                ax.axvline(x=event_time, linestyle='--', linewidth=1)
                label_text = f"{next_epoch_labels[i]}\n{event_time.strftime('%I:%M:%S %p')}"
                y_lim = ax.get_ylim()
                ax.text(event_time, y_lim[1], label_text,
                        rotation=90, verticalalignment='top', horizontalalignment='center',
                        fontsize=6, alpha=0.8)
    except Exception as e:
        logging.warning(f"⚠️ Could not draw epoch markers: {e}")

    # X-axis ticks
    tick = start_time.replace(minute=(start_time.minute // 30) * 30, second=0, microsecond=0)
    time_ticks = []
    while tick <= now:
        time_ticks.append(tick)
        tick += timedelta(minutes=30)
    ax.set_xticks(time_ticks)
    ax.set_xticklabels([t.strftime('%I:%M %p') for t in time_ticks], rotation=0)
    ax.set_xlim(start_time, now)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc='lower left', fontsize=8)


# --- Start Animation ---
try:
    ani = animation.FuncAnimation(fig, plot, interval=5000, cache_frame_data=False)
    logging.info("🎞️ FuncAnimation initialized successfully")
except Exception as e:
    logging.error(f"❌ Failed to initialize FuncAnimation: {e}", exc_info=True)
try:
    plt.show()
except Exception as e:
    logging.error(f"❌ Crash during plt.show(): {e}", exc_info=True)
