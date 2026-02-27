# main/engine/graphing/graphrounds.py

import os
import importlib.util
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import datetime
import logging
import pytz
import sys

# Ensure project root (the folder that contains "main/") is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))  # -> DogeBets/
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- graphrounds weekly CSV logger (standalone; does not touch indicators_log.py)
GRAPHROUNDS_LOGGER = None
_grlog_path = os.path.join(SCRIPT_DIR, "logs", "graphrounds_log.py")
try:
    print("[graphrounds] graphrounds_log.py path:", _grlog_path)
    if os.path.exists(_grlog_path):
        spec = importlib.util.spec_from_file_location("graphrounds_log", _grlog_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("spec/loader is None for graphrounds_log.py")
        _mod = importlib.util.module_from_spec(spec)
        # ✅ Fix for dataclass loading: register module before executing it
        import sys
        sys.modules[spec.name] = _mod
        spec.loader.exec_module(_mod)  # type: ignore
        GRAPHROUNDS_LOGGER = _mod

        print("[graphrounds] ✅ graphrounds_log loaded OK")
    else:
        print("[graphrounds] ❌ graphrounds_log.py not found at path above")
except Exception as _e:
    GRAPHROUNDS_LOGGER = None
    print("[graphrounds] ❌ graphrounds_log failed to load:", repr(_e))


# Canonical imports (single module instance everywhere)
from main.engine.indicators import indicators
from main.engine.indicators.gaussian import indicators_gauss8, indicators_gauss23, indicators_gauss38
from main.engine.indicators.gaussian import indicators_gauss53, indicators_gauss68, indicators_gauss83

HAVE_G8 = HAVE_G23 = HAVE_G38 = HAVE_G53 = HAVE_G68 = HAVE_G83 = True
print("INDICATORS IMPORTED FROM:", getattr(indicators, "__file__", "<no file>"))
print("GAUSSIANS IMPORTED FROM:", getattr(indicators_gauss8, "__file__", "<no file>"))


# Import volume indicator module (optional but enabled here)
try:
    import indicators_volume
    HAVE_VOL = True
except Exception:
    indicators_volume = None
    HAVE_VOL = False


# Configuration to enable or disable logging
LOGGING_ENABLED = False

# epoch_log v1 live console preview
EPOCH_LOG_V1_PREVIEW = True

# Control whether indicator traces are on by default
SHOW_IND_VOLUME_DEFAULT = False
SHOW_IND_GAUSS8_DEFAULT = False
SHOW_IND_GAUSS23_DEFAULT = False
SHOW_IND_GAUSS38_DEFAULT = False
SHOW_IND_GAUSS53_DEFAULT = False
SHOW_IND_GAUSS68_DEFAULT = False
SHOW_IND_GAUSS83_DEFAULT = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def log_info(message):
    if LOGGING_ENABLED:
        logging.info(message)


def log_error(message):
    logging.error(message)


app = dash.Dash(__name__)

# Start background processing (warm start + live updates)
indicators.start()

# Start indicator(s) processing/logging
if HAVE_VOL and indicators_volume is not None:
    indicators_volume.start()
if HAVE_G8 and indicators_gauss8 is not None:
    indicators_gauss8.start()
if HAVE_G23 and indicators_gauss23 is not None:
    indicators_gauss23.start()
if HAVE_G38 and indicators_gauss38 is not None:
    indicators_gauss38.start()
if HAVE_G53 and indicators_gauss53 is not None:
    indicators_gauss53.start()
if HAVE_G68 and indicators_gauss68 is not None:
    indicators_gauss68.start()
if HAVE_G83 and indicators_gauss83 is not None:
    indicators_gauss83.start()

app.layout = html.Div(
    style={"height": "98vh", "width": "95vw", "backgroundColor": "lightgrey"},
    children=[
        dcc.Graph(id="btc-graph", style={"height": "90vh"}),
        # ---- X-axis window control (minutes) ----
        dcc.Input(
            id="x-window-minutes",
            type="number",
            value=32,
            min=1,
            step=1,
            debounce=True,  # only fires when you press enter / defocus
            style={"width": "120px", "margin": "6px"},
        ),
        # Drive both graph refresh + epoch_log v1 console preview at 2.5s cadence
        dcc.Interval(id="interval-component", interval=2500, n_intervals=0),
        dcc.Interval(id="epoch-interval-component", interval=1 * 1000, n_intervals=0),
        html.Div(id="epoch-display", style={"textAlign": "center", "fontSize": 24, "color": "black"}),
        html.Div(id="console-dummy", style={"display": "none"}),
    ],

)

@app.callback(
    Output("epoch-display", "children"),
    Input("epoch-interval-component", "n_intervals"),
)
def update_epoch(_n):
    try:
        return indicators.get_epoch_display()
    except Exception as e:
        log_error(f"Error updating epoch display: {e}")
        return "Error updating epoch"


# -----------------------------
# epoch_log v1 console preview
# -----------------------------
_last_printed_ts_utc = None
_last_marker_epoch = None
_last_marker_ts = None
  # ISO string


def _to_sec(x):
    """Normalize a timestamp to integer epoch-seconds for stable matching."""
    try:
        # datetime (aware or naive)
        if hasattr(x, "timestamp"):
            return int(x.timestamp())
        # ISO string
        if isinstance(x, str):
            # Handles 'Z' and offsets
            s = x.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return int(dt.timestamp())
    except Exception:
        return None
    return None


def _build_ts_to_val(plot: dict, key: str) -> dict:
    """Build a sec->val map from an indicator plot series."""
    try:
        xs = plot.get("ts", []) or []
        ys = plot.get(key, []) or []
        if not xs or not ys:
            return {}
        tail = 400  # slightly bigger buffer
        xs = xs[-tail:]
        ys = ys[-tail:]
        out = {}
        for x, y in zip(xs, ys):
            sec = _to_sec(x)
            if sec is not None:
                out[sec] = y
        return out
    except Exception:
        return {}


@app.callback(
    Output("console-dummy", "children"),
    Input("interval-component", "n_intervals"),
)
def epoch_log_v1_preview(_n):
    """Print one aligned row per new 2.5s bar (between epoch markers)."""
    global _last_printed_ts_utc

    if not EPOCH_LOG_V1_PREVIEW:
        return ""

    try:
        snap, _markers = indicators.get_processed_snapshot()

        # Detect epoch boundary markers (so we can safely archive weekly CSV without splitting an epoch)
        global _last_marker_epoch, _last_marker_ts
        if _markers:
            try:
                last_m = _markers[-1]
                m_ep = last_m.get("epoch")
                m_ts = last_m.get("ts")  # may be datetime or iso string
                # normalize m_ts to datetime (local/EST already used throughout graphrounds)
                if isinstance(m_ts, str):
                    try:
                        m_ts = datetime.datetime.fromisoformat(m_ts)
                    except Exception:
                        m_ts = None
                if m_ep is not None and m_ep != _last_marker_epoch:
                    _last_marker_epoch = m_ep
                    _last_marker_ts = m_ts
                    if GRAPHROUNDS_LOGGER and m_ts is not None:
                        GRAPHROUNDS_LOGGER.log_epoch_boundary_event(m_ep, m_ts)
                        GRAPHROUNDS_LOGGER.epochlog_v1_on_epoch_boundary(m_ts)
            except Exception as _e:
                # never let logging break the graph loop
                pass
        ts_list = snap.get("timestamp", []) or []      # datetime (EST)
        ts_utc_list = snap.get("ts_utc", []) or []     # ISO string
        if not ts_list or not ts_utc_list:
            return ""

        # Build gaussian lookup maps (datetime -> value)
        g8_map = _build_ts_to_val(indicators_gauss8.get_plot_series(), "g8") if HAVE_G8 and indicators_gauss8 else {}
        g23_map = _build_ts_to_val(indicators_gauss23.get_plot_series(), "g23") if HAVE_G23 and indicators_gauss23 else {}
        g38_map = _build_ts_to_val(indicators_gauss38.get_plot_series(), "g38") if HAVE_G38 and indicators_gauss38 else {}
        g53_map = _build_ts_to_val(indicators_gauss53.get_plot_series(), "g53") if HAVE_G53 and indicators_gauss53 else {}
        g68_map = _build_ts_to_val(indicators_gauss68.get_plot_series(), "g68") if HAVE_G68 and indicators_gauss68 else {}
        g83_map = _build_ts_to_val(indicators_gauss83.get_plot_series(), "g83") if HAVE_G83 and indicators_gauss83 else {}

        # Find new rows since last print
        start_i = 0
        if _last_printed_ts_utc is not None:
            try:
                start_i = ts_utc_list.index(_last_printed_ts_utc) + 1
            except ValueError:
                # if history rolled, just print the tail
                start_i = max(0, len(ts_utc_list) - 3)

        if start_i >= len(ts_utc_list):
            return ""

        epoch_snap = indicators.get_epoch_snapshot() if hasattr(indicators, "get_epoch_snapshot") else {}
        ep = epoch_snap.get("epoch")
        next_ep = epoch_snap.get("next_epoch")
        cd = epoch_snap.get("countdown_s")
        nxt = epoch_snap.get("next_round_time_est")
        if ep is None or next_ep is None:
            print("[epoch_log_v1_preview] epoch snapshot empty:", epoch_snap)

        for i in range(start_i, len(ts_utc_list)):
            ts = ts_list[i]
            ts_utc = ts_utc_list[i]
            o = snap.get("BTC_open", [None])[i]
            h = snap.get("BTC_high", [None])[i]
            l = snap.get("BTC_low", [None])[i]
            c = snap.get("BTC_close", [None])[i]
            v = snap.get("BTC_volume", [None])[i]

            def fmt(x):
                return "NA" if x is None else f"{float(x):.2f}"

            def fmtv(x):
                return "NA" if x is None else f"{float(x):.4f}"

            ts_sec = int(ts.timestamp())
            g8 = g8_map.get(ts_sec)
            g23 = g23_map.get(ts_sec)
            g38 = g38_map.get(ts_sec)
            g53 = g53_map.get(ts_sec)
            g68 = g68_map.get(ts_sec)
            g83 = g83_map.get(ts_sec)

         #   if g23_map:
         #       print("[gauss_debug] sample keys:", list(g23_map.keys())[-3:])


            # Persist this row to weekly CSV (7-day rollover, archived on epoch boundary)
            if GRAPHROUNDS_LOGGER:
                try:
                    row = {
                        "ts_utc": ts_utc,
                        "ts_est": ts.isoformat(),
                        "epoch": ep,
                        "next_epoch": next_ep,
                        "countdown_s": cd,
                        "next_round_time_est": str(nxt),
                        "open": o,
                        "high": h,
                        "low": l,
                        "close": c,
                        "volume": v,
                        "g8": g8,
                        "g23": g23,
                        "g38": g38,
                        "g53": g53,
                        "g68": g68,
                        "g83": g83,
                    }
                    GRAPHROUNDS_LOGGER.log_epochlog_v1_row(row, ts)
                except Exception:
                    pass

            print(
                f"{ts.strftime('%H:%M:%S')} | ep={ep} -> next={next_ep} | cd={cd}s | next_round={nxt} | "
                f"O={fmt(o)} H={fmt(h)} L={fmt(l)} C={fmt(c)} V={fmt(v)} | "
                f"G8={fmtv(g8)} G23={fmtv(g23)} G38={fmtv(g38)} G53={fmtv(g53)} G68={fmtv(g68)} G83={fmtv(g83)}"
            )

            _last_printed_ts_utc = ts_utc

    except Exception as e:
        log_error(f"epoch_log_v1_preview error: {e}")

    return ""

# -----------------------------
# Color maps
# -----------------------------
GAUSS_COLORS = {
    8:  "yellow",
    23: "orange",
    38: "cyan",
    53: "dodgerblue",
    68: "purple",
    83: "magenta",
}

BTC_COLORS = {
    "close": "black",
    "open":  "purple",
    "high":  "red",
    "low":   "blue",
    "volume": "orange",
}

# -----------------------------
# X-axis window helper
# -----------------------------
def compute_x_range(minutes: float):
    """
    Returns (x_start, x_end) for the graph based on a lookback window in minutes.
    Keeps the 10-second forward buffer you requested.
    """
    est = pytz.timezone("America/New_York")
    now_est = datetime.datetime.now(est)

    # guardrails
    try:
        m = float(minutes)
    except Exception:
        m = 32.0
    if m <= 0:
        m = 32.0

    x_start = now_est - datetime.timedelta(minutes=m)
    x_end = now_est + datetime.timedelta(seconds=10)  # keep your 10s blank buffer
    return x_start, x_end

@app.callback(
    Output("btc-graph", "figure"),
    Input("interval-component", "n_intervals"),
    Input("x-window-minutes", "value"),
    State("btc-graph", "relayoutData"),
    State("btc-graph", "figure"),
)

def update_graph(n, x_window_minutes, relayoutData, prev_fig):
    log_info(f"Updating graph for interval {n}")
    try:
        fig = go.Figure()

        snap, markers = indicators.get_processed_snapshot()
        ts = snap["timestamp"]

        traces = [
            go.Scatter(
                x=[t for t in ts],
                y=snap["BTC_close"],
                mode="lines",
                name="BTC Close",
                line=dict(color=BTC_COLORS["close"], width=2),
                customdata=[t.strftime("%d/%m/%y %I:%M:%S %p") for t in ts],
                hovertemplate="Time: %{customdata}<br>Price: %{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=[t for t in ts],
                y=snap["BTC_open"],
                mode="lines",
                name="BTC Open",
                line=dict(color=BTC_COLORS["open"], width=2),
                customdata=[t.strftime("%d/%m/%y %I:%M:%S %p") for t in ts],
                hovertemplate="Time: %{customdata}<br>Price: %{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=[t for t in ts],
                y=snap["BTC_high"],
                mode="lines",
                name="BTC High",
                line=dict(color=BTC_COLORS["high"], width=2),
                customdata=[t.strftime("%d/%m/%y %I:%M:%S %p") for t in ts],
                hovertemplate="Time: %{customdata}<br>Price: %{y:.2f}<extra></extra>",
            ),
            go.Scatter(
                x=[t for t in ts],
                y=snap["BTC_low"],
                mode="lines",
                name="BTC Low",
                line=dict(color=BTC_COLORS["low"], width=2),
                customdata=[t.strftime("%d/%m/%y %I:%M:%S %p") for t in ts],
                hovertemplate="Time: %{customdata}<br>Price: %{y:.2f}<extra></extra>",
            ),
            go.Bar(
                x=[t for t in ts],
                y=snap["BTC_volume"],
                name="BTC Volume",
                marker=dict(color=BTC_COLORS["volume"]),
                customdata=[t.strftime("%d/%m/%y %I:%M:%S %p") for t in ts],
                hovertemplate="Time: %{customdata}<br>Volume: %{y}<extra></extra>",
                yaxis="y2",
            ),
        ]

        # ---- Add IND_Volume trace (CVD) ----
        if HAVE_VOL and indicators_volume is not None:
            vol_series = indicators_volume.get_plot_series()
            x_ind = vol_series["ts"]
            y_ind = vol_series["cvd"]

            ind_trace = go.Scatter(
                x=x_ind,
                y=y_ind,
                mode="lines",
                name="IND_Volume",
                yaxis="y3",
                hovertemplate="Time: %{x}<br>CVD: %{y:.2f}<extra></extra>",
            )
            if not SHOW_IND_VOLUME_DEFAULT:
                ind_trace.visible = "legendonly"
            traces.append(ind_trace)

        # Helper to add gauss traces with consistent color
        def _add_gauss_trace(mod, key: str, sigma: int, show_default: bool, label: str):
            series = mod.get_plot_series()
            x_g = series.get("ts", [])
            y_g = series.get(key, [])
            if not x_g:
                return
            tr = go.Scatter(
                x=x_g,
                y=y_g,
                mode="lines",
                name=label,
                line=dict(color=GAUSS_COLORS.get(sigma, "white"), width=2),
                hovertemplate=f"Time: %{{x}}<br>{label}: %{{y:.2f}}<extra></extra>",
            )
            if not show_default:
                tr.visible = "legendonly"
            traces.append(tr)

        # ---- Gaussian traces ----
        if HAVE_G8 and indicators_gauss8 is not None:
            _add_gauss_trace(indicators_gauss8, "g8", 8, SHOW_IND_GAUSS8_DEFAULT, "IND_gauss_8")
        if HAVE_G23 and indicators_gauss23 is not None:
            _add_gauss_trace(indicators_gauss23, "g23", 23, SHOW_IND_GAUSS23_DEFAULT, "IND_gauss_23")
        if HAVE_G38 and indicators_gauss38 is not None:
            _add_gauss_trace(indicators_gauss38, "g38", 38, SHOW_IND_GAUSS38_DEFAULT, "IND_gauss_38")
        if HAVE_G53 and indicators_gauss53 is not None:
            _add_gauss_trace(indicators_gauss53, "g53", 53, SHOW_IND_GAUSS53_DEFAULT, "IND_gauss_53")
        if HAVE_G68 and indicators_gauss68 is not None:
            _add_gauss_trace(indicators_gauss68, "g68", 68, SHOW_IND_GAUSS68_DEFAULT, "IND_gauss_68")
        if HAVE_G83 and indicators_gauss83 is not None:
            _add_gauss_trace(indicators_gauss83, "g83", 83, SHOW_IND_GAUSS83_DEFAULT, "IND_gauss_83")

        # Preserve trace visibility state (including indicator toggles)
        if prev_fig and "data" in prev_fig:
            for i, trace in enumerate(traces):
                if i < len(prev_fig["data"]) and "visible" in prev_fig["data"][i]:
                    trace.visible = prev_fig["data"][i]["visible"]

        for trace in traces:
            fig.add_trace(trace)

        est = pytz.timezone("America/New_York")
        now_est = datetime.datetime.now(est)

        fig.update_layout(
            xaxis=dict(
                title="",
                showgrid=True,
                showticklabels=True,
                gridcolor="white",
                range=list(compute_x_range(x_window_minutes)),
                tickformat="%I:%M %p",
            ),
            yaxis=dict(title="Price", tickformat=".2f", gridcolor="white"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", gridcolor="white"),

            # axis for IND_Volume / CVD
            yaxis3=dict(
                title="IND_Volume",
                overlaying="y",
                side="right",
                showgrid=False,
                position=0.98,
            ),

            legend=dict(x=0, y=1),
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor="rgba(81,81,81,1)",
            plot_bgcolor="rgba(150,150,150,1)",
            font=dict(color="white"),
        )

        # Epoch markers
        try:
            if markers:
                shapes = []
                annotations = []
                for m in markers:
                    x = m.get("ts")
                    ep = m.get("epoch")
                    if not x or ep is None:
                        continue

                    shapes.append(
                        dict(
                            type="line",
                            xref="x",
                            yref="paper",
                            x0=x,
                            x1=x,
                            y0=0,
                            y1=1,
                            line=dict(width=2, dash="dash"),
                        )
                    )
                    annotations.append(
                        dict(
                            x=x,
                            y=1.02,
                            xref="x",
                            yref="paper",
                            text=f"{ep}",
                            showarrow=False,
                            font=dict(size=11, color="white"),
                            align="center",
                        )
                    )
                fig.update_layout(shapes=shapes, annotations=annotations)
        except Exception as e:
            log_error(f"Error adding epoch markers: {e}")

        # Preserve zoom level
        if relayoutData and "xaxis.range[0]" in relayoutData and "xaxis.range[1]" in relayoutData:
            fig.update_layout(xaxis=dict(range=[relayoutData["xaxis.range[0]"], relayoutData["xaxis.range[1]"]]))

        return fig

    except Exception as e:
        log_error(f"Error updating graph: {e}")
        return go.Figure()


if __name__ == "__main__":
    log_info("Starting server...")
    app.run(debug=True, use_reloader=False)
