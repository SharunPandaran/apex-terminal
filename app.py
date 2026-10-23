"""
Apex Institutional Terminal  (v2)
Run:   streamlit run app.py
Token: put UPSTOX_ACCESS_TOKEN in .streamlit/secrets.toml  (or env var, or paste in sidebar)
"""
import datetime as dt
import gzip
import json
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
import ta
import yfinance as yf
from bs4 import BeautifulSoup
from plotly.subplots import make_subplots

IST = ZoneInfo("Asia/Kolkata")

st.set_page_config(
    page_title="Apex Institutional Terminal | Trade Signal Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# THEME
# ==============================================================================
THEMES = {
    "light": dict(
        bg_main="#f8fafc", bg_card="#ffffff", bg_sub="#f1f5f9", border="#e2e8f0",
        text_main="#0f172a", text_sub="#475569", accent="#0284c7",
        stat_bg="#e0f2fe", chart_bg="#ffffff", grid="#f1f5f9", plotly="plotly_white",
        up="#059669", down="#dc2626", warn="#d97706",
    ),
    "dark": dict(
        bg_main="#0e131f", bg_card="#151d30", bg_sub="#1c263e", border="#283756",
        text_main="#ffffff", text_sub="#94a3b8", accent="#38bdf8",
        stat_bg="rgba(56,189,248,0.2)", chart_bg="#111728", grid="#1e293b", plotly="plotly_dark",
        up="#34d399", down="#f87171", warn="#fbbf24",
    ),
}

st.sidebar.markdown("### 🎨 Terminal Interface")
_qp = st.query_params
theme_mode = st.sidebar.radio(
    "Display theme", ["Light", "Dark"],
    index=1 if _qp.get("theme") == "dark" else 0, horizontal=True,
)
theme_key = "light" if theme_mode == "Light" else "dark"
T = THEMES[theme_key]
is_light = theme_key == "light"
st.query_params["theme"] = theme_key  # survives page reloads (ribbon click)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@600;800&family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color:@bg_main@ !important; font-family:'Plus Jakarta Sans',sans-serif !important; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 { color:@text_main@ !important; }
.block-container { padding-top:1.2rem !important; padding-bottom:2rem !important; max-width:98% !important; }
[data-testid="stSidebar"] { background-color:@bg_card@ !important; border-right:1.5px solid @border@ !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background-color:@bg_card@; border-color:@border@ !important; border-radius:12px; }
.terminal-header { font-size:26px; font-weight:800; letter-spacing:-0.5px; margin-bottom:2px; }
.terminal-subtitle { font-size:13px; font-weight:600; color:@text_sub@; margin-bottom:16px; }
.hud-card { background:@bg_card@; border:1.5px solid @border@; border-top:3.5px solid @accent@; border-radius:14px;
    padding:22px 24px; margin:14px 0 18px 0; box-shadow:0 4px 20px rgba(0,0,0,.05); }
.hud-grid { margin-top:18px; display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:14px; }
.cell { background:@bg_sub@; border:1px solid @border@; border-radius:10px; padding:12px 14px; }
.cell-l { font-size:11px; color:@text_sub@; font-weight:800; margin-bottom:4px; text-transform:uppercase; }
.cell-v { font-size:16px; font-family:'JetBrains Mono',monospace; font-weight:800; }
.pill { display:inline-flex; align-items:center; padding:8px 18px; border-radius:9999px; font-size:15px; font-weight:800; border:2px solid; }
.pill-buy { background:#ecfdf5; border-color:#10b981; color:#047857; }
.pill-sell { background:#fef2f2; border-color:#ef4444; color:#b91c1c; }
.pill-neutral { background:#fffbeb; border-color:#f59e0b; color:#b45309; }
[data-testid="stMetric"] { background:@bg_card@ !important; border:1.5px solid @border@ !important; border-radius:10px !important; padding:12px 16px !important; }
[data-testid="stMetricLabel"] p { font-size:11px !important; letter-spacing:.4px; color:@text_sub@ !important; font-weight:800 !important; text-transform:uppercase; }
[data-testid="stMetricValue"] div { font-family:'JetBrains Mono',monospace !important; font-size:17px !important; font-weight:800 !important; }
.tag { font-family:'JetBrains Mono',monospace; font-weight:800; padding:3px 8px; border-radius:5px; border:1px solid; }
.tag-stat { color:@accent@; background:@stat_bg@; border-color:@accent@; }
.tag-up { color:#059669; background:#d1fae5; border-color:#10b981; }
.tag-down { color:#dc2626; background:#fee2e2; border-color:#ef4444; }
.plan-lines { font-size:14px; line-height:2.1; }
.sec-title { font-size:17px; font-weight:800; margin-bottom:10px; }
</style>
"""
for _k, _v in T.items():
    CSS = CSS.replace(f"@{_k}@", _v)
st.markdown(CSS, unsafe_allow_html=True)

# ==============================================================================
# SECRETS / TOKEN  (never hard-code the token in source!)
# ==============================================================================
def _secret_token() -> str:
    try:
        tok = st.secrets.get("UPSTOX_ACCESS_TOKEN", "")
    except Exception:
        tok = ""
    return tok or os.getenv("UPSTOX_ACCESS_TOKEN", "")


UPSTOX_TOKEN = (
    st.sidebar.text_input("Upstox token (optional override)", type="password",
                          help="Upstox tokens expire every day. Better: keep it in .streamlit/secrets.toml")
    or _secret_token()
)

# ==============================================================================
# INSTRUMENT MASTER  (NSE + BSE equities, F&O lot sizes)
# ==============================================================================
FALLBACK_STOCKS = [
    ("RELIANCE", "Reliance Industries Ltd", "NSE_EQ|INE002A01018"),
    ("TCS", "Tata Consultancy Services", "NSE_EQ|INE467B01029"),
    ("HDFCBANK", "HDFC Bank Ltd", "NSE_EQ|INE040A01034"),
    ("INFY", "Infosys Ltd", "NSE_EQ|INE009A01021"),
    ("SUZLON", "Suzlon Energy Ltd", "NSE_EQ|INE040H01021"),
    ("IRFC", "Indian Railway Finance Corp", "NSE_EQ|INE053F01010"),
    ("EICHERMOT", "Eicher Motors Ltd", "NSE_EQ|INE066A01021"),
]


def _gunzip_json(content: bytes):
    try:
        content = gzip.decompress(content)
    except OSError:  # server already decompressed it
        pass
    return json.loads(content)


@st.cache_data(ttl=86400, show_spinner="Loading NSE/BSE instrument master…")
def load_instruments():
    frames = []
    for ex in ("NSE", "BSE"):
        try:
            r = requests.get(f"https://assets.upstox.com/market-quote/instruments/exchange/{ex}.json.gz", timeout=40)
            r.raise_for_status()
            frames.append(pd.DataFrame(_gunzip_json(r.content)))
        except Exception:
            continue

    if not frames:
        fb = pd.DataFrame(
            [dict(trading_symbol=s, name=n, exchange="NSE", instrument_key=k, exchange_token="") for s, n, k in FALLBACK_STOCKS]
        )
        fb["search_label"] = fb["trading_symbol"] + " — " + fb["name"] + " [NSE]"
        return fb, {}, False

    raw = pd.concat(frames, ignore_index=True)

    eq = raw[raw["segment"].isin(["NSE_EQ", "BSE_EQ"])].copy()
    eq = eq[(eq["segment"] == "BSE_EQ") | (eq["instrument_type"] == "EQ")]
    eq = eq.dropna(subset=["trading_symbol"])
    eq["name"] = eq["name"].fillna(eq["trading_symbol"])
    eq["exchange"] = np.where(eq["segment"] == "NSE_EQ", "NSE", "BSE")
    eq["search_label"] = eq["trading_symbol"].astype(str) + " — " + eq["name"].astype(str) + " [" + eq["exchange"] + "]"
    eq["_rank"] = (eq["exchange"] != "NSE").astype(int)  # NSE first
    eq = eq.sort_values(["_rank", "trading_symbol"]).drop_duplicates("search_label").reset_index(drop=True)
    cols = ["trading_symbol", "name", "exchange", "instrument_key", "exchange_token", "search_label"]
    eq = eq[[c for c in cols if c in eq.columns]]
    if "exchange_token" not in eq.columns:
        eq["exchange_token"] = ""

    lots = {}
    try:
        fo = raw[(raw["segment"] == "NSE_FO") & (raw["instrument_type"] == "FUT")].sort_values("expiry")
        lots = fo.groupby("underlying_symbol")["lot_size"].first().astype(int).to_dict()
    except Exception:
        pass
    return eq, lots, True


ALL_STOCKS, FNO_LOTS, MASTER_OK = load_instruments()
if not MASTER_OK:
    st.sidebar.warning("Upstox instrument master load aayilla — 7 stocks mathram (offline list).")

GLOBAL_ASSETS = {"GOLD (XAU/USD)": "GC=F", "EUR/USD": "EURUSD=X", "BITCOIN (BTC)": "BTC-USD"}
INDICES = {
    "NIFTY 50": dict(yf="^NSEI", upstox="NSE_INDEX|Nifty 50", step=50, fno="NIFTY"),
    "BANK NIFTY": dict(yf="^NSEBANK", upstox="NSE_INDEX|Nifty Bank", step=100, fno="BANKNIFTY"),
    "BSE SENSEX": dict(yf="^BSESN", upstox="BSE_INDEX|SENSEX", step=100, fno="SENSEX"),
    "FINNIFTY": dict(yf="NIFTY_FIN_SERVICE.NS", upstox="NSE_INDEX|Nifty Fin Service", step=50, fno="FINNIFTY"),
}
ALIASES = {"SENSEX": "BSE SENSEX"}

INTERVALS = {
    "1 Minute (Live)": dict(yf="1m", unit="minutes", n=1, yf_period="5d", days_back=5, mins=1),
    "5 Minutes": dict(yf="5m", unit="minutes", n=5, yf_period="1mo", days_back=20, mins=5),
    "15 Minutes": dict(yf="15m", unit="minutes", n=15, yf_period="60d", days_back=30, mins=15),
    "30 Minutes": dict(yf="30m", unit="minutes", n=30, yf_period="60d", days_back=30, mins=30),
    "1 Day (Daily)": dict(yf="1d", unit="days", n=1, yf_period="2y", days_back=500, mins=1440),
}

# ==============================================================================
# MARKET DATA LAYER
# ==============================================================================
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna(subset=["Close"]).copy()
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df[~df.index.duplicated(keep="last")].sort_index()


@st.cache_data(ttl=45, show_spinner=False)
def fetch_yf(sym: str, period: str, interval: str) -> pd.DataFrame:
    try:
        return _clean(yf.download(sym, period=period, interval=interval, progress=False, auto_adjust=True))
    except Exception:
        return pd.DataFrame()


def _upstox_get(url: str, token: str):
    r = requests.get(url, headers={"Accept": "application/json", "Authorization": f"Bearer {token}"}, timeout=8)
    if r.status_code == 401:
        raise PermissionError("Upstox token expired / invalid")
    r.raise_for_status()
    return r.json().get("data", {}).get("candles", [])


def _candles_df(candles) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    d = pd.DataFrame(candles).iloc[:, :6]
    d.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date").sort_index().astype(float)
    if d.index.tz is not None:
        d.index = d.index.tz_convert(IST)
    return d


@st.cache_data(ttl=20, show_spinner=False)
def fetch_upstox(inst_key: str, unit: str, n: int, days_back: int, token: str) -> pd.DataFrame:
    """History (up to yesterday) + today's intraday candles, via Upstox v3 API."""
    k = quote(inst_key, safe="")
    base = "https://api.upstox.com/v3/historical-candle"
    today = dt.date.today()
    frames = []
    try:
        url = f"{base}/{k}/{unit}/{n}/{today - dt.timedelta(days=1)}/{today - dt.timedelta(days=days_back)}"
        frames.append(_candles_df(_upstox_get(url, token)))
    except PermissionError:
        raise
    except Exception:
        pass
    try:
        frames.append(_candles_df(_upstox_get(f"{base}/intraday/{k}/{unit}/{n}", token)))
    except PermissionError:
        raise
    except Exception:
        pass
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames)
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_candles(inst_key, yf_syms, cfg):
    notes = []
    if UPSTOX_TOKEN and inst_key:
        try:
            d = fetch_upstox(inst_key, cfg["unit"], cfg["n"], cfg["days_back"], UPSTOX_TOKEN)
            if len(d) >= 30:
                return d, "Upstox v3 API", notes
        except PermissionError as e:
            notes.append(f"⚠️ {e} — yfinance fallback use cheyyunnu.")
        except Exception:
            pass
    for sym in yf_syms:
        for period in (cfg["yf_period"], "5d" if cfg["yf"] != "1d" else "1y"):
            d = fetch_yf(sym, period, cfg["yf"])
            if len(d) >= 30:
                return d, f"yfinance ({sym})", notes
    return pd.DataFrame(), "none", notes


@st.cache_data(ttl=600, show_spinner=False)
def prev_session_hlc(sym: str):
    d = fetch_yf(sym, "1mo", "1d")
    if len(d) < 2:
        return None
    last_date = d.index[-1].date()
    row = d.iloc[-2] if last_date >= dt.datetime.now(IST).date() else d.iloc[-1]
    return float(row["High"]), float(row["Low"]), float(row["Close"])


# ==============================================================================
# TOP RIBBON + PRE-MARKET RADAR  (batched downloads → fast)
# ==============================================================================
RIBBON = {
    "NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN", "FINNIFTY": "NIFTY_FIN_SERVICE.NS",
    "GOLD (XAU/USD)": "GC=F", "BITCOIN (BTC)": "BTC-USD", "EUR/USD": "EURUSD=X",
}


@st.cache_data(ttl=30, show_spinner=False)
def ribbon_data():
    out = []
    try:
        raw = yf.download(list(RIBBON.values()), period="5d", interval="1d", progress=False,
                          group_by="ticker", auto_adjust=True, threads=True)
    except Exception:
        return out
    for label, sym in RIBBON.items():
        try:
            c = raw[sym]["Close"].dropna()
            ltp, prev = float(c.iloc[-1]), float(c.iloc[-2])
            out.append((label, ltp, (ltp - prev) / prev * 100))
        except Exception:
            continue
    return out


def render_ribbon():
    items = ribbon_data()
    if not items:
        return
    badges = ""
    for label, ltp, pct in items:
        up = pct >= 0
        if "EUR" in label:
            p = f"{ltp:.4f}"
        elif "BTC" in label:
            p = f"${ltp:,.0f}"
        elif "GOLD" in label:
            p = f"${ltp:,.2f}"
        else:
            p = f"₹{ltp:,.2f}"
        badges += (
            f'<span class="ti" onclick="openAsset(\'{label}\')"><span class="tn">{label}</span>'
            f'<span class="tp">{p}</span><span class="tc" style="color:{"#059669" if up else "#dc2626"};'
            f'background:{"#d1fae5" if up else "#fee2e2"}">{"▲" if up else "▼"} {pct:+.2f}%</span></span>'
        )
    html = """
    <html><head><style>
    body{margin:0;overflow:hidden;background:transparent;font-family:'Plus Jakarta Sans',sans-serif}
    .wrap{overflow:hidden;background:@bg_card@;border:1.5px solid @border@;border-radius:10px;padding:8px 0;white-space:nowrap}
    .move{display:inline-block;animation:m 30s linear infinite}.move:hover{animation-play-state:paused}
    @keyframes m{0%{transform:translate3d(0,0,0)}100%{transform:translate3d(-50%,0,0)}}
    .ti{display:inline-flex;align-items:center;margin:0 14px;padding:5px 14px;background:@bg_sub@;border:1px solid @border@;border-radius:7px;cursor:pointer}
    .tn{color:@text_main@;font-weight:800;font-size:12.5px;margin-right:8px}
    .tp{color:@accent@;font-weight:800;font-family:monospace;font-size:13px;margin-right:8px}
    .tc{font-weight:800;font-size:11.5px;padding:2px 7px;border-radius:4px}
    </style><script>
    function openAsset(n){try{const p=new URLSearchParams(window.parent.location.search);
      p.set('select_asset',n);p.set('theme','@theme@');window.parent.location.search=p.toString();}catch(e){}}
    </script></head><body><div class="wrap"><div class="move">@badges@@badges@</div></div></body></html>
    """
    for k, v in {**T, "theme": theme_key, "badges": badges}.items():
        html = html.replace(f"@{k}@", str(v))
    components.html(html, height=58)


@st.cache_data(ttl=600, show_spinner=False)
def premarket_table():
    idx = {"NIFTY 50": "^NSEI", "BANK NIFTY": "^NSEBANK", "SENSEX": "^BSESN", "NIFTY IT": "^CNXIT", "INDIA VIX": "^INDIAVIX"}
    try:
        raw = yf.download(list(idx.values()), period="3mo", interval="1d", progress=False, group_by="ticker", auto_adjust=True)
    except Exception:
        return pd.DataFrame()
    rows = []
    for name, sym in idx.items():
        try:
            d = raw[sym].dropna(subset=["Close"])
            ltp, prev = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
            chg = (ltp - prev) / prev * 100
            if name == "INDIA VIX":
                rows.append(dict(Index=name, LTP=f"{ltp:.2f}", Change=f"{chg:+.2f}%",
                                 Trend="⚠️ High Volatility" if ltp > 16 else "🟢 Cool / Stable", Support="-", Resistance="-"))
                continue
            e20 = float(d["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
            e50 = float(d["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            trend = "🟢 Strong Bullish" if ltp > e20 > e50 else ("🔴 Strong Bearish" if ltp < e20 < e50 else "🟡 Neutral / Sideways")
            rows.append(dict(Index=name, LTP=f"₹{ltp:,.2f}", Change=f"{chg:+.2f}%", Trend=trend,
                             Support=f"₹{d['Low'].tail(20).min():,.0f}", Resistance=f"₹{d['High'].tail(20).max():,.0f}"))
        except Exception:
            continue
    return pd.DataFrame(rows)


# ==============================================================================
# SCREENER.IN
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screener(symbol: str):
    sym = symbol.replace(".NS", "").replace(".BO", "").strip().upper()
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122 Safari/537.36"})
    resp = None
    for url in (f"https://www.screener.in/company/{sym}/consolidated/", f"https://www.screener.in/company/{sym}/"):
        try:
            r = s.get(url, timeout=6)
            if r.status_code == 200 and "Search results" not in r.text:
                resp = r
                break
        except Exception:
            continue
    if resp is None:
        try:
            sr = s.get(f"https://www.screener.in/api/company/search/?q={sym}", timeout=5)
            if sr.status_code == 200 and sr.json():
                resp = s.get("https://www.screener.in" + sr.json()[0]["url"], timeout=6)
        except Exception:
            pass
    if resp is None or resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.content, "html.parser")
    data = {"ratios": {}, "quarterly": None, "pros": [], "cons": []}
    for li in soup.select("#top-ratios li"):
        n, v = li.select_one(".name"), li.select_one(".number")
        if n and v:
            data["ratios"][n.get_text(strip=True)] = v.get_text(strip=True)

    sec = soup.find("section", id="quarters")
    table = sec.find("table") if sec else None
    if table:
        head = [th.get_text(strip=True) for th in table.find("thead").find_all("th")] if table.find("thead") else []
        rows = []
        tbody = table.find("tbody")
        for tr in (tbody.find_all("tr") if tbody else []):
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if cells:
                cells[0] = cells[0].rstrip("+ ").strip()  # "Sales +" → "Sales" (does NOT touch hyphens)
                rows.append(cells)
        if rows and head:
            dfq = pd.DataFrame(rows)
            if head and not head[0]:
                head[0] = "Metric"
            if len(dfq.columns) == len(head):
                dfq.columns = head
            data["quarterly"] = dfq
    data["pros"] = [p.get_text(strip=True) for p in soup.select(".pros ul li")]
    data["cons"] = [c.get_text(strip=True) for c in soup.select(".cons ul li")]
    return data


# ==============================================================================
# QUANT ENGINE
# ==============================================================================
def supertrend(df: pd.DataFrame, period=10, mult=3.0):
    """Standard SuperTrend with proper final-band logic (numpy, fast)."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    n = len(df)
    atr = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=period).values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * atr, hl2 - mult * atr
    fub, flb = ub.copy(), lb.copy()
    line = np.full(n, np.nan)
    d = np.zeros(n, dtype=int)
    if n <= period:
        return pd.Series(line, index=df.index), pd.Series(d, index=df.index)
    s = period - 1
    d[s] = 1 if c[s] >= hl2[s] else -1
    line[s] = flb[s] if d[s] == 1 else fub[s]
    for i in range(s + 1, n):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == -1 and c[i] > fub[i - 1]:
            d[i] = 1
        elif d[i - 1] == 1 and c[i] < flb[i - 1]:
            d[i] = -1
        else:
            d[i] = d[i - 1]
        line[i] = flb[i] if d[i] == 1 else fub[i]
    return pd.Series(line, index=df.index), pd.Series(d, index=df.index)


def session_vwap(df: pd.DataFrame, intraday: bool):
    """VWAP that resets every day. Not meaningful for daily bars / no-volume feeds."""
    if not intraday or df["Volume"].sum() <= 0:
        return None
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = pd.Series(df.index.date, index=df.index)
    return (tp * df["Volume"]).groupby(day).cumsum() / df["Volume"].groupby(day).cumsum().replace(0, np.nan)


def add_indicators(df: pd.DataFrame, intraday: bool):
    c = df["Close"]
    for w in (20, 50, 200):
        df[f"EMA_{w}"] = c.ewm(span=w, adjust=False).mean()
    df["RSI"] = ta.momentum.rsi(c, window=14)
    m = ta.trend.MACD(c)
    df["MACD"], df["MACD_Signal"], df["MACD_Diff"] = m.macd(), m.macd_signal(), m.macd_diff()
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["BB_Upper"], df["BB_Lower"] = bb.bollinger_hband(), bb.bollinger_lband()
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], c, window=14)
    df["ST"], df["ST_Dir"] = supertrend(df)
    v = session_vwap(df, intraday)
    df["VWAP"] = v if v is not None else np.nan
    return df


def swing_points(sub: pd.DataFrame, k=3):
    hi, lo = sub["High"].values, sub["Low"].values
    sh, sl = [], []
    for i in range(k, len(sub) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            sh.append(i)
        if lo[i] == lo[i - k:i + k + 1].min():
            sl.append(i)
    return sh, sl


def detect_patterns(df: pd.DataFrame, atr: float):
    out = []
    if len(df) < 30:
        return out
    sub = df.tail(120)
    hi, lo, cl = sub["High"].values, sub["Low"].values, sub["Close"].values
    sh, sl = swing_points(sub)
    n = len(sub)

    # --- Double Bottom (two real swing lows, similar level, meaningful bounce between them)
    if len(sl) >= 2:
        i1, i2 = sl[-2], sl[-1]
        l1, l2 = lo[i1], lo[i2]
        if i2 - i1 >= 6 and n - 1 - i2 <= 40 and abs(l1 - l2) <= max(0.6 * atr, 0.002 * l1):
            neck = hi[i1:i2 + 1].max()
            if neck - max(l1, l2) >= 1.5 * atr:
                ok = cl[-1] > neck
                out.append(dict(
                    name="Double Bottom (W-Pattern)", type="Bullish Reversal", idx=sub.index[i2], price=float(l2),
                    explanation="വില രണ്ടുതവണ ഒരേ സപ്പോർട്ട് ലെവലിൽ തട്ടി മുകളിലേക്ക് പോയി. ശക്തമായ Bullish Reversal സൂചന.",
                    note=f"Neckline {neck:,.2f} — " + ("ബ്രേക്ക്ഔട്ട് സ്ഥിരീകരിച്ചു ✅" if ok else "ബ്രേക്ക്ഔട്ടിനായി കാത്തിരിക്കുക ⏳")))
    # --- Double Top
    if len(sh) >= 2:
        i1, i2 = sh[-2], sh[-1]
        h1, h2 = hi[i1], hi[i2]
        if i2 - i1 >= 6 and n - 1 - i2 <= 40 and abs(h1 - h2) <= max(0.6 * atr, 0.002 * h1):
            neck = lo[i1:i2 + 1].min()
            if min(h1, h2) - neck >= 1.5 * atr:
                ok = cl[-1] < neck
                out.append(dict(
                    name="Double Top (M-Pattern)", type="Bearish Reversal", idx=sub.index[i2], price=float(h2),
                    explanation="വില രണ്ടുതവണ ഒരേ റെസിസ്റ്റൻസ് കടക്കാൻ കഴിയാതെ തിരിഞ്ഞു. ശക്തമായ Bearish Reversal സൂചന.",
                    note=f"Neckline {neck:,.2f} — " + ("ബ്രേക്ക്ഡൗൺ സ്ഥിരീകരിച്ചു ✅" if ok else "ബ്രേക്ക്ഡൗണിനായി കാത്തിരിക്കുക ⏳")))

    # --- Candlestick patterns on the latest bar
    o, h, l, c = (df[x].iloc[-1] for x in ("Open", "High", "Low", "Close"))
    po, pc = df["Open"].iloc[-2], df["Close"].iloc[-2]
    body, rng = abs(c - o), h - l
    up_w, lo_w = h - max(o, c), min(o, c) - l
    idx = df.index[-1]
    if pc < po and c > o and c >= po and o <= pc and body > abs(pc - po):
        out.append(dict(name="Bullish Engulfing", type="Bullish Continuation/Reversal", idx=idx, price=float(l),
                        explanation="പച്ച കാൻഡിൽ മുൻപത്തെ ചുവപ്പ് കാൻഡിലിനെ പൂർണ്ണമായി വിഴുങ്ങി. ബയർമാർ ശക്തമായി തിരിച്ചെത്തി.", note=""))
    elif pc > po and c < o and c <= po and o >= pc and body > abs(pc - po):
        out.append(dict(name="Bearish Engulfing", type="Bearish Continuation/Reversal", idx=idx, price=float(h),
                        explanation="ചുവപ്പ് കാൻഡിൽ മുൻപത്തെ പച്ച കാൻഡിലിനെ കവർ ചെയ്തു. സെല്ലർമാർ നിയന്ത്രണം ഏറ്റെടുത്തു.", note=""))
    if rng > 0.3 * atr:
        if body <= 0.35 * rng and lo_w >= 2 * body and up_w <= 0.2 * rng:
            out.append(dict(name="Hammer (Bullish Pin Bar)", type="Bullish Reversal", idx=idx, price=float(l),
                            explanation="താഴേക്ക് പോയ വിലയെ ബയർമാർ ശക്തമായി മുകളിലേക്ക് തള്ളി. താഴത്തെ ലെവലിൽ ഡിമാൻഡ് ഉണ്ട്.", note=""))
        elif body <= 0.35 * rng and up_w >= 2 * body and lo_w <= 0.2 * rng:
            out.append(dict(name="Shooting Star (Bearish Pin Bar)", type="Bearish Reversal", idx=idx, price=float(h),
                            explanation="മുകളിലേക്ക് കയറിയ വില ശക്തമായ സെല്ലിംഗിൽ താഴേക്ക് വന്നു. റെസിസ്റ്റൻസിൽ സപ്ലൈ ഉണ്ട്.", note=""))
        if body / rng < 0.1:
            out.append(dict(name="Doji (Indecision)", type="Neutral / Alert", idx=idx, price=float(c),
                            explanation="ഓപ്പണും ക്ലോസും ഏകദേശം ഒരേ വില. ബയർ-സെല്ലർ തുല്യശക്തി; അടുത്ത കാൻഡിലിന്റെ ബ്രേക്ക്ഔട്ട് നോക്കുക.", note=""))
    return out


def score_signal(price, df, ema200_ok):
    last = df.iloc[-1]
    rows = []
    if ema200_ok:
        if price > last["EMA_200"] and last["EMA_50"] > last["EMA_200"]:
            rows.append(("Primary trend (EMA 50/200)", +2, "Price & EMA50 above EMA200"))
        elif price < last["EMA_200"] and last["EMA_50"] < last["EMA_200"]:
            rows.append(("Primary trend (EMA 50/200)", -2, "Price & EMA50 below EMA200"))
        else:
            rows.append(("Primary trend (EMA 50/200)", 0, "Mixed"))
    else:
        rows.append(("Primary trend (EMA 50/200)", 0, "Not enough bars for EMA200 (<200)"))
    rows.append(("Short trend (EMA 20)", +1 if price > last["EMA_20"] else -1, "Price above EMA20" if price > last["EMA_20"] else "Price below EMA20"))
    rows.append(("MACD", +1 if last["MACD"] > last["MACD_Signal"] else -1, "MACD above signal" if last["MACD"] > last["MACD_Signal"] else "MACD below signal"))
    rsi = last["RSI"]
    if rsi < 35:
        rows.append(("RSI(14)", +1, f"Oversold ({rsi:.0f})"))
    elif rsi > 70:
        rows.append(("RSI(14)", -1, f"Overbought ({rsi:.0f})"))
    else:
        rows.append(("RSI(14)", 0, f"Balanced ({rsi:.0f})"))
    rows.append(("SuperTrend", +1 if last["ST_Dir"] == 1 else -1, "Bullish" if last["ST_Dir"] == 1 else "Bearish"))
    return sum(r[1] for r in rows), rows


def build_plan(score, price, atr, r1, r2):
    if score >= 3:
        p = dict(label="STRONG BUY / LONG", cls="buy", icon="🟢", action="BUY / LONG ACCUMULATION", bias=1,
                 sl=price - 1.75 * atr, rr=(1.5, 2.5), strat="BULL CALL SPREAD / LONG CE", opt="CE")
    elif score >= 1:
        p = dict(label="BUY ON DIPS", cls="buy", icon="🟢", action="MILD BUY / PULLBACK ENTRY", bias=1,
                 sl=price - 1.5 * atr, rr=(1.5, 2.0), strat="LONG CE (ATM) OR BULL PUT SPREAD", opt="CE")
    elif score == 0:
        p = dict(label="NEUTRAL / WAIT", cls="neutral", icon="🟡", action="NO DIRECT TRADE (RANGE)", bias=0,
                 sl=price - atr, rr=None, strat="IRON CONDOR / STRANGLE", opt="")
    elif score >= -2:
        p = dict(label="SELL ON RISE", cls="sell", icon="🔴", action="MILD SHORT / SELL ON PULLBACK", bias=-1,
                 sl=price + 1.5 * atr, rr=(1.5, 2.0), strat="LONG PE (ATM) OR BEAR CALL SPREAD", opt="PE")
    else:
        p = dict(label="STRONG SELL / SHORT", cls="sell", icon="🔴", action="SHORT / DIRECT EXIT", bias=-1,
                 sl=price + 1.75 * atr, rr=(1.5, 2.5), strat="BEAR PUT SPREAD / LONG PE", opt="PE")
    risk = abs(price - p["sl"])
    if p["rr"]:
        p["t1"], p["t2"] = price + p["bias"] * p["rr"][0] * risk, price + p["bias"] * p["rr"][1] * risk
        p["t1_label"], p["t2_label"] = f"TARGET 1 (R:R {p['rr'][0]})", f"TARGET 2 (R:R {p['rr'][1]})"
    else:
        p["t1"], p["t2"], p["t1_label"], p["t2_label"] = r1, r2, "RANGE HIGH (R1)", "BREAKOUT (R2)"
    p["risk"] = risk
    return p


def stock_strike_step(p):
    for lim, step in ((100, 2.5), (500, 5), (1000, 10), (2500, 20), (5000, 50)):
        if p < lim:
            return step
    return 100


def fs(x):  # strike formatter
    return f"{x:.2f}".rstrip("0").rstrip(".")


# ==============================================================================
# HEADER + RIBBON + RADAR
# ==============================================================================
st.markdown('<div class="terminal-header">🏛️ Institutional Equity & Derivatives Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="terminal-subtitle">Full-universe NSE/BSE equities • Pattern recognition • Previous-session Pivot S/R • Risk-sized trade plans</div>', unsafe_allow_html=True)
render_ribbon()

with st.expander("🔔 Daily Morning Pre-Open Radar & Market Outlook", expanded=False):
    radar = premarket_table()
    if not radar.empty:
        st.dataframe(radar, use_container_width=True, hide_index=True)

# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
if "select_asset" in st.query_params:
    a = st.query_params["select_asset"]
    st.session_state["_asset_pref"] = ALIASES.get(a, a)
    del st.query_params["select_asset"]
pref = st.session_state.get("_asset_pref", "NIFTY 50")

st.sidebar.markdown("### ⚡ Terminal Navigation")
class_idx = 0 if pref in GLOBAL_ASSETS else (1 if pref in INDICES else 2)
asset_type = st.sidebar.radio(
    "Asset Class",
    ["Global Forex & Spot Gold", "Indian Indices (NSE / BSE)", "Indian Equities (All NSE / BSE Stocks)"],
    index=class_idx, key="asset_class",
)

inst_key, yf_syms, step_fixed, fno_name = None, [], None, None
is_equity = False
stock_row = None

if asset_type.startswith("Global"):
    opts = list(GLOBAL_ASSETS)
    display_name = st.sidebar.selectbox("Select Asset:", opts, index=opts.index(pref) if pref in opts else 0)
    yf_syms = ["GC=F", "XAUUSD=X"] if "GOLD" in display_name else [GLOBAL_ASSETS[display_name]]
elif asset_type.startswith("Indian Indices"):
    opts = list(INDICES)
    display_name = st.sidebar.selectbox("Select Index:", opts, index=opts.index(pref) if pref in opts else 0)
    meta = INDICES[display_name]
    inst_key, yf_syms, step_fixed, fno_name = meta["upstox"], [meta["yf"]], meta["step"], meta["fno"]
else:
    is_equity = True
    labels = ALL_STOCKS["search_label"].tolist()
    want = pref if pref not in GLOBAL_ASSETS and pref not in INDICES else "RELIANCE"
    hit = ALL_STOCKS.index[(ALL_STOCKS["trading_symbol"] == want) & (ALL_STOCKS["exchange"] == "NSE")]
    default_i = int(hit[0]) if len(hit) else 0
    label = st.sidebar.selectbox("🔍 Search any stock (symbol / company):", labels, index=default_i)
    stock_row = ALL_STOCKS[ALL_STOCKS["search_label"] == label].iloc[0]
    display_name = stock_row["trading_symbol"]
    inst_key = stock_row["instrument_key"]
    if stock_row["exchange"] == "NSE":
        yf_syms = [f"{display_name}.NS"]
    else:  # Yahoo uses the BSE scrip code, not the symbol
        code = str(stock_row.get("exchange_token", "") or "")
        yf_syms = [f"{code}.BO"] if code else [f"{display_name}.BO"]
    fno_name = display_name

st.sidebar.markdown("---")
interval_label = st.sidebar.selectbox("Chart Resolution", list(INTERVALS), index=1)
cfg = INTERVALS[interval_label]
intraday = cfg["yf"] != "1d"
bars_shown = st.sidebar.slider("Bars to display", 60, 500, 200, 20)

st.sidebar.markdown("### 📈 Overlays")
show_st = st.sidebar.checkbox("SuperTrend (10, 3)", True)
show_vwap = st.sidebar.checkbox("VWAP (session)", True)
show_piv = st.sidebar.checkbox("Pivot S/R (previous session)", True)
show_bb = st.sidebar.checkbox("Bollinger Bands", False)

st.sidebar.markdown("### 💼 Risk Manager")
capital = st.sidebar.number_input("Capital (₹)", 10000, 100000000, 200000, 10000)
risk_pct = st.sidebar.slider("Risk per trade (%)", 0.25, 3.0, 1.0, 0.25)

st.sidebar.markdown("---")
load_fund = st.sidebar.checkbox("Load Screener.in fundamentals", True)
refresh_s = st.sidebar.selectbox("Auto-refresh", [0, 15, 30, 60], format_func=lambda x: "Off" if x == 0 else f"{x}s")
if refresh_s:
    try:
        from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
        st_autorefresh(interval=refresh_s * 1000, key="auto_refresh")
    except ImportError:
        st.sidebar.caption("pip install streamlit-autorefresh  (auto-refresh-inu)")

# ==============================================================================
# LOAD DATA
# ==============================================================================
with st.spinner(f"Connecting to feed for {display_name}…"):
    df, data_source, notes = load_candles(inst_key, yf_syms, cfg)
for n_ in notes:
    st.warning(n_)
if df.empty:
    st.error(f"⚠️ '{display_name}' ന് ഡാറ്റ കിട്ടിയില്ല. മറ്റൊരു resolution (5 Minutes / Daily) try ചെയ്യൂ.")
    st.stop()

df = add_indicators(df, intraday)
n_bars = len(df)
ema200_ok = n_bars >= 200
last = df.iloc[-1]
price = float(last["Close"])
prev_price = float(df["Close"].iloc[-2])
pct = (price - prev_price) / prev_price * 100 if prev_price else 0.0
atr = float(last["ATR"]) if pd.notna(last["ATR"]) else price * 0.01
rsi = float(last["RSI"]) if pd.notna(last["RSI"]) else 50.0

# Pivots from the PREVIOUS SESSION's daily H/L/C (not the previous candle!)
hlc = prev_session_hlc(yf_syms[0]) or (float(df["High"].iloc[-2]), float(df["Low"].iloc[-2]), float(df["Close"].iloc[-2]))
hp, lp, cp = hlc
pp = (hp + lp + cp) / 3
r1, s1 = 2 * pp - lp, 2 * pp - hp
r2, s2 = pp + (hp - lp), pp - (hp - lp)

patterns = detect_patterns(df, atr)
score, breakdown = score_signal(price, df, ema200_ok)
plan = build_plan(score, price, atr, r1, r2)

cur = "$" if ("GOLD" in display_name or "BITCOIN" in display_name) else ("" if display_name == "EUR/USD" else "₹")
dec = 5 if display_name == "EUR/USD" else 2
P = lambda x: f"{cur}{x:,.{dec}f}"

# data freshness
last_ts = df.index[-1]
now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tzinfo else pd.Timestamp.now()
age_min = (now - last_ts).total_seconds() / 60
fresh = f"Last candle {last_ts.strftime('%d %b %H:%M')}" + (f" • ⏱ {age_min:.0f} min old (market closed / delayed?)" if intraday and age_min > 3 * cfg["mins"] else "")

pill = f'<div class="pill pill-{plan["cls"]}">{plan["icon"]} {plan["label"]}</div>'
col_action = T["up"] if plan["bias"] > 0 else (T["down"] if plan["bias"] < 0 else T["warn"])
zone_lo, zone_hi = price - 0.25 * atr, price + 0.25 * atr


def cell(label, value, color):
    return f'<div class="cell"><div class="cell-l">{label}</div><div class="cell-v" style="color:{color}">{value}</div></div>'


has_fno = (fno_name in FNO_LOTS) if is_equity else (fno_name is not None)
lot = FNO_LOTS.get(fno_name) if fno_name else None
step = step_fixed or stock_strike_step(price)
atm = round(price / step) * step

hud_cells = (
    cell("Suggested entry zone", f"{P(zone_lo)} – {P(zone_hi)}", T["accent"])
    + cell("Hard stop loss", P(plan["sl"]), "#dc2626")
    + cell(plan["t1_label"], P(plan["t1"]), "#059669")
    + cell(plan["t2_label"], P(plan["t2"]), "#059669")
)
if has_fno and plan["bias"] != 0:
    hud_cells += cell("Options contract", f"{fs(atm)} {plan['opt']}", "#7c3aed")

st.markdown(f"""
<div class="hud-card">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
    <div>
      <div style="font-size:12px;color:{T['accent']};font-weight:800;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">
        QUANT SIGNAL • {data_source} • score {score:+d}</div>
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <span style="font-size:30px;font-weight:900;font-family:'JetBrains Mono',monospace;">{display_name}</span>{pill}
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:12px;color:{T['text_sub']};font-weight:700;text-transform:uppercase;">LAST PRICE</div>
      <div style="font-size:28px;font-weight:900;font-family:'JetBrains Mono',monospace;">{P(price)}
        <span style="font-size:17px;color:{col_action};">({pct:+.2f}%)</span></div>
      <div style="font-size:11px;color:{T['text_sub']};">{fresh}</div>
    </div>
  </div>
  <div class="hud-grid">{hud_cells}</div>
</div>
""", unsafe_allow_html=True)

m = st.columns(7)
PM = (lambda x: f"{cur}{x:,.0f}") if price >= 1000 else P
m[0].metric("Pivot", PM(pp))
m[1].metric("R1", PM(r1))
m[2].metric("S1", PM(s1))
m[3].metric("RSI (14)", f"{rsi:.1f}", "Oversold" if rsi < 35 else ("Overbought" if rsi > 70 else "Balanced"))
m[4].metric("SuperTrend", "BULLISH" if last["ST_Dir"] == 1 else "BEARISH")
m[5].metric("MACD", "Bullish" if last["MACD"] > last["MACD_Signal"] else "Bearish")
m[6].metric("ATR (14)", PM(atr) if price >= 1000 else P(atr))

# ==============================================================================
# CHART  (price + volume + MACD + RSI)
# ==============================================================================
pdf = df.tail(bars_shown)
has_vol = pdf["Volume"].sum() > 0
panels = ["price"] + (["vol"] if has_vol else []) + ["macd", "rsi"]
heights = {"price": 0.56, "vol": 0.12, "macd": 0.16, "rsi": 0.16}
row_of = {p: i + 1 for i, p in enumerate(panels)}
tot = sum(heights[p] for p in panels)
fig = make_subplots(rows=len(panels), cols=1, shared_xaxes=True, vertical_spacing=0.025,
                    row_heights=[heights[p] / tot for p in panels])

fig.add_trace(go.Candlestick(
    x=pdf.index, open=pdf["Open"], high=pdf["High"], low=pdf["Low"], close=pdf["Close"], name="Price",
    increasing=dict(line=dict(color="#10b981", width=1.2), fillcolor="#10b981"),
    decreasing=dict(line=dict(color="#ef4444", width=1.2), fillcolor="#ef4444")), row=1, col=1)
fig.add_trace(go.Scatter(x=pdf.index, y=pdf["EMA_20"], name="EMA 20", line=dict(color=T["accent"], width=1.3)), row=1, col=1)
fig.add_trace(go.Scatter(x=pdf.index, y=pdf["EMA_50"], name="EMA 50", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
if ema200_ok:
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf["EMA_200"], name="EMA 200", line=dict(color="#ec4899", width=2)), row=1, col=1)
if show_st:
    up_line = pdf["ST"].where(pdf["ST_Dir"] == 1)
    dn_line = pdf["ST"].where(pdf["ST_Dir"] == -1)
    fig.add_trace(go.Scatter(x=pdf.index, y=up_line, name="SuperTrend ▲", line=dict(color="#10b981", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=pdf.index, y=dn_line, name="SuperTrend ▼", line=dict(color="#ef4444", width=1.8)), row=1, col=1)
if show_vwap and pdf["VWAP"].notna().any():
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf["VWAP"], name="VWAP", line=dict(color="#8b5cf6", width=1.6, dash="dash")), row=1, col=1)
if show_bb:
    for col_ in ("BB_Upper", "BB_Lower"):
        fig.add_trace(go.Scatter(x=pdf.index, y=pdf[col_], name=col_.replace("_", " "),
                                 line=dict(color="rgba(148,163,184,.5)", dash="dot")), row=1, col=1)
if show_piv:
    for val, nm, colr, dash in ((r2, "R2", "#b91c1c", "dash"), (r1, "R1", "#ef4444", "dash"), (pp, "PP", "#64748b", "dot"),
                                (s1, "S1", "#10b981", "dash"), (s2, "S2", "#047857", "dash")):
        fig.add_hline(y=val, line_dash=dash, line_color=colr, line_width=1, annotation_text=nm,
                      annotation_position="top left", row=1, col=1)
fig.add_hline(y=price, line_dash="dot", line_color=T["warn"], line_width=1.8, annotation_text=f"LTP {P(price)}",
              annotation_position="right", annotation_font=dict(color=T["warn"], size=12), row=1, col=1)

for p_ in patterns:
    if p_["idx"] in pdf.index:
        fig.add_annotation(x=p_["idx"], y=p_["price"], text=f"📍 {p_['name']}", showarrow=True, arrowhead=2,
                           arrowcolor=T["accent"], font=dict(size=11, color="#ffffff"), bgcolor="#0f172a",
                           bordercolor=T["accent"], row=1, col=1)

if has_vol:
    vcol = np.where(pdf["Close"] >= pdf["Open"], "#10b981", "#ef4444")
    fig.add_trace(go.Bar(x=pdf.index, y=pdf["Volume"], marker_color=vcol, name="Volume", showlegend=False), row=row_of["vol"], col=1)

r = row_of["macd"]
fig.add_trace(go.Scatter(x=pdf.index, y=pdf["MACD"], name="MACD", line=dict(color="#2563eb", width=1.4)), row=r, col=1)
fig.add_trace(go.Scatter(x=pdf.index, y=pdf["MACD_Signal"], name="Signal", line=dict(color="#f97316", width=1.4)), row=r, col=1)
fig.add_trace(go.Bar(x=pdf.index, y=pdf["MACD_Diff"], marker_color=["#10b981" if v >= 0 else "#ef4444" for v in pdf["MACD_Diff"]],
                     name="Hist", showlegend=False), row=r, col=1)

r = row_of["rsi"]
fig.add_trace(go.Scatter(x=pdf.index, y=pdf["RSI"], name="RSI", line=dict(color="#8b5cf6", width=1.5)), row=r, col=1)
for lvl, colr in ((70, "#ef4444"), (30, "#10b981")):
    fig.add_hline(y=lvl, line_dash="dot", line_color=colr, line_width=1, row=r, col=1)
fig.update_yaxes(range=[0, 100], row=r, col=1)

fig.add_annotation(text=f"<b>{display_name}</b> • {interval_label}", xref="paper", yref="paper", x=0.01, y=0.99,
                   showarrow=False, font=dict(size=16, color=T["text_main"]), bgcolor=T["bg_card"],
                   bordercolor=T["accent"], borderwidth=1.2, borderpad=5)

if not asset_type.startswith("Global"):  # hide nights / weekends for NSE-BSE
    breaks = [dict(bounds=["sat", "mon"])]
    if intraday:
        breaks.append(dict(bounds=[15.5, 9.25], pattern="hour"))
    fig.update_xaxes(rangebreaks=breaks)
else:
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] if display_name != "BITCOIN (BTC)" else [])

fig.update_layout(
    template=T["plotly"], height=760 if has_vol else 700, margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor=T["chart_bg"], plot_bgcolor=T["chart_bg"], hovermode="x unified", dragmode="pan",
    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11, color=T["text_main"])),
    font=dict(color=T["text_main"]),
    xaxis_rangeslider_visible=False,
)
fig.update_xaxes(gridcolor=T["grid"], rangeslider_visible=False)
fig.update_yaxes(gridcolor=T["grid"])
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})

# ==============================================================================
# TABS
# ==============================================================================
tab_plan, tab_pat, tab_fund = st.tabs(["📋 Trade Plan & Signal Breakdown", "🔍 Pattern Analysis", "📑 Fundamentals"])

with tab_plan:
    c1, c2 = st.columns(2)
    qty_txt = ""
    if not asset_type.startswith("Global") and plan["risk"] > 0:
        qty = int((capital * risk_pct / 100) // plan["risk"])
        qty_txt = f"{qty:,} units"
        if lot:
            qty_txt += f"  (≈ {qty // lot} lot × {lot})"
        qty_txt += f" — risk ₹{capital * risk_pct / 100:,.0f}"

    with c1:
        with st.container(border=True):
            st.markdown(f"""
            <div class="sec-title">📋 Trade Blueprint</div>
            <div class="plan-lines">
              • <b>Action:</b> <span style="color:{col_action};font-weight:900;">{plan['action']}</span><br>
              • <b>Execution zone:</b> <span class="tag tag-stat">{P(zone_lo)} – {P(zone_hi)}</span><br>
              • <b>Stop loss:</b> <span class="tag tag-down">{P(plan['sl'])}</span><br>
              • <b>{plan['t1_label'].title()}:</b> <span class="tag tag-up">{P(plan['t1'])}</span><br>
              • <b>{plan['t2_label'].title()}:</b> <span class="tag tag-up">{P(plan['t2'])}</span><br>
              {'• <b>Position size:</b> <span class="tag tag-stat">' + qty_txt + '</span>' if qty_txt and plan['bias'] != 0 else ''}
            </div>""", unsafe_allow_html=True)

    with c2:
        with st.container(border=True):
            if has_fno:
                if plan["bias"] != 0:
                    itm = atm - plan["bias"] * step
                    body_ = (f"• <b>Strategy:</b> <span style='color:#7c3aed;font-weight:900;'>{plan['strat']}</span><br>"
                             f"• <b>ATM strike:</b> <span class='tag tag-stat'>{fs(atm)} {plan['opt']}</span><br>"
                             f"• <b>ITM (≈0.65Δ) strike:</b> <span class='tag tag-stat'>{fs(itm)} {plan['opt']}</span><br>"
                             f"• <b>Underlying SL trigger:</b> <span class='tag tag-down'>{P(price - plan['bias'] * atr)}</span><br>")
                else:
                    sp = np.floor(s1 / step) * step
                    sc = np.ceil(r1 / step) * step
                    body_ = (f"• <b>Strategy:</b> <span style='color:#7c3aed;font-weight:900;'>{plan['strat']}</span><br>"
                             f"• <b>Short strikes (S1 / R1):</b> <span class='tag tag-stat'>{fs(sp)} PE</span> "
                             f"<span class='tag tag-stat'>{fs(sc)} CE</span><br>")
                body_ += f"• <b>Lot size:</b> {lot if lot else 'n/a'} • strike step {fs(step)}{'' if step_fixed else ' (approx — option chain check cheyyuka)'}<br>"
                body_ += "• <b>Rule:</b> exit decisions underlying spot price base cheyyuka, premium base alla."
                st.markdown(f'<div class="sec-title">⚡ Derivatives Engine</div><div class="plan-lines">{body_}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="sec-title">⚡ Derivatives Engine</div>', unsafe_allow_html=True)
                st.caption("ഈ അസറ്റിന് NSE F&O contract ഇല്ല (അല്ലെങ്കിൽ options ബാധകമല്ല) — അതുകൊണ്ട് options ശുപാർശ കാണിക്കുന്നില്ല.")

    st.markdown("##### 🧮 Signal score breakdown")
    bd = pd.DataFrame(breakdown, columns=["Factor", "Points", "Reading"])
    bd.loc[len(bd)] = ["TOTAL", score, f"≥+3 strong buy • +1..+2 buy dips • 0 wait • −1..−2 sell rise • ≤−3 strong sell"]
    st.dataframe(bd, use_container_width=True, hide_index=True)
    if not ema200_ok:
        st.caption(f"ℹ️ {n_bars} bars മാത്രമേ ഉള്ളൂ — EMA 200 വിശ്വസനീയമല്ല, അതുകൊണ്ട് ചാർട്ടിൽ നിന്നും സ്കോറിൽ നിന്നും ഒഴിവാക്കി.")

with tab_pat:
    st.caption("അവസാന കാൻഡിൽ ഇപ്പോഴും രൂപപ്പെടുകയാണെങ്കിൽ (live) ഈ പാറ്റേൺ മാറാം. Candle close ആകുമ്പോൾ confirm ചെയ്യുക.")
    if patterns:
        cols = st.columns(min(len(patterns), 3))
        for i, p_ in enumerate(patterns):
            bull, bear = "Bullish" in p_["type"], "Bearish" in p_["type"]
            bg_, bd_, tc_ = ("#ecfdf5", "#10b981", "#047857") if bull else (("#fef2f2", "#ef4444", "#b91c1c") if bear else ("#fffbeb", "#f59e0b", "#b45309"))
            with cols[i % len(cols)]:
                st.markdown(f"""
                <div style="background:{T['bg_sub']};border:1.5px solid {T['border']};border-radius:12px;padding:16px;margin-bottom:12px;">
                  <div style="font-size:16px;font-weight:800;margin-bottom:6px;">{p_['name']}</div>
                  <span style="display:inline-block;background:{bg_};border:1px solid {bd_};color:{tc_};font-weight:800;font-size:11px;padding:2px 8px;border-radius:12px;margin-bottom:10px;">{p_['type']}</span>
                  <div style="font-size:13px;line-height:1.7;color:{T['text_sub']};"><b>വിശദീകരണം:</b> {p_['explanation']}</div>
                  <div style="font-size:12.5px;margin-top:8px;font-weight:700;">{p_['note']}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.info("ℹ️ നിലവിൽ വ്യക്തമായ ചാർട്ട് / കാൻഡിൽ പാറ്റേൺ രൂപപ്പെട്ടിട്ടില്ല.")

with tab_fund:
    if not is_equity:
        st.caption("Fundamentals ഇന്ത്യൻ equities-ന് മാത്രം.")
    elif not load_fund:
        st.caption("Sidebar-ൽ 'Load Screener.in fundamentals' tick ചെയ്യൂ.")
    else:
        sym_for_screener = display_name if stock_row["exchange"] == "NSE" else (str(stock_row.get("exchange_token", "")) or display_name)
        with st.spinner("Screener.in data എടുക്കുന്നു…"):
            sd = fetch_screener(sym_for_screener)
        if sd and sd.get("ratios"):
            rt = sd["ratios"]
            g = lambda k, pre="", suf="": f"{pre}{rt[k]}{suf}" if k in rt else "N/A"
            f = st.columns(6)
            f[0].metric("Market Cap", g("Market Cap", "₹", " Cr"))
            f[1].metric("Current Price", g("Current Price", "₹"))
            f[2].metric("Stock P/E", g("Stock P/E"))
            f[3].metric("ROCE", g("ROCE", "", "%"))
            f[4].metric("ROE", g("ROE", "", "%"))
            f[5].metric("Book Value", g("Book Value", "₹"))
            if sd["quarterly"] is not None:
                st.markdown("##### 📊 Quarterly results (₹ Cr)")
                st.dataframe(sd["quarterly"], use_container_width=True, hide_index=True)
            pc_, cc_ = st.columns(2)
            with pc_:
                st.success("**Strengths**\n\n" + ("\n".join(f"- {x}" for x in sd["pros"]) or "No specific strengths published."))
            with cc_:
                st.error("**Risks**\n\n" + ("\n".join(f"- {x}" for x in sd["cons"]) or "No specific risks published."))
        else:
            st.caption("Screener.in-ൽ ഈ സ്റ്റോക്കിന്റെ ഡാറ്റ കിട്ടിയില്ല.")

st.caption("⚠️ Educational tool only — not investment advice. Signals are rule-based, not backtested; trade at your own risk.")
