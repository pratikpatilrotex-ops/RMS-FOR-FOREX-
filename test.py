"""
╔══════════════════════════════════════════════════════════════╗
║   BROKER RISK & TRADE INTELLIGENCE  — RMS Dashboard v3.1    ║
║   Full rewrite with all fixes applied                       ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import requests

# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RMS · Broker Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""<style>
[data-testid="collapsedControl"] { display:none !important; }
section[data-testid="stSidebar"]  { display:none !important; }
</style>""", unsafe_allow_html=True)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif !important;}
.main,.block-container{background-color:#070d18 !important;color:#cbd5e1 !important;padding-top:1rem !important;max-width:100% !important;}
[data-testid="stHeader"]{background:transparent !important;}
[data-testid="stToolbar"]{display:none !important;}
footer{display:none !important;}
.rms-header{background:linear-gradient(135deg,#0d1a2e 0%,#0a1628 60%,#0f2040 100%);border:1px solid rgba(30,58,95,.4);border-radius:14px;padding:22px 32px;margin-bottom:24px;display:flex;align-items:center;justify-content:space-between;}
.rms-header-left h1{font-size:26px;font-weight:800;background:linear-gradient(90deg,#60a5fa,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;letter-spacing:-.5px;}
.rms-header-left p{font-size:12px;color:#475569;margin:4px 0 0 0;letter-spacing:1.5px;text-transform:uppercase;}
.rms-badge{background:linear-gradient(135deg,#1d4ed8,#0ea5e9);color:#fff;font-size:11px;font-weight:700;padding:5px 14px;border-radius:20px;letter-spacing:1.5px;text-transform:uppercase;}
.kpi-grid{display:flex;gap:14px;margin-bottom:20px;flex-wrap:wrap;}
.kpi-card{background:#0d1a2e;border:1px solid rgba(30,58,95,.3);border-radius:12px;padding:18px 22px;flex:1;min-width:160px;position:relative;overflow:hidden;}
.kpi-card::after{content:'';position:absolute;top:0;left:0;right:0;height:2px;border-radius:12px 12px 0 0;}
.kpi-card.green::after{background:linear-gradient(90deg,#22d3ee,#0d9488);}
.kpi-card.blue::after{background:linear-gradient(90deg,#3b82f6,#60a5fa);}
.kpi-card.amber::after{background:linear-gradient(90deg,#f59e0b,#fbbf24);}
.kpi-card.red::after{background:linear-gradient(90deg,#ef4444,#f87171);}
.kpi-card.purple::after{background:linear-gradient(90deg,#7c3aed,#a78bfa);}
.kpi-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:8px;}
.kpi-value{font-family:'Space Mono',monospace;font-size:24px;font-weight:700;line-height:1;}
.kpi-value.green{color:#22d3ee;}.kpi-value.blue{color:#60a5fa;}.kpi-value.amber{color:#fbbf24;}.kpi-value.red{color:#f87171;}.kpi-value.purple{color:#a78bfa;}
.kpi-sub{font-size:11px;color:#475569;margin-top:5px;}
.section-title{font-size:13px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#475569;border-left:3px solid #3b82f6;padding-left:10px;margin:24px 0 14px 0;}
.alert{border-radius:8px;padding:10px 16px;margin:6px 0;font-size:12px;display:flex;align-items:flex-start;gap:10px;}
.alert.high{background:#1a0a0a;border:1px solid #7f1d1d;color:#fca5a5;}
.alert.medium{background:#1a1200;border:1px solid #78350f;color:#fde68a;}
.alert.low{background:#0a1520;border:1px solid rgba(30,58,95,.7);color:#93c5fd;}
.alert.good{background:#051a0f;border:1px solid #166534;color:#4ade80;}
.insight-card{background:#0a1321;border:1px solid rgba(30,58,95,.3);border-radius:10px;padding:14px 16px;margin:6px 0;}
.insight-title{font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;margin-bottom:6px;}
.insight-body{font-size:12px;color:#94a3b8;line-height:1.6;}
[data-testid="stTabs"] button{font-family:'Inter',sans-serif !important;font-size:12px !important;font-weight:600 !important;letter-spacing:.8px !important;text-transform:uppercase !important;color:#475569 !important;padding:10px 18px !important;}
[data-testid="stTabs"] button[aria-selected="true"]{color:#60a5fa !important;border-bottom:2px solid #3b82f6 !important;}
[data-testid="stDataFrame"]{border:1px solid rgba(30,58,95,.3) !important;border-radius:10px !important;}
[data-testid="metric-container"]{background:#0d1a2e !important;border:1px solid rgba(30,58,95,.3) !important;border-radius:10px !important;padding:14px !important;}
[data-testid="stMetricValue"]{font-family:'Space Mono',monospace !important;color:#60a5fa !important;}
[data-testid="stMetricLabel"]{color:#475569 !important;font-size:11px !important;}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:#0d1a2e !important;border:1px solid rgba(30,58,95,.7) !important;color:#e2e8f0 !important;border-radius:8px !important;}
.stSelectbox>div>div{background:#0d1a2e !important;border:1px solid rgba(30,58,95,.7) !important;color:#e2e8f0 !important;}
[data-testid="stFileUploader"]{background:#0d1a2e !important;border:1px dashed rgba(30,58,95,.7) !important;border-radius:12px !important;}
</style>""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(13,26,46,0.6)",
    font=dict(family="Inter", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="rgba(30,58,95,0.2)", zerolinecolor="rgba(30,58,95,0.33)"),
    yaxis=dict(gridcolor="rgba(30,58,95,0.2)", zerolinecolor="rgba(30,58,95,0.33)"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(30,58,95,0.27)"),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ── HELPERS ──────────────────────────────────────────────────
def safe_dt(s):
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def fmt_usd(v):
    try:
        return f"${float(v):,.2f}"
    except:
        return "$0.00"


def fmt_pct(v):
    try:
        return f"{float(v):.2f}%"
    except:
        return "0.00%"


def to_xl(sheets):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nm, df in sheets.items():
            df.to_excel(w, index=False, sheet_name=nm[:31])
    buf.seek(0)
    return buf.read()


def to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


def dl(label, sheets, csv_df, key):
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        st.download_button(
            f"📥 Excel — {label}", to_xl(sheets), f"rms_{key}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xl_{key}"
        )
    with c2:
        st.download_button(f"📄 CSV — {label}", to_csv(csv_df), f"rms_{key}.csv", "text/csv", key=f"csv_{key}")


def kpi(label, value, color="blue", sub=""):
    return (
        f'<div class="kpi-card {color}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {color}">{value}</div>'
        + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
        + '</div>'
    )


def alert(msg, level="low"):
    icons = {"high": "⚠️", "medium": "◉", "low": "●", "good": "✓"}
    return f'<div class="alert {level}">{icons.get(level, "•")} {msg}</div>'


def insight(title, body, color="#60a5fa"):
    return (
        f'<div class="insight-card"><div class="insight-title" style="color:{color}">{title}</div>'
        f'<div class="insight-body">{body}</div></div>'
    )


# ── COLUMN ID ────────────────────────────────────────────────
def id_cols(raw):
    raw = raw.dropna(how="all")
    hdr = raw.iloc[0]
    df = raw.iloc[1:].reset_index(drop=True)
    df.columns = hdr
    cols = list(df.columns)
    ti = [i for i, c in enumerate(cols) if str(c).strip().lower() == "time"]
    pi = [i for i, c in enumerate(cols) if str(c).strip().lower() == "price"]
    if len(ti) >= 1: cols[ti[0]] = "Open Time"
    if len(ti) >= 2: cols[ti[1]] = "Close Time"
    if len(pi) >= 1: cols[pi[0]] = "Open Price"
    if len(pi) >= 2: cols[pi[1]] = "Close Price"
    for nm in ["Volume", "Profit", "Type", "Symbol", "Position", "Commission", "Swap", "S / L", "T / P"]:
        for i, c in enumerate(cols):
            if str(c).strip().lower() == nm.lower():
                cols[i] = nm
    df.columns = cols
    return df


# ── DEALS PARSER ─────────────────────────────────────────────
def parse_deals(df_raw):
    idx = df_raw.index[df_raw[0].astype(str).str.contains("Deals", case=False, na=False)].tolist()
    if not idx:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    raw = df_raw.iloc[idx[0] + 1:].dropna(how="all").reset_index(drop=True)
    if len(raw) < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    hdr = [str(c).strip() if not pd.isna(c) else f"_c{i}" for i, c in enumerate(raw.iloc[0])]
    data = raw.iloc[1:].reset_index(drop=True)
    data.columns = range(len(data.columns))
    hl = [h.lower() for h in hdr]

    def ci(nm):
        return next((i for i, h in enumerate(hl) if nm in h), None)

    tc, tyc, pc, cc = ci("time"), ci("type"), ci("profit"), ci("comment")
    if tyc is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bal = data[data[tyc].astype(str).str.lower().str.strip() == "balance"].copy()
    if len(bal) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    bal["_profit_num"] = pd.to_numeric(
        bal[pc].astype(str).str.replace(r"[\s,]", "", regex=True), errors="coerce"
    ) if pc is not None else 0.0

    def classify(row):
        comment = str(row[cc]).upper() if cc is not None else ""
        if "DEPOSIT" in comment:  return "Deposit"
        if "WITHDRAW" in comment: return "Withdrawal"
        return "Deposit" if row["_profit_num"] > 0 else "Withdrawal"

    bal["_cashtype"] = bal.apply(classify, axis=1)

    def bdf(rows):
        d = pd.DataFrame()
        if tc is not None: d["Time"] = safe_dt(rows[tc].astype(str))
        d["Amount"] = rows["_profit_num"].values
        if cc is not None: d["Comment"] = rows[cc].astype(str).str.strip()
        d["CashType"] = rows["_cashtype"].values
        return d.reset_index(drop=True)

    all_cf = bdf(bal)
    deps = bdf(bal[bal["_cashtype"] == "Deposit"])
    wths = bdf(bal[bal["_cashtype"] == "Withdrawal"])
    return deps, wths, all_cf


# ── EQUITY METRICS ───────────────────────────────────────────
def equity_metrics(df, start):
    df = df.sort_values("Close Time").copy()
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0)
    df["Commission"] = pd.to_numeric(df.get("Commission", 0), errors="coerce").fillna(0)
    df["Swap"] = pd.to_numeric(df.get("Swap", 0), errors="coerce").fillna(0)
    df["NetPnL"] = df["Profit"] + df["Commission"] + df["Swap"]
    df["CumProfit"] = df["NetPnL"].cumsum()
    df["Equity"] = start + df["CumProfit"]
    df["EquityBefore"] = start + df["CumProfit"].shift(1).fillna(0)
    df["TradeReturnPct"] = np.where(df["EquityBefore"] != 0, df["NetPnL"] / df["EquityBefore"] * 100, 0)

    def rl(p):
        if p <= -50: return "Very High (≤ -50%)"
        if p <= -30: return "High (≤ -30%)"
        return "Normal"

    df["RiskBucket"] = df["TradeReturnPct"].apply(rl)
    df["EquityPeak"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] - df["EquityPeak"]
    df["DrawdownPct"] = np.where(df["EquityPeak"] != 0, df["Drawdown"] / df["EquityPeak"] * 100, 0)
    mr = df.loc[df["Drawdown"].idxmin()] if len(df) else None
    return df, mr["Drawdown"] if mr is not None else 0, mr["DrawdownPct"] if mr is not None else 0


# ── SWAP NIGHTS ──────────────────────────────────────────────
def swap_nights(ot, ct, triday):
    if pd.isna(ot) or pd.isna(ct) or ct <= ot:
        return 0, 0
    n = t = 0
    cur = ot.date()
    last = (ct - timedelta(seconds=1)).date()
    while cur <= last:
        night = datetime(cur.year, cur.month, cur.day, 23, 59)
        if ot <= night < ct:
            wd = night.weekday()
            if wd not in (5, 6):
                if wd == triday:
                    t += 1
                else:
                    n += 1
        cur += timedelta(days=1)
    return n, t


# ── BOT/EA ───────────────────────────────────────────────────
def detect_burst(df, ms=2):
    df = df.copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df = df.sort_values("Open Time").reset_index(drop=True)
    if len(df) < 2:
        return df.iloc[0:0], 0, 0
    gs, cur = [], [0]
    for i in range(1, len(df)):
        dt = (df.loc[i, "Open Time"] - df.loc[i - 1, "Open Time"]).total_seconds()
        if 0 < dt <= ms:
            cur.append(i)
        else:
            if len(cur) >= 2: gs.append(cur)
            cur = [i]
    if len(cur) >= 2: gs.append(cur)
    idx = sorted({i for g in gs for i in g})
    return df.loc[idx].copy(), len(gs), max((len(g) for g in gs), default=0)


def detect_reversal(df, ms=20):
    df = df.copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df = df.sort_values(["Symbol", "Open Time"]).reset_index(drop=True)
    if len(df) < 2:
        return df.iloc[0:0], 0
    ri, ev = [], []
    i = 0
    while i < len(df) - 1:
        r1, r2 = df.loc[i], df.loc[i + 1]
        dt = (r2["Open Time"] - r1["Open Time"]).total_seconds()
        if r1["Symbol"] == r2["Symbol"] and r1["Type"] != r2["Type"] and 0 < dt <= ms:
            ri += [i, i + 1]
            ev.append((i, i + 1))
            i += 2
        else:
            i += 1
    return df.loc[sorted(set(ri))].copy(), len(ev)


def ea_score(bn, rn, tot, offp, avgs):
    s = 0
    if tot > 0:
        s += min(bn / tot * 100 * 2, 35)
        s += min(rn / tot * 100 * 1.5, 25)
    s += min(offp * .5, 20)
    s += 20 if avgs < 30 else (10 if avgs < 120 else 0)
    return min(int(s), 100)


# ── RISK SCORE ───────────────────────────────────────────────
def risk_score(sc, ddp, tox, r30, eas):
    score = 0
    dd = min(abs(ddp) * .5, 25)
    score += dd
    conc = min(max(sc - 50, 0) * .4, 20)
    score += conc
    tx = min(tox * .5, 20)
    score += tx
    rk = min(r30 * 2, 20)
    score += rk
    ea = eas * .15
    score += ea
    rows = [
        {"Component": "Max Drawdown",        "Points": round(dd, 1),   "Max": 25},
        {"Component": "Symbol Concentration","Points": round(conc, 1), "Max": 20},
        {"Component": "Scalping Toxicity",   "Points": round(tx, 1),   "Max": 20},
        {"Component": "Trade Sizing Risk",   "Points": round(rk, 1),   "Max": 20},
        {"Component": "EA / Bot Probability","Points": round(ea, 1),   "Max": 15},
    ]
    total = min(int(score), 100)
    if total >= 75:   lv, co = "CRITICAL", "#ef4444"
    elif total >= 55: lv, co = "HIGH",     "#f97316"
    elif total >= 35: lv, co = "MEDIUM",   "#fbbf24"
    else:             lv, co = "LOW",      "#22c55e"
    return total, lv, co, pd.DataFrame(rows)


# ── STREAK ANALYSIS ──────────────────────────────────────────
def streak_analysis(net_pnl_series):
    profits = list(net_pnl_series)
    max_w = max_l = cur_w = cur_l = 0
    streak_data = []
    total_win_streaks = []
    total_loss_streaks = []

    for p in profits:
        if p > 0:
            cur_w += 1; cur_l = 0
            max_w = max(max_w, cur_w)
            streak_data.append(cur_w)
        else:
            cur_l += 1; cur_w = 0
            max_l = max(max_l, cur_l)
            streak_data.append(-cur_l)

    run = 0
    for i, p in enumerate(profits):
        if p > 0:
            run += 1
            if i + 1 >= len(profits) or profits[i + 1] <= 0:
                total_win_streaks.append(run); run = 0
        else:
            run += 1
            if i + 1 >= len(profits) or profits[i + 1] > 0:
                total_loss_streaks.append(run); run = 0

    avg_w = round(np.mean(total_win_streaks), 1) if total_win_streaks else 0
    avg_l = round(np.mean(total_loss_streaks), 1) if total_loss_streaks else 0
    streak_df = pd.DataFrame({"Trade": range(1, len(streak_data) + 1), "Streak": streak_data})
    return max_w, max_l, avg_w, avg_l, streak_df


# ── 7 KPI CALCULATOR ─────────────────────────────────────────
def calc_seven_kpis(df, raw, mdd_a, mdd_p):
    net_pnl = df["NetPnL_trade"]
    wins = net_pnl[net_pnl > 0]
    losses = net_pnl[net_pnl < 0]

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    rr_ratio = round(avg_win / abs(avg_loss), 2) if avg_loss != 0 else np.nan

    date_range = (df["Open Time"].max() - df["Open Time"].min()).days + 1
    trades_per_day = round(raw["tot_t"] / date_range, 1) if date_range > 0 else float(raw["tot_t"])

    gross_profit = float(df.loc[net_pnl > 0, "NetPnL_trade"].sum())
    gross_loss = float(abs(df.loc[net_pnl < 0, "NetPnL_trade"].sum()))
    pf = round(gross_profit / gross_loss, 3) if gross_loss > 0 else np.nan

    max_w, max_l, avg_w, avg_l, streak_df = streak_analysis(net_pnl)

    return {
        "net_profit":        round(float(raw["net_p"]), 2),
        "profit_per_trade":  round(float(raw["net_p"]) / raw["tot_t"], 2) if raw["tot_t"] else 0.0,
        "profit_factor":     pf,
        "win_rate":          round(float(raw["wr"]), 1),
        "total_wins":        int(len(wins)),
        "total_losses":      int(len(losses)),
        "rr_ratio":          rr_ratio,
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "mdd_abs":           round(float(mdd_a), 2),
        "mdd_pct":           round(float(mdd_p), 2),
        "total_trades":      int(raw["tot_t"]),
        "trades_per_day":    trades_per_day,
        "date_range_days":   int(date_range),
        "avg_hold_sec":      round(float(raw["avhs"]), 0),
        "max_consec_wins":   int(max_w),
        "max_consec_losses": int(max_l),
        "avg_consec_wins":   float(avg_w),
        "avg_consec_losses": float(avg_l),
        "streak_df":         streak_df,
    }


# ══════════════════════════════════════════════════════════════
#  SYMBOL PIP VALUE TABLE  (per-symbol accurate pip values)
# ══════════════════════════════════════════════════════════════
SYM_PIP_VAL = {
    # Forex majors — standard $10/pip per lot
    "EURUSD":10.0,"GBPUSD":10.0,"AUDUSD":10.0,"NZDUSD":10.0,
    "USDCAD":7.7, "USDCHF":11.0,
    # JPY pairs — $8–9/pip
    "USDJPY":9.0,"EURJPY":9.0,"GBPJPY":9.0,"AUDJPY":9.0,"CADJPY":9.0,
    # Gold / Silver
    "XAUUSD":1.0,"GOLD":1.0,"XAGUSD":50.0,
    # Indices — per point
    "US30":1.0,"NAS100":1.0,"SPX500":1.0,"GER40":1.0,"UK100":1.0,
    "NDX":1.0,"DAX":1.0,"FTSE":1.0,
    # Oil
    "XTIUSD":10.0,"XBRUSD":10.0,"WTI":10.0,"BRENT":10.0,
    # Crypto
    "BTCUSD":1.0,"ETHUSD":1.0,"XRPUSD":1.0,
}

def get_pip_val(symbol):
    """Return pip value for a symbol, fallback $8 blended average."""
    s = str(symbol).upper().replace(".","").replace("_","").replace("-","")
    for k, v in SYM_PIP_VAL.items():
        if k in s: return v
    return 8.0  # blended fallback


# ══════════════════════════════════════════════════════════════
#  ADVANCED BROKER CONTROLS ENGINE
#  Rules: Slippage (trade-count), Delay (hold-time %), Market-open,
#         Symbol-wise spread, Order-type asymmetry, EV-based routing
# ══════════════════════════════════════════════════════════════

# Market session open windows (UTC minutes) — first 5 min = high spread zone
SESS_OPEN_UTC = {
    "Sydney":   21 * 60,
    "Tokyo":    0  * 60,
    "London":   7  * 60,
    "New York": 13 * 60,
}

def calc_market_open_trades(tdf, broker_gmt_offset):
    """Count trades opened within 5 minutes of any session open."""
    off_min = int(round(broker_gmt_offset * 60))
    count = 0
    lots  = 0.0
    rows  = []
    for _, r in tdf.iterrows():
        ot = r.get("Open Time")
        if pd.isna(ot): continue
        utc_min = (ot.hour * 60 + ot.minute - off_min) % 1440
        for sess, s_utc in SESS_OPEN_UTC.items():
            diff = (utc_min - s_utc) % 1440
            if diff < 5:  # within first 5 minutes
                count += 1
                lots  += float(r.get("Volume", 0))
                rows.append({
                    "Open Time": ot,
                    "Symbol":    r.get("Symbol",""),
                    "Type":      r.get("Type",""),
                    "Volume":    r.get("Volume",0),
                    "Profit":    r.get("Profit",0),
                    "Session":   sess,
                    "MinuteAfterOpen": int(diff),
                })
                break
    return count, lots, pd.DataFrame(rows)


def calc_symbol_controls(tdf, sym_dir):
    """
    Per-symbol: scalp%, win rate, avg hold, profit, pip_val, recommended markup.
    Returns a DataFrame — one row per Symbol+Type.
    """
    rows = []
    if "Symbol" not in tdf.columns or "Hold_Time" not in tdf.columns:
        return pd.DataFrame()

    for sym, grp in tdf.groupby("Symbol", dropna=True):
        pip_val  = get_pip_val(sym)
        tot      = len(grp)
        scalp_n  = int((grp["Hold_Time"].dt.total_seconds() < 180).sum())
        scalp_p  = scalp_n / tot * 100 if tot else 0
        wr       = float(grp["is_win"].mean() * 100) if "is_win" in grp else 0
        avg_h    = float(grp["Hold_Time"].dt.total_seconds().mean())
        net_p    = float(grp["NetPnL_trade"].sum()) if "NetPnL_trade" in grp else 0
        vol      = float(grp["Volume"].sum())

        # Recommended markup based on symbol toxicity
        if scalp_p >= 50 or pip_val <= 1.0:   mu = 0.8  # gold/indices or heavy scalp
        elif scalp_p >= 25:                    mu = 0.5
        elif net_p > 0 and wr >= 55:           mu = 0.4  # profitable symbol
        else:                                   mu = 0.2

        rows.append({
            "Symbol":       sym,
            "PipValue($)":  pip_val,
            "Trades":       tot,
            "Volume(lots)": round(vol, 2),
            "Scalp%":       round(scalp_p, 1),
            "WinRate%":     round(wr, 1),
            "AvgHold(s)":   round(avg_h, 0),
            "NetProfit":    round(net_p, 2),
            "Rec.Markup(pip)": mu,
            "Est.MarkupRev":   round(mu * pip_val * vol, 2),
        })
    return pd.DataFrame(rows).sort_values("Volume(lots)", ascending=False)


def calc_order_asymmetry(sym_dir):
    """
    Detect directional bias: if BUY profit >> SELL profit (or vice versa),
    broker should apply higher markup on the profitable direction.
    Returns analysis DataFrame + asymmetry flag.
    """
    if sym_dir is None or len(sym_dir) == 0:
        return pd.DataFrame(), False

    buy  = sym_dir[sym_dir["Type"].str.lower() == "buy"].copy()
    sell = sym_dir[sym_dir["Type"].str.lower() == "sell"].copy()

    rows = []
    asymmetric_syms = []
    for sym in set(buy["Symbol"].tolist() + sell["Symbol"].tolist()):
        b = buy[buy["Symbol"]==sym]
        s = sell[sell["Symbol"]==sym]
        bp = float(b["NetProfit"].sum()) if len(b) else 0
        sp = float(s["NetProfit"].sum()) if len(s) else 0
        bw = float(b["WinRate"].mean()) if len(b) else 0
        sw = float(s["WinRate"].mean()) if len(s) else 0
        bv = float(b["Volume"].sum()) if len(b) else 0
        sv = float(s["Volume"].sum()) if len(s) else 0

        # Asymmetry = one direction is profitable, other losing
        asym = (bp > 0 and sp < 0) or (sp > 0 and bp < 0)
        if asym: asymmetric_syms.append(sym)

        bias = "BUY bias" if bp > sp else ("SELL bias" if sp > bp else "Neutral")
        action = "Markup profitable direction" if asym else ("Monitor" if abs(bp-sp)>100 else "Normal")

        rows.append({
            "Symbol":         sym,
            "BUY Profit":     round(bp, 2),
            "SELL Profit":    round(sp, 2),
            "BUY WinRate%":   round(bw, 1),
            "SELL WinRate%":  round(sw, 1),
            "BUY Volume":     round(bv, 2),
            "SELL Volume":    round(sv, 2),
            "Directional Bias": bias,
            "Broker Action":  action,
        })
    return pd.DataFrame(rows), len(asymmetric_syms) > 0


def calc_ev_routing(kpi7, tdf):
    """
    Expected Value = (WR * AvgWin) + ((1-WR) * AvgLoss)
    Positive EV + high frequency = A-Book risk.
    Returns ev, ev_label, trend analysis.
    """
    wr   = kpi7["win_rate"] / 100
    aw   = kpi7["avg_win"]
    al   = kpi7["avg_loss"]   # negative number
    ev   = (wr * aw) + ((1 - wr) * al)
    ev_per_day = ev * kpi7["trades_per_day"]

    # Trend: split trades into 3 periods, check if EV improving
    trend_rows = []
    if "NetPnL_trade" in tdf.columns and "Open Time" in tdf.columns and len(tdf) >= 9:
        tdf_s   = tdf.sort_values("Open Time").reset_index(drop=True)
        chunk   = max(len(tdf_s) // 3, 1)
        labels  = ["Early","Mid","Recent"]
        for i, lbl in enumerate(labels):
            seg = tdf_s.iloc[i*chunk:(i+1)*chunk]
            sp  = float(seg["NetPnL_trade"].sum())
            sw  = float((seg["NetPnL_trade"] > 0).mean() * 100)
            trend_rows.append({"Period": lbl, "Trades": len(seg),
                                "Net P&L": round(sp, 2), "Win Rate%": round(sw, 1)})
    trend_df = pd.DataFrame(trend_rows)

    # Is trend improving?
    improving = False
    if len(trend_df) == 3:
        improving = trend_df.iloc[2]["Net P&L"] > trend_df.iloc[0]["Net P&L"]

    # EV routing override
    if ev > 0 and kpi7["trades_per_day"] >= 10:
        ev_label = "A-Book Override ⚠️"
        ev_color = "#f97316"
        ev_reason = f"EV=${ev:.2f}/trade × {kpi7['trades_per_day']:.0f} trades/day = ${ev_per_day:.0f}/day broker exposure. Must A-Book or apply full controls."
    elif ev > 0 and kpi7["trades_per_day"] >= 5:
        ev_label = "A-Book Recommended"
        ev_color = "#fbbf24"
        ev_reason = f"Positive EV=${ev:.2f}/trade. Profitable trader — B-Booking creates broker liability."
    elif ev > 0:
        ev_label = "Monitor"
        ev_color = "#60a5fa"
        ev_reason = f"Slight positive EV=${ev:.2f}/trade at low frequency. Low risk, but track performance."
    else:
        ev_label = "B-Book Safe"
        ev_color = "#22d3a0"
        ev_reason = f"Negative EV=${ev:.2f}/trade. Trader has no statistical edge. Safe to B-Book."

    return round(ev, 4), round(ev_per_day, 2), ev_label, ev_color, ev_reason, trend_df, improving


def calc_slippage_controls(tdf, trades_per_day, scalp_pct, slippage_threshold=30):
    """
    Rule-based slippage engine:
    - Trigger 1: ≥ threshold trades/day  → add slippage on ALL trades
    - Trigger 2: ≥ 30% of trades < 60s  → add slippage on short trades only
    - Slippage amount scales with frequency
    Returns per-trade slippage recommendation + revenue estimate.
    """
    short_trades = pd.DataFrame()
    if "Hold_Time" in tdf.columns:
        short_trades = tdf[tdf["Hold_Time"].dt.total_seconds() < 60].copy()
    short_pct = len(short_trades) / len(tdf) * 100 if len(tdf) else 0

    # Determine slippage pips
    if trades_per_day >= 100:
        slip_pip  = 1.0; reason = f"Ultra HFT ({trades_per_day:.0f}/day)"
    elif trades_per_day >= 50:
        slip_pip  = 0.7; reason = f"Very high freq ({trades_per_day:.0f}/day)"
    elif trades_per_day >= slippage_threshold:
        slip_pip  = 0.5; reason = f"High freq ≥{slippage_threshold} trades/day"
    elif short_pct >= 30:
        slip_pip  = 0.4; reason = f"{short_pct:.0f}% trades <60s"
    else:
        slip_pip  = 0.0; reason = "Below threshold — no slippage"

    triggered = slip_pip > 0

    # Revenue calculation per symbol pip value
    rev = 0.0
    if triggered and "Symbol" in tdf.columns and "Volume" in tdf.columns:
        for sym, grp in tdf.groupby("Symbol"):
            pv   = get_pip_val(sym)
            lots = float(grp["Volume"].sum())
            rev += slip_pip * pv * lots

    return {
        "triggered":    triggered,
        "slip_pip":     slip_pip,
        "reason":       reason,
        "short_trades": len(short_trades),
        "short_pct":    round(short_pct, 1),
        "trades_per_day": trades_per_day,
        "threshold":    slippage_threshold,
        "est_revenue":  round(rev, 2),
    }


def calc_delay_controls(tdf, kpi7, ea_sc):
    """
    Execution delay engine:
    - Trigger: % of trades with hold < 120s  OR  EA score >= 40
    - Delay applied only to short-hold trades
    - Estimates % of scalp profit captured back by delay
    """
    if "Hold_Time" not in tdf.columns:
        return {"triggered": False, "delay_ms": 0, "reason": "No hold time data",
                "scalp_trade_count": 0, "scalp_profit": 0, "captured": 0}

    scalp60  = tdf[tdf["Hold_Time"].dt.total_seconds() < 60]
    scalp120 = tdf[tdf["Hold_Time"].dt.total_seconds() < 120]
    pct60    = len(scalp60)  / len(tdf) * 100 if len(tdf) else 0
    pct120   = len(scalp120) / len(tdf) * 100 if len(tdf) else 0

    # Determine delay
    if ea_sc >= 70 or pct60 >= 40:
        delay_ms = 200; reason = f"EA={ea_sc} or {pct60:.0f}% trades <60s → 200ms"
    elif ea_sc >= 40 or pct120 >= 30:
        delay_ms = 100; reason = f"EA={ea_sc} or {pct120:.0f}% trades <120s → 100ms"
    elif pct120 >= 15:
        delay_ms = 50;  reason = f"{pct120:.0f}% trades <120s → 50ms"
    else:
        delay_ms = 0;   reason = "Normal hold times — no delay needed"

    triggered = delay_ms > 0

    # Capture estimate: delay degrades scalp edge by 20–40%
    scalp_p = float(scalp60["NetPnL_trade"].sum()) if "NetPnL_trade" in scalp60.columns else 0
    capture_rate = 0.35 if delay_ms >= 200 else (0.25 if delay_ms >= 100 else 0.15)
    captured = max(scalp_p * capture_rate, 0)

    return {
        "triggered":        triggered,
        "delay_ms":         delay_ms,
        "reason":           reason,
        "pct60":            round(pct60, 1),
        "pct120":           round(pct120, 1),
        "scalp_trade_count":len(scalp60),
        "scalp_profit":     round(scalp_p, 2),
        "capture_rate":     capture_rate,
        "captured":         round(captured, 2),
    }


def broker_revenue_v2(tdf, lev, kpi7, routing_label, risk_tier, ea_sc,
                      slip_ctrl, delay_ctrl, sym_ctrl, mkt_open_lots,
                      broker_gmt_offset=2.0):
    """
    Corrected 5-stream revenue calculator.
    routing_label: "A-Book" or "B-Book" only.
    risk_tier:     "High Risk" | "Medium Risk" | "Low Risk"

    KEY FIXES vs old version:
    - Spread markup and slippage do NOT double-count the same lots
      Slippage applies only to short-hold trades; markup on the rest
    - Delay capture capped at 10% of scalp profit (realistic broker edge)
    - Leverage carry uses trading_days not trade_count (correct denominator)
    - Market open lots use per-symbol pip val not hardcoded $8
    """
    tot_l = float(tdf["Volume"].sum()) if "Volume" in tdf.columns else 0

    # ── Determine markup pip based on routing + risk ──────────
    if routing_label == "A-Book" and risk_tier == "High Risk":
        mu_pips = 0.6   # A-Book high risk: aggressive markup to reduce their edge
    elif routing_label == "A-Book":
        mu_pips = 0.3
    elif routing_label == "B-Book" and risk_tier == "High Risk":
        mu_pips = 0.4
    else:
        mu_pips = 0.2   # B-Book low risk: minimal markup

    # ── 1. SPREAD MARKUP — only on NON-scalp lots to avoid double-count ──
    scalp_lots = 0.0
    non_scalp_lots = tot_l
    if "Hold_Time" in tdf.columns and "Symbol" in tdf.columns:
        scalp_mask = tdf["Hold_Time"].dt.total_seconds() < 120
        scalp_lots    = float(tdf.loc[scalp_mask, "Volume"].sum()) if "Volume" in tdf.columns else 0
        non_scalp_lots = tot_l - scalp_lots

    # Per-symbol pip-value-weighted markup on non-scalp lots
    mu_rev = 0.0
    if "Symbol" in tdf.columns and "Volume" in tdf.columns:
        for sym, grp in tdf.iterrows() if False else tdf.groupby("Symbol"):
            pv   = get_pip_val(sym)
            # Only non-scalp portion for this symbol
            if "Hold_Time" in tdf.columns:
                ns_lots = float(grp.loc[grp["Hold_Time"].dt.total_seconds() >= 120, "Volume"].sum())
            else:
                ns_lots = float(grp["Volume"].sum())
            mu_rev += mu_pips * pv * ns_lots
    else:
        mu_rev = mu_pips * 8.0 * non_scalp_lots

    apply_mu = f"+{mu_pips}pip on non-scalp lots ({non_scalp_lots:.1f} lots)"

    # ── 2. SLIPPAGE — only on scalp/short lots (not all lots) ──
    # slip_ctrl already has est_revenue calculated on short trades only
    slip_rev = slip_ctrl["est_revenue"]
    apply_sl = "ACTIVE ✅" if slip_ctrl["triggered"] else f"Not triggered (<{slip_ctrl['threshold']}/day)"

    # ── 3. EXECUTION DELAY — realistic 10% capture of scalp profit ──
    if "Hold_Time" in tdf.columns and "NetPnL_trade" in tdf.columns:
        scalp60_profit = float(tdf.loc[
            tdf["Hold_Time"].dt.total_seconds() < 60, "NetPnL_trade"].sum())
    else:
        scalp60_profit = 0.0
    # Realistic: delay degrades scalp entries by ~10% of their profit
    delay_capture_rate = 0.10 if delay_ctrl["triggered"] else 0.0
    delay_rev = max(scalp60_profit * delay_capture_rate, 0)
    apply_dl  = f"ACTIVE {delay_ctrl['delay_ms']}ms — 10% of scalp P&L ✅" \
                if delay_ctrl["triggered"] else "Not triggered"

    # ── 4. MARKET OPEN SPREAD ─────────────────────────────────
    mkt_markup_pip = 1.0
    mkt_open_rev   = 0.0
    if "Symbol" in tdf.columns and len(tdf) > 0 and mkt_open_lots > 0:
        # Estimate blended pip val for open trades
        blended_pv = float(sym_ctrl["PipValue($)"].mean()) if len(sym_ctrl) else 8.0
        blended_pv = min(blended_pv, 10.0)  # cap at $10 — open lots are mostly forex
        mkt_open_rev = mkt_markup_pip * blended_pv * mkt_open_lots
    apply_mkt = "Yes — session open trades" if mkt_open_lots > 0 else "No open-window trades"

    # ── 5. SWAP CARRY ─────────────────────────────────────────
    if "Hold_Time" in tdf.columns:
        ov_mask  = tdf["Hold_Time"].dt.total_seconds() > 82800
        ov_lots  = float(tdf.loc[ov_mask, "Volume"].sum()) if "Volume" in tdf.columns else 0
        avg_nights = min(tdf["Hold_Time"].mean().total_seconds() / 86400, 30) if len(tdf) else 0
    else:
        ov_lots = 0; avg_nights = 0
    swap_rate = 2.5
    swap_rev  = swap_rate * ov_lots * avg_nights
    apply_sw  = "Yes" if avg_nights > 0.5 else "Not applicable (intraday)"

    # NOTE: Leverage carry removed — it was multiplying by trade_count which
    # made it absurdly large. Real brokers charge leverage fees daily, not per-trade.
    # Swap carry already captures overnight cost. Avoided double-count.

    streams = {
        "Spread Markup": {
            "revenue": round(mu_rev, 2),
            "rate":    f"+{mu_pips} pip (non-scalp lots only)",
            "apply":   apply_mu,
            "formula": f"{mu_pips}pip × pip_val × {non_scalp_lots:.1f} non-scalp lots (scalp lots go to Slippage)"
        },
        "Slippage (freq-triggered)": {
            "revenue": round(slip_rev, 2),
            "rate":    f"+{slip_ctrl['slip_pip']} pip (scalp lots only)",
            "apply":   apply_sl,
            "formula": f"{slip_ctrl['reason']} | applied to {slip_ctrl['short_trades']} trades <60s"
        },
        "Execution Delay": {
            "revenue": round(delay_rev, 2),
            "rate":    f"{delay_ctrl['delay_ms']}ms → 10% edge reduction",
            "apply":   apply_dl,
            "formula": f"10% × ${scalp60_profit:,.2f} scalp P&L = ${delay_rev:,.2f} captured"
        },
        "Market Open Spread": {
            "revenue": round(mkt_open_rev, 2),
            "rate":    f"+{mkt_markup_pip} pip at session open",
            "apply":   apply_mkt,
            "formula": f"{mkt_markup_pip}pip × ${min(float(sym_ctrl['PipValue($)'].mean()) if len(sym_ctrl) else 8.0, 10.0):.1f}/pip × {mkt_open_lots:.2f} open lots"
        },
        "Swap Carry": {
            "revenue": round(swap_rev, 2),
            "rate":    f"${swap_rate}/lot/night",
            "apply":   apply_sw,
            "formula": f"${swap_rate} × {ov_lots:.2f} overnight lots × {avg_nights:.1f} avg nights"
        },
    }
    total = sum(v["revenue"] for v in streams.values())
    return streams, round(total, 2)


# ══════════════════════════════════════════════════════════════
#  ML ENGINE  v2 — adds EV feature, fixes contamination
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def build_ml_models():
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    import warnings; warnings.filterwarnings("ignore")

    rng = np.random.default_rng(42)

    def gen(n, params):
        rows = []
        for _ in range(n):
            row = {}
            for k, v in params.items():
                if isinstance(v, tuple) and len(v) == 3:
                    row[k] = int(rng.integers(int(v[0]), int(v[1])))
                elif isinstance(v, tuple) and len(v) == 2:
                    row[k] = float(rng.uniform(float(v[0]), float(v[1])))
                else:
                    row[k] = v
            rows.append(row)
        return rows

    # ── EV helper for synthetic data ──────────────────────────
    def make_ev(wr, rr):
        # EV = WR*RR*1 - (1-WR)*1  (normalised to 1-unit loss)
        return round((wr/100) * rr - (1 - wr/100), 4)

    # ── A-Book: profitable, positive EV, disciplined ──────────
    AB = dict(win_rate=(52.0,72.0), profit_factor=(1.3,3.5), rr_ratio=(1.2,3.5),
              avg_hold_sec=(1800.0,86400.0), scalp_pct=(0.0,15.0), ea_score=(0.0,30.0),
              mdd_pct=(5.0,28.0), trades_per_day=(0.5,8.0), max_cons_loss=(1,7,3),
              off_hours_pct=(0.0,12.0), leverage=(10.0,100.0))

    # ── B-Book: losing, negative EV, high DD ──────────────────
    BB = dict(win_rate=(25.0,48.0), profit_factor=(0.3,0.95), rr_ratio=(0.3,0.9),
              avg_hold_sec=(300.0,7200.0), scalp_pct=(5.0,50.0), ea_score=(5.0,50.0),
              mdd_pct=(30.0,80.0), trades_per_day=(1.0,30.0), max_cons_loss=(5,20,3),
              off_hours_pct=(5.0,40.0), leverage=(50.0,500.0))

    # ── Toxic: profitable, positive EV, HFT/EA/scalp ──────────
    # KEY FIX: Toxic CAN have low win rate but HIGH RR → positive EV
    TX = dict(win_rate=(30.0,85.0), profit_factor=(1.1,3.0), rr_ratio=(1.5,4.0),
              avg_hold_sec=(2.0,300.0), scalp_pct=(40.0,100.0), ea_score=(40.0,100.0),
              mdd_pct=(5.0,50.0), trades_per_day=(20.0,200.0), max_cons_loss=(1,10,3),
              off_hours_pct=(20.0,80.0), leverage=(100.0,500.0))

    rows_ab = gen(700, AB)
    rows_bb = gen(900, BB)
    rows_tx = gen(600, TX)

    # Inject EV feature into each profile
    for r in rows_ab: r["ev"] = make_ev(r["win_rate"], r["rr_ratio"]); r["label"] = 0
    for r in rows_bb: r["ev"] = make_ev(r["win_rate"], r["rr_ratio"]); r["label"] = 1
    for r in rows_tx: r["ev"] = make_ev(r["win_rate"], r["rr_ratio"]); r["label"] = 2

    all_rows = rows_ab + rows_bb + rows_tx
    rng.shuffle(all_rows)
    df_tr = pd.DataFrame(all_rows)

    # 12 features — added ev
    FEATURES = ["win_rate","profit_factor","rr_ratio","avg_hold_sec",
                "scalp_pct","ea_score","mdd_pct","trades_per_day",
                "max_cons_loss","off_hours_pct","leverage","ev"]

    for col in FEATURES:
        df_tr[col] = pd.to_numeric(df_tr[col], errors="coerce").fillna(0.0)

    X = df_tr[FEATURES].values.astype(float)
    y = df_tr["label"].values.astype(int)

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=12,
        random_state=42, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X, y)

    # contamination = 0.08 (realistic — 8% of accounts are anomalous)
    iso = IsolationForest(contamination=0.08, n_estimators=200, random_state=42)
    iso.fit(X[:, [3, 4, 5, 7, 9]])  # hold, scalp, ea, tpd, off_hours

    fi = dict(zip(FEATURES, clf.feature_importances_))
    return clf, iso, FEATURES, fi


def ml_predict(clf, iso, FEATURES, raw, kpi7, ea_sc):
    pf_val  = kpi7["profit_factor"]
    rr_val  = kpi7["rr_ratio"]
    pf_safe = float(pf_val) if not (isinstance(pf_val, float) and np.isnan(pf_val)) else 0.8
    rr_safe = float(rr_val) if not (isinstance(rr_val, float) and np.isnan(rr_val)) else 0.8

    wr_frac = raw["wr"] / 100
    ev_val  = float((wr_frac * rr_safe) - (1 - wr_frac))

    feats = [
        float(raw["wr"]),
        pf_safe, rr_safe,
        float(raw["avhs"]),
        float(raw["tox"]),
        float(ea_sc),
        float(abs(raw["mdd_p"])),
        float(kpi7["trades_per_day"]),
        float(kpi7["max_consec_losses"]),
        float(raw["offp"]),
        float(raw["lev"]),
        ev_val,
    ]

    feat_arr   = np.array([feats], dtype=float)
    pred_raw   = clf.predict(feat_arr)[0]
    proba      = clf.predict_proba(feat_arr)[0]
    iso_feats  = np.array([[raw["avhs"], raw["tox"], ea_sc,
                            kpi7["trades_per_day"], raw["offp"]]], dtype=float)
    toxic_flag = iso.predict(iso_feats)[0] == -1

    # ── ROUTING: only A-Book or B-Book ───────────────────────
    # Map: 0=A-Book, 1=B-Book, 2=was-Toxic → now treated as A-Book with high risk
    # "Toxic" from ML = skilled HFT/scalper = A-Book. Never a separate routing.
    if pred_raw == 2:
        # Model flagged as high-frequency profitable → A-Book
        routing = "A-Book"
    else:
        routing = "A-Book" if pred_raw == 0 else "B-Book"

    # EV override: positive EV + high frequency forces A-Book
    ev_override = False
    if routing == "B-Book" and ev_val > 0 and kpi7["trades_per_day"] >= 10:
        routing = "A-Book"
        ev_override = True
    elif routing == "B-Book" and ev_val > 0.05 and kpi7["trades_per_day"] >= 5:
        routing = "A-Book"
        ev_override = True

    # ── RISK TIER (separate from routing) ────────────────────
    # Answers: "how dangerous is this account?"
    risk_score = 0
    if toxic_flag:                          risk_score += 3
    if pred_raw == 2:                       risk_score += 2   # ML said toxic profile
    if ev_val > 0.1:                        risk_score += 2
    if kpi7["trades_per_day"] >= 30:        risk_score += 2
    if float(raw["tox"]) >= 40:             risk_score += 1
    if ea_sc >= 60:                         risk_score += 1

    if risk_score >= 6:    risk_tier = "High Risk"
    elif risk_score >= 3:  risk_tier = "Medium Risk"
    else:                  risk_tier = "Low Risk"

    # ── HYBRID recommendation ─────────────────────────────────
    # A-Book + controls = real-world "hybrid" for high-risk profitable traders
    if routing == "A-Book" and risk_tier == "High Risk":
        routing_display = "A-Book + Full Controls"
    elif routing == "A-Book" and risk_tier == "Medium Risk":
        routing_display = "A-Book + Monitor"
    elif routing == "B-Book" and risk_tier == "High Risk":
        routing_display = "B-Book + Controls"
    else:
        routing_display = routing

    # Low confidence flag
    low_conf = float(max(proba)) < 0.65

    return (pred_raw, routing, routing_display, risk_tier,
            proba, toxic_flag, feats, FEATURES,
            ev_val, ev_override, low_conf)


# ── keep old broker_revenue for fallback ─────────────────────
def broker_revenue(tdf, lev, total_dep, kpi7, ab_label, ea_sc):
    """Legacy wrapper — calls broker_revenue_v2 with defaults."""
    sym_ctrl   = calc_symbol_controls(tdf, None)
    slip_ctrl  = calc_slippage_controls(tdf, kpi7["trades_per_day"], kpi7.get("scalp_pct", 0))
    delay_ctrl = calc_delay_controls(tdf, kpi7, ea_sc)
    _, mkt_lots, _ = calc_market_open_trades(tdf, 2.0)
    risk_tier = "High Risk" if ab_label == "Toxic" else "Low Risk"
    routing   = "A-Book" if ab_label in ("A-Book","Toxic") else "B-Book"
    return broker_revenue_v2(tdf, lev, kpi7, routing, risk_tier, ea_sc,
                             slip_ctrl, delay_ctrl, sym_ctrl, mkt_lots)


# ── SESSION ANALYSIS ─────────────────────────────────────────
SESS_UTC_MIN = {
    "Sydney":   (21 * 60, 6 * 60),
    "Tokyo":    (0 * 60,  9 * 60),
    "London":   (7 * 60,  16 * 60),
    "New York": (13 * 60, 22 * 60),
}
SESS_IST = {
    "Sydney":   ("02:30 AM", "11:30 AM"),
    "Tokyo":    ("05:30 AM", "02:30 PM"),
    "London":   ("12:30 PM", "09:30 PM"),
    "New York": ("06:30 PM", "03:30 AM +1"),
}


def _fmt_min(m):
    m = int(m) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def assign_sess(dt, broker_offset_min):
    if pd.isna(dt): return "Unknown"
    utc_min = (dt.hour * 60 + dt.minute - broker_offset_min) % 1440
    active = []
    for nm, (s, e) in SESS_UTC_MIN.items():
        if s < e:
            if s <= utc_min < e: active.append(nm)
        else:
            if utc_min >= s or utc_min < e: active.append(nm)
    if not active: return "Off-Hours"
    if len(active) > 1: return " + ".join(sorted(active))
    return active[0]


def session_analysis(df, broker_gmt_offset):
    df = df.copy()
    off_min = int(round(broker_gmt_offset * 60))
    df["Session"] = df["Open Time"].apply(lambda t: assign_sess(t, off_min))
    df["DayOfWeek"] = df["Open Time"].dt.day_name()
    df["Hour"] = df["Open Time"].dt.hour

    ss = df.groupby("Session").agg(
        Trades=("Profit", "count"), Profit=("Profit", "sum"),
        AvgProfit=("Profit", "mean"),
        WinCount=("Profit", lambda x: (x > 0).sum())
    ).reset_index()
    ss["WinRate"] = (ss["WinCount"] / ss["Trades"] * 100).round(1)
    ss["Profit"] = ss["Profit"].round(2)
    ss["AvgProfit"] = ss["AvgProfit"].round(2)

    dow = df.groupby("DayOfWeek").agg(
        Trades=("Profit", "count"), Profit=("Profit", "sum")
    ).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]).dropna().reset_index()

    hrly = df.groupby("Hour").agg(
        Trades=("Profit", "count"), Profit=("Profit", "sum")
    ).reset_index()

    wins = []
    for nm, (s_utc, e_utc) in SESS_UTC_MIN.items():
        s_srv = (s_utc + off_min) % 1440
        e_srv = (e_utc + off_min) % 1440
        wins.append({
            "Session": nm,
            "IST Open": SESS_IST[nm][0], "IST Close": SESS_IST[nm][1],
            "UTC Open": _fmt_min(s_utc),  "UTC Close": _fmt_min(e_utc),
            f"Server Open  (GMT{broker_gmt_offset:+.1f})": _fmt_min(s_srv),
            f"Server Close (GMT{broker_gmt_offset:+.1f})": _fmt_min(e_srv),
        })
    return ss, dow, hrly, pd.DataFrame(wins)


# ── IP LOOKUP ────────────────────────────────────────────────
def lookup_ip(ip):
    try:
        r = requests.get(f"http://ip-api.com/json/{ip}", timeout=5).json()
        if r.get("status") != "success":
            return {"status": "fail", "message": r.get("message", "")}
        return {
            "status": "success", "ip": ip,
            "country": r.get("country"), "region": r.get("regionName"),
            "city": r.get("city"), "isp": r.get("isp"), "org": r.get("org"),
            "timezone": r.get("timezone"), "lat": r.get("lat"), "lon": r.get("lon")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── CONTRACT DEFAULTS ────────────────────────────────────────
def contract_defaults(sym):
    s = str(sym).upper()
    if "XAG" in s: return 5000, .001
    if "XAU" in s: return 100, .01
    if "JPY" in s: return 100000, .001
    if any(x in s for x in ["US30","NAS100","SPX500","GER40","UK100"]): return 1, .01
    if any(x in s for x in ["WTI","BRENT","OIL"]): return 1000, .01
    if any(x in s for x in ["BTC","ETH","XRP"]): return 1, .01
    return 100000, .00001


# ══════════════════════════════════════════════════════════════
#  MAIN ANALYZER
# ══════════════════════════════════════════════════════════════
def analyze(uploaded, lev):
    try:
        df_raw = pd.read_excel(uploaded, sheet_name=0, header=None)
        deps, wths, all_cf = parse_deals(df_raw)
        total_dep = float(deps["Amount"].sum()) if len(deps) else 0.0
        total_wth = float(abs(wths["Amount"].sum())) if len(wths) else 0.0
        eff_eq = total_dep if total_dep > 0 else 0.0

        pi = df_raw.index[df_raw[0].astype(str).str.contains("Positions", case=False, na=False)].tolist()
        oi = df_raw.index[df_raw[0].astype(str).str.contains("Orders", case=False, na=False)].tolist()
        if not pi:
            st.error("❌ 'Positions' section not found.")
            return None
        start = pi[0] + 1
        end = oi[0] if oi else len(df_raw)
        df = id_cols(df_raw.iloc[start:end])

        df["Open Time"]  = safe_dt(df["Open Time"])
        df["Close Time"] = safe_dt(df["Close Time"])
        df["Profit"]     = pd.to_numeric(df.get("Profit", 0),  errors="coerce").fillna(0)
        df["Volume"]     = pd.to_numeric(df.get("Volume", 0),  errors="coerce").fillna(0)
        cc_col = next((c for c in df.columns if "commission" in str(c).lower()), None)
        df["Commission"] = pd.to_numeric(df.get(cc_col, 0), errors="coerce").fillna(0) if cc_col else 0.0
        for col in ["Open Price", "Close Price"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        sc2 = next((c for c in df.columns if str(c).strip().lower() == "swap"), None)
        df["Swap"] = pd.to_numeric(df.get(sc2, 0), errors="coerce").fillna(0) if sc2 else 0.0

        df["Hold_Time"]    = df["Close Time"] - df["Open Time"]
        df["is_scalp"]     = df["Hold_Time"] <= timedelta(minutes=3)
        df["NetPnL_trade"] = df["Profit"] + df["Commission"] + df["Swap"]
        df["is_win"]       = df["NetPnL_trade"] > 0
        df["Date"]         = df["Close Time"].dt.date

        eq_df, mdd_a, mdd_p = equity_metrics(df, eff_eq)
        fin_eq = float(eq_df["Equity"].iloc[-1]) if len(eq_df) else eff_eq

        gross_p   = float(df["Profit"].sum())
        tot_c     = float(df["Commission"].sum())
        tot_swap  = float(df["Swap"].sum())
        net_p     = gross_p + tot_c + tot_swap
        tot_t     = len(df)
        tot_l     = float(df["Volume"].sum())
        scn       = int(df["is_scalp"].sum())
        scprf     = float(df.loc[df["is_scalp"], "NetPnL_trade"].sum())
        tox       = scn / tot_t * 100 if tot_t else 0.0
        wr        = float(df["is_win"].sum()) / tot_t * 100 if tot_t else 0.0

        pf = (
            float(df.loc[df["NetPnL_trade"] > 0, "NetPnL_trade"].sum()) /
            float(abs(df.loc[df["NetPnL_trade"] < 0, "NetPnL_trade"].sum()))
            if any(df["NetPnL_trade"] < 0) else np.nan
        )

        current_balance = total_dep + net_p - total_wth
        gpct = (current_balance - total_dep) / total_dep * 100 if total_dep else 0.0

        dpnl = df.groupby("Date")["NetPnL_trade"].sum().reset_index()
        dpnl.columns = ["Date", "Profit"]
        dpnl["Balance"] = eff_eq + dpnl["Profit"].cumsum()

        ss = df.groupby("Symbol", dropna=True).agg(
            Volume=("Volume","sum"), Trades=("NetPnL_trade","count"),
            GrossProfit=("Profit","sum"), Commission=("Commission","sum"),
            Swap=("Swap","sum"), NetProfit=("NetPnL_trade","sum")
        ).reset_index()
        if len(ss):
            ss["AvgNetProfit"] = (ss["NetProfit"] / ss["Trades"]).round(2)
            ss["WinRate%"] = ss["Symbol"].map(df.groupby("Symbol")["is_win"].mean() * 100).round(1)
            ss = ss.sort_values("Volume", ascending=False)
            vt = float(ss["Volume"].sum())
            sc = float(ss.head(3)["Volume"].sum() / vt * 100) if vt else 0.0
            tsym = ss.iloc[0]["Symbol"]
            tsh  = float(ss.iloc[0]["Volume"] / vt * 100) if vt else 0.0
        else:
            sc = tsym = tsh = 0

        sdir = df.groupby(["Symbol","Type"], dropna=True).agg(
            Trades=("NetPnL_trade","count"), Volume=("Volume","sum"),
            NetProfit=("NetPnL_trade","sum"), WinCount=("is_win","sum")
        ).reset_index()
        sdir["WinRate"]      = (sdir["WinCount"] / sdir["Trades"] * 100).round(1)
        sdir["AvgNetProfit"] = (sdir["NetProfit"] / sdir["Trades"]).round(2)
        sdir["NetProfit"]    = sdir["NetProfit"].round(2)

        r30  = int((eq_df["TradeReturnPct"] <= -30).sum())
        r50  = int((eq_df["TradeReturnPct"] <= -50).sum())
        r30p = r30 / len(eq_df) * 100 if len(eq_df) else 0.0
        r50p = r50 / len(eq_df) * 100 if len(eq_df) else 0.0
        avhs = float(df["Hold_Time"].mean().total_seconds()) if len(df) else 0.0
        offn = int(df["Open Time"].apply(lambda t: t.hour < 6 if not pd.isna(t) else False).sum())
        offp = offn / tot_t * 100 if tot_t else 0.0

        summ = {
            "Total Trades":        tot_t,
            "Gross Profit ($)":    round(gross_p, 2),
            "Commission ($)":      round(tot_c, 2),
            "Swap ($)":            round(tot_swap, 2),
            "Net Profit ($)":      round(net_p, 2),
            "Total Lots":          round(tot_l, 2),
            "Win Rate":            fmt_pct(wr),
            "Profit Factor":       f"{pf:.2f}" if not np.isnan(pf) else "N/A",
            "Avg Hold Time":       str(df["Hold_Time"].mean()).split(".")[0],
            "Scalping %":          fmt_pct(tox),
            "Symbol Concentration":fmt_pct(sc),
            "Total Deposits":      fmt_usd(total_dep),
            "Total Withdrawals":   fmt_usd(total_wth),
            "Current Balance":     fmt_usd(current_balance),
            "Growth %":            fmt_pct(gpct),
            "Max Drawdown ($)":    fmt_usd(mdd_a),
            "Max Drawdown (%)":    fmt_pct(mdd_p),
        }

        raw = {
            "gross_p": gross_p, "net_p": net_p, "tot_l": tot_l, "wr": wr, "tox": tox,
            "mdd_p": float(mdd_p), "mdd_a": float(mdd_a), "sc": sc,
            "r30p": r30p, "r50p": r50p,
            "gpct": gpct, "fin_eq": fin_eq, "eff_eq": eff_eq,
            "tot_c": tot_c, "tot_swap": tot_swap, "scn": scn, "scprf": scprf,
            "tsym": tsym, "tsh": tsh, "avhs": avhs, "offp": offp, "lev": float(lev),
            "pf": pf, "tot_t": tot_t, "total_dep": total_dep, "total_wth": total_wth,
            "current_balance": current_balance,
        }

        recs = []
        if tsym and tsh > 60:
            recs.append(("medium", f"{tsym} = {tsh:.1f}% of volume. Consider markup on this symbol."))
        if tox >= 40:
            recs.append(("high", "High scalping (≥40%). Widen spread +0.3 pips + execution delay."))
        elif tox >= 20:
            recs.append(("medium", "Moderate scalping. Monitor high-frequency windows."))
        if mdd_p <= -50:
            recs.append(("high", f"Deep drawdown ({mdd_p:.1f}%). Increase margin, reduce leverage."))
        elif gpct > 50:
            recs.append(("medium", f"Strong growth ({gpct:.1f}%). Monitor payout risk."))
        if r50p > 5:
            recs.append(("high", f"{r50p:.1f}% of trades risk >50% equity. Enforce volume limits."))
        elif r30p > 5:
            recs.append(("medium", f"{r30p:.1f}% of trades risk >30% equity. Reduce leverage."))
        if abs(tot_c) == 0 and tot_l > 0:
            recs.append(("low", "No commission detected. Add markup to monetize volume."))
        if lev > 0 and mdd_p <= -50:
            recs.append(("medium", f"Reduce leverage from {lev:.0f}x to ~{max(lev/2,10):.0f}x."))

        kpi7 = calc_seven_kpis(df, raw, mdd_a, mdd_p)
        raw["max_consec_wins"]   = kpi7["max_consec_wins"]
        raw["max_consec_losses"] = kpi7["max_consec_losses"]
        raw["rr_ratio"]          = kpi7["rr_ratio"]
        raw["trades_per_day"]    = kpi7["trades_per_day"]
        raw["avg_win"]           = kpi7["avg_win"]
        raw["avg_loss"]          = kpi7["avg_loss"]
        raw["profit_per_trade"]  = kpi7["profit_per_trade"]

        return summ, ss, sdir, dpnl, eq_df, recs, df, raw, deps, wths, all_cf, kpi7

    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback; st.code(traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════════
#  UI — HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""<div class="rms-header">
  <div class="rms-header-left">
    <h1>🛡️ Broker Risk &amp; Trade Intelligence Dashboard</h1>
    <p>RMS · MT4 / MT5 Report Analyzer · v3.1</p>
  </div>
  <div><span class="rms-badge">v3.1 Pro</span></div>
</div>""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Upload MT5 / MT4 Trade Report (.xlsx)</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["xlsx"], label_visibility="collapsed",
                             help="Export trade history from MT4/MT5 as .xlsx")

st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
sc1, sc2 = st.columns([1, 3])
with sc1:
    lev = st.number_input("Account Leverage (e.g. 500 for 1:500)", value=500.0, step=10.0)
with sc2:
    st.info("💡 **Initial Equity auto-detected** from Deals section deposit/withdrawal entries.")
st.markdown("---")

if uploaded is None:
    st.markdown("""<div style='text-align:center;padding:60px 0;'>
        <div style='font-size:56px;'>📂</div>
        <div style='font-size:17px;font-weight:700;margin-top:14px;color:#475569;'>
            Upload a Trade History report above to begin</div>
        <div style='font-size:12px;color:#334155;margin-top:8px;'>
            Supports MT4 &amp; MT5 Excel (.xlsx) Trade History exports</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

with st.spinner("🔍 Analyzing report..."):
    result = analyze(uploaded, lev)
if result is None:
    st.stop()

summ, sym_sum, sym_dir, dpnl, eq_df, recs, tdf, rm, deps, wths, all_cf, kpi7 = result

# ── KPI STRIP ────────────────────────────────────────────────
_np   = rm["net_p"]
_gp   = rm["gross_p"]
_cb   = rm["current_balance"]
_comm = rm["tot_c"]
_swap = rm["tot_swap"]
_wr   = rm["wr"]
_mddp = rm["mdd_p"]

if abs(_comm) + abs(_swap) > 0 and _gp != 0:
    fee_drag = (abs(_comm) + abs(_swap)) / abs(_gp) * 100
    st.markdown(alert(
        f"💰 <b>Fee Breakdown —</b> "
        f"Gross Profit: <b>{fmt_usd(_gp)}</b> | "
        f"Commission: <b>{fmt_usd(_comm)}</b> | "
        f"Swap: <b>{fmt_usd(_swap)}</b> | "
        f"Total Fee Drag: <b>{fmt_usd(_comm + _swap)}</b> ({fee_drag:.1f}% of gross) | "
        f"<b>True Net Profit: {fmt_usd(_np)}</b>",
        "medium" if fee_drag > 20 else "low"), unsafe_allow_html=True)

kp = '<div class="kpi-grid">'
kp += kpi("Net Profit",      fmt_usd(_np),  "green" if _np >= 0 else "red",  "After commission + swap costs")
kp += kpi("Total Deposits",  fmt_usd(rm["total_dep"]), "blue", f"Withdrawals: {fmt_usd(rm['total_wth'])}")
kp += kpi("Current Balance", fmt_usd(_cb),  "green" if _cb >= rm["total_dep"] else "red",
          f"Dep + NetP&L − Wth  |  {rm['gpct']:+.1f}%")
kp += kpi("Total Trades",    f"{rm['tot_t']:,}", "blue", f"Lots: {rm['tot_l']:.2f}  |  PF: {summ['Profit Factor']}")
kp += kpi("Win Rate",        fmt_pct(_wr),
          "green" if _wr >= 55 else "amber" if _wr >= 45 else "red",
          "≥55% Healthy  |  45-55% Marginal  |  <45% Poor")
kp += kpi("Max Drawdown",    fmt_usd(rm["mdd_a"]),
          "red" if abs(_mddp) >= 30 else "amber",
          fmt_pct(_mddp) + " from peak equity")
kp += '</div>'
st.markdown(kp, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
(T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11) = st.tabs([
    "📊 Overview", "💰 Cash Flows", "📈 Equity",
    "💹 Symbols",  "⚡ Trade Risk", "🤖 Bot / EA",
    "🕐 Sessions", "🎯 Risk Score", "🌍 IP Intel",
    "🔄 Swap",     "📋 Broker Actions"
])

# ── TAB 1 — OVERVIEW ─────────────────────────────────────────
with T1:
    st.markdown('<div class="section-title">Daily P&L</div>', unsafe_allow_html=True)
    f = go.Figure(go.Bar(
        x=dpnl["Date"].astype(str), y=dpnl["Profit"],
        marker_color=["#22d3a0" if v >= 0 else "#f87171" for v in dpnl["Profit"]]
    ))
    f.update_layout(title="Daily Profit / Loss", **CHART_LAYOUT)
    st.plotly_chart(f, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        def bk(td):
            if pd.isna(td): return "Unknown"
            s = td.total_seconds()
            if s <= 180:   return "0-3 min (Scalp)"
            if s <= 3600:  return "3-60 min"
            if s <= 10800: return "1-3 hr"
            if s <= 86400: return "3-24 hr"
            return "1+ day"

        eq_df["HoldBucket"] = eq_df["Hold_Time"].apply(bk)
        hc = eq_df["HoldBucket"].value_counts().reset_index(); hc.columns = ["Bucket","Count"]
        f2 = px.pie(hc, names="Bucket", values="Count",
                    color_discrete_sequence=["#3b82f6","#22d3a0","#fbbf24","#f87171","#a78bfa"])
        f2.update_layout(title="Hold Time Distribution", **CHART_LAYOUT)
        st.plotly_chart(f2, use_container_width=True)
    with c2:
        f3 = px.histogram(eq_df, x="Profit", nbins=60, color_discrete_sequence=["#3b82f6"])
        f3.update_layout(title="P&L per Trade", **CHART_LAYOUT)
        st.plotly_chart(f3, use_container_width=True)

    dl("Overview", {"Daily PnL": dpnl, "Equity": eq_df[["Close Time","Profit","Equity","Drawdown"]]},
       dpnl, "overview")

# ── TAB 2 — CASH FLOWS ───────────────────────────────────────
with T2:
    st.markdown('<div class="section-title">💰 Cash Flow — Auto-Extracted from Deals Section</div>',
                unsafe_allow_html=True)
    ch = '<div class="kpi-grid">'
    ch += kpi("Total Deposits",   fmt_usd(rm["total_dep"]), "green", f"{len(deps)} entries")
    ch += kpi("Total Withdrawals", fmt_usd(rm["total_wth"]), "red",  f"{len(wths)} entries")
    ch += kpi("Net Capital (Dep−Wth)", fmt_usd(rm["total_dep"] - rm["total_wth"]), "blue",
              "Deposits − Withdrawals")
    ch += '</div>'
    st.markdown(ch, unsafe_allow_html=True)

    sd, sw, sa = st.tabs(["🟢 Deposits", "🔴 Withdrawals", "📋 Overall"])
    with sd:
        if len(deps) > 0:
            f = go.Figure(go.Bar(x=deps["Time"].astype(str), y=deps["Amount"], marker_color="#22d3a0"))
            f.update_layout(title="Deposits Over Time", **CHART_LAYOUT); st.plotly_chart(f, use_container_width=True)
            st.dataframe(deps, use_container_width=True)
            dl("Deposits", {"Deposits": deps}, deps, "deposits")
        else:
            st.markdown(alert("No deposit entries found in Deals section.", "medium"), unsafe_allow_html=True)
    with sw:
        if len(wths) > 0:
            f = go.Figure(go.Bar(x=wths["Time"].astype(str), y=wths["Amount"], marker_color="#f87171"))
            f.update_layout(title="Withdrawals Over Time", **CHART_LAYOUT); st.plotly_chart(f, use_container_width=True)
            st.dataframe(wths, use_container_width=True)
            dl("Withdrawals", {"Withdrawals": wths}, wths, "withdrawals")
        else:
            st.markdown(alert("No withdrawal entries found in Deals section.", "low"), unsafe_allow_html=True)
    with sa:
        if len(all_cf) > 0:
            all2 = all_cf.copy()
            all2["DisplayAmount"] = all2["Amount"].abs()
            f = go.Figure()
            for tp, co in [("Deposit","#22d3a0"),("Withdrawal","#f87171")]:
                sub = all2[all2["CashType"] == tp]
                if len(sub):
                    f.add_trace(go.Bar(x=sub["Time"].astype(str), y=sub["DisplayAmount"], name=tp, marker_color=co))
            f.update_layout(title="All Cash Flow Entries", barmode="group", **CHART_LAYOUT)
            st.plotly_chart(f, use_container_width=True)
            disp_cols = ["Time","Amount","CashType","Comment"] if "Comment" in all2.columns else ["Time","Amount","CashType"]
            st.dataframe(all2[disp_cols], use_container_width=True)
            dl("CashFlows", {"All": all2, "Deposits": deps, "Withdrawals": wths}, all2, "cashflows")
        else:
            st.markdown(alert("No balance entries found in Deals section.", "medium"), unsafe_allow_html=True)

# ── TAB 3 — EQUITY ───────────────────────────────────────────
with T3:
    st.caption("📌 Equity curve & drawdown computed on each **trade close time** — when P&L is actually realised.")
    f = go.Figure()
    f.add_trace(go.Scatter(x=eq_df["Close Time"], y=eq_df["Equity"],     mode="lines",
                           line=dict(color="#60a5fa", width=2), name="Equity"))
    f.add_trace(go.Scatter(x=eq_df["Close Time"], y=eq_df["EquityPeak"], mode="lines",
                           line=dict(color="#22d3a0", width=1, dash="dot"), name="Peak"))
    f.update_layout(title="Equity Curve vs Peak", **CHART_LAYOUT); st.plotly_chart(f, use_container_width=True)

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=eq_df["Close Time"], y=eq_df["DrawdownPct"],
                            fill="tozeroy", line=dict(color="#f87171"), name="Drawdown %"))
    f2.update_layout(title="Drawdown % Over Time", **CHART_LAYOUT); st.plotly_chart(f2, use_container_width=True)
    dl("Equity", {"Equity Curve": eq_df[["Close Time","Equity","EquityPeak","Drawdown","DrawdownPct"]]},
       eq_df[["Close Time","Equity","DrawdownPct"]], "equity")

# ── TAB 4 — SYMBOLS ──────────────────────────────────────────
with T4:
    st.markdown('<div class="section-title">Overall Symbol Performance</div>', unsafe_allow_html=True)
    st.dataframe(sym_sum, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        f = px.pie(sym_sum.head(6), names="Symbol", values="Volume", title="Volume by Symbol",
                   color_discrete_sequence=px.colors.sequential.Blues_r)
        f.update_layout(**CHART_LAYOUT); st.plotly_chart(f, use_container_width=True)
    with c2:
        f2 = px.bar(sym_sum.head(10), x="Symbol", y="NetProfit", color="NetProfit",
                    color_continuous_scale=["#f87171","#fbbf24","#22d3a0"], title="Net Profit by Symbol")
        f2.update_layout(**CHART_LAYOUT); st.plotly_chart(f2, use_container_width=True)

    st.markdown('<div class="section-title">Symbol Profit: BUY vs SELL Direction</div>', unsafe_allow_html=True)
    st.dataframe(sym_dir, use_container_width=True)
    buy_d  = sym_dir[sym_dir["Type"].str.lower() == "buy"]
    sell_d = sym_dir[sym_dir["Type"].str.lower() == "sell"]
    fb = go.Figure()
    fb.add_trace(go.Bar(x=buy_d["Symbol"],  y=buy_d["NetProfit"],  name="BUY",  marker_color="#22d3a0"))
    fb.add_trace(go.Bar(x=sell_d["Symbol"], y=sell_d["NetProfit"], name="SELL", marker_color="#f87171"))
    fb.update_layout(title="Profit: BUY vs SELL per Symbol", barmode="group", **CHART_LAYOUT)
    st.plotly_chart(fb, use_container_width=True)
    fw = go.Figure()
    fw.add_trace(go.Bar(x=buy_d["Symbol"],  y=buy_d["WinRate"],  name="BUY Win%",  marker_color="#3b82f6"))
    fw.add_trace(go.Bar(x=sell_d["Symbol"], y=sell_d["WinRate"], name="SELL Win%", marker_color="#a78bfa"))
    fw.update_layout(title="Win Rate %: BUY vs SELL per Symbol", barmode="group",
                     yaxis_title="Win Rate %", **CHART_LAYOUT)
    st.plotly_chart(fw, use_container_width=True)

    if "Symbol" in tdf.columns:
        st.markdown('<div class="section-title">Profit Heatmap: Symbol × Day</div>', unsafe_allow_html=True)
        tdf["DOW"] = tdf["Open Time"].dt.day_name()
        heat = tdf.groupby(["Symbol","DOW"])["Profit"].sum().reset_index()
        hp = heat.pivot(index="Symbol", columns="DOW", values="Profit").fillna(0)
        hp = hp[[c for c in ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"] if c in hp.columns]]
        fh = px.imshow(hp, color_continuous_scale="RdYlGn", title="Profit Heatmap")
        fh.update_layout(**CHART_LAYOUT); st.plotly_chart(fh, use_container_width=True)

    dl("Symbols", {"Symbol Summary": sym_sum, "Symbol By Direction": sym_dir}, sym_sum, "symbols")

# ── TAB 5 — TRADE RISK ───────────────────────────────────────
with T5:
    st.markdown('<div class="section-title">Per-Trade Risk vs Equity</div>', unsafe_allow_html=True)
    f = px.scatter(eq_df.reset_index(), x="index", y="TradeReturnPct", color="RiskBucket",
                   color_discrete_map={"Normal":"#3b82f6","High (≤ -30%)":"#fbbf24","Very High (≤ -50%)":"#f87171"},
                   title="Each trade's P&L as % of equity at close",
                   labels={"index":"Trade #","TradeReturnPct":"Return % of Equity"},
                   hover_data=["Profit","EquityBefore"])
    f.add_hline(y=-30, line_dash="dot", line_color="#fbbf24", annotation_text="-30%")
    f.add_hline(y=-50, line_dash="dot", line_color="#f87171", annotation_text="-50%")
    f.update_layout(**CHART_LAYOUT); st.plotly_chart(f, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        rc = eq_df["RiskBucket"].value_counts().reset_index(); rc.columns = ["RiskBucket","Trades"]
        st.dataframe(rc, use_container_width=True)
    with c2:
        if "S / L" in tdf.columns and "T / P" in tdf.columns:
            nsl = tdf["S / L"].isna().sum(); ntp = tdf["T / P"].isna().sum()
            st.markdown(alert(f"NO Stop Loss: {nsl} trades ({nsl/len(tdf)*100:.1f}%)",
                              "high" if nsl/len(tdf) > .2 else "medium"), unsafe_allow_html=True)
            st.markdown(alert(f"NO Take Profit: {ntp} trades ({ntp/len(tdf)*100:.1f}%)", "medium"),
                        unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔎 Broker Insights from Trade Risk Data</div>', unsafe_allow_html=True)
    ih = ""
    dt2 = tdf.groupby(tdf["Open Time"].dt.date).size(); avg_d = dt2.mean(); max_d = dt2.max(); mx_day = dt2.idxmax()
    if max_d > avg_d * 3:
        ih += insight("🔥 Overtrading Detected",
                      f"Peak: <b>{int(max_d)} trades on {mx_day}</b> vs avg {avg_d:.0f}/day. "
                      "Indicates panic, revenge trading or EA burst. "
                      "<b>Broker action:</b> Monitor margin on peak days.", "#f87171")
    else:
        ih += insight("✅ Trade Frequency Normal",
                      f"Max {int(max_d)} trades/day vs avg {avg_d:.0f}. No overtrading spikes.", "#22d3a0")

    vols = eq_df["Volume"].dropna()
    if len(vols) > 5:
        cv = vols.std() / vols.mean() * 100 if vols.mean() else 0
        if cv > 80:
            ih += insight("⚠️ Erratic Lot Sizing",
                          f"Lot CV = <b>{cv:.0f}%</b>. Wildly varying sizes suggest martingale/grid. "
                          "<b>Broker action:</b> Progressive margin tiers.", "#f97316")
        elif cv > 40:
            ih += insight("◉ Moderate Lot Variation", f"Lot CV = {cv:.0f}%. Watch for martingale patterns.", "#fbbf24")
        else:
            ih += insight("✅ Consistent Lot Sizing", f"Lot CV = {cv:.0f}%. Uniform sizing — lower tail risk.", "#22d3a0")

    gp2 = tdf.loc[tdf["Profit"] > 0, "Profit"].sum()
    if gp2 > 0 and abs(rm["tot_c"]) > 0:
        cd = abs(rm["tot_c"]) / gp2 * 100
        ih += insight("💸 Commission Drag",
                      f"Commission = <b>{fmt_usd(abs(rm['tot_c']))}</b> = <b>{cd:.1f}% of gross profit</b>. "
                      + ("Commission-sensitive — spread widening may push client away." if cd > 30 else "Moderate drag."),
                      "#a78bfa")
    elif abs(rm["tot_c"]) == 0:
        ih += insight("💡 Zero Commission",
                      f"No commission charged. <b>Broker opportunity:</b> Introduce per-lot commission on "
                      f"{rm['tot_l']:.1f} total lots.", "#60a5fa")

    cl2 = ms2 = 0; st2_list = []
    for p in eq_df["Profit"]:
        cl2 = (cl2 + 1 if p < 0 else (st2_list.append(cl2) or 0)); ms2 = max(ms2, cl2)
    avs2 = np.mean([s for s in st2_list if s > 0]) if any(s > 0 for s in st2_list) else 0
    ih += insight(f"📉 Max Consecutive Loss Streak: {ms2}",
                  f"Avg streak = {avs2:.1f}. " + (
                      f"<b>Broker risk:</b> {ms2} consecutive losses → likely precedes large revenge trade."
                      if ms2 >= 10 else "Within normal range."),
                  "#f87171" if ms2 >= 10 else "#fbbf24")

    wd_p = dpnl["Profit"].min(); wd_d = dpnl.loc[dpnl["Profit"].idxmin(), "Date"]
    wdpct = abs(wd_p) / rm["total_dep"] * 100 if rm["total_dep"] else 0
    ih += insight(f"📅 Worst Day: {wd_d}",
                  f"Loss of <b>{fmt_usd(wd_p)}</b> = <b>{wdpct:.1f}% of net capital</b>. "
                  + ("<b>Broker action:</b> Implement daily loss limit — auto-freeze after 20% daily equity loss."
                     if wdpct > 20 else "Single-day loss manageable."),
                  "#f87171" if wdpct > 20 else "#fbbf24")
    st.markdown(ih, unsafe_allow_html=True)
    dl("Trade Risk",
       {"Trade Risk": eq_df[["Close Time","Profit","TradeReturnPct","RiskBucket","Equity"]]},
       eq_df[["Close Time","Profit","TradeReturnPct","RiskBucket"]], "traderisk")

# ── TAB 6 — BOT/EA ───────────────────────────────────────────
with T6:
    st.markdown('<div class="section-title">🤖 Automated Trading Detection</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1: bs = st.slider("Burst window (sec)", 1, 10, 2, key="bs")
    with b2: rs = st.slider("Reversal window (sec)", 5, 60, 20, key="rs")

    bdf_ea, bev, bmx = detect_burst(tdf, bs)
    rdf_ea, rev      = detect_reversal(tdf, rs)
    avhs2 = tdf["Hold_Time"].mean().total_seconds() if "Hold_Time" in tdf.columns else 0
    eas = ea_score(len(bdf_ea), len(rdf_ea), len(tdf), rm["offp"], avhs2)

    fg = go.Figure(go.Indicator(
        mode="gauge+number", value=eas,
        title={"text":"EA / Bot Probability","font":{"color":"#94a3b8"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"#475569"},
               "bar":{"color":"#f87171" if eas>60 else "#fbbf24" if eas>30 else "#22d3a0"},
               "steps":[{"range":[0,30],"color":"#0d2a1a"},{"range":[30,60],"color":"#1a1500"},
                         {"range":[60,100],"color":"#1a0505"}],
               "threshold":{"line":{"color":"#f87171","width":3},"thickness":.75,"value":70}},
        number={"suffix":"/100","font":{"color":"#e2e8f0","size":36}}))
    fg.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#94a3b8"}, height=280, margin=dict(t=60,b=20))
    st.plotly_chart(fg, use_container_width=True)

    bh = '<div class="kpi-grid">'
    bh += kpi("Burst Trades",   str(len(bdf_ea)), "red" if len(bdf_ea)>20 else "blue", f"{bev} events · max {bmx}/burst")
    bh += kpi("Reversal Pairs", str(len(rdf_ea)), "amber" if len(rdf_ea)>10 else "blue", f"{rev} events")
    bh += kpi("Off-Hours %",    fmt_pct(rm["offp"]), "red" if rm["offp"]>15 else "blue", "00:00–06:00 server time")
    bh += kpi("Avg Hold",       f"{avhs2:.0f}s", "red" if avhs2<30 else "green", "< 30s = likely EA")
    bh += '</div>'
    st.markdown(bh, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if len(bdf_ea) > 0:
            st.markdown('<div class="section-title">Burst Trades</div>', unsafe_allow_html=True)
            st.dataframe(bdf_ea[["Open Time","Symbol","Type","Volume","Profit"]].head(100), use_container_width=True)
        else:
            st.markdown(alert("No burst trades detected.", "good"), unsafe_allow_html=True)
    with c2:
        if len(rdf_ea) > 0:
            st.markdown('<div class="section-title">Reversal Trades</div>', unsafe_allow_html=True)
            st.dataframe(rdf_ea[["Open Time","Symbol","Type","Volume","Profit"]].head(100), use_container_width=True)
        else:
            st.markdown(alert("No reversal trades detected.", "good"), unsafe_allow_html=True)

    hr2 = tdf.copy(); hr2["Hour"] = hr2["Open Time"].dt.hour
    fhr = px.bar(hr2.groupby("Hour").size().reset_index(name="Trades"), x="Hour", y="Trades",
                 color="Trades", color_continuous_scale="Reds", title="Trades by Server Hour")
    fhr.update_layout(**CHART_LAYOUT); st.plotly_chart(fhr, use_container_width=True)

    cmb = pd.concat([bdf_ea.assign(DetectionType="Burst"), rdf_ea.assign(DetectionType="Reversal")]).drop_duplicates()
    dl("Bot/EA", {"Burst": bdf_ea, "Reversal": rdf_ea, "Combined": cmb}, cmb, "bot_ea")

# ── TAB 7 — SESSIONS ─────────────────────────────────────────
with T7:
    st.markdown('<div class="section-title">Trading Session Breakdown</div>', unsafe_allow_html=True)
    gs1, gs2 = st.columns([1, 3])
    with gs1:
        gmt = st.number_input("Broker Server GMT Offset\n(e.g. 2 = GMT+2, 5.5 = IST)",
                              value=2.0, step=0.5, min_value=-12.0, max_value=14.0, key="gmt")
    with gs2:
        lon_srv_min = (7 * 60 + int(round(gmt * 60))) % 1440
        lon_srv_str = f"{lon_srv_min // 60:02d}:{lon_srv_min % 60:02d}"
        st.markdown(alert(
            f"Session windows are <b>IST-verified</b>. "
            f"Broker server = <b>GMT{gmt:+.1f}</b>. "
            f"London opens 07:00 UTC → <b>{lon_srv_str} server time</b>.",
            "low"), unsafe_allow_html=True)

    ss_s, dow_s, hrly_s, wins_s = session_analysis(tdf, gmt)

    st.markdown('<div class="section-title">Session Reference — IST · UTC · Server Time</div>', unsafe_allow_html=True)
    st.dataframe(wins_s, use_container_width=True)

    fs = px.bar(ss_s, x="Session", y="Profit", color="WinRate",
                color_continuous_scale="RdYlGn",
                text=ss_s["WinRate"].apply(lambda x: f"{x:.0f}% WR"),
                title=f"Net Profit by Session  (broker server = GMT{gmt:+.1f})")
    fs.update_layout(**CHART_LAYOUT); st.plotly_chart(fs, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fd = px.bar(dow_s, x="DayOfWeek", y="Profit", color="Profit",
                    color_continuous_scale="RdYlGn", title="P&L by Day of Week")
        fd.update_layout(**CHART_LAYOUT); st.plotly_chart(fd, use_container_width=True)
    with c2:
        fh2 = px.bar(hrly_s, x="Hour", y="Trades", color="Profit",
                     color_continuous_scale="RdYlGn", title=f"Trades by Server Hour (GMT{gmt:+.1f})")
        fh2.update_layout(**CHART_LAYOUT); st.plotly_chart(fh2, use_container_width=True)

    st.markdown('<div class="section-title">Session Stats Table</div>', unsafe_allow_html=True)
    st.dataframe(ss_s, use_container_width=True)
    dl("Sessions", {"Session Stats": ss_s, "DOW": dow_s, "Hourly": hrly_s, "Reference": wins_s}, ss_s, "sessions")

# ── TAB 8 — RISK SCORE ───────────────────────────────────────
with T8:
    avhs3 = tdf["Hold_Time"].mean().total_seconds() if "Hold_Time" in tdf.columns else 0
    bd3, _, _ = detect_burst(tdf, 2); rd3, _ = detect_reversal(tdf, 20)
    eas3 = ea_score(len(bd3), len(rd3), len(tdf), rm["offp"], avhs3)
    rs_val, rs_lv, rs_co, rs_br = risk_score(rm["sc"], rm["mdd_p"], rm["tox"], rm["r30p"], eas3)

    c1, c2 = st.columns([1, 2])
    with c1:
        fg2 = go.Figure(go.Indicator(
            mode="gauge+number", value=rs_val,
            gauge={"axis":{"range":[0,100],"tickcolor":"#475569"},
                   "bar":{"color":rs_co},
                   "steps":[{"range":[0,35],"color":"#052010"},{"range":[35,55],"color":"#1a1500"},
                             {"range":[55,75],"color":"#1a0a00"},{"range":[75,100],"color":"#1a0505"}]},
            number={"suffix":"/100","font":{"color":"#e2e8f0","size":48}},
            title={"text":f"Risk Level: {rs_lv}","font":{"color":rs_co,"size":14}}))
        fg2.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#94a3b8"}, height=320, margin=dict(t=40,b=20))
        st.plotly_chart(fg2, use_container_width=True)
    with c2:
        fbr = px.bar(rs_br, x="Points", y="Component", orientation="h", color="Points",
                     color_continuous_scale="Reds",
                     text=rs_br.apply(lambda r: f"{r['Points']:.1f}/{r['Max']}", axis=1))
        fbr.update_traces(textposition="outside")
        fbr.update_layout(title="Risk Score Components", **CHART_LAYOUT)
        st.plotly_chart(fbr, use_container_width=True)

    mm = {
        "CRITICAL": ("high",   "CRITICAL risk. Reduce leverage, add margin calls, manual review all trades."),
        "HIGH":     ("high",   "HIGH risk. Apply tighter position limits and enhanced monitoring."),
        "MEDIUM":   ("medium", "MEDIUM risk. Standard monitoring with periodic reviews."),
        "LOW":      ("good",   "LOW risk client. Normal monitoring applies."),
    }
    st.markdown(alert(mm[rs_lv][1], mm[rs_lv][0]), unsafe_allow_html=True)
    st.dataframe(rs_br, use_container_width=True)
    dl("Risk Score", {"Risk Score": rs_br}, rs_br, "risk_score")

# ── TAB 9 — IP INTEL ─────────────────────────────────────────
with T9:
    st.markdown('<div class="section-title">🌍 IP Address Intelligence</div>', unsafe_allow_html=True)
    ip_in = st.text_input("Enter IP Address", placeholder="e.g. 185.220.101.1", key="ip_in")
    if ip_in:
        with st.spinner(f"Looking up {ip_in}..."):
            ipd = lookup_ip(ip_in.strip())
        if ipd.get("status") == "success":
            ih2 = '<div class="kpi-grid">'
            ih2 += kpi("Country",  ipd.get("country","N/A"),  "blue")
            ih2 += kpi("City",     ipd.get("city","N/A"),     "blue")
            ih2 += kpi("ISP",      ipd.get("isp","N/A"),      "amber")
            ih2 += kpi("Timezone", ipd.get("timezone","N/A"), "green")
            ih2 += '</div>'
            st.markdown(ih2, unsafe_allow_html=True)
            st.markdown(f"**Region:** {ipd.get('region','N/A')} | **Org:** {ipd.get('org','N/A')}")
            if ipd.get("lat") and ipd.get("lon"):
                st.map(pd.DataFrame([{"lat": ipd["lat"], "lon": ipd["lon"]}]), zoom=4)
            kws = ["vpn","proxy","datacenter","hosting","tor","socks","anonymous","relay"]
            fl = [k for k in kws if k in (ipd.get("org","") + ipd.get("isp","")).lower()]
            st.markdown(
                alert(f"VPN/Proxy keywords found: {', '.join(fl)}", "high")
                if fl else alert("No VPN/proxy detected.", "good"),
                unsafe_allow_html=True)
            ip_df = pd.DataFrame([ipd]); dl("IP", {"IP Lookup": ip_df}, ip_df, "ip")
        else:
            st.error(f"Lookup failed: {ipd.get('message','')}")

    st.markdown("---")
    st.markdown('<div class="section-title">Bulk IP Lookup</div>', unsafe_allow_html=True)
    bulk = st.text_area("Paste IPs (one per line)", height=100, key="bulk_ip")
    if st.button("🔍 Lookup All", key="bulk_btn") and bulk.strip():
        ips = [x.strip() for x in bulk.splitlines() if x.strip()]
        res = []; pr = st.progress(0)
        for i, ip in enumerate(ips[:50]):
            res.append(lookup_ip(ip)); pr.progress((i + 1) / len(ips))
        pr.empty()
        bdf2 = pd.DataFrame([r for r in res if r.get("status") == "success"])
        if len(bdf2):
            st.dataframe(bdf2, use_container_width=True); dl("Bulk IP", {"Bulk": bdf2}, bdf2, "bulk_ip")
        else:
            st.warning("No successful lookups.")

# ── TAB 10 — SWAP ────────────────────────────────────────────
with T10:
    st.markdown('<div class="section-title">🔄 Swap Calculator</div>', unsafe_allow_html=True)
    sw1, sw2 = st.columns(2)
    with sw1:
        pfx = int(st.number_input("Symbol prefix length", min_value=3, max_value=10, value=6, key="sw_pfx"))
    with sw2:
        DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
        tri_name = st.selectbox("Triple Swap Day", DAYS, index=2, key="sw_tri")
        tri_idx  = DAYS.index(tri_name)

    srows = []
    for _, r in tdf.iterrows():
        nd, td = swap_nights(r["Open Time"], r["Close Time"], tri_idx)
        srows.append({
            "Symbol": r.get("Symbol",""), "SymbolPrefix": str(r.get("Symbol",""))[:pfx],
            "OrderType": r.get("Type",""), "Volume": r.get("Volume",0),
            "Open Time": r["Open Time"], "Close Time": r["Close Time"],
            "NormalDays": nd, "TripleDays": td, "SwapDays_Final": nd + td * 3
        })
    sdf = pd.DataFrame(srows)

    us = sdf[["SymbolPrefix"]].drop_duplicates().copy()
    us[["ContractSize","Point"]] = us.apply(lambda r: pd.Series(contract_defaults(r["SymbolPrefix"])), axis=1)
    if "sw_master" not in st.session_state: st.session_state.sw_master = us.copy()
    st.markdown("**Symbol Configuration**")
    scfg = st.data_editor(st.session_state.sw_master, use_container_width=True, key="sw_sym_cfg")
    st.session_state.sw_master = scfg

    uc = sdf[["SymbolPrefix","OrderType"]].drop_duplicates().copy(); uc["SwapRate"] = 0.0
    if "sw_rates" not in st.session_state: st.session_state.sw_rates = uc.copy()
    st.markdown(f"**Swap Rate Override** (Triple day = **{tri_name}**, charged ×3)")
    srt = st.data_editor(st.session_state.sw_rates, use_container_width=True, key="sw_rates_ed")
    st.session_state.sw_rates = srt

    sdf = sdf.merge(scfg[["SymbolPrefix","ContractSize","Point"]], on="SymbolPrefix", how="left")
    sdf = sdf.merge(srt[["SymbolPrefix","OrderType","SwapRate"]], on=["SymbolPrefix","OrderType"], how="left")
    for col, dv in [("SwapRate",0),("Volume",0),("NormalDays",0),("TripleDays",0),
                    ("SwapDays_Final",0),("ContractSize",100000),("Point",0.00001)]:
        sdf[col] = pd.to_numeric(sdf[col], errors="coerce").fillna(dv)

    sdf["NormalSwapMoney"] = sdf["SwapRate"] * sdf["ContractSize"] * sdf["Point"] * sdf["Volume"] * sdf["NormalDays"]
    sdf["TripleSwapMoney"] = sdf["SwapRate"] * sdf["ContractSize"] * sdf["Point"] * sdf["Volume"] * sdf["TripleDays"] * 3
    sdf["TotalSwapMoney"]  = sdf["NormalSwapMoney"] + sdf["TripleSwapMoney"]

    st.markdown('<div class="section-title">Swap Per Trade</div>', unsafe_allow_html=True)
    dcols = ["Symbol","OrderType","Volume","Open Time","Close Time","NormalDays","TripleDays",
             "SwapDays_Final","SwapRate","NormalSwapMoney","TripleSwapMoney","TotalSwapMoney"]
    st.dataframe(sdf[dcols], use_container_width=True)

    cf2 = sdf[sdf["SwapDays_Final"] > 0]
    ssumm = cf2.groupby(["SymbolPrefix","OrderType"], dropna=True).agg(
        Trades=("Symbol","count"), Volume=("Volume","sum"),
        NormalDays=("NormalDays","sum"), TripleDays=("TripleDays","sum"),
        TotalSwapDays=("SwapDays_Final","sum"),
        NormalSwap=("NormalSwapMoney","sum"), TripleSwap=("TripleSwapMoney","sum"),
        TotalSwap=("TotalSwapMoney","sum")
    ).reset_index()
    tr = pd.DataFrame([{"SymbolPrefix":"TOTAL","OrderType":"",
                        "Trades":ssumm["Trades"].sum(),"Volume":ssumm["Volume"].sum(),
                        "NormalDays":ssumm["NormalDays"].sum(),"TripleDays":ssumm["TripleDays"].sum(),
                        "TotalSwapDays":ssumm["TotalSwapDays"].sum(),
                        "NormalSwap":ssumm["NormalSwap"].sum(),"TripleSwap":ssumm["TripleSwap"].sum(),
                        "TotalSwap":ssumm["TotalSwap"].sum()}])
    ssumm = pd.concat([ssumm, tr], ignore_index=True)
    st.markdown(f'<div class="section-title">BUY vs SELL Summary (Triple = {tri_name})</div>', unsafe_allow_html=True)
    st.dataframe(ssumm, use_container_width=True)
    dl("Swap", {"Swap Per Trade": sdf[dcols], "Swap Summary": ssumm, "Symbol Config": scfg, "Swap Rates": srt},
       sdf[dcols], "swap")

# ── TAB 11 — BROKER ACTIONS v2 ───────────────────────────────
with T11:
    with st.spinner("🧠 Loading ML models (v2 — EV feature, 12 inputs)..."):
        _clf, _iso, _FEAT, _fi = build_ml_models()

    # ── Core detections ───────────────────────────────────────
    _bd2, _, _  = detect_burst(tdf, 2)
    _rv2, _     = detect_reversal(tdf, 20)
    _avhs2      = float(tdf["Hold_Time"].mean().total_seconds()) if "Hold_Time" in tdf.columns else 0.0
    _ea_sc      = ea_score(len(_bd2), len(_rv2), len(tdf), rm["offp"], _avhs2)
    _avh        = kpi7["avg_hold_sec"]
    _vol_lim    = "Reduce 50%" if abs(rm["mdd_p"])>=50 else ("Reduce 25%" if abs(rm["mdd_p"])>=30 else "Normal")

    # ── New engines ───────────────────────────────────────────
    _gmt_val    = 2.0   # default; user can change in Sessions tab
    _slip_ctrl  = calc_slippage_controls(tdf, kpi7["trades_per_day"], rm["tox"], slippage_threshold=30)
    _delay_ctrl = calc_delay_controls(tdf, kpi7, _ea_sc)
    _sym_ctrl   = calc_symbol_controls(tdf, sym_dir)
    _asym_df, _has_asym = calc_order_asymmetry(sym_dir)
    _ev, _ev_pd, _ev_label, _ev_color, _ev_reason, _trend_df, _improving = calc_ev_routing(kpi7, tdf)
    _mkt_cnt, _mkt_lots, _mkt_df = calc_market_open_trades(tdf, _gmt_val)

    # ── ML predict (only A-Book / B-Book routing + risk tier) ──
    (_pred_raw, _routing, _routing_display, _risk_tier,
     _proba, _tox_flag, _feats, _FEATURES,
     _ev_feat, _ev_override, _low_conf) = ml_predict(_clf, _iso, _FEAT, rm, kpi7, _ea_sc)

    # ── Revenue v2 (corrected formulas) ───────────────────────
    _streams, _total_rev = broker_revenue_v2(
        tdf, rm["lev"], kpi7, _routing, _risk_tier, _ea_sc,
        _slip_ctrl, _delay_ctrl, _sym_ctrl, _mkt_lots, _gmt_val)

    # ── Snap to history ───────────────────────────────────────
    if "acct_history" not in st.session_state:
        st.session_state["acct_history"] = []
    _snap = {
        "Account":        uploaded.name if hasattr(uploaded,"name") else "Unknown",
        "Routing":        _routing_display,
        "Risk Tier":      _risk_tier,
        "EV/trade":       _ev_feat,
        "EV Override":    "YES" if _ev_override else "No",
        "A-Book%":        round(_proba[0]*100,1),
        "B-Book%":        round(_proba[1]*100,1),
        "Toxic Profile%": round(_proba[2]*100,1),
        "Low Conf":       "⚠️" if _low_conf else "OK",
        "Toxic Flow":     "YES ⚠️" if _tox_flag else "No",
        "Net Profit":     round(rm["net_p"],2),
        "Win Rate%":      round(rm["wr"],1),
        "PF":             round(float(kpi7["profit_factor"]),2) if not np.isnan(kpi7["profit_factor"]) else 0,
        "RR":             round(float(kpi7["rr_ratio"]),2) if not np.isnan(kpi7["rr_ratio"]) else 0,
        "Avg Hold(s)":    round(_avh,0),
        "Scalp%":         round(rm["tox"],1),
        "Trades/Day":     round(kpi7["trades_per_day"],1),
        "Slippage":       f"{_slip_ctrl['slip_pip']}pip" if _slip_ctrl["triggered"] else "None",
        "Delay":          f"{_delay_ctrl['delay_ms']}ms" if _delay_ctrl["triggered"] else "None",
        "Mkt Open Trades":_mkt_cnt,
        "EA Score":       _ea_sc,
        "Max DD%":        round(rm["mdd_p"],2),
        "Deposits":       round(rm["total_dep"],2),
        "Est.Rev($)":     _total_rev,
    }
    _existing = [h["Account"] for h in st.session_state["acct_history"]]
    if _snap["Account"] not in _existing:
        st.session_state["acct_history"].append(_snap)

    _vc  = {"A-Book":"#22d3a0", "A-Book + Full Controls":"#f97316",
            "A-Book + Monitor":"#fbbf24", "B-Book":"#f87171",
            "B-Book + Controls":"#fb923c"}
    _rc  = _vc.get(_routing_display, "#fbbf24")

    _risk_clr = {"High Risk":"#f87171","Medium Risk":"#fbbf24","Low Risk":"#22d3a0"}.get(_risk_tier,"#94a3b8")

    _badges = ""
    if _ev_override:
        _badges += '<span style="background:#7f3800;color:#fed7aa;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;margin-left:10px;">🔄 EV OVERRIDE → A-Book</span>'
    if _tox_flag:
        _badges += '<span style="background:#7f1d1d;color:#fca5a5;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;margin-left:10px;">⚠️ HIGH-FREQ FLOW</span>'
    if _low_conf:
        _badges += '<span style="background:#1e3a5f;color:#93c5fd;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;margin-left:10px;">❓ REVIEW MANUALLY</span>'

    # A-Book prob = proba[0] + proba[2] (both mean "profitable/skilled")
    # B-Book prob = proba[1]
    _abook_pct = (_proba[0] + _proba[2]) * 100
    _bbook_pct = _proba[1] * 100

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1a2e,#0a1628);
        border:1px solid rgba(30,58,95,.5);border-radius:14px;
        padding:22px 28px;margin-bottom:16px;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
        <div>
          <div style="font-size:11px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:6px;">
            🧠 ML Decision — Routing &amp; Risk Tier</div>
          <div style="font-size:36px;font-weight:800;color:{_rc};font-family:Space Mono,monospace;display:flex;align-items:center;flex-wrap:wrap;gap:8px;">
            {_routing_display}{_badges}</div>
          <div style="margin-top:10px;">
            <span style="background:{_risk_clr}22;color:{_risk_clr};padding:4px 14px;
              border-radius:20px;font-size:12px;font-weight:700;border:1px solid {_risk_clr}55;">
              {_risk_tier}</span>
            <span style="color:#475569;font-size:11px;margin-left:12px;">
              EV={_ev_feat:+.4f}/trade · EV/day=${_ev_pd:+.2f} · Confidence {max(_proba)*100:.0f}%</span>
          </div>
        </div>
        <div style="display:flex;gap:20px;flex-wrap:wrap;align-items:center;">
          <div style="text-align:center;background:#052010;border-radius:10px;padding:12px 18px;">
            <div style="font-size:10px;letter-spacing:1.5px;color:#4ade80;text-transform:uppercase;">A-Book Signal</div>
            <div style="font-size:34px;font-weight:800;color:#22d3a0;font-family:Space Mono,monospace;">{_abook_pct:.0f}%</div>
            <div style="font-size:9px;color:#475569;">A-Book + Toxic profile</div>
          </div>
          <div style="text-align:center;background:#1f0505;border-radius:10px;padding:12px 18px;">
            <div style="font-size:10px;letter-spacing:1.5px;color:#f87171;text-transform:uppercase;">B-Book Signal</div>
            <div style="font-size:34px;font-weight:800;color:#f87171;font-family:Space Mono,monospace;">{_bbook_pct:.0f}%</div>
            <div style="font-size:9px;color:#475569;">losing/no-edge profile</div>
          </div>
          <div style="text-align:center;background:#0a0a1e;border-radius:10px;padding:12px 18px;">
            <div style="font-size:10px;letter-spacing:1.5px;color:#a78bfa;text-transform:uppercase;">Est. Revenue</div>
            <div style="font-size:34px;font-weight:800;color:#a78bfa;font-family:Space Mono,monospace;">{fmt_usd(_total_rev)}</div>
            <div style="font-size:9px;color:#475569;">from controls applied</div>
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # EV explanation alert
    st.markdown(alert(_ev_reason,
        "high" if "Override" in _ev_label else
        "medium" if "Recommended" in _ev_label else
        "low" if "Monitor" in _ev_label else "good"),
        unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION B — 3 CONTROL TRIGGER CARDS
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🎯 Rule-Based Execution Controls</div>', unsafe_allow_html=True)
    _ta, _tb, _tc = st.columns(3)

    with _ta:
        _s = _slip_ctrl
        _sc_clr = "#f87171" if _s["triggered"] else "#22d3a0"
        st.markdown(f"""<div style="background:#0d1a2e;border-radius:12px;padding:16px;
            border:1px solid rgba(30,58,95,.3);border-top:3px solid {_sc_clr};">
          <div style="font-size:10px;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
            ⚡ SLIPPAGE CONTROL</div>
          <div style="font-size:28px;font-weight:800;color:{_sc_clr};font-family:Space Mono,monospace;">
            {"ACTIVE" if _s["triggered"] else "INACTIVE"}</div>
          <div style="font-size:20px;font-weight:700;color:{_sc_clr};margin:6px 0;">
            {_s["slip_pip"]} pip</div>
          <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
            <b>Trigger:</b> {_s["reason"]}<br>
            <b>Trades/day:</b> {_s["trades_per_day"]:.1f} (threshold {_s["threshold"]})<br>
            <b>Short trades:</b> {_s["short_trades"]} ({_s["short_pct"]}% &lt;60s)<br>
            <b>Est. Revenue:</b> {fmt_usd(_s["est_revenue"])}
          </div>
        </div>""", unsafe_allow_html=True)

    with _tb:
        _d = _delay_ctrl
        _dc_clr = "#f87171" if _d["triggered"] else "#22d3a0"
        st.markdown(f"""<div style="background:#0d1a2e;border-radius:12px;padding:16px;
            border:1px solid rgba(30,58,95,.3);border-top:3px solid {_dc_clr};">
          <div style="font-size:10px;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
            ⏱️ EXECUTION DELAY</div>
          <div style="font-size:28px;font-weight:800;color:{_dc_clr};font-family:Space Mono,monospace;">
            {"ACTIVE" if _d["triggered"] else "INACTIVE"}</div>
          <div style="font-size:20px;font-weight:700;color:{_dc_clr};margin:6px 0;">
            {_d["delay_ms"]} ms</div>
          <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
            <b>Trigger:</b> {_d["reason"]}<br>
            <b>Trades &lt;60s:</b> {_d["pct60"]}%<br>
            <b>Trades &lt;120s:</b> {_d["pct120"]}%<br>
            <b>Scalp P&amp;L:</b> {fmt_usd(_d["scalp_profit"])}<br>
            <b>Captured ({_d["capture_rate"]*100:.0f}%):</b> {fmt_usd(_d["captured"])}
          </div>
        </div>""", unsafe_allow_html=True)

    with _tc:
        _mk_clr = "#fbbf24" if _mkt_cnt > 0 else "#22d3a0"
        st.markdown(f"""<div style="background:#0d1a2e;border-radius:12px;padding:16px;
            border:1px solid rgba(30,58,95,.3);border-top:3px solid {_mk_clr};">
          <div style="font-size:10px;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
            🔔 MARKET OPEN SPREAD</div>
          <div style="font-size:28px;font-weight:800;color:{_mk_clr};font-family:Space Mono,monospace;">
            {"ACTIVE" if _mkt_cnt>0 else "INACTIVE"}</div>
          <div style="font-size:20px;font-weight:700;color:{_mk_clr};margin:6px 0;">
            +1.5 pip at open</div>
          <div style="font-size:11px;color:#94a3b8;line-height:1.6;">
            <b>Trades at session open:</b> {_mkt_cnt}<br>
            <b>Lots at open:</b> {_mkt_lots:.2f}<br>
            <b>Sessions targeted:</b> London · NY · Tokyo · Sydney<br>
            <b>Window:</b> First 5 min after session open
          </div>
        </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION C — SYMBOL-WISE CONTROLS
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">💹 Symbol-Wise Spread Control (Per-Symbol Pip Values)</div>',
                unsafe_allow_html=True)
    if len(_sym_ctrl) > 0:
        _sym_c1, _sym_c2 = st.columns([2,1])
        with _sym_c1:
            st.dataframe(_sym_ctrl, use_container_width=True, hide_index=True)
        with _sym_c2:
            _sf = px.bar(_sym_ctrl.head(8), x="Symbol", y="Est.MarkupRev",
                         color="Rec.Markup(pip)", color_continuous_scale="Reds",
                         title="Est. Markup Revenue by Symbol")
            _sf.update_layout(height=260, **CHART_LAYOUT)
            st.plotly_chart(_sf, use_container_width=True)
    else:
        st.markdown(alert("No symbol data available.", "low"), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION D — ORDER TYPE ASYMMETRY
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📐 Order Type Asymmetry — BUY vs SELL Bias Detection</div>',
                unsafe_allow_html=True)
    if _has_asym:
        st.markdown(alert(
            "⚠️ <b>Directional bias detected</b> — trader is consistently profitable on one side only. "
            "Apply higher markup on profitable direction or hedge that directional exposure.", "medium"),
            unsafe_allow_html=True)
    else:
        st.markdown(alert("No significant directional bias. BUY/SELL roughly balanced.", "good"),
            unsafe_allow_html=True)
    if len(_asym_df) > 0:
        _as1, _as2 = st.columns([2,1])
        with _as1:
            st.dataframe(_asym_df, use_container_width=True, hide_index=True)
        with _as2:
            _asym_plot = _asym_df.melt(id_vars="Symbol",
                value_vars=["BUY Profit","SELL Profit"], var_name="Direction", value_name="Profit")
            _afig = px.bar(_asym_plot, x="Symbol", y="Profit", color="Direction",
                           color_discrete_map={"BUY Profit":"#22d3a0","SELL Profit":"#f87171"},
                           barmode="group", title="BUY vs SELL Profit per Symbol")
            _afig.update_layout(height=260, **CHART_LAYOUT)
            st.plotly_chart(_afig, use_container_width=True)

    # Market open trades table
    if len(_mkt_df) > 0:
        st.markdown('<div class="section-title">🔔 Market Open Trade Log (First 5 Min per Session)</div>',
                    unsafe_allow_html=True)
        st.dataframe(_mkt_df, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════
    # SECTION E — EV TREND + REVENUE BREAKDOWN
    # ════════════════════════════════════════════════════════
    _ee1, _ee2 = st.columns([1,2])
    with _ee1:
        st.markdown('<div class="section-title">📈 Trader Edge Trend (Early→Mid→Recent)</div>',
                    unsafe_allow_html=True)
        if len(_trend_df) > 0:
            _trend_clr = ["#f87171","#fbbf24","#22d3a0" if _improving else "#f97316"]
            _tf = go.Figure(go.Bar(
                x=_trend_df["Period"], y=_trend_df["Net P&L"],
                marker_color=_trend_clr,
                text=_trend_df.apply(lambda r: f"WR:{r['Win Rate%']:.0f}%", axis=1),
                textposition="outside"))
            _tf.update_layout(title=f"Trend: {'📈 IMPROVING' if _improving else '📉 DECLINING'}",
                              height=250, **CHART_LAYOUT)
            st.plotly_chart(_tf, use_container_width=True)
            st.markdown(alert(
                f"Trader edge is <b>{'IMPROVING ✅' if _improving else 'DECLINING ⚠️'}</b> — "
                f"Recent P&L {fmt_usd(_trend_df.iloc[-1]['Net P&L'])} vs "
                f"Early {fmt_usd(_trend_df.iloc[0]['Net P&L'])}. "
                + ("Increasing A-Book risk." if _improving else "B-Book becoming safer."),
                "medium" if _improving else "good"), unsafe_allow_html=True)

    with _ee2:
        st.markdown('<div class="section-title">💰 6 Revenue Streams (Upgraded)</div>',
                    unsafe_allow_html=True)
        _rev_rows = [{"Stream": s, "Est.Revenue": fmt_usd(i["revenue"]),
                      "Rate": i["rate"], "Status": i["apply"]}
                     for s, i in _streams.items()]
        st.dataframe(pd.DataFrame(_rev_rows), use_container_width=True, hide_index=True)
        _rev_colors = {
            "Spread Markup (symbol-aware)":"#3b82f6",
            "Slippage (freq-triggered)":"#f97316",
            "Execution Delay Capture":"#a78bfa",
            "Market Open Spread":"#fbbf24",
            "Swap Carry":"#22d3a0",
            "Leverage Carry":"#60a5fa"
        }
        _rv_fig = go.Figure(go.Bar(
            x=[s.split("(")[0].strip() for s in _streams.keys()],
            y=[v["revenue"] for v in _streams.values()],
            marker_color=[_rev_colors.get(s,"#60a5fa") for s in _streams],
            text=[fmt_usd(v["revenue"]) for v in _streams.values()],
            textposition="outside"))
        _rv_fig.update_layout(
            title=f"Total: {fmt_usd(_total_rev)}",
            height=240, **CHART_LAYOUT)
        st.plotly_chart(_rv_fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # SECTION F — ML GAUGES + FEATURE IMPORTANCE
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🧠 ML Decision Detail</div>', unsafe_allow_html=True)
    _g1, _g2, _g3 = st.columns(3)
    for _col, _name, _val, _color in [
        (_g1,"A-Book Probability",_proba[0]*100,"#22d3a0"),
        (_g2,"B-Book Probability",_proba[1]*100,"#f87171"),
        (_g3,"Toxic Probability", _proba[2]*100,"#f97316"),
    ]:
        with _col:
            _fg = go.Figure(go.Indicator(
                mode="gauge+number", value=_val,
                title={"text":_name,"font":{"color":"#94a3b8","size":12}},
                gauge={"axis":{"range":[0,100],"tickcolor":"#475569"},
                       "bar":{"color":_color},
                       "steps":[{"range":[0,40],"color":"rgba(13,26,46,0.8)"},
                                 {"range":[40,65],"color":"rgba(13,26,46,0.5)"},
                                 {"range":[65,100],"color":"rgba(13,26,46,0.3)"}],
                       "threshold":{"line":{"color":"#fbbf24","width":2},"thickness":0.75,"value":65}},
                number={"suffix":"%","font":{"color":"#e2e8f0","size":30}}))
            _fg.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color":"#94a3b8"},
                              height=230, margin=dict(t=50,b=5))
            _col.plotly_chart(_fg, use_container_width=True)

    if _low_conf:
        st.markdown(alert(
            f"⚠️ <b>Low confidence prediction</b> — top class only {max(_proba)*100:.0f}%. "
            "This account sits on the boundary between categories. Manual review recommended before routing decision.",
            "medium"), unsafe_allow_html=True)

    _fi_df = pd.DataFrame({
        "Feature":    _FEATURES,
        "Importance": [_fi[f] for f in _FEATURES],
        "AccountVal": [round(_feats[i],3) for i in range(len(_feats))]
    }).sort_values("Importance", ascending=True)
    _fi_fig = go.Figure(go.Bar(
        x=_fi_df["Importance"], y=_fi_df["Feature"], orientation="h",
        marker_color="#3b82f6",
        text=_fi_df["AccountVal"].apply(lambda v: f"→ {v}"),
        textposition="outside"))
    _fi_fig.update_layout(title="Feature Importances (→ = your account's value)",
                          height=350, **CHART_LAYOUT)
    st.plotly_chart(_fi_fig, use_container_width=True)

    # ════════════════════════════════════════════════════════
    # SECTION G — FULL CONTROL PANEL
    # ════════════════════════════════════════════════════════
    _cp1, _cp2 = st.columns([1,2])
    with _cp1:
        st.markdown('<div class="section-title">⚙️ Full Execution Control Panel</div>', unsafe_allow_html=True)
        _mu_pip  = {"High Risk": 0.6, "Medium Risk": 0.3}.get(_risk_tier, 0.2) \
                   if _routing == "A-Book" else \
                   {"High Risk": 0.4}.get(_risk_tier, 0.2)
        _ctrl_rows = [
            ("📚","Book Routing",       _routing_display,
             "#22d3a0" if _routing=="A-Book" else "#f87171"),
            ("🎯","Risk Tier",          _risk_tier,
             "#f87171" if _risk_tier=="High Risk" else "#fbbf24" if _risk_tier=="Medium Risk" else "#22d3a0"),
            ("🔄","EV Override",        "YES — Forced A-Book" if _ev_override else "No",
             "#f97316" if _ev_override else "#475569"),
            ("📈","Spread Markup",      f"+{_mu_pip} pip (symbol-aware)",
             "#f87171" if _mu_pip>=0.5 else "#fbbf24" if _mu_pip>0 else "#22d3a0"),
            ("⚡","Slippage",           f"+{_slip_ctrl['slip_pip']} pip — {_slip_ctrl['reason'][:30]}"
                                        if _slip_ctrl["triggered"] else "Not triggered",
             "#f87171" if _slip_ctrl["triggered"] else "#22d3a0"),
            ("⏱️","Execution Delay",    f"{_delay_ctrl['delay_ms']}ms — {_delay_ctrl['reason'][:25]}"
                                        if _delay_ctrl["triggered"] else "Not triggered",
             "#fbbf24" if _delay_ctrl["triggered"] else "#22d3a0"),
            ("🔔","Market Open Spread", f"+1.0pip on {_mkt_cnt} open trades"
                                        if _mkt_cnt>0 else "No session-open trades",
             "#fbbf24" if _mkt_cnt>0 else "#475569"),
            ("📉","Volume Limit",       _vol_lim,
             "#f87171" if "50" in _vol_lim else "#fbbf24" if "25" in _vol_lim else "#22d3a0"),
            ("📐","Direction Markup",   "Apply on profitable side" if _has_asym else "No bias detected",
             "#fbbf24" if _has_asym else "#22d3a0"),
            ("📊","Trend Signal",       "IMPROVING — monitor" if _improving else "DECLINING — safer",
             "#fbbf24" if _improving else "#22d3a0"),
            ("👁️","Monitoring Level",
             "Real-time" if _risk_tier=="High Risk"
             else "Daily" if _routing=="A-Book" and _ev_feat>0
             else "Weekly",
             "#f87171" if _risk_tier=="High Risk" else "#fbbf24" if _ev_feat>0 else "#22d3a0"),
        ]
        for icon, lbl, val, clr in _ctrl_rows:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 12px;margin:3px 0;background:#0d1a2e;border-radius:7px;
                border-left:3px solid {clr};">
                <span style="font-size:11px;color:#94a3b8;">{icon} {lbl}</span>
                <span style="font-family:Space Mono,monospace;font-size:10px;font-weight:700;
                             color:{clr};max-width:55%;text-align:right;">{val}</span>
            </div>""", unsafe_allow_html=True)

    with _cp2:
        # ── STREAK CHART ──────────────────────────────────────
        st.markdown('<div class="section-title">📈 Win/Loss Streak</div>', unsafe_allow_html=True)
        _sdf_s = kpi7["streak_df"].copy()
        _sdf_s["Color"] = _sdf_s["Streak"].apply(lambda v: "#22d3a0" if v > 0 else "#f87171")
        _sf2 = go.Figure(go.Bar(
            x=_sdf_s["Trade"], y=_sdf_s["Streak"],
            marker_color=_sdf_s["Color"],
            hovertemplate="Trade #%{x}<br>Streak: %{y}<extra></extra>"))
        _sf2.add_hline(y=0, line_color="#475569", line_width=1)
        _sf2.update_layout(
            title=f"Max Win: {kpi7['max_consec_wins']} | Max Loss: {kpi7['max_consec_losses']}",
            height=200, **CHART_LAYOUT)
        st.plotly_chart(_sf2, use_container_width=True)

        # ── TOXIC DETAIL ──────────────────────────────────────
        st.markdown('<div class="section-title">☠️ Toxic / EA Detail</div>', unsafe_allow_html=True)
        _tox_items = [
            ("Isolation Forest",  "ANOMALY ⚠️" if _tox_flag else "Normal",
             "#f97316" if _tox_flag else "#22d3a0"),
            ("EA/Bot Score",      f"{_ea_sc}/100",
             "#f87171" if _ea_sc>=70 else "#fbbf24" if _ea_sc>=40 else "#22d3a0"),
            ("EV per Trade",      f"${_ev_feat:+.4f}",
             "#f87171" if _ev_feat>0.05 else "#fbbf24" if _ev_feat>0 else "#22d3a0"),
            ("EV per Day",        f"${_ev_pd:+.2f}",
             "#f87171" if _ev_pd>50 else "#fbbf24" if _ev_pd>10 else "#22d3a0"),
            ("Burst Trades",      f"{len(_bd2)} detected",
             "#f87171" if len(_bd2)>20 else "#22d3a0"),
            ("Reversal Pairs",    f"{len(_rv2)} detected",
             "#f87171" if len(_rv2)>10 else "#22d3a0"),
            ("Trades <60s",       f"{_delay_ctrl['pct60']}%",
             "#f87171" if _delay_ctrl['pct60']>30 else "#fbbf24" if _delay_ctrl['pct60']>15 else "#22d3a0"),
            ("Off-Hours %",       f"{rm['offp']:.1f}%",
             "#f87171" if rm["offp"]>20 else "#22d3a0"),
        ]
        for lbl3, val3, clr3 in _tox_items:
            st.markdown(f"""<div style="display:flex;justify-content:space-between;padding:6px 12px;
                margin:2px 0;background:#0d1a2e;border-radius:6px;border-left:3px solid {clr3};">
                <span style="font-size:11px;color:#94a3b8;">{lbl3}</span>
                <span style="font-family:Space Mono,monospace;font-size:11px;
                             font-weight:700;color:{clr3};">{val3}</span>
            </div>""", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════
    # SECTION H — CUSTOM SIMULATOR + ACCOUNT HISTORY
    # ════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">🔧 Custom Markup Revenue Simulator</div>', unsafe_allow_html=True)
    _ms1, _ms2, _ms3 = st.columns(3)
    with _ms1: mp = st.number_input("Markup (pips)", value=float(_mu_pip), step=0.1, key="mk_p")
    with _ms2:
        sym_options = ["ALL"] + list(sym_sum["Symbol"].tolist())
        mks = st.selectbox("Apply to", sym_options, key="mk_s")
    with _ms3:
        pv_default = get_pip_val(mks) if mks != "ALL" else 8.0
        pv = st.number_input("Pip value ($)", value=float(pv_default), step=0.5, key="mk_v")
    la = rm["tot_l"] if mks=="ALL" else (
        sym_sum[sym_sum["Symbol"]==mks]["Volume"].values[0]
        if len(sym_sum[sym_sum["Symbol"]==mks]) else 0)
    er = mp * pv * la
    st.markdown(f"""<div style="background:#052010;border:1px solid #166534;
        border-radius:10px;padding:16px 22px;margin-top:6px;display:flex;
        justify-content:space-between;align-items:center;">
        <div>
          <div style="font-size:10px;color:#4ade80;letter-spacing:2px;text-transform:uppercase;">
            Estimated Markup Revenue</div>
          <div style="font-size:32px;font-weight:800;font-family:Space Mono,monospace;color:#22d3a0;">
            {fmt_usd(er)}</div>
          <div style="font-size:11px;color:#475569;margin-top:4px;">
            {mp}pip × ${pv}/pip × {la:.2f} lots ({mks}) — pip val auto-loaded per symbol</div>
        </div>
        <div style="text-align:right;">
          <div style="font-size:10px;color:#475569;">Symbol Pip Value</div>
          <div style="font-size:24px;font-weight:800;color:#60a5fa;font-family:Space Mono,monospace;">
            ${pv}/pip</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗂️ Account History Dataset</div>', unsafe_allow_html=True)
    _hist = st.session_state["acct_history"]
    if len(_hist) >= 1:
        _hist_df = pd.DataFrame(_hist)
        st.dataframe(_hist_df, use_container_width=True, hide_index=True)
        st.caption(f"📌 {len(_hist_df)} accounts · EV Override column shows which accounts were corrected by EV rule · Export → manually label → retrain.")
        dl("ML History", {"Account History": _hist_df}, _hist_df, "acct_history")
    else:
        st.markdown(alert("Upload more accounts to build the dataset.", "low"), unsafe_allow_html=True)

    # ── FULL EXPORT ───────────────────────────────────────────
    st.markdown('<div class="section-title">📥 Full Report Export</div>', unsafe_allow_html=True)
    _sdf2  = pd.DataFrame(list(summ.items()), columns=["Metric","Value"])
    _rdf2  = pd.DataFrame(recs, columns=["Level","Recommendation"]) if recs else pd.DataFrame()
    _kex   = {k:v for k,v in kpi7.items() if k != "streak_df"}
    _kdf   = pd.DataFrame([_kex])
    _edf   = pd.DataFrame([{"Stream":s,**{kk:vv for kk,vv in i.items() if kk!="formula"},"Formula":i["formula"]}
                            for s,i in _streams.items()])
    _ml_df = pd.DataFrame([{
        "Routing": _routing_display, "Risk_Tier": _risk_tier,
        "EV_per_Trade": _ev_feat, "EV_per_Day": _ev_pd,
        "EV_Override": _ev_override, "Low_Conf": _low_conf,
        "A_Book_Signal%": round((_proba[0]+_proba[2])*100,1),
        "B_Book_Signal%": round(_proba[1]*100,1),
        "Toxic_Profile%": round(_proba[2]*100,1),
        "Toxic_Flow": _tox_flag, "EA_Score": _ea_sc,
        "Slippage_pip": _slip_ctrl["slip_pip"], "Slip_Triggered": _slip_ctrl["triggered"],
        "Delay_ms": _delay_ctrl["delay_ms"], "Delay_Triggered": _delay_ctrl["triggered"],
        "Mkt_Open_Trades": _mkt_cnt, "Total_Est_Revenue": _total_rev
    }])
    _slip_export = pd.DataFrame([_slip_ctrl])
    _delay_export = pd.DataFrame([{k:v for k,v in _delay_ctrl.items()}])
    dl("Full Report v2", {
        "Summary":_sdf2,"7 KPIs":_kdf,"ML Decision v2":_ml_df,
        "Revenue Streams":_edf,"Slippage Control":_slip_export,
        "Delay Control":_delay_export,"Symbol Controls":_sym_ctrl,
        "Order Asymmetry":_asym_df,"Trend Analysis":_trend_df,
        "Market Open Trades":_mkt_df if len(_mkt_df)>0 else pd.DataFrame(),
        "Streaks":kpi7["streak_df"],
        "Symbol Summary":sym_sum,"Symbol By Direction":sym_dir,
        "Daily PnL":dpnl,
        "Equity":eq_df[["Close Time","Profit","Equity","Drawdown","DrawdownPct","TradeReturnPct","RiskBucket"]],
        "Deposits":deps,"Withdrawals":wths,"Recommendations":_rdf2
    }, _sdf2, "full_report_v2")
