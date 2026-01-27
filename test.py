# 29sixtab.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
import plotly.express as px
import requests

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Broker Risk & Trade Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# THEME TOGGLE (Dark / Light)
# --------------------------------------------------
dark_mode = st.toggle("🌙 Dark theme", value=True)

DARK_CSS = """
<style>
body, .main, .block-container {
    background-color: #111827 !important;
    color: #e5e7eb !important;
}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stSidebar"] {background-color: #020617 !important;}
div[data-baseweb="tab"] {
    background-color: #020617 !important;
    color: #e5e7eb !important;
}
</style>
"""

LIGHT_CSS = """
<style>
body, .main, .block-container {
    background-color: #f9fafb !important;
    color: #111827 !important;
}
[data-testid="stHeader"] {background: transparent !important;}
[data-testid="stSidebar"] {background-color: #ffffff !important;}
</style>
"""

st.markdown(DARK_CSS if dark_mode else LIGHT_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def safe_to_datetime(series):
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def format_currency(val):
    return f"${val:,.2f}"

def format_percent(val):
    return f"{val:.2f}%"

def identify_columns_mt5_by_index(positions_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-identify MT5 'Positions' columns:
    - First 'Time'  -> Open Time
    - Second 'Time' -> Close Time
    - First 'Price' -> Open Price
    - Second 'Price'-> Close Price
    - Keep Volume / Profit / Type / Symbol etc.
    """
    positions_raw = positions_raw.dropna(how="all")
    header = positions_raw.iloc[0]
    positions_raw = positions_raw[1:]
    positions_df = positions_raw.reset_index(drop=True)
    positions_df.columns = header

    cols = list(positions_df.columns)

    time_idx = [i for i, c in enumerate(cols) if str(c).strip().lower() == "time"]
    price_idx = [i for i, c in enumerate(cols) if str(c).strip().lower() == "price"]

    # Map by index to avoid duplicate-name collisions
    if len(time_idx) >= 1:
        cols[time_idx[0]] = "Open Time"
    if len(time_idx) >= 2:
        cols[time_idx[1]] = "Close Time"
    if len(price_idx) >= 1:
        cols[price_idx[0]] = "Open Price"
    if len(price_idx) >= 2:
        cols[price_idx[1]] = "Close Price"

    # Other direct mappings
    simple_names = ["Volume", "Profit", "Type", "Symbol", "Position", "Commission", "Swap"]
    for name in simple_names:
        for i, c in enumerate(cols):
            if str(c).strip().lower() == name.lower():
                cols[i] = name

    positions_df.columns = cols
    return positions_df

def compute_equity_metrics(df: pd.DataFrame, start_equity: float):
    """
    Build equity curve per trade (sorted by Close Time),
    compute max drawdown and per-trade risk % of equity.
    """
    df = df.sort_values("Close Time").copy()
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0)

    df["CumProfit"] = df["Profit"].cumsum()
    df["Equity"] = start_equity + df["CumProfit"]

    # Equity before each trade closes
    df["EquityBefore"] = start_equity + df["CumProfit"].shift(1).fillna(0)

    # Trade return as % of equity before trade
    df["TradeReturnPct"] = np.where(
        df["EquityBefore"] != 0,
        df["Profit"] / df["EquityBefore"] * 100.0,
        0.0
    )

    # Risk label
    def risk_label(pct):
        if pct <= -50:
            return "Very High Risk (≤ -50%)"
        elif pct <= -30:
            return "High Risk (≤ -30%)"
        else:
            return "Normal"

    df["RiskBucket"] = df["TradeReturnPct"].apply(risk_label)

    # Max drawdown
    df["EquityPeak"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] - df["EquityPeak"]
    df["DrawdownPct"] = np.where(
        df["EquityPeak"] != 0,
        df["Drawdown"] / df["EquityPeak"] * 100.0,
        0.0
    )

    max_dd_row = df.loc[df["Drawdown"].idxmin()] if len(df) > 0 else None
    if max_dd_row is not None and not df["Drawdown"].isna().all():
        max_dd_amount = max_dd_row["Drawdown"]
        max_dd_pct = max_dd_row["DrawdownPct"]
    else:
        max_dd_amount = 0.0
        max_dd_pct = 0.0

    return df, max_dd_amount, max_dd_pct

# ---- NEW: swap-night counting logic ----
def count_swap_nights(open_time: datetime, close_time: datetime, triple_weekday_index: int):
    """
    Count swap nights between open_time and close_time.
    Swap is charged on nightly timestamp = date at 23:59:00.
    We charge for a night N if open_time <= night_ts < close_time.
    Saturday (5) and Sunday (6) nights are skipped.
    triple_weekday_index: 0=Monday .. 4=Friday - which weekday is triple-swap.
    Returns: (normal_count_excluding_triple, triple_count)
    """
    if pd.isna(open_time) or pd.isna(close_time):
        return 0, 0

    # if close <= open -> no nights
    if close_time <= open_time:
        return 0, 0

    normal = 0
    triple = 0

    # iterate days from open.date() to close.date() inclusive of possible night
    # We'll check nights for each date d where night_ts = datetime(d.year,d.month,d.day,23,59)
    # We need to cover nights whose night timestamp might be after open and before close.
    current_date = open_time.date()
    last_date = (close_time - timedelta(seconds=1)).date()  # nights strictly before close_time

    # iterate from current_date to last_date (inclusive)
    while current_date <= last_date:
        night_ts = datetime(current_date.year, current_date.month, current_date.day, 23, 59, 0)

        # Charge if night_ts in [open_time, close_time)
        if (night_ts >= open_time) and (night_ts < close_time):
            wd = night_ts.weekday()  # 0..6
            # skip Saturday & Sunday nights
            if wd in (5, 6):
                pass
            else:
                if wd == triple_weekday_index:
                    triple += 1
                else:
                    normal += 1

        current_date = current_date + timedelta(days=1)

    return normal, triple
# --------------------------------------------------
# Improved Burst Trades Detection (Slider Ready)
# --------------------------------------------------

def detect_burst_trades(df, max_seconds=2):
    """
    Detect burst trades:
    ≥2 trades opened within max_seconds of each other (time proximity only)

    Parameters:
        df           : DataFrame with trades
        max_seconds : int, from Streamlit slider (e.g., 1–5 seconds)

    Returns:
        burst_df            -> DataFrame of all burst trades
        burst_event_count  -> Number of burst events (groups)
        max_burst_size     -> Largest burst group size
    """

    df = df.copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df = df.sort_values("Open Time").reset_index(drop=True)

    if len(df) < 2:
        return df.iloc[0:0], 0, 0

    burst_groups = []
    current_group = [0]

    for i in range(1, len(df)):
        dt = (df.loc[i, "Open Time"] - df.loc[i - 1, "Open Time"]).total_seconds()

        if 0 < dt <= max_seconds:
            current_group.append(i)
        else:
            if len(current_group) >= 2:
                burst_groups.append(current_group)
            current_group = [i]

    if len(current_group) >= 2:
        burst_groups.append(current_group)

    burst_indices = sorted({idx for group in burst_groups for idx in group})
    burst_df = df.loc[burst_indices].copy()

    burst_event_count = len(burst_groups)
    max_burst_size = max([len(g) for g in burst_groups], default=0)

    return burst_df, burst_event_count, max_burst_size


# --------------------------------------------------
# Improved Reversal Trades Detection (Slider Ready)
# --------------------------------------------------

def detect_reversal_trades(df, max_seconds=20):
    """
    Detect reversal trades:
    Same Symbol + Opposite Type + within max_seconds

    Parameters:
        df           : DataFrame with trades
        max_seconds : int, from Streamlit slider (e.g., 5–60 seconds)

    Returns:
        reversal_df           -> DataFrame of all reversal trades
        reversal_event_count -> Number of reversal events (pairs)
    """

    df = df.copy()
    df["Open Time"] = pd.to_datetime(df["Open Time"])
    df = df.sort_values(["Symbol", "Open Time"]).reset_index(drop=True)

    if len(df) < 2:
        return df.iloc[0:0], 0

    reversal_indices = []
    reversal_events = []

    i = 0
    while i < len(df) - 1:
        r1 = df.loc[i]
        r2 = df.loc[i + 1]

        same_symbol = r1["Symbol"] == r2["Symbol"]
        opposite_type = r1["Type"] != r2["Type"]
        dt = (r2["Open Time"] - r1["Open Time"]).total_seconds()

        if same_symbol and opposite_type and 0 < dt <= max_seconds:
            reversal_indices.append(i)
            reversal_indices.append(i + 1)
            reversal_events.append((i, i + 1))
            i += 2  # skip to avoid chain double counting
        else:
            i += 1

    reversal_indices = sorted(set(reversal_indices))
    reversal_df = df.loc[reversal_indices].copy()

    reversal_event_count = len(reversal_events)

    return reversal_df, reversal_event_count


# --------------------------------------------------
# IP Lookup Logic (Simple, Streamlit Friendly)
# --------------------------------------------------

def lookup_ip(ip_address):
    """
    Simple IP lookup using ip-api.com (free public API)

    Returns:
        dict with country, city, isp, org, timezone, status
    """

    try:
        url = f"http://ip-api.com/json/{ip_address}"
        resp = requests.get(url, timeout=5)
        data = resp.json()

        if data.get("status") != "success":
            return {
                "status": "fail",
                "message": data.get("message", "Lookup failed")
            }

        return {
            "status": "success",
            "ip": ip_address,
            "country": data.get("country"),
            "region": data.get("regionName"),
            "city": data.get("city"),
            "isp": data.get("isp"),
            "org": data.get("org"),
            "timezone": data.get("timezone"),
            "lat": data.get("lat"),
            "lon": data.get("lon"),
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# --------------------------------------------------
# Example Integration Helper (Slider Driven)
# --------------------------------------------------

def analyze_burst_and_reversal_with_sliders(df, burst_seconds=2, reversal_seconds=20):
    """
    Helper function to integrate both detections using slider values.

    Parameters:
        df               : trades DataFrame
        burst_seconds   : from Streamlit slider
        reversal_seconds: from Streamlit slider

    Returns:
        dict with all burst & reversal metrics
    """

    # --- Burst ---
    burst_df, burst_event_count, max_burst_size = detect_burst_trades(
        df, max_seconds=burst_seconds
    )

    burst_count = len(burst_df)
    burst_profit = burst_df["Profit"].sum() if len(burst_df) > 0 else 0
    burst_percentage = (burst_count / len(df) * 100) if len(df) > 0 else 0

    # --- Reversal ---
    reversal_df, reversal_event_count = detect_reversal_trades(
        df, max_seconds=reversal_seconds
    )

    reversal_count = len(reversal_df)
    reversal_profit = reversal_df["Profit"].sum() if len(reversal_df) > 0 else 0
    reversal_percentage = (reversal_count / len(df) * 100) if len(df) > 0 else 0

    return {
        "burst_df": burst_df,
        "burst_count": burst_count,
        "burst_profit": burst_profit,
        "burst_percentage": burst_percentage,
        "burst_event_count": burst_event_count,
        "max_burst_size": max_burst_size,

        "reversal_df": reversal_df,
        "reversal_count": reversal_count,
        "reversal_profit": reversal_profit,
        "reversal_percentage": reversal_percentage,
        "reversal_event_count": reversal_event_count,
    }


if __name__ == "__main__":
    print("Utility module with Burst, Reversal, Slider integration, and IP lookup logic.")

# --------------------------------------------------
# Analyzer
# --------------------------------------------------
def analyze_trades(uploaded_file, start_capital, net_cash_flows, current_leverage):
    try:
        df_raw = pd.read_excel(uploaded_file, sheet_name=0, header=None)

        # --------------------------
        # Find Positions section
        # --------------------------
        start_idx = df_raw.index[df_raw[0].astype(str).str.contains("Positions", case=False, na=False)].tolist()
        end_idx = df_raw.index[df_raw[0].astype(str).str.contains("Orders", case=False, na=False)].tolist()

        if not start_idx:
            st.error("❌ Could not find 'Positions' section in the report.")
            return None

        start = start_idx[0] + 1
        end = end_idx[0] if end_idx else len(df_raw)
        positions_raw = df_raw.iloc[start:end]

        # Identify columns
        df = identify_columns_mt5_by_index(positions_raw)

        # --------------------------
        # Core cleaning
        # --------------------------
        df["Open Time"] = safe_to_datetime(df["Open Time"])
        df["Close Time"] = safe_to_datetime(df["Close Time"])

        df["Profit"] = pd.to_numeric(df.get("Profit", 0), errors="coerce").fillna(0.0)
        df["Volume"] = pd.to_numeric(df.get("Volume", 0), errors="coerce").fillna(0.0)

        comm_col = next((c for c in df.columns if "commission" in str(c).lower()), None)
        df["Commission"] = pd.to_numeric(df.get(comm_col, 0), errors="coerce").fillna(0.0) if comm_col else 0.0

        df["Hold_Time"] = df["Close Time"] - df["Open Time"]
        df["is_scalp"] = df["Hold_Time"] <= timedelta(minutes=3)
        df["is_win"] = df["Profit"] > 0

        # Daily PnL
        df["Date"] = df["Close Time"].dt.date
        daily_pnl = df.groupby("Date")["Profit"].sum().reset_index()
        daily_pnl["Balance"] = (start_capital + net_cash_flows) + daily_pnl["Profit"].cumsum()

        # --------------------------
        # Equity + max drawdown + trade risk
        # --------------------------
        effective_start_equity = start_capital + net_cash_flows
        equity_df, max_dd_amount, max_dd_pct = compute_equity_metrics(df, effective_start_equity)

        final_equity = equity_df["Equity"].iloc[-1] if len(equity_df) > 0 else effective_start_equity
        total_profit = equity_df["Profit"].sum() if len(equity_df) > 0 else 0.0

        # --------------------------
        # Aggregate statistics
        # --------------------------
        total_trades = len(df)
        total_lots = df["Volume"].sum()
        total_comm = df["Commission"].sum()
        scalping_count = int(df["is_scalp"].sum())
        scalping_profit = df.loc[df["is_scalp"], "Profit"].sum()

        toxic_ratio = (scalping_count / total_trades * 100.0) if total_trades else 0.0
        comm_pct = (abs(total_comm) / abs(total_profit) * 100.0) if total_profit != 0 else 0.0

        avg_hold_time = df["Hold_Time"].mean()
        median_hold_time = df["Hold_Time"].median()
        win_rate = (df["is_win"].sum() / total_trades * 100.0) if total_trades else 0.0

        if any(df["Profit"] < 0):
            profit_factor = (
                df.loc[df["Profit"] > 0, "Profit"].sum() /
                abs(df.loc[df["Profit"] < 0, "Profit"].sum())
            )
        else:
            profit_factor = np.nan

        profit_per_lot = total_profit / total_lots if total_lots > 0 else 0.0

        # Equity behaviour
        if max_dd_pct <= -50:
            equity_behaviour = "Equity Tanks Often (Losing Trader)"
        elif total_profit > 0 and final_equity > effective_start_equity * 1.5:
            equity_behaviour = "Equity Grows Fast (Winning Trader)"
        else:
            equity_behaviour = "Mixed / Normal Equity Behaviour"

        # --------------------------
        # Symbol summary & concentration
        # --------------------------
        symbol_summary = df.groupby("Symbol", dropna=True).agg(
            Volume=("Volume", "sum"),
            Trades=("Profit", "count"),
            Profit=("Profit", "sum"),
            Commission=("Commission", "sum")
        ).reset_index()

        if len(symbol_summary) > 0:
            symbol_summary["Avg Profit/Trade"] = symbol_summary["Profit"] / symbol_summary["Trades"]
            symbol_summary = symbol_summary.sort_values("Volume", ascending=False)
            top_symbols = symbol_summary.head(3)
            volume_total = symbol_summary["Volume"].sum()
            symbol_concentration = (
                top_symbols["Volume"].sum() / volume_total * 100.0 if volume_total > 0 else 0.0
            )
            top_symbol = symbol_summary.iloc[0]["Symbol"]
            top_symbol_share = (
                symbol_summary.iloc[0]["Volume"] / volume_total * 100.0 if volume_total > 0 else 0.0
            )
        else:
            top_symbols = pd.DataFrame()
            symbol_concentration = 0.0
            top_symbol = None
            top_symbol_share = 0.0

        # Daily PnL volatility
        pnl_volatility = daily_pnl["Profit"].std() if len(daily_pnl) > 1 else 0.0

        growth_pct = (
            (final_equity - effective_start_equity) / effective_start_equity * 100.0
            if effective_start_equity > 0 else 0.0
        )

        # --------------------------
        # Trade risk stats
        # --------------------------
        risky_30 = (equity_df["TradeReturnPct"] <= -30).sum()
        risky_50 = (equity_df["TradeReturnPct"] <= -50).sum()
        risky_30_pct = (
            risky_30 / len(equity_df) * 100.0 if len(equity_df) > 0 else 0.0
        )
        risky_50_pct = (
            risky_50 / len(equity_df) * 100.0 if len(equity_df) > 0 else 0.0
        )

        # --------------------------
        # Broker Recommendations
        # --------------------------
        recommendations = []

        # 1. Symbol concentration
        if top_symbol is not None and top_symbol_share > 60:
            recommendations.append(
                f"Symbol {top_symbol} carries {top_symbol_share:.1f}% of total volume. "
                "Broker can widen spread or add markup only on this symbol to improve revenue."
            )

        # 2. Toxic / scalping behaviour
        if toxic_ratio >= 40:
            recommendations.append(
                "High share of scalping trades detected. Consider increasing spread by about 0.3 pips "
                "and applying execution delay for ultra-fast flows."
            )
        elif toxic_ratio >= 20:
            recommendations.append(
                "Moderate scalping behaviour. Broker can keep spreads stable but watch high-frequency periods."
            )

        # 3. Drawdown / equity behaviour
        if max_dd_pct <= -50:
            recommendations.append(
                "Large historical drawdowns suggest equity is often reduced heavily. "
                "Broker is relatively safe from payout risk and can increase margin requirements "
                "or reduce maximum leverage."
            )
        elif total_profit > 0 and growth_pct > 50:
            recommendations.append(
                "Trader has strong equity growth. Broker should monitor for payout risk and consider "
                "tighter risk controls such as symbol-specific limits or dynamic markups."
            )

        # 4. Per-trade risk
        if risky_50_pct > 5:
            recommendations.append(
                "Many trades risk more than 50% of equity per position. "
                "Recommend cutting leverage sharply and enforcing strict volume limits per trade."
            )
        elif risky_30_pct > 5:
            recommendations.append(
                "A noticeable number of trades risk more than 30% of equity. "
                "Broker can lower max leverage and introduce progressive margin requirements."
            )

        # 5. Commission / markup opportunity
        if abs(total_comm) == 0 and total_lots > 0:
            recommendations.append(
                "No commission detected on this account while there is active volume. "
                "Broker can introduce a small commission or markup to convert this flow into revenue."
            )

        # 6. Generic leverage comment
        if current_leverage > 0:
            if max_dd_pct <= -50:
                suggested_leverage = max(current_leverage / 2, 10)
                recommendations.append(
                    f"Given the deep drawdowns, consider reducing account leverage from "
                    f"{current_leverage}x to about {suggested_leverage}x to stabilise risk."
                )
            elif growth_pct > 30 and total_profit > 0:
                recommendations.append(
                    "Equity is growing with current leverage. Broker may keep the same leverage but "
                    "monitor for periods of extreme profitability to manage payout exposure."
                )

        # --------------------------
        # Summary dictionary
        # --------------------------
        summary = {
            "Total Trades": total_trades,
            "Total Profit ($)": round(total_profit, 2),
            "Total Lots": round(total_lots, 2),
            "Avg Profit per Lot ($)": round(profit_per_lot, 2),
            "Total Commission ($)": round(total_comm, 2),
            "Commission % of Profit": format_percent(comm_pct),
            "Win Rate": format_percent(win_rate),
            "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A",
            "Avg Hold Time": str(avg_hold_time).split(".")[0],
            "Median Hold Time": str(median_hold_time).split(".")[0],
            "Total Scalping Trades": scalping_count,
            "Scalping Profit ($)": round(scalping_profit, 2),
            "Scalping % of Total": format_percent(toxic_ratio),
            "Symbol Concentration (Top 3 Volume)": format_percent(symbol_concentration),
            "Daily PnL Volatility ($)": f"{pnl_volatility:.2f}",
            "Trading Pattern": (
                "Toxic Trading" if toxic_ratio >= 40
                else "Moderate Scalping" if toxic_ratio >= 20
                else "Normal Trading"
            ),
            "Final Equity": format_currency(final_equity),
            "Net Capital (Initial + CF)": format_currency(effective_start_equity),
            "Net Profit on Equity": format_currency(total_profit),
            "Profit % on Equity": format_percent(growth_pct),
            "Max Drawdown ($)": format_currency(max_dd_amount),
            "Max Drawdown (%)": format_percent(max_dd_pct),
            "Equity Behaviour": equity_behaviour,
            "Trades Risk >30% Equity": f"{risky_30} ({format_percent(risky_30_pct)})",
            "Trades Risk >50% Equity": f"{risky_50} ({format_percent(risky_50_pct)})",
        }

        # IMPORTANT: also return raw df for Swap tab
        return summary, symbol_summary, daily_pnl, equity_df, recommendations, df

    except Exception as e:
        st.error(f"Error while processing file: {e}")
        return None

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📊 Broker Risk & Trade Intelligence Dashboard")

uploaded = st.file_uploader("Upload MT5/MT4 Trade Report (.xlsx)", type=["xlsx"])

st.markdown("### Capital & Leverage Settings (manual input)")
col_cap1, col_cap2, col_cap3 = st.columns(3)
with col_cap1:
    start_capital = st.number_input("Initial Equity / Net Capital", value=10000.0, step=100.0)
with col_cap2:
    net_cash_flows = st.number_input(
        "Net Deposits / Withdrawals (total, negative for withdrawals)",
        value=0.0, step=100.0
    )
with col_cap3:
    current_leverage = st.number_input("Current Account Leverage (e.g. 100 for 1:100)", value=100.0, step=10.0)

# triple swap day input (for Swap tab)
triple_day = st.selectbox(
    "Triple Swap Day (broker setting)",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    index=2
)
triple_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(triple_day)

st.markdown("---")

if uploaded is not None:
    with st.spinner("Analyzing report..."):
        result = analyze_trades(uploaded, start_capital, net_cash_flows, current_leverage)

    if result is not None:
        summary, symbol_summary, daily_pnl, equity_df, recommendations, trades_df = result

        # --------------------------
        # TOP SUMMARY
        # --------------------------
        st.subheader("📄 Account Summary")
        top_cols = st.columns(4)
        keys_main = [
            "Total Trades",
            "Total Profit ($)",
            "Final Equity",
            "Profit % on Equity"
        ]
        for col, key in zip(top_cols, keys_main):
            with col:
                st.metric(key, summary[key])

        st.markdown("#### Detailed Stats")
        # Visualize the main numeric stats in a concise layout (cards)
        stats_cols = st.columns(4)
        stats_small = {
            "Total Trades": summary["Total Trades"],
            "Total Profit ($)": summary["Total Profit ($)"],
            "Total Lots": summary["Total Lots"],
            "Avg Profit per Lot ($)": summary["Avg Profit per Lot ($)"]
        }
        for c, (k, v) in zip(stats_cols, stats_small.items()):
            with c:
                st.metric(k, v)

        # Show remaining stats in two columns
        left, right = st.columns(2)
        left.write("**Performance & Risk**")
        left.write(f"- Win Rate: {summary['Win Rate']}")
        left.write(f"- Profit Factor: {summary['Profit Factor']}")
        left.write(f"- Trading Pattern: {summary['Trading Pattern']}")
        left.write(f"- Avg Hold Time: {summary['Avg Hold Time']}")
        left.write(f"- Median Hold Time: {summary['Median Hold Time']}")

        right.write("**Costs & Drawdown**")
        right.write(f"- Total Commission ($): {summary['Total Commission ($)']}")
        right.write(f"- Commission % of Profit: {summary['Commission % of Profit']}")
        right.write(f"- Max Drawdown ($): {summary['Max Drawdown ($)']}")
        right.write(f"- Max Drawdown (%): {summary['Max Drawdown (%)']}")
        right.write(f"- Equity Behaviour: {summary['Equity Behaviour']}")

        st.markdown("---")

        # --------------------------
        # TABS FOR VISUALISATION (5 old + 1 new Swap tab)
        # --------------------------
        tab_overview, tab_equity, tab_symbols, tab_trades, tab_actions, tab_swap = st.tabs(
            ["Overview", "Equity & Drawdown", "Symbol Risk", "Trade Risk", "Broker Actions", "Swap"]
        )

        # ---- Overview Tab ----
        with tab_overview:
            st.markdown("### Daily PnL & Account Growth")

            if len(daily_pnl) > 0:
                fig_pnl = px.bar(
                    daily_pnl,
                    x="Date",
                    y="Profit",
                    title="Daily Profit / Loss",
                    labels={"Profit": "Daily PnL ($)", "Date": "Date"}
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

                fig_bal = px.line(
                    daily_pnl,
                    x="Date",
                    y="Balance",
                    title="Account Balance Over Time",
                    markers=True,
                    labels={"Balance": "Equity ($)", "Date": "Date"}
                )
                st.plotly_chart(fig_bal, use_container_width=True)

            st.markdown("### Scalping vs Normal Trades")
            if "is_scalp" in equity_df.columns:
                scalp_counts = equity_df["is_scalp"].value_counts().rename(index={True: "Scalping", False: "Non-Scalping"})
                pie_data = pd.DataFrame({"Category": scalp_counts.index, "Count": scalp_counts.values})
                fig_scalp = px.pie(
                    pie_data,
                    names="Category",
                    values="Count",
                    title="Trade Type Distribution"
                )
                st.plotly_chart(fig_scalp, use_container_width=True)

            st.markdown("### PnL Distribution (Profit Behaviour View)")
            fig_hist = px.histogram(equity_df, x="Profit", nbins=80, title="PnL per Trade Distribution")
            st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### Hold Time Distribution (Trade Duration Behaviour View)")
            # bucket hold times
            def bucket_hold(td):
                if pd.isna(td):
                    return "Unknown"
                s = td.total_seconds()
                if s <= 180:
                    return "0-3 min (Scalp)"
                if s <= 3600:
                    return "3-60 min (Fast Intraday)"
                if s <= 10800:
                    return "60-180 min (Intraday)"
                if s <= 86400:
                    return "4-24 hours (Swing Intraday)"
                return "1+ day (Position)"
            equity_df["HoldBucket"] = equity_df["Hold_Time"].apply(bucket_hold)
            hold_counts = equity_df["HoldBucket"].value_counts().reset_index()
            hold_counts.columns = ["Bucket", "Count"]
            fig_hold = px.bar(hold_counts, x="Bucket", y="Count", title="Hold Time Buckets")
            st.plotly_chart(fig_hold, use_container_width=True)

        # ---- Equity & Drawdown Tab ----
        with tab_equity:
            st.markdown("### Equity Curve & Drawdown (Stability View)")
            if len(equity_df) > 0:
                fig_eq = px.line(
                    equity_df,
                    x="Close Time",
                    y="Equity",
                    title="Equity Curve per Trade",
                    labels={"Equity": "Equity ($)", "Close Time": "Close Time"}
                )
                st.plotly_chart(fig_eq, use_container_width=True)

                fig_dd = px.area(
                    equity_df,
                    x="Close Time",
                    y="Drawdown",
                    title="Drawdown Over Time (Equity vs Peak)",
                    labels={"Drawdown": "Drawdown ($)", "Close Time": "Close Time"}
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            st.markdown("### Risk-Per-Trade Scatter (Risk Discipline View)")
            if len(equity_df) > 0:
                fig_risk = px.scatter(
                    equity_df.reset_index(),
                    x="index",
                    y="TradeReturnPct",
                    color=equity_df["TradeReturnPct"].apply(lambda x: "Loss" if x < 0 else "Win"),
                    title="Trade Return (% of Equity Before Trade)",
                    labels={"index": "Trade Number", "TradeReturnPct": "Return (% of Equity)"},
                    hover_data=["Profit", "EquityBefore", "Equity"]
                )
                st.plotly_chart(fig_risk, use_container_width=True)

        # ---- Symbol Risk Tab ----
        with tab_symbols:
            st.markdown("### Symbol Exposure & Profitability (Focus Behaviour View)")
            if len(symbol_summary) > 0:
                # show commission per symbol as requested
                symbol_summary_display = symbol_summary.copy()
                symbol_summary_display["Commission % of Profit"] = (
                    np.where(
                        symbol_summary_display["Profit"] != 0,
                        (symbol_summary_display["Commission"].abs() / symbol_summary_display["Profit"].abs()) * 100,
                        0
                    )
                ).round(2)
                st.dataframe(symbol_summary_display, use_container_width=True)

                # Pie chart for top 3 concentration
                top3 = symbol_summary.sort_values("Volume", ascending=False).head(3)
                other_sum = symbol_summary["Volume"].sum() - top3["Volume"].sum()
                pie_df = top3[["Symbol","Volume"]].copy()
                if other_sum > 0:
                    pie_df = pd.concat(
                        [pie_df, pd.DataFrame([{"Symbol": "Other", "Volume": other_sum}])],
                        ignore_index=True
                )

                fig_pie = px.pie(pie_df, names="Symbol", values="Volume", title="Top Symbols by Volume Share")
                st.plotly_chart(fig_pie, use_container_width=True)

                # Commission Impact Chart per symbol (Gross Profit vs Commission vs Net)
                comp = symbol_summary.copy()
                comp["Gross"] = comp["Profit"] + comp["Commission"]  # before commission
                comp = comp.sort_values("Gross", ascending=False).head(20)
                fig_comm = px.bar(comp, x="Symbol", y=["Gross","Commission"], title="Gross Profit & Commission by Symbol")
                st.plotly_chart(fig_comm, use_container_width=True)
                # Add net profit as line
                # plotly express doesn't support combined easily here; keep separate small table
                st.markdown("Net Profit per symbol (Gross - Commission):")
                comp["Net"] = comp["Gross"] - comp["Commission"]
                st.dataframe(comp[["Symbol","Gross","Commission","Net"]], use_container_width=True)
            else:
                st.info("No symbol data available.")

        # ---- Trade Risk Tab ----
        with tab_trades:
            st.markdown("### Per-Trade Risk vs Equity")
            if len(equity_df) > 0:
                fig_tr_pct = px.scatter(
                    equity_df.reset_index(),
                    x="index",
                    y="TradeReturnPct",
                    color="RiskBucket",
                    title="Trade Return (% of Equity Before Trade)",
                    labels={"index": "Trade Number", "TradeReturnPct": "Return (% of Equity)"},
                    hover_data=["Profit", "EquityBefore", "Equity"]
                )
                st.plotly_chart(fig_tr_pct, use_container_width=True)

                st.markdown("#### Risk Buckets")
                risk_counts = equity_df["RiskBucket"].value_counts().reset_index()
                risk_counts.columns = ["RiskBucket", "Trades"]
                st.dataframe(risk_counts, use_container_width=True)
            else:
                st.info("No trade data available.")

        # ---- Broker Actions Tab ----
        with tab_actions:
            st.markdown("### Broker Risk Management & Profit Opportunities")
            if recommendations:
                for r in recommendations:
                    st.markdown(f"- {r}")
            else:
                st.success("No immediate broker actions detected — flow looks balanced.")

            st.markdown("#### Download Basic Report")
            with BytesIO() as buffer:
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    symbol_summary.to_excel(writer, index=False, sheet_name="Symbol Summary")
                    daily_pnl.to_excel(writer, index=False, sheet_name="Daily PnL")
                    equity_df.to_excel(writer, index=False, sheet_name="Equity Per Trade")
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Excel Report",
                    data=buffer,
                    file_name="broker_risk_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        # ---- Swap Tab ----
        def safe_to_datetime(x):
            try:
                return pd.to_datetime(x)
            except:
                return pd.NaT


        # ----------------------------------------------
        # Helper: Count swap nights with triple logic
        # ----------------------------------------------
        def count_swap_nights(open_time, close_time, triple_idx=2):
            """
            triple_idx = weekday of triple swap (0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri)
            Returns: (normal_count, triple_count)
            """

            if pd.isna(open_time) or pd.isna(close_time):
                return 0, 0

            # Swap charged at 23:59
            start_date = open_time.date()
            end_date = close_time.date()

            normal_cnt = 0
            triple_cnt = 0

            current = start_date
            while current < end_date:
                weekday = current.weekday()  # 0=Mon ... 6=Sun

                # Skip Saturday & Sunday
                if weekday in [5, 6]:
                    current += timedelta(days=1)
                    continue

                # Triple swap day
                if weekday == triple_idx:
                    triple_cnt += 1
                else:
                    normal_cnt += 1

                current += timedelta(days=1)

            return normal_cnt, triple_cnt


        # ----------------------------------------------
        # ---- Swap Tab UI ----
        # ----------------------------------------------
        st.subheader("🔄 Swap Days & Swap Money Calculator")

        st.markdown("""
        ✔ Swap charged at **23:59**  
        ✔ Saturday & Sunday skipped  
        ✔ Triple day applied  
        ✔ Symbol suffix handled automatically  
        ✔ BUY / SELL swap separated  
        """)

        # ----------------------------------------------
        # CONFIG
        # ----------------------------------------------
        PREFIX_LEN = st.number_input(
            "Symbol prefix length for matching (recommended 5–6)",
            min_value=3,
            max_value=10,
            value=6
        )

        triple_idx = st.selectbox(
            "Triple Swap Weekday",
            options=[
                ("Monday", 0),
                ("Tuesday", 1),
                ("Wednesday", 2),
                ("Thursday", 3),
                ("Friday", 4),
            ],
            index=2
        )[1]

        # ----------------------------------------------
        # Load Swap Rates File
        # ----------------------------------------------
        st.markdown("### Upload Swap Rate File")
        swap_rate_file = st.file_uploader(
            "Upload Swap Rate CSV / Excel",
            type=["csv", "xlsx"],
            key="swap_rates"
        )

        if swap_rate_file is not None:
            if swap_rate_file.name.endswith(".csv"):
                swap_rates = pd.read_csv(swap_rate_file)
            else:
                swap_rates = pd.read_excel(swap_rate_file)

            # Expect columns: Symbol, OrderType, SwapRate
            swap_rates["SymbolPrefix"] = swap_rates["Symbol"].astype(str).str[:PREFIX_LEN]
        else:
            swap_rates = pd.DataFrame(
                columns=["Symbol", "OrderType", "SwapRate", "SymbolPrefix"]
            )

        # ----------------------------------------------
        # Prepare Trades Data
        # trades_df MUST exist before this block
        # Required columns:
        # Symbol, Type, Volume, Open Time, Close Time
        # ----------------------------------------------
        trades = trades_df.copy()

        trades["Open Time"] = trades["Open Time"].apply(safe_to_datetime)
        trades["Close Time"] = trades["Close Time"].apply(safe_to_datetime)

        # ----------------------------------------------
        # Build Per-Trade Swap Table
        # ----------------------------------------------
        swap_rows = []

        for _, r in trades.iterrows():
            normal_cnt, triple_cnt = count_swap_nights(
                r["Open Time"], r["Close Time"], triple_idx
            )

            final_cnt = normal_cnt + (triple_cnt * 3)

            symbol = r.get("Symbol", "")
            symbol_prefix = str(symbol)[:PREFIX_LEN]

            swap_rows.append({
                "Symbol": symbol,
                "SymbolPrefix": symbol_prefix,
                "OrderType": r.get("Type", ""),
                "Volume": r.get("Volume", 0),
                "Open Time": r["Open Time"],
                "Close Time": r["Close Time"],
                "SwapDays_Normal": normal_cnt,
                "SwapDays_Triple": triple_cnt,
                "SwapDays_Final": final_cnt
            })

        swap_df = pd.DataFrame(swap_rows)

        # ----------------------------------------------
        # Merge Swap Rates using PREFIX + BUY/SELL
        # ----------------------------------------------
        swap_df = swap_df.merge(
            swap_rates[["SymbolPrefix", "OrderType", "SwapRate"]],
            on=["SymbolPrefix", "OrderType"],
            how="left"
        )

        # ----------------------------------------------
        # Manual Swap Rate Override
        # ----------------------------------------------
        st.markdown("### Manual Swap Rate Override (Optional)")

        manual_rates = st.data_editor(
            swap_df[["SymbolPrefix", "OrderType", "SwapRate"]]
            .drop_duplicates()
            .reset_index(drop=True),
            use_container_width=True,
            num_rows="dynamic"
        )

        # Replace SwapRate with manual values
        swap_df = swap_df.drop(columns=["SwapRate"], errors="ignore").merge(
            manual_rates,
            on=["SymbolPrefix", "OrderType"],
            how="left"
        )

        # ----------------------------------------------
        # Swap Money Calculation
        # ----------------------------------------------
        swap_df["SwapRate"] = pd.to_numeric(swap_df["SwapRate"], errors="coerce").fillna(0)

        swap_df["SwapMoney"] = (
                swap_df["SwapRate"] *
                swap_df["Volume"] *
                swap_df["SwapDays_Final"]
        )

        # ----------------------------------------------
        # Display Per-Trade Table
        # ----------------------------------------------
        st.markdown("### Swap Per Trade")
        st.dataframe(swap_df, use_container_width=True)

        # ----------------------------------------------
        # BUY vs SELL Carry-Forward Summary
        # ----------------------------------------------
        cf_trades = swap_df[swap_df["SwapDays_Final"] > 0]

        buy_sell_summary = cf_trades.groupby(
            ["SymbolPrefix", "OrderType"], dropna=True
        ).agg(
            CarryForwardTrades=("Symbol", "count"),
            CarryForwardVolume=("Volume", "sum"),
            SwapDays=("SwapDays_Final", "sum"),
            TotalSwapMoney=("SwapMoney", "sum")
        ).reset_index()

        st.markdown("### BUY vs SELL Carry-Forward Swap Summary")
        st.dataframe(buy_sell_summary, use_container_width=True)

        # ----------------------------------------------
        # Download Excel
        # ----------------------------------------------
        buf = BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            swap_df.to_excel(writer, index=False, sheet_name="Swap Per Trade")
            buy_sell_summary.to_excel(
                writer, index=False, sheet_name="Swap BUY SELL Summary"
            )

        buf.seek(0)

        st.download_button(
            "📥 Download Swap Excel",
            data=buf,
            file_name="swap_with_rates.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        )
