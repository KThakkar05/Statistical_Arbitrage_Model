"""Interactive Streamlit dashboard for the pairs trading library.

Two pages:
    - Pair Analysis: cointegration diagnostics and z-score signal for one pair.
    - Pair Rankings: scans every combination within an industry, ranked by
      ADF statistic.

All statistics are computed via ``src.cointegration`` and ``src.signals`` —
this file only handles data download orchestration, layout, and charts.
"""
import itertools

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

from src.cointegration import adf_test, calc_spread, fit_ols, scan_pairs
from src.signals import calc_zscore

st.set_page_config(page_title="Pairs Trading Dashboard", layout="wide")

# ── Top-50 companies: label -> (ticker, industry) ────────────────────────────
TOP50 = {
    "Apple (AAPL)":                      ("AAPL",         "Technology"),
    "Microsoft (MSFT)":                  ("MSFT",         "Technology"),
    "NVIDIA (NVDA)":                     ("NVDA",         "Technology"),
    "Alphabet / Google (GOOGL)":         ("GOOGL",        "Technology"),
    "Meta Platforms (META)":             ("META",         "Technology"),
    "TSMC (TSM)":                        ("TSM",          "Technology"),
    "Broadcom (AVGO)":                   ("AVGO",         "Technology"),
    "Samsung Electronics (005930.KS)":   ("005930.KS",    "Technology"),
    "Oracle (ORCL)":                     ("ORCL",         "Technology"),
    "ASML (ASML)":                       ("ASML",         "Technology"),
    "AMD (AMD)":                         ("AMD",          "Technology"),
    "SAP (SAP)":                         ("SAP",          "Technology"),
    "Accenture (ACN)":                   ("ACN",          "Technology"),
    "Salesforce (CRM)":                  ("CRM",          "Technology"),
    "Tencent (0700.HK)":                 ("0700.HK",      "Technology"),
    "Amazon (AMZN)":                     ("AMZN",         "Consumer Discretionary"),
    "Tesla (TSLA)":                      ("TSLA",         "Consumer Discretionary"),
    "Walmart (WMT)":                     ("WMT",          "Consumer Discretionary"),
    "Home Depot (HD)":                   ("HD",           "Consumer Discretionary"),
    "Costco (COST)":                     ("COST",         "Consumer Discretionary"),
    "Netflix (NFLX)":                    ("NFLX",         "Consumer Discretionary"),
    "Alibaba (BABA)":                    ("BABA",         "Consumer Discretionary"),
    "Toyota (TM)":                       ("TM",           "Consumer Discretionary"),
    "LVMH (MC.PA)":                      ("MC.PA",        "Luxury & Retail"),
    "Hermes (RMS.PA)":                   ("RMS.PA",       "Luxury & Retail"),
    "L'Oreal (OR.PA)":                   ("OR.PA",        "Luxury & Retail"),
    "Procter & Gamble (PG)":             ("PG",           "Consumer Staples"),
    "Nestle (NESN.SW)":                  ("NESN.SW",      "Consumer Staples"),
    "JPMorgan Chase (JPM)":              ("JPM",          "Financials"),
    "Visa (V)":                          ("V",            "Financials"),
    "Mastercard (MA)":                   ("MA",           "Financials"),
    "Berkshire Hathaway (BRK-B)":        ("BRK-B",        "Financials"),
    "Bank of America (BAC)":             ("BAC",          "Financials"),
    "Goldman Sachs (GS)":                ("GS",           "Financials"),
    "HSBC (HSBC)":                       ("HSBC",         "Financials"),
    "Eli Lilly (LLY)":                   ("LLY",          "Healthcare"),
    "UnitedHealth (UNH)":                ("UNH",          "Healthcare"),
    "Novo Nordisk (NVO)":                ("NVO",          "Healthcare"),
    "Johnson & Johnson (JNJ)":           ("JNJ",          "Healthcare"),
    "AbbVie (ABBV)":                     ("ABBV",         "Healthcare"),
    "Roche (ROG.SW)":                    ("ROG.SW",       "Healthcare"),
    "Saudi Aramco (2222.SR)":            ("2222.SR",      "Energy"),
    "ExxonMobil (XOM)":                  ("XOM",          "Energy"),
    "Chevron (CVX)":                     ("CVX",          "Energy"),
    "Shell (SHEL)":                      ("SHEL",         "Energy"),
    "Reliance Industries (RELIANCE.NS)": ("RELIANCE.NS",  "Energy"),
    "Caterpillar (CAT)":                 ("CAT",          "Industrials"),
    "Siemens (SIE.DE)":                  ("SIE.DE",       "Industrials"),
    "T-Mobile (TMUS)":                   ("TMUS",         "Telecoms"),
    "Custom ticker...":                  ("__custom__",   "__custom__"),
}

INDUSTRIES = sorted({v[1] for v in TOP50.values() if v[1] != "__custom__"})


def filtered_labels(industry: str) -> list:
    if industry == "All Industries":
        labels = [k for k, v in TOP50.items() if v[1] != "__custom__"]
    else:
        labels = [k for k, v in TOP50.items() if v[1] == industry]
    return labels + ["Custom ticker..."]


def stock_selector(label: str, industry: str, default_label: str, custom_default: str) -> str:
    labels = filtered_labels(industry)
    default_idx = labels.index(default_label) if default_label in labels else 0
    selected = st.selectbox(label, labels, index=default_idx)
    if TOP50[selected][0] == "__custom__":
        return st.text_input(f"{label} - enter ticker", value=custom_default).strip().upper()
    return TOP50[selected][0]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Page", ["Pair Analysis", "Pair Rankings"], label_visibility="collapsed")
    st.divider()

    st.header("Date Range")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=pd.Timestamp("2018-01-01"))
    with col2:
        end_date = st.date_input("End", value=pd.Timestamp("2023-12-31"))

    st.divider()

    if page == "Pair Analysis":
        st.header("Pair Settings")
        industry = st.selectbox("Filter by industry", ["All Industries"] + INDUSTRIES)
        stock_1 = stock_selector("Stock 1", industry, "Apple (AAPL)", "AAPL")
        stock_2 = stock_selector("Stock 2", industry, "Microsoft (MSFT)", "MSFT")
        window = st.slider("Rolling window (days)", 10, 120, 60, 5)
        entry_threshold = st.slider("Entry z-score threshold", 1.0, 3.0, 2.0, 0.25)
        run = st.button("Run Analysis", type="primary", use_container_width=True)

    else:  # Pair Rankings
        st.header("Scan Settings")
        scan_industry = st.selectbox(
            "Industry to scan",
            ["All Industries"] + INDUSTRIES,
            help="'All Industries' tests every possible pair across all 49 stocks (~1,176 pairs) - this takes a few minutes.",
        )
        min_rows = st.slider("Min. trading days required", 100, 500, 200, 50)
        top_n = st.slider("Show top N pairs", 10, 100, 25, 5)
        run_rankings = st.button("Run Rankings", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 - PAIR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
if page == "Pair Analysis":
    st.title("Pair Analysis")
    st.markdown("Select two stocks in the sidebar to test cointegration and visualise the z-score signal.")

    if not run:
        st.info("Configure the pair in the sidebar, then click **Run Analysis**.")
        st.stop()

    if stock_1 == stock_2:
        st.error("Please select two different tickers.")
        st.stop()

    with st.spinner(f"Downloading {stock_1} and {stock_2}..."):
        raw = yf.download([stock_1, stock_2], start=str(start_date), end=str(end_date),
                           auto_adjust=True, progress=False)

    if raw.empty or "Close" not in raw.columns:
        st.error("No data returned - check tickers and date range.")
        st.stop()

    prices = raw["Close"][[stock_1, stock_2]].dropna()

    if len(prices) < window + 10:
        st.error(f"Only {len(prices)} rows - not enough for a {window}-day window.")
        st.stop()

    s1, s2 = prices[stock_1], prices[stock_2]

    alpha, beta, _ = fit_ols(s1, s2)
    spread = calc_spread(s1, s2, alpha, beta)
    adf = adf_test(spread)
    zscore = calc_zscore(spread, window)

    # Metrics row
    st.subheader(f"{stock_1} / {stock_2}  -  {start_date}  to  {end_date}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Hedge ratio (β)", f"{beta:.4f}")
    m2.metric("Intercept (α)",   f"{alpha:.4f}")
    m3.metric("ADF statistic",   f"{adf['adf_stat']:.4f}")
    m4.metric("ADF p-value",     f"{adf['p_value']:.4f}")
    m5.metric("Z-score (latest)", f"{zscore.iloc[-1]:.3f}")

    # ADF panel
    with st.expander("ADF Test - full results", expanded=True):
        if adf["p_value"] < 0.05:
            st.success(f"Spread is **stationary** at the 5% level (p = {adf['p_value']:.4f}). Evidence of cointegration.")
        else:
            st.warning(f"Spread is **not stationary** at the 5% level (p = {adf['p_value']:.4f}). Use caution.")

        crit_df = pd.DataFrame({"Critical value": {k: f"{v:.4f}" for k, v in adf["crit_values"].items()}})
        crit_df.index.name = "Significance level"
        ca, cb = st.columns([1, 2])
        with ca:
            st.dataframe(crit_df, use_container_width=True)
        with cb:
            st.markdown(f"""
| Parameter | Value |
|---|---|
| ADF statistic | `{adf['adf_stat']:.4f}` |
| p-value | `{adf['p_value']:.4f}` |
| Lags used | `{adf['lags']}` |
| Observations | `{adf['nobs']}` |
""")

    # Normalised price chart
    st.subheader("Normalised Price Series")
    fig_price = go.Figure()
    for series, name in [(s1, stock_1), (s2, stock_2)]:
        norm = (series / series.iloc[0]) * 100
        fig_price.add_trace(go.Scatter(
            x=norm.index, y=norm, name=name, line=dict(width=1.5),
            hovertemplate="%{x|%Y-%m-%d}<br>" + name + ": %{y:.2f}<extra></extra>",
        ))
    fig_price.update_layout(yaxis_title="Rebased (100 = first day)",
                             hovermode="x unified", height=300, margin=dict(t=30, b=30))
    st.plotly_chart(fig_price, use_container_width=True)

    # Spread + Z-score chart
    st.subheader("Spread & Rolling Z-Score")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=(f"{stock_1} / {stock_2} - Raw Spread",
                                         f"Rolling Z-Score ({window}-day window)"),
                         vertical_spacing=0.08)

    fig.add_trace(go.Scatter(x=spread.index, y=spread, name="Spread",
                              line=dict(color="#2c7bb6", width=1),
                              hovertemplate="%{x|%Y-%m-%d}<br>Spread: %{y:.4f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=spread.mean(), line=dict(color="grey", dash="dash", width=0.8),
                  annotation_text="Mean", annotation_position="top left", row=1, col=1)

    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, name="Z-Score",
                              line=dict(color="#1a1a2e", width=1),
                              hovertemplate="%{x|%Y-%m-%d}<br>Z-Score: %{y:.3f}<extra></extra>"),
                  row=2, col=1)

    for level, color, lbl in [(entry_threshold, "#d7191c", f"+{entry_threshold}"),
                               (-entry_threshold, "#1a9641", f"-{entry_threshold}"),
                               (0, "grey", "")]:
        fig.add_hline(y=level,
                      line=dict(color=color, dash="dash" if level != 0 else "solid",
                                width=1.2 if level != 0 else 0.6),
                      annotation_text=lbl, annotation_position="top left", row=2, col=1)

    above = zscore.where(zscore > entry_threshold)
    below = zscore.where(zscore < -entry_threshold)
    fig.add_trace(go.Scatter(
        x=pd.concat([above.index.to_series(), above.index.to_series()[::-1]]),
        y=pd.concat([above.fillna(entry_threshold),
                     pd.Series([entry_threshold] * len(above), index=above.index)[::-1]]),
        fill="toself", fillcolor="rgba(215,25,28,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([below.index.to_series(), below.index.to_series()[::-1]]),
        y=pd.concat([below.fillna(-entry_threshold),
                     pd.Series([-entry_threshold] * len(below), index=below.index)[::-1]]),
        fill="toself", fillcolor="rgba(26,150,65,0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)

    fig.update_layout(hovermode="x unified", height=600, margin=dict(t=50, b=30),
                       yaxis_title="Spread", yaxis2_title="Z-Score")
    st.plotly_chart(fig, use_container_width=True)

    # Signal log
    st.subheader("Signal Log")
    signal = pd.Series("-", index=zscore.index)
    signal[zscore > entry_threshold] = f"SHORT spread (z > +{entry_threshold})"
    signal[zscore < -entry_threshold] = f"LONG spread  (z < -{entry_threshold})"
    signal_log = (pd.DataFrame({"Z-Score": zscore.round(3), "Signal": signal})
                  .loc[signal != "-"].tail(50))
    if signal_log.empty:
        st.info("No signals triggered at the selected threshold.")
    else:
        st.dataframe(signal_log, use_container_width=True)

    with st.expander("Raw price data"):
        st.dataframe(prices.tail(100), use_container_width=True)
        st.download_button("Download CSV", prices.to_csv().encode(),
                            file_name=f"{stock_1}_{stock_2}_prices.csv", mime="text/csv")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 - PAIR RANKINGS
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.title("Pair Rankings")
    st.markdown("Scans every combination of stocks in the selected industry and ranks them by ADF "
                "statistic (most negative = strongest evidence of cointegration).")

    if not run_rankings:
        st.info("Choose an industry in the sidebar and click **Run Rankings**.")
        st.stop()

    if scan_industry == "All Industries":
        scan_entries = [(label, ticker) for label, (ticker, _) in TOP50.items() if ticker != "__custom__"]
    else:
        scan_entries = [(label, ticker) for label, (ticker, ind) in TOP50.items() if ind == scan_industry]

    label_by_ticker = dict(scan_entries)
    all_tickers = [t for _, t in scan_entries]
    n_pairs = len(list(itertools.combinations(all_tickers, 2)))

    st.info(f"Scanning **{len(all_tickers)} stocks** -> **{n_pairs} pairs**. Downloading prices...")

    with st.spinner("Downloading price data and running ADF tests..."):
        df = scan_pairs(all_tickers, str(start_date), str(end_date), min_rows=min_rows)

    if df.empty:
        st.error("No valid pairs found - try a wider date range or lower the minimum observations.")
        st.stop()

    df["Stock 1"] = df["ticker_1"].map(label_by_ticker)
    df["Stock 2"] = df["ticker_2"].map(label_by_ticker)
    df = df.rename(columns={
        "rank": "Rank", "ticker_1": "Ticker 1", "ticker_2": "Ticker 2",
        "adf_stat": "ADF Statistic", "p_value": "P-Value", "beta": "Hedge Ratio (β)",
        "n_obs": "Observations",
    })
    df["Stationary 5%"] = df["stationary_5pct"].map({True: "Yes", False: "No"})
    df["Stationary 1%"] = df["stationary_1pct"].map({True: "Yes", False: "No"})

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_stationary_5 = (df["Stationary 5%"] == "Yes").sum()
    n_stationary_1 = (df["Stationary 1%"] == "Yes").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pairs tested", len(df))
    c2.metric("Stationary @ 5%", n_stationary_5)
    c3.metric("Stationary @ 1%", n_stationary_1)
    c4.metric("Best ADF statistic", f"{df['ADF Statistic'].iloc[0]:.4f}")

    # ── Bar chart - top N by ADF stat ─────────────────────────────────────────
    st.subheader(f"Top {top_n} Pairs by ADF Statistic")
    top_df = df.head(top_n).copy()
    top_df["Pair"] = top_df["Ticker 1"] + " / " + top_df["Ticker 2"]
    top_df["Color"] = top_df["P-Value"].apply(
        lambda p: "#2ca02c" if p < 0.01 else ("#ffbf00" if p < 0.05 else "#d62728")
    )

    fig_bar = go.Figure(go.Bar(
        x=top_df["ADF Statistic"],
        y=top_df["Pair"],
        orientation="h",
        marker_color=top_df["Color"],
        customdata=top_df[["P-Value", "Hedge Ratio (β)", "Observations"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "ADF Statistic: %{x:.4f}<br>"
            "P-Value: %{customdata[0]:.4f}<br>"
            "Hedge Ratio: %{customdata[1]:.4f}<br>"
            "Observations: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig_bar.update_layout(
        xaxis_title="ADF Statistic (more negative = more stationary)",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 28),
        margin=dict(t=20, b=40, l=180),
        showlegend=False,
    )
    fig_bar.add_annotation(
        text="<span style='color:#2ca02c'>■</span> p < 1%  "
             "<span style='color:#ffbf00'>■</span> p < 5%  "
             "<span style='color:#d62728'>■</span> p >= 5%",
        xref="paper", yref="paper", x=1, y=1.02,
        showarrow=False, align="right",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Full ranked table ─────────────────────────────────────────────────────
    st.subheader("Full Rankings Table")

    def highlight_row(row):
        p = row["P-Value"]
        if p < 0.01:
            bg = "background-color: rgba(44,160,44,0.15)"
        elif p < 0.05:
            bg = "background-color: rgba(255,191,0,0.15)"
        else:
            bg = ""
        return [bg] * len(row)

    display_cols = ["Rank", "Stock 1", "Stock 2", "ADF Statistic",
                     "P-Value", "Hedge Ratio (β)", "Observations",
                     "Stationary 5%", "Stationary 1%"]

    styled = df[display_cols].style.apply(highlight_row, axis=1).format({
        "ADF Statistic": "{:.4f}",
        "P-Value": "{:.4f}",
        "Hedge Ratio (β)": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True, height=500)

    st.download_button(
        "Download full results CSV",
        df[display_cols].to_csv(index=False).encode(),
        file_name=f"pair_rankings_{scan_industry.replace(' ', '_')}.csv",
        mime="text/csv",
    )
