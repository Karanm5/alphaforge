"""
AlphaForge — Institutional Quantitative Research Platform
Streamlit dashboard tying all five engines into a single interface.
"""

import sys, os
sys.path.insert(0, "/home/claude")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── Engine Imports ───
from alphaforge.data.loader import fetch_prices, fetch_returns, fetch_fred, FRED_SERIES
from alphaforge.engines.stat_arb import (
    KalmanPairsFilter, StatArbSignal, fit_ou, engle_granger_test
)
from alphaforge.engines.regime_detection import (
    HMMRegimeDetector, RegimeGARCH, realised_vol
)
from alphaforge.engines.portfolio import (
    BlackLitterman, CVaROptimiser, BLView, portfolio_metrics
)
from alphaforge.engines.execution import (
    AlmgrenChriss, MarketParams, estimate_market_params, participation_rate_schedule
)
from alphaforge.engines.risk_engine import (
    build_risk_report, monte_carlo_var_es, parametric_var_es,
    historical_var_es, stress_test_portfolio, STRESS_SCENARIOS
)

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AlphaForge",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  Design System
# ─────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', monospace !important;
}

/* Dark terminal theme */
.stApp { background: #0a0c0f; color: #c9d1d9; }

[data-testid="stSidebar"] {
    background: #080a0d !important;
    border-right: 1px solid #1e2530;
}

.metric-card {
    background: #0d1117;
    border: 1px solid #1e2530;
    border-left: 3px solid #00d4aa;
    padding: 14px 18px;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    margin-bottom: 8px;
}
.metric-card .label {
    font-size: 10px;
    color: #6e7681;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 22px;
    font-weight: 600;
    color: #e6edf3;
}
.metric-card.green { border-left-color: #00d4aa; }
.metric-card.red   { border-left-color: #f85149; }
.metric-card.amber { border-left-color: #d29922; }
.metric-card.blue  { border-left-color: #388bfd; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #00d4aa;
    border-bottom: 1px solid #1e2530;
    padding-bottom: 6px;
    margin: 24px 0 16px 0;
}

.engine-badge {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    color: #00d4aa;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 2px;
    letter-spacing: 0.08em;
}

.stButton button {
    background: #161b22 !important;
    color: #00d4aa !important;
    border: 1px solid #00d4aa !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
    border-radius: 3px !important;
    transition: all 0.15s ease !important;
}
.stButton button:hover {
    background: #00d4aa20 !important;
    box-shadow: 0 0 12px #00d4aa30 !important;
}

.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}

.stSlider > div > div > div { background: #00d4aa !important; }

div[data-testid="stMetric"] {
    background: #0d1117;
    border: 1px solid #1e2530;
    padding: 12px;
    border-radius: 4px;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.05em;
    color: #6e7681;
}
.stTabs [aria-selected="true"] { color: #00d4aa !important; }
.stTabs [data-baseweb="tab-border"] { background-color: #00d4aa !important; }

.stDataFrame { font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; }

.logo-text {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #e6edf3;
}
.logo-accent { color: #00d4aa; }

p, .stMarkdown { color: #8b949e !important; font-size: 13px; }
h1, h2, h3 { color: #e6edf3 !important; font-family: 'IBM Plex Mono', monospace !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Plot theme
# ─────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="#0a0c0f",
    plot_bgcolor="#0d1117",
    font=dict(family="IBM Plex Mono", color="#8b949e", size=11),
    xaxis=dict(gridcolor="#161b22", linecolor="#30363d", zerolinecolor="#30363d"),
    yaxis=dict(gridcolor="#161b22", linecolor="#30363d", zerolinecolor="#30363d"),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="#0d1117", bordercolor="#30363d", borderwidth=1),
)

ACCENT   = "#00d4aa"
RED      = "#f85149"
AMBER    = "#d29922"
BLUE     = "#388bfd"
PURPLE   = "#a371f7"
ORANGE   = "#db6d28"


def styled_fig(title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(color="#c9d1d9", size=13)))
    return fig


# ─────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding: 12px 0 20px 0; border-bottom: 1px solid #1e2530; margin-bottom: 20px;">
        <div class="logo-text">⚡ <span class="logo-accent">ALPHA</span>FORGE</div>
        <div style="font-family:'IBM Plex Mono';font-size:10px;color:#6e7681;letter-spacing:0.1em;margin-top:4px;">
          QUANTITATIVE RESEARCH PLATFORM v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

    fred_key = st.text_input(
        "FRED API Key", type="password",
        placeholder="Enter your FRED API key",
        help="Get free key at fred.stlouisfed.org"
    )

    st.markdown('<div class="section-header">Universe</div>', unsafe_allow_html=True)

    DEFAULT_TICKERS = "SPY,QQQ,GLD,TLT,XLF,XLE,MSFT,JPM"
    tickers_raw = st.text_input("Tickers (comma-separated)", value=DEFAULT_TICKERS)
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    start_str  = start_date.strftime("%Y-%m-%d")

    st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)

    engine = st.radio(
        "Engine",
        options=[
            "📊  Dashboard",
            "📈  Stat Arb",
            "🔮  Regimes",
            "⚖️  Portfolio",
            "🎯  Execution",
            "🛡️  Risk Engine",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="section-header">Data</div>', unsafe_allow_html=True)
    load_btn = st.button("⟳  Load / Refresh Data", use_container_width=True)


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(tickers, start):
    prices  = fetch_prices(tickers, start=start)
    returns = np.log(prices / prices.shift(1)).dropna()
    return prices, returns


if "prices" not in st.session_state or load_btn:
    with st.spinner("Fetching market data..."):
        try:
            prices, returns = load_market_data(tuple(tickers), start_str)
            st.session_state["prices"]  = prices
            st.session_state["returns"] = returns
            st.session_state["tickers"] = list(prices.columns)
        except Exception as e:
            st.error(f"Data error: {e}")
            st.stop()

prices  = st.session_state["prices"]
returns = st.session_state["returns"]
tickers = st.session_state["tickers"]


# ─────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────

if "Dashboard" in engine:
    st.markdown('<h2 style="margin-bottom:4px;">Market Overview</h2>', unsafe_allow_html=True)
    st.markdown(f'<span class="engine-badge">LIVE DATA · {len(tickers)} ASSETS · {len(returns)} DAYS</span>', unsafe_allow_html=True)

    # KPI Row
    ann = 252
    cols = st.columns(5)
    metrics = []
    for t in tickers[:5]:
        if t not in returns.columns:
            continue
        r   = returns[t].dropna()
        ret = r.mean() * ann
        vol = r.std() * np.sqrt(ann)
        sr  = ret / vol if vol > 0 else 0
        metrics.append((t, ret, vol, sr))

    for i, (t, ret, vol, sr) in enumerate(metrics):
        colour = "green" if ret > 0 else "red"
        cols[i % 5].markdown(f"""
        <div class="metric-card {colour}">
            <div class="label">{t}</div>
            <div class="value">{ret*100:+.1f}%</div>
            <div style="font-family:'IBM Plex Mono';font-size:10px;color:#6e7681;margin-top:4px;">
                Vol {vol*100:.1f}% · SR {sr:.2f}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Correlation heatmap + rolling returns
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-header">Return Correlation Matrix</div>', unsafe_allow_html=True)
        corr = returns.corr()
        fig  = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale=[[0, RED], [0.5, "#0d1117"], [1, ACCENT]],
            zmid=0, text=corr.round(2).values,
            texttemplate="%{text}", textfont=dict(size=10),
        ))
        fig.update_layout(**PLOT_LAYOUT, height=380, title="")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">Cumulative Returns</div>', unsafe_allow_html=True)
        fig = styled_fig()
        colors_list = [ACCENT, BLUE, AMBER, RED, PURPLE, ORANGE, "#c9d1d9", "#6e7681"]
        for i, t in enumerate(tickers[:8]):
            if t not in returns.columns:
                continue
            cum = (1 + returns[t]).cumprod()
            fig.add_trace(go.Scatter(
                x=cum.index, y=cum,
                name=t, mode="lines",
                line=dict(color=colors_list[i % len(colors_list)], width=1.5),
            ))
        fig.update_layout(height=380, yaxis_title="Cumulative Return")
        st.plotly_chart(fig, use_container_width=True)

    # Distribution of returns
    st.markdown('<div class="section-header">Return Distributions</div>', unsafe_allow_html=True)
    sel_ticker = st.selectbox("Asset", tickers, key="dash_dist")
    r_sel = returns[sel_ticker].dropna() * 100

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Rolling Volatility"])
    fig.add_trace(go.Histogram(
        x=r_sel, nbinsx=80, name="Returns",
        marker_color=ACCENT, opacity=0.7,
        histnorm="probability density",
    ), row=1, col=1)

    # Normal overlay
    x_range = np.linspace(r_sel.min(), r_sel.max(), 200)
    from scipy.stats import norm
    pdf = norm.pdf(x_range, r_sel.mean(), r_sel.std())
    fig.add_trace(go.Scatter(
        x=x_range, y=pdf, name="Normal fit",
        line=dict(color=RED, dash="dash", width=1.5),
    ), row=1, col=1)

    # Rolling vol
    rv = returns[sel_ticker].rolling(21).std() * np.sqrt(252) * 100
    fig.add_trace(go.Scatter(
        x=rv.index, y=rv, name="21d Vol",
        line=dict(color=AMBER, width=1.5), fill="tozeroy",
        fillcolor=AMBER + "20",
    ), row=1, col=2)

    fig.update_layout(**PLOT_LAYOUT, height=320, showlegend=False)
    fig.update_xaxes(gridcolor="#161b22")
    fig.update_yaxes(gridcolor="#161b22")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  ENGINE 1: STAT ARB
# ─────────────────────────────────────────────

elif "Stat Arb" in engine:
    st.markdown('<h2>Statistical Arbitrage</h2>', unsafe_allow_html=True)
    st.markdown('<span class="engine-badge">KALMAN FILTER · ORNSTEIN-UHLENBECK · PAIRS TRADING</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    asset_y   = c1.selectbox("Asset Y (dependent)",  tickers, index=0)
    asset_x   = c2.selectbox("Asset X (independent)", tickers, index=1)
    entry_z   = c3.slider("Entry Z-score", 1.0, 3.5, 2.0, 0.1)

    c4, c5    = st.columns(2)
    exit_z    = c4.slider("Exit Z-score", 0.1, 1.5, 0.5, 0.1)
    delta     = c5.select_slider("Kalman δ", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3], value=1e-4)

    if asset_y == asset_x:
        st.warning("Select two different assets.")
        st.stop()

    run_btn = st.button("▶  Run Statistical Arbitrage", use_container_width=False)

    if run_btn:
        with st.spinner("Running Kalman filter + OU estimation..."):
            price_y = prices[asset_y].dropna()
            price_x = prices[asset_x].dropna()
            common  = price_y.index.intersection(price_x.index)
            price_y, price_x = price_y[common], price_x[common]

            # Cointegration test
            coint = engle_granger_test(price_y, price_x)

            # Kalman + signal
            engine_sa = StatArbSignal(entry_z=entry_z, exit_z=exit_z, delta=delta)
            results   = engine_sa.fit(price_y, price_x)
            ou_params = results.attrs["ou_params"]
            perf      = engine_sa.performance_summary(results)

        # Cointegration metrics
        st.markdown('<div class="section-header">Cointegration Test (Engle-Granger)</div>', unsafe_allow_html=True)
        cc = st.columns(4)
        is_coint = coint["Cointegrated"]
        cc[0].markdown(f'<div class="metric-card {"green" if is_coint else "red"}"><div class="label">Cointegrated</div><div class="value">{"✓ YES" if is_coint else "✗ NO"}</div></div>', unsafe_allow_html=True)
        cc[1].markdown(f'<div class="metric-card blue"><div class="label">ADF Statistic</div><div class="value">{coint["ADF Statistic"]}</div></div>', unsafe_allow_html=True)
        cc[2].markdown(f'<div class="metric-card blue"><div class="label">p-value</div><div class="value">{coint["p-value"]}</div></div>', unsafe_allow_html=True)
        cc[3].markdown(f'<div class="metric-card blue"><div class="label">Hedge Ratio β</div><div class="value">{coint["Hedge Ratio β"]}</div></div>', unsafe_allow_html=True)

        # OU params
        st.markdown('<div class="section-header">Ornstein-Uhlenbeck Parameters</div>', unsafe_allow_html=True)
        oc = st.columns(5)
        ou_data = [
            ("θ (Mean-Rev)", f"{ou_params.theta:.4f}"),
            ("μ (LR Mean)",  f"{ou_params.mu:.4f}"),
            ("σ (Diffusion)",f"{ou_params.sigma:.4f}"),
            ("Half-Life",    f"{ou_params.half_life:.1f}d"),
            ("σ_eq",         f"{ou_params.sigma_eq:.4f}"),
        ]
        for i, (lbl, val) in enumerate(ou_data):
            oc[i].markdown(f'<div class="metric-card blue"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

        # Performance
        st.markdown('<div class="section-header">Strategy Performance</div>', unsafe_allow_html=True)
        pc = st.columns(len(perf))
        colours = ["green", "green", "red", "green", "red", "blue", "blue"]
        for i, (k, v) in enumerate(perf.items()):
            colour = colours[i % len(colours)]
            pc[i].markdown(f'<div class="metric-card {colour}"><div class="label">{k}</div><div class="value">{v}</div></div>', unsafe_allow_html=True)

        # Charts
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=["Price Series + Hedge Ratio", "Kalman Spread", "Z-Score + Signals", "Cumulative PnL"],
            row_heights=[0.25, 0.25, 0.25, 0.25])

        # Price series
        fig.add_trace(go.Scatter(x=price_y.index, y=price_y/price_y.iloc[0],
            name=asset_y, line=dict(color=ACCENT, width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_x.index, y=price_x/price_x.iloc[0],
            name=asset_x, line=dict(color=BLUE, width=1.5)), row=1, col=1)

        # Hedge ratio on secondary axis
        fig.add_trace(go.Scatter(x=results.index, y=results["hedge_ratio"],
            name="β (Kalman)", line=dict(color=AMBER, dash="dot", width=1)),
            row=1, col=1)

        # Spread
        fig.add_trace(go.Scatter(x=results.index, y=results["spread"],
            name="Spread", line=dict(color=PURPLE, width=1), fill="tozeroy",
            fillcolor=PURPLE + "15"), row=2, col=1)

        # Z-score
        fig.add_trace(go.Scatter(x=results.index, y=results["z_score"],
            name="Z-score", line=dict(color=ACCENT, width=1.5)), row=3, col=1)
        for level, col_name in [(entry_z, RED), (-entry_z, ACCENT), (exit_z, "#ffffff40"), (-exit_z, "#ffffff40")]:
            fig.add_hline(y=level, line_color=col_name, line_dash="dash",
                         line_width=1, row=3, col=1)

        # Signal markers
        long_idx  = results.index[results["signal"] == 1]
        short_idx = results.index[results["signal"] == -1]

        # PnL
        cum_pnl = (1 + results["pair_pnl"]).cumprod()
        fig.add_trace(go.Scatter(x=cum_pnl.index, y=cum_pnl,
            name="Pair PnL", line=dict(color=ACCENT, width=2)), row=4, col=1)

        fig.update_layout(**PLOT_LAYOUT, height=800, showlegend=True)
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#161b22", row=i, col=1)
            fig.update_yaxes(gridcolor="#161b22", row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Configure parameters and click **Run Statistical Arbitrage** to begin.")


# ─────────────────────────────────────────────
#  ENGINE 2: REGIME DETECTION
# ─────────────────────────────────────────────

elif "Regime" in engine:
    st.markdown('<h2>Volatility Regime Detection</h2>', unsafe_allow_html=True)
    st.markdown('<span class="engine-badge">GAUSSIAN HMM · GARCH(1,1) · REGIME-CONDITIONAL FORECASTING</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    reg_ticker  = c1.selectbox("Asset", tickers, key="reg_ticker")
    n_regimes   = c2.selectbox("Number of Regimes", [2, 3, 4], index=1)
    n_iter_hmm  = c3.slider("HMM Iterations", 50, 500, 200, 50)

    run_btn = st.button("▶  Detect Regimes", use_container_width=False)

    if run_btn:
        with st.spinner("Fitting HMM + GARCH models..."):
            r_series = returns[reg_ticker].dropna()

            # HMM
            detector  = HMMRegimeDetector(n_components=n_regimes, n_iter=n_iter_hmm)
            reg_model = detector.fit(r_series)

            # GARCH per regime
            garch_models = RegimeGARCH()
            garch_results = garch_models.fit_all(r_series, reg_model)
            garch_full    = garch_models.fit_unconditional(r_series)

        # Regime stats table
        st.markdown('<div class="section-header">Regime Statistics</div>', unsafe_allow_html=True)
        st.dataframe(
            reg_model.regime_stats.style.background_gradient(
                subset=["Ann Vol"], cmap="RdYlGn_r"
            ).format({"Ann Return": "{:.2f}%", "Ann Vol": "{:.2f}%",
                      "Sharpe": "{:.3f}", "Freq (%)": "{:.1f}%"}),
            use_container_width=True,
        )

        # Transition matrix
        st.markdown('<div class="section-header">Regime Transition Matrix</div>', unsafe_allow_html=True)
        trans_df = pd.DataFrame(
            reg_model.transition_matrix,
            index=[reg_model.regime_labels.get(i, f"R{i}") for i in range(n_regimes)],
            columns=[reg_model.regime_labels.get(i, f"R{i}") for i in range(n_regimes)],
        )
        fig_trans = go.Figure(go.Heatmap(
            z=trans_df.values, x=trans_df.columns, y=trans_df.index,
            colorscale=[[0, "#0d1117"], [1, ACCENT]],
            text=trans_df.round(3).values, texttemplate="%{text}",
            textfont=dict(size=12),
        ))
        fig_trans.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig_trans, use_container_width=True)

        # State sequence + GARCH vol
        regime_colors_map = {0: ACCENT, 1: AMBER, 2: RED, 3: PURPLE}
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
            subplot_titles=[f"{reg_ticker} Price + Regimes", "Posterior Regime Probabilities", "GARCH Conditional Volatility"],
            row_heights=[0.35, 0.3, 0.35])

        # Price with regime background
        price_series = prices[reg_ticker]
        price_align  = price_series.reindex(reg_model.states.index, method="ffill")
        fig.add_trace(go.Scatter(x=price_align.index, y=price_align,
            name=reg_ticker, line=dict(color="#e6edf3", width=1.5)), row=1, col=1)

        # Colour bands for regimes
        states_s = reg_model.states
        prev_state = states_s.iloc[0]
        band_start = states_s.index[0]
        for i in range(1, len(states_s)):
            if states_s.iloc[i] != prev_state or i == len(states_s) - 1:
                fig.add_vrect(
                    x0=band_start, x1=states_s.index[i],
                    fillcolor=regime_colors_map.get(int(prev_state), "#ffffff") + "25",
                    layer="below", line_width=0, row=1, col=1
                )
                band_start  = states_s.index[i]
                prev_state  = states_s.iloc[i]

        # Regime probabilities
        for k in range(n_regimes):
            col_name = f"P(regime_{k})"
            if col_name in reg_model.state_probs.columns:
                lbl = reg_model.regime_labels.get(k, f"R{k}")
                fig.add_trace(go.Scatter(
                    x=reg_model.state_probs.index,
                    y=reg_model.state_probs[col_name],
                    name=lbl, mode="lines",
                    line=dict(color=regime_colors_map.get(k, "#ffffff"), width=1.5),
                    fill="tonexty" if k > 0 else "tozeroy",
                    fillcolor=regime_colors_map.get(k, "#ffffff") + "30",
                ), row=2, col=1)

        # GARCH vol per regime
        for k, g_res in garch_results.items():
            fig.add_trace(go.Scatter(
                x=g_res.conditional_vol.index,
                y=g_res.conditional_vol * np.sqrt(252) * 100,
                name=f"GARCH {g_res.regime_label}",
                line=dict(color=regime_colors_map.get(k, AMBER), width=1, dash="dot"),
            ), row=3, col=1)

        # Full GARCH vol
        fig.add_trace(go.Scatter(
            x=garch_full.conditional_vol.index,
            y=garch_full.conditional_vol * np.sqrt(252) * 100,
            name="GARCH (full)", line=dict(color=ACCENT, width=1.5),
        ), row=3, col=1)

        fig.update_layout(**PLOT_LAYOUT, height=750)
        for i in range(1, 4):
            fig.update_xaxes(gridcolor="#161b22", row=i, col=1)
            fig.update_yaxes(gridcolor="#161b22", row=i, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # GARCH parameter table
        if garch_results:
            st.markdown('<div class="section-header">GARCH(1,1) Parameters by Regime</div>', unsafe_allow_html=True)
            garch_table = []
            for k, g in garch_results.items():
                garch_table.append({
                    "Regime":       g.regime_label,
                    "ω":            g.omega,
                    "α (ARCH)":     g.alpha,
                    "β (GARCH)":    g.beta,
                    "Persistence":  g.persistence,
                    "Uncond. Ann. Vol (%)": g.unconditional_vol,
                    "1d Forecast (%)":  g.vol_forecast_1d,
                    "5d Forecast (%)":  g.vol_forecast_5d,
                    "Obs":          g.n_obs,
                })
            st.dataframe(pd.DataFrame(garch_table), use_container_width=True)

    else:
        st.info("Configure parameters and click **Detect Regimes** to begin.")


# ─────────────────────────────────────────────
#  ENGINE 3: PORTFOLIO
# ─────────────────────────────────────────────

elif "Portfolio" in engine:
    st.markdown('<h2>Portfolio Construction</h2>', unsafe_allow_html=True)
    st.markdown('<span class="engine-badge">BLACK-LITTERMAN · CVAR OPTIMISATION · EFFICIENT FRONTIER</span>', unsafe_allow_html=True)

    tabs = st.tabs(["Black-Litterman", "CVaR Optimisation", "Efficient Frontier"])

    with tabs[0]:
        st.markdown('<div class="section-header">Black-Litterman Model</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        tau_val = c1.slider("τ (prior uncertainty)", 0.01, 0.20, 0.05, 0.01)
        lam_val = c2.slider("λ (risk aversion)", 1.0, 5.0, 2.5, 0.1)

        st.markdown("**Investor Views**")
        n_views = st.number_input("Number of views", 0, 5, 1)
        views   = []

        for i in range(int(n_views)):
            vc1, vc2, vc3 = st.columns(3)
            view_asset = vc1.selectbox(f"View {i+1} asset", tickers, key=f"va_{i}")
            view_ret   = vc2.number_input(f"Expected return (%)", -30.0, 50.0, 10.0, 1.0, key=f"vr_{i}") / 100
            view_conf  = vc3.slider(f"Confidence", 0.1, 0.9, 0.5, 0.05, key=f"vc_{i}")
            views.append(BLView(
                assets=[view_asset], weights=[1.0],
                expected_return=view_ret,
                confidence=view_conf,
            ))

        if st.button("▶  Run Black-Litterman", key="run_bl"):
            with st.spinner("Computing BL posterior..."):
                port_assets = tickers[:8]
                port_rets   = returns[port_assets].dropna()
                bl          = BlackLitterman(tau=tau_val, risk_aversion=lam_val)
                bl_result   = bl.fit(port_rets, views)

            # Expected returns comparison
            er_df = pd.DataFrame({
                "Prior Π":     bl_result.prior_returns,
                "Posterior μ": bl_result.posterior_returns,
            }) * 100

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=er_df.index, y=er_df["Prior Π"],
                name="Equilibrium Π", marker_color=BLUE + "cc",
            ))
            fig.add_trace(go.Bar(
                x=er_df.index, y=er_df["Posterior μ"],
                name="BL Posterior μ", marker_color=ACCENT + "cc",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=350,
                title="Expected Returns: Prior vs Posterior (%)",
                barmode="group")
            st.plotly_chart(fig, use_container_width=True)

            # Weights
            fig2 = go.Figure(go.Bar(
                x=bl_result.bl_weights.index,
                y=bl_result.bl_weights.values * 100,
                marker_color=ACCENT,
                text=(bl_result.bl_weights * 100).round(1),
                textposition="outside",
            ))
            fig2.update_layout(**PLOT_LAYOUT, height=320,
                title="Optimal Portfolio Weights (%)")
            st.plotly_chart(fig2, use_container_width=True)

            # Performance
            metrics = portfolio_metrics(bl_result.bl_weights, port_rets)
            mc = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                val_f = str(v)
                colour = "green" if "Return" in k and v > 0 else ("red" if "Drawdown" in k else "blue")
                mc[i].markdown(f'<div class="metric-card {colour}"><div class="label">{k}</div><div class="value">{val_f}{"%" if k not in ["Sharpe Ratio","Calmar Ratio"] else ""}</div></div>', unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="section-header">CVaR Portfolio Optimisation</div>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        cvar_alpha   = c1.slider("Confidence α", 0.90, 0.99, 0.95, 0.01)
        min_wt       = c2.slider("Min weight", -0.10, 0.10, 0.0, 0.01)
        rf_rate      = c3.slider("Risk-free rate (%)", 0.0, 8.0, 5.0, 0.25) / 100

        if st.button("▶  Optimise CVaR Portfolio", key="run_cvar"):
            with st.spinner("Solving CVaR optimisation (CVXPY/CLARABEL)..."):
                port_assets = tickers[:8]
                port_rets   = returns[port_assets].dropna()
                optimiser   = CVaROptimiser(alpha=cvar_alpha, min_weight=min_wt, risk_free_rate=rf_rate)
                cvar_result = optimiser.optimise(port_rets)

            st.markdown(f'<span class="engine-badge">SOLVER: {cvar_result.optimization_status.upper()}</span>', unsafe_allow_html=True)

            kc = st.columns(4)
            kc[0].markdown(f'<div class="metric-card green"><div class="label">Expected Return</div><div class="value">{cvar_result.expected_return*100:.2f}%</div></div>', unsafe_allow_html=True)
            kc[1].markdown(f'<div class="metric-card red"><div class="label">Portfolio CVaR</div><div class="value">{cvar_result.portfolio_cvar*100:.2f}%</div></div>', unsafe_allow_html=True)
            kc[2].markdown(f'<div class="metric-card blue"><div class="label">Sharpe Ratio</div><div class="value">{cvar_result.sharpe_ratio:.3f}</div></div>', unsafe_allow_html=True)
            kc[3].markdown(f'<div class="metric-card amber"><div class="label">Diversification</div><div class="value">{cvar_result.diversification:.3f}</div></div>', unsafe_allow_html=True)

            fig = go.Figure(go.Bar(
                x=cvar_result.weights.index,
                y=cvar_result.weights.values * 100,
                marker_color=[ACCENT if w >= 0 else RED for w in cvar_result.weights],
                text=(cvar_result.weights * 100).round(1),
                textposition="outside",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=350, title="CVaR-Optimal Weights (%)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown('<div class="section-header">CVaR Efficient Frontier</div>', unsafe_allow_html=True)
        n_pts = st.slider("Frontier points", 10, 40, 20)
        if st.button("▶  Compute Frontier", key="run_frontier"):
            with st.spinner("Sweeping return targets..."):
                port_assets = tickers[:8]
                port_rets   = returns[port_assets].dropna()
                opt         = CVaROptimiser(alpha=0.95)
                frontier    = opt.efficient_frontier(port_rets, n_points=n_pts)
            frontier_clean = frontier.dropna(subset=["cvar"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frontier_clean["cvar"] * 100,
                y=frontier_clean["expected_return"] * 100,
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
                marker=dict(color=frontier_clean["sharpe"],
                            colorscale="RdYlGn", size=8,
                            colorbar=dict(title="Sharpe")),
                name="Efficient Frontier",
            ))
            fig.update_layout(**PLOT_LAYOUT, height=420,
                xaxis_title="CVaR (%)", yaxis_title="Expected Return (%)",
                title="CVaR Efficient Frontier")
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
#  ENGINE 4: EXECUTION
# ─────────────────────────────────────────────

elif "Execution" in engine:
    st.markdown('<h2>Optimal Execution</h2>', unsafe_allow_html=True)
    st.markdown('<span class="engine-badge">ALMGREN-CHRISS 2000 · IMPLEMENTATION SHORTFALL · EXECUTION FRONTIER</span>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    exec_ticker  = c1.selectbox("Asset", tickers, key="exec_ticker")
    exec_side    = c2.selectbox("Direction", ["Sell", "Buy"])

    c3, c4, c5  = st.columns(3)
    shares       = c3.number_input("Shares", 1_000, 10_000_000, 100_000, 10_000)
    T_days       = c4.slider("Horizon (days)", 1, 30, 5)
    n_steps      = c5.slider("Intervals", 5, 100, 20)

    c6, c7       = st.columns(2)
    risk_lam_exp = c6.slider("Risk aversion (log10 λ)", -8, -3, -6)
    risk_lam     = 10 ** risk_lam_exp

    if exec_side == "Buy":
        shares = -shares

    if st.button("▶  Compute Optimal Schedule", key="run_exec"):
        with st.spinner("Solving Almgren-Chriss trajectory..."):
            ohlcv   = None
            try:
                from alphaforge.data.loader import fetch_ohlcv
                ohlcv = fetch_ohlcv(exec_ticker, start="2022-01-01")
            except Exception:
                pass

            if ohlcv is not None and len(ohlcv) > 30:
                mkt_params = estimate_market_params(ohlcv)
            else:
                mkt_params = MarketParams(
                    sigma=0.02, eta=1e-6, gamma=5e-7,
                    adv=5_000_000, spread=0.001, price=100.0,
                )

            ac     = AlmgrenChriss(mkt_params)
            result = ac.optimal_trajectory(abs(shares), T_days, n_steps, risk_lam)

            # Frontier
            frontier_df = ac.efficient_frontier(abs(shares), T_days, n_steps)

        # Cost metrics
        st.markdown('<div class="section-header">Execution Cost Breakdown</div>', unsafe_allow_html=True)
        kc = st.columns(5)
        kc[0].markdown(f'<div class="metric-card red"><div class="label">Total IS (bps)</div><div class="value">{result.implementation_shortfall:.1f}</div></div>', unsafe_allow_html=True)
        kc[1].markdown(f'<div class="metric-card red"><div class="label">Total Cost ($)</div><div class="value">${result.total_cost:,.0f}</div></div>', unsafe_allow_html=True)
        kc[2].markdown(f'<div class="metric-card amber"><div class="label">Temp. Impact ($)</div><div class="value">${result.temporary_cost:,.0f}</div></div>', unsafe_allow_html=True)
        kc[3].markdown(f'<div class="metric-card amber"><div class="label">Perm. Impact ($)</div><div class="value">${result.permanent_cost:,.0f}</div></div>', unsafe_allow_html=True)
        kc[4].markdown(f'<div class="metric-card blue"><div class="label">vs TWAP</div><div class="value">-{max(0,(result.twap_cost-result.total_cost)/result.twap_cost*100):.1f}%</div></div>', unsafe_allow_html=True)

        # Trajectory chart
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=["Inventory Schedule", "Trade Rate (shares/day)", "Participation Rate (%)", "Execution Frontier"])

        # Inventory
        for traj_y, traj_name, color in [
            (result.inventory, "AC Optimal", ACCENT),
            (np.linspace(abs(shares), 0, n_steps+1), "TWAP", BLUE),
        ]:
            fig.add_trace(go.Scatter(
                x=result.times, y=traj_y,
                name=traj_name, line=dict(color=color, width=2)),
                row=1, col=1)

        # Trade rate
        fig.add_trace(go.Bar(
            x=result.times[1:], y=np.abs(result.trade_rate),
            name="Shares/day", marker_color=ACCENT + "cc"),
            row=1, col=2)

        # Participation
        sched = participation_rate_schedule(result, mkt_params.adv)
        fig.add_trace(go.Scatter(
            x=sched["time_day"], y=sched["participation"],
            name="Participation %", line=dict(color=AMBER, width=2),
            fill="tozeroy", fillcolor=AMBER + "20"),
            row=2, col=1)
        fig.add_hline(y=20, line_color=RED, line_dash="dash",
                     line_width=1, annotation_text="20% cap", row=2, col=1)

        # Frontier
        fig.add_trace(go.Scatter(
            x=frontier_df["std_cost"] / 1000,
            y=frontier_df["E_cost"]  / 1000,
            mode="lines", name="E[C] vs σ(C)",
            line=dict(color=ACCENT, width=2)),
            row=2, col=2)
        fig.add_trace(go.Scatter(
            x=[np.sqrt(max(result.execution_variance, 0)) / 1000],
            y=[result.total_cost / 1000],
            mode="markers", name="Selected",
            marker=dict(color=RED, size=10, symbol="star")),
            row=2, col=2)

        fig.update_layout(**PLOT_LAYOUT, height=620)
        for r_ in range(1, 3):
            for c_ in range(1, 3):
                fig.update_xaxes(gridcolor="#161b22", row=r_, col=c_)
                fig.update_yaxes(gridcolor="#161b22", row=r_, col=c_)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Configure trade parameters and click **Compute Optimal Schedule** to begin.")


# ─────────────────────────────────────────────
#  ENGINE 5: RISK ENGINE
# ─────────────────────────────────────────────

elif "Risk Engine" in engine:
    st.markdown('<h2>Risk Engine</h2>', unsafe_allow_html=True)
    st.markdown('<span class="engine-badge">VAR · EXPECTED SHORTFALL · MONTE CARLO · STRESS TESTING · KUPIEC</span>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    risk_tickers_sel = c1.multiselect("Portfolio Assets", tickers, default=tickers[:4])
    risk_alpha       = c2.selectbox("Confidence Level", [0.95, 0.99], index=1)
    mc_sims          = c3.select_slider("MC Simulations", [5000, 10000, 25000, 50000], value=25000)

    if len(risk_tickers_sel) < 2:
        st.warning("Select at least 2 assets for portfolio risk analysis.")
        st.stop()

    if st.button("▶  Run Full Risk Analysis", key="run_risk"):
        with st.spinner(f"Running VaR, ES, Monte Carlo ({mc_sims:,} sims), stress tests..."):
            port_rets = returns[risk_tickers_sel].dropna()
            # Equal weight portfolio
            weights   = pd.Series(
                np.ones(len(risk_tickers_sel)) / len(risk_tickers_sel),
                index=risk_tickers_sel,
            )
            port_pnl  = (port_rets @ weights).dropna()
            report    = build_risk_report(weights, port_rets, mc_sims=mc_sims)

            # Monte Carlo for distribution plot
            mc_result = monte_carlo_var_es(weights.values, port_rets,
                                           alpha=risk_alpha, horizon=1, n_simulations=mc_sims)

        # Summary metrics
        st.markdown('<div class="section-header">Value at Risk Summary</div>', unsafe_allow_html=True)
        rc = st.columns(6)
        risk_metrics = [
            ("Hist VaR 99% 1d", f"{report.hist_var_99}%", "red"),
            ("Param VaR 99% 1d", f"{report.param_var_99}%", "red"),
            ("MC VaR 99% 1d",    f"{report.mc_var_99}%",    "red"),
            ("ES 99% 1d",       f"{report.es_99_1d}%",      "red"),
            ("VaR 99% 10d",     f"{report.var_99_10d}%",    "amber"),
            ("ES 99% 10d",      f"{report.es_99_10d}%",     "amber"),
        ]
        for i, (lbl, val, col) in enumerate(risk_metrics):
            rc[i].markdown(f'<div class="metric-card {col}"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Distribution Statistics</div>', unsafe_allow_html=True)
        dc = st.columns(5)
        dist_data = [
            ("Ann Return", f"{report.mean_return:.2f}%",  "green"),
            ("Ann Vol",    f"{report.volatility:.2f}%",   "blue"),
            ("Skewness",   f"{report.skewness:.4f}",      "blue"),
            ("Excess Kurt", f"{report.kurtosis:.4f}",     "amber"),
            ("JB p-value", f"{report.jarque_bera_p:.4f}", "amber" if report.jarque_bera_p < 0.05 else "green"),
        ]
        for i, (lbl, val, col) in enumerate(dist_data):
            dc[i].markdown(f'<div class="metric-card {col}"><div class="label">{lbl}</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

        # Main risk charts
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=[
                "P&L Distribution + VaR / ES",
                "Rolling Volatility (21d)",
                "Monte Carlo P&L Distribution",
                "Rolling VaR (252d window)"
            ])

        # Historical distribution
        port_pnl_pct = port_pnl * 100
        fig.add_trace(go.Histogram(
            x=port_pnl_pct, nbinsx=100, name="Daily P&L",
            marker_color=BLUE + "aa", histnorm="probability density"),
            row=1, col=1)

        var_line = -report.hist_var_99
        es_line  = -report.es_99_1d
        fig.add_vline(x=var_line, line_color=AMBER, line_dash="dash",
                     annotation_text=f"VaR {var_line:.2f}%", row=1, col=1)
        fig.add_vline(x=es_line, line_color=RED, line_dash="dash",
                     annotation_text=f"ES {es_line:.2f}%", row=1, col=1)

        # Rolling vol
        fig.add_trace(go.Scatter(
            x=report.rolling_vol.index, y=report.rolling_vol * 100,
            name="21d Vol", line=dict(color=ACCENT, width=1.5),
            fill="tozeroy", fillcolor=ACCENT + "20"),
            row=1, col=2)

        # MC distribution
        mc_pnl_pct = mc_result["pnl_distribution"] * 100
        fig.add_trace(go.Histogram(
            x=mc_pnl_pct, nbinsx=100, name="MC P&L",
            marker_color=PURPLE + "aa", histnorm="probability density"),
            row=2, col=1)
        mc_var_line = -mc_result["VaR"] * 100
        fig.add_vline(x=mc_var_line, line_color=RED, line_dash="dash",
                     annotation_text=f"MC VaR {mc_var_line:.2f}%", row=2, col=1)

        # Rolling VaR
        rv_clean = report.rolling_var.dropna() * 100
        fig.add_trace(go.Scatter(
            x=rv_clean.index, y=rv_clean,
            name="252d Rolling VaR", line=dict(color=RED, width=1.5),
            fill="tozeroy", fillcolor=RED + "15"),
            row=2, col=2)

        fig.update_layout(**PLOT_LAYOUT, height=650, showlegend=False)
        for r_ in range(1, 3):
            for c_ in range(1, 3):
                fig.update_xaxes(gridcolor="#161b22", row=r_, col=c_)
                fig.update_yaxes(gridcolor="#161b22", row=r_, col=c_)
        st.plotly_chart(fig, use_container_width=True)

        # Stress tests
        st.markdown('<div class="section-header">Historical Stress Tests</div>', unsafe_allow_html=True)
        stress_df = report.stress_results[["Scenario", "Description", "Portfolio P&L (%)"]].copy()
        stress_df = stress_df.sort_values("Portfolio P&L (%)")

        fig_stress = go.Figure(go.Bar(
            x=stress_df["Portfolio P&L (%)"],
            y=stress_df["Scenario"],
            orientation="h",
            marker_color=[RED if v < 0 else ACCENT for v in stress_df["Portfolio P&L (%)"]],
            text=stress_df["Portfolio P&L (%)"].apply(lambda x: f"{x:+.2f}%"),
            textposition="outside",
        ))
        fig_stress.update_layout(**PLOT_LAYOUT, height=360,
            title="Portfolio P&L Under Stress Scenarios",
            xaxis_title="P&L (%)", yaxis_title="")
        st.plotly_chart(fig_stress, use_container_width=True)

        # Full stress table
        with st.expander("Full Stress Test Table"):
            cols_show = ["Scenario", "Description", "Portfolio P&L (%)"]
            st.dataframe(report.stress_results[cols_show].style.background_gradient(
                subset=["Portfolio P&L (%)"], cmap="RdYlGn"
            ), use_container_width=True)

    else:
        st.info("Select assets and click **Run Full Risk Analysis** to begin.")
