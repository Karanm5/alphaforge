# ⚡ AlphaForge

**Institutional-grade quantitative research platform** built in Python. Five self-contained engines covering the full quant workflow — from signal generation to execution optimisation and risk management.

---

## Engines

| # | Engine | Methods |
|---|--------|---------|
| 1 | **Statistical Arbitrage** | Kalman Filter dynamic hedge ratio, Ornstein-Uhlenbeck process, Engle-Granger cointegration, Z-score signal generation |
| 2 | **Volatility Regime Detection** | Gaussian HMM (2–4 regimes), GARCH(1,1) per regime, Viterbi decoding, regime-conditional volatility forecasting |
| 3 | **Portfolio Construction** | Black-Litterman model (He & Litterman 1999), CVaR optimisation via linear programming (Rockafellar & Uryasev 2000), CVaR efficient frontier |
| 4 | **Optimal Execution** | Almgren-Chriss (2000) trajectory optimisation, implementation shortfall minimisation, execution efficient frontier, TWAP/VWAP benchmarking |
| 5 | **Risk Engine** | Historical, parametric (Gaussian + Student-t), and Monte Carlo VaR/ES; 7 historical stress scenarios; Kupiec (1995) backtesting |

---

## Mathematics

### Engine 1 — Kalman Filter + Ornstein-Uhlenbeck

State equation: `β_t = β_{t-1} + w_t`, `w_t ~ N(0, Q)`  
Observation equation: `y_t = x_t β_t + v_t`, `v_t ~ N(0, R)`  

OU mean reversion: `dS_t = θ(μ - S_t)dt + σ dW_t`  
Half-life: `τ = ln(2) / θ`

### Engine 2 — HMM + GARCH

Emission: `P(r_t | s_t) = N(μ_k, σ_k²)`  
GARCH(1,1): `σ_t² = ω + α ε_{t-1}² + β σ_{t-1}²`  
Persistence: `α + β < 1`

### Engine 3 — Black-Litterman

Equilibrium returns: `Π = λ Σ w_mkt`  
Posterior: `E[r] = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]`  

CVaR optimisation (LP form):  
`min_{w,ζ} ζ + 1/((1-α)T) Σ max(-r_t'w - ζ, 0)`

### Engine 4 — Almgren-Chriss

Optimal trajectory: `x(t) = X · sinh(κ(T-t)) / sinh(κT)`  
where `κ² = λσ² / η̃` and `η̃ = η - γτ/2`

Objective: `min E[C] + λ Var[C]`

### Engine 5 — Risk

Parametric ES (Gaussian): `ES = -(μ - σ · φ(z_α)/(1-α))`  
Kupiec LR test: `LR = -2[T₀ ln(p₀) + T₁ ln(1-p₀) - T₀ ln(p̂) - T₁ ln(1-p̂)]`

---

## Quickstart

### Install

```bash
pip install numpy scipy pandas yfinance fredapi hmmlearn arch filterpy \
            cvxpy statsmodels scikit-learn streamlit plotly pyportfolioopt
```

### Run the dashboard

```bash
streamlit run alphaforge/app.py
```

Enter your [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) in the sidebar, select tickers, and load data.

### Programmatic usage

```python
from alphaforge.data.loader import fetch_returns
from alphaforge.engines.stat_arb import StatArbSignal
from alphaforge.engines.regime_detection import HMMRegimeDetector, RegimeGARCH
from alphaforge.engines.portfolio import BlackLitterman, CVaROptimiser, BLView
from alphaforge.engines.execution import AlmgrenChriss, MarketParams
from alphaforge.engines.risk_engine import build_risk_report

returns = fetch_returns(["SPY", "QQQ", "GLD", "TLT"], start="2018-01-01")

# Pairs trading signal
sa = StatArbSignal(entry_z=2.0, exit_z=0.5)
result = sa.fit(prices["SPY"], prices["QQQ"])

# Regime detection
model = HMMRegimeDetector(n_components=3).fit(returns["SPY"])

# Black-Litterman with a view
bl_result = BlackLitterman().fit(
    returns,
    views=[BLView(["SPY"], [1.0], expected_return=0.12, confidence=0.6)]
)

# Optimal execution
mkt  = MarketParams(sigma=0.018, eta=1.2e-6, gamma=6e-7, adv=8_000_000, spread=0.0005, price=430.0)
traj = AlmgrenChriss(mkt).optimal_trajectory(X=250_000, T=5, n=20, risk_aversion=1e-6)
print(f"Implementation shortfall: {traj.implementation_shortfall:.1f} bps")

# Risk report
import pandas as pd
weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=["SPY", "QQQ", "GLD", "TLT"])
report  = build_risk_report(weights, returns)
print(f"VaR 99% 1d: {report.var_99_1d}%  |  ES 99% 1d: {report.es_99_1d}%")
```

---

## Project Structure

```
alphaforge/
├── app.py                    # Streamlit dashboard
├── data/
│   └── loader.py             # yfinance + FRED data layer
└── engines/
    ├── stat_arb.py           # Kalman Filter + OU + pairs trading
    ├── regime_detection.py   # Gaussian HMM + GARCH per regime
    ├── portfolio.py          # Black-Litterman + CVaR optimisation
    ├── execution.py          # Almgren-Chriss optimal execution
    └── risk_engine.py        # VaR, ES, Monte Carlo, stress tests
```

---

## Stress Scenarios

The risk engine includes seven calibrated historical scenarios:

| Scenario | Equity Shock | Vol Shock |
|----------|-------------|-----------|
| 2000 Dot-com | -40% | +30% |
| 2008 GFC | -45% | +60% |
| 2010 Flash Crash | -9% | +40% |
| 2020 COVID | -34% | +80% |
| 2022 Rate Shock | -25% | +25% |
| Liquidity Crisis | -20% | +100% |
| Stagflation | -30% | +20% |

---

## Data Sources

- **Market data** — [yfinance](https://github.com/ranaroussi/yfinance) (equities, ETFs)
- **Macro data** — [FRED](https://fred.stlouisfed.org/) via `fredapi` (VIX, Fed Funds Rate, yield curve, CPI, HY spreads)

A free FRED API key is available at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html).

---

## References

- Almgren, R. & Chriss, N. (2000). *Optimal execution of portfolio transactions.* Journal of Risk.
- He, G. & Litterman, R. (1999). *The intuition behind Black-Litterman model portfolios.* Goldman Sachs.
- Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of conditional value-at-risk.* Journal of Risk.
- Kupiec, P. (1995). *Techniques for verifying the accuracy of risk measurement models.* Journal of Derivatives.
- Idzorek, T. (2005). *A step-by-step guide to the Black-Litterman model.* Zephyr Associates.

---

## Tech Stack

`Python 3.10+` · `NumPy` · `SciPy` · `pandas` · `CVXPY` · `hmmlearn` · `arch` · `filterpy` · `statsmodels` · `Streamlit` · `Plotly`
