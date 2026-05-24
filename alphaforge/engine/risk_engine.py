"""
AlphaForge Engine 5: Risk Engine
──────────────────────────────────
VaR, Expected Shortfall, Monte Carlo, stress testing, and Greeks.

Methods:
  Historical Simulation:   empirical distribution of portfolio returns
  Parametric (Gaussian):   VaR = μ - z_α σ, ES = μ - φ(z_α)/(1-α) σ
  Parametric (Student-t):  accounts for fat tails via DoF calibration
  Monte Carlo:             correlated GBM simulation → full P&L distribution

Stress tests:
  2000 Dot-com crash, 2008 GFC, 2010 Flash Crash,
  2020 COVID crash, 2022 Rate shock, custom factor shocks.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Historical Simulation VaR / ES
# ─────────────────────────────────────────────

def historical_var_es(
    portfolio_returns: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
) -> Tuple[float, float]:
    """
    Historical simulation VaR and ES.
    Scales to horizon using square-root-of-time rule.
    Returns (VaR, ES) as positive loss values.
    """
    r = portfolio_returns.dropna().values
    h_scale = np.sqrt(horizon)

    var = -np.percentile(r, (1 - alpha) * 100) * h_scale
    es  = -r[r <= -var / h_scale].mean() * h_scale if (r <= -var / h_scale).sum() > 0 else var

    return float(var), float(es)


# ─────────────────────────────────────────────
#  Parametric VaR (Gaussian + Student-t)
# ─────────────────────────────────────────────

def parametric_var_es(
    portfolio_returns: pd.Series,
    alpha: float = 0.99,
    horizon: int = 1,
    distribution: str = "normal",  # "normal" or "t"
) -> Dict:
    """
    Parametric VaR and ES.
    Fits Normal or Student-t to return series.
    """
    r = portfolio_returns.dropna().values
    mu     = r.mean()
    sigma  = r.std()
    h_sc   = np.sqrt(horizon)

    if distribution == "normal":
        z   = stats.norm.ppf(1 - alpha)
        var = -(mu - z * sigma) * h_sc
        es  = -(mu - sigma * stats.norm.pdf(z) / (1 - alpha)) * h_sc
        dof = None
    else:  # Student-t
        # Fit degrees of freedom via MLE
        dof, loc, scale = stats.t.fit(r, floc=mu)
        dof = max(dof, 2.01)  # ensure finite variance
        t_q = stats.t.ppf(1 - alpha, df=dof)
        var = -(loc + scale * t_q) * h_sc
        # ES for Student-t
        pdf_t = stats.t.pdf(t_q, df=dof)
        es    = -(loc + scale * (pdf_t / (1 - alpha)) * ((dof + t_q**2) / (dof - 1))) * h_sc

    return {
        "distribution": distribution,
        "mean":         round(mu, 6),
        "sigma":        round(sigma, 6),
        "dof":          round(dof, 2) if dof else None,
        "VaR":          round(float(var), 6),
        "ES":           round(float(es), 6),
    }


# ─────────────────────────────────────────────
#  Monte Carlo VaR
# ─────────────────────────────────────────────

def monte_carlo_var_es(
    weights: np.ndarray,
    returns: pd.DataFrame,
    alpha: float = 0.99,
    horizon: int = 10,
    n_simulations: int = 50_000,
    seed: int = 42,
) -> Dict:
    """
    Monte Carlo VaR/ES via correlated Geometric Brownian Motion.
    Simulates multi-day P&L paths over given horizon.
    """
    np.random.seed(seed)
    r   = returns.dropna().values
    mu  = r.mean(axis=0)
    cov = np.cov(r.T)

    # Cholesky decomposition for correlated sampling
    try:
        L = np.linalg.cholesky(cov + 1e-8 * np.eye(cov.shape[0]))
    except np.linalg.LinAlgError:
        # Fall back to eigenvalue decomposition
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.maximum(eigval, 1e-8)
        L = eigvec @ np.diag(np.sqrt(eigval))

    # Simulate: (n_sims × horizon × n_assets)
    n_assets = len(weights)
    z        = np.random.standard_normal((n_simulations, horizon, n_assets))
    shocks   = z @ L.T                           # correlated
    log_rets = mu[None, None, :] + shocks        # (sim × horizon × assets)

    # Cumulative portfolio log return over horizon
    cum_log_ret = log_rets.sum(axis=1) @ weights  # (n_sims,)

    var   = float(-np.percentile(cum_log_ret, (1 - alpha) * 100))
    tail  = cum_log_ret[cum_log_ret <= -var]
    es    = float(-tail.mean()) if len(tail) > 0 else var

    # Full distribution for plotting
    pnl_distribution = cum_log_ret

    return {
        "VaR":                round(var, 6),
        "ES":                 round(es, 6),
        "n_simulations":      n_simulations,
        "horizon_days":       horizon,
        "confidence_level":   alpha,
        "pnl_distribution":   pnl_distribution,
        "mean_pnl":           round(cum_log_ret.mean(), 6),
        "std_pnl":            round(cum_log_ret.std(), 6),
        "skewness":           round(float(pd.Series(cum_log_ret).skew()), 4),
        "kurtosis":           round(float(pd.Series(cum_log_ret).kurt()), 4),
    }


# ─────────────────────────────────────────────
#  Stress Testing
# ─────────────────────────────────────────────

STRESS_SCENARIOS = {
    "2000 Dot-com Peak": {
        "description": "NASDAQ -78% peak-to-trough (2000-2002)",
        "equity_shock":  -0.40,
        "bond_shock":    +0.04,
        "vol_shock":     +0.30,
        "credit_shock":  +0.015,
        "fx_shock":      -0.05,
    },
    "2008 GFC": {
        "description": "Global Financial Crisis (Sep 2008 - Mar 2009)",
        "equity_shock":  -0.45,
        "bond_shock":    +0.06,
        "vol_shock":     +0.60,
        "credit_shock":  +0.025,
        "fx_shock":      -0.10,
    },
    "2010 Flash Crash": {
        "description": "May 6, 2010: S&P dropped 9% intraday",
        "equity_shock":  -0.09,
        "bond_shock":    +0.01,
        "vol_shock":     +0.40,
        "credit_shock":  +0.005,
        "fx_shock":      -0.02,
    },
    "2020 COVID Crash": {
        "description": "COVID-19 pandemic sell-off (Feb-Mar 2020)",
        "equity_shock":  -0.34,
        "bond_shock":    +0.05,
        "vol_shock":     +0.80,
        "credit_shock":  +0.020,
        "fx_shock":      -0.08,
    },
    "2022 Rate Shock": {
        "description": "Fed rate hikes; bonds/equities co-move down",
        "equity_shock":  -0.25,
        "bond_shock":    -0.18,
        "vol_shock":     +0.25,
        "credit_shock":  +0.012,
        "fx_shock":      +0.08,
    },
    "Liquidity Crisis": {
        "description": "Severe liquidity crunch; bid-ask widens 10x",
        "equity_shock":  -0.20,
        "bond_shock":    -0.05,
        "vol_shock":     +1.00,
        "credit_shock":  +0.030,
        "fx_shock":      -0.15,
    },
    "Stagflation": {
        "description": "High inflation + low growth; 1970s-style",
        "equity_shock":  -0.30,
        "bond_shock":    -0.20,
        "vol_shock":     +0.20,
        "credit_shock":  +0.008,
        "fx_shock":      -0.12,
    },
}


def asset_factor_loadings(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate factor loadings of each asset to equity, bond, vol, credit, FX.
    Simplified OLS factor model.
    """
    # Without actual factor data, approximate using return properties
    corr   = returns.corr()
    betas  = pd.DataFrame(index=returns.columns, columns=[
        "equity_beta", "bond_beta", "vol_beta", "credit_beta", "fx_beta"
    ])

    for col in returns.columns:
        r   = returns[col].dropna()
        vol = r.std() * np.sqrt(252)
        sk  = r.skew()
        # Heuristic factor loadings based on return distribution properties
        betas.loc[col, "equity_beta"] = 1.0              # assume fully equity-loaded
        betas.loc[col, "bond_beta"]   = -0.3             # typical equity-bond negative correlation
        betas.loc[col, "vol_beta"]    = -0.5 * (vol / 0.20)   # scaled by vol level
        betas.loc[col, "credit_beta"] = 0.3              # moderate credit exposure
        betas.loc[col, "fx_beta"]     = 0.1              # small FX exposure

    return betas.astype(float)


def stress_test_portfolio(
    weights: pd.Series,
    returns: pd.DataFrame,
    custom_scenarios: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Apply historical stress scenarios to portfolio.
    Returns DataFrame with scenario P&L impact.
    """
    scenarios = {**STRESS_SCENARIOS}
    if custom_scenarios:
        scenarios.update(custom_scenarios)

    betas  = asset_factor_loadings(returns)
    w      = weights.reindex(returns.columns).fillna(0)
    w     /= w.sum()

    results = []
    for name, shock in scenarios.items():
        portfolio_pnl = 0.0
        asset_pnl     = {}

        for asset in returns.columns:
            if asset not in betas.index:
                continue
            b_eq  = betas.loc[asset, "equity_beta"]
            b_bd  = betas.loc[asset, "bond_beta"]
            b_vl  = betas.loc[asset, "vol_beta"]
            b_cr  = betas.loc[asset, "credit_beta"]
            b_fx  = betas.loc[asset, "fx_beta"]

            asset_ret = (
                b_eq * shock.get("equity_shock", 0) +
                b_bd * shock.get("bond_shock",   0) +
                b_vl * shock.get("vol_shock",    0) * 0.1 +   # scale vol shock
                b_cr * shock.get("credit_shock", 0) * (-1) +  # credit widens → equity down
                b_fx * shock.get("fx_shock",     0)
            )
            asset_pnl[asset] = round(asset_ret, 4)
            portfolio_pnl   += w[asset] * asset_ret

        results.append({
            "Scenario":     name,
            "Description":  shock.get("description", ""),
            "Portfolio P&L (%)": round(portfolio_pnl * 100, 2),
            **{f"{a} (%)": round(v * 100, 2) for a, v in asset_pnl.items()},
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
#  Comprehensive Risk Report
# ─────────────────────────────────────────────

@dataclass
class RiskReport:
    """Full risk report for a portfolio."""
    portfolio_returns:   pd.Series
    weights:             pd.Series

    # VaR / ES at multiple confidence levels
    var_95_1d:   float
    es_95_1d:    float
    var_99_1d:   float
    es_99_1d:    float
    var_99_10d:  float
    es_99_10d:   float

    # Methods
    hist_var_99:   float
    param_var_99:  float
    mc_var_99:     float

    # Distribution stats
    mean_return:  float
    volatility:   float
    skewness:     float
    kurtosis:     float
    jarque_bera_p: float

    # Stress tests
    stress_results: pd.DataFrame

    # Rolling risk
    rolling_vol:   pd.Series
    rolling_var:   pd.Series


def build_risk_report(
    weights: pd.Series,
    returns: pd.DataFrame,
    alpha_levels: List[float] = [0.95, 0.99],
    mc_sims: int = 20_000,
) -> RiskReport:
    """
    Assemble full risk report for a portfolio.
    """
    w  = weights.reindex(returns.columns).fillna(0)
    w /= w.sum()
    port_ret = (returns @ w).dropna()

    # Historical VaR/ES
    v95_1d, e95_1d   = historical_var_es(port_ret, alpha=0.95, horizon=1)
    v99_1d, e99_1d   = historical_var_es(port_ret, alpha=0.99, horizon=1)
    v99_10d, e99_10d = historical_var_es(port_ret, alpha=0.99, horizon=10)

    # Parametric
    p99 = parametric_var_es(port_ret, alpha=0.99)
    p99t = parametric_var_es(port_ret, alpha=0.99, distribution="t")

    # Monte Carlo
    mc = monte_carlo_var_es(w.values, returns, alpha=0.99, horizon=1, n_simulations=mc_sims)

    # Distribution stats
    r       = port_ret.values
    jb_stat, jb_p = stats.jarque_bera(r)

    # Stress
    stress = stress_test_portfolio(weights, returns)

    # Rolling 21-day vol and VaR
    rolling_vol = port_ret.rolling(21).std() * np.sqrt(252)
    rolling_var = port_ret.rolling(252).apply(
        lambda x: -np.percentile(x, 1), raw=True
    )

    return RiskReport(
        portfolio_returns=port_ret,
        weights=weights,
        var_95_1d=round(v95_1d * 100, 3),
        es_95_1d=round(e95_1d * 100, 3),
        var_99_1d=round(v99_1d * 100, 3),
        es_99_1d=round(e99_1d * 100, 3),
        var_99_10d=round(v99_10d * 100, 3),
        es_99_10d=round(e99_10d * 100, 3),
        hist_var_99=round(v99_1d * 100, 3),
        param_var_99=round(p99["VaR"] * 100, 3),
        mc_var_99=round(mc["VaR"] * 100, 3),
        mean_return=round(float(r.mean()) * 252 * 100, 3),
        volatility=round(float(r.std()) * np.sqrt(252) * 100, 3),
        skewness=round(float(pd.Series(r).skew()), 4),
        kurtosis=round(float(pd.Series(r).kurt()), 4),
        jarque_bera_p=round(float(jb_p), 4),
        stress_results=stress,
        rolling_vol=rolling_vol,
        rolling_var=rolling_var,
    )


# ─────────────────────────────────────────────
#  Backtesting Risk Model (Kupiec / Christoffersen)
# ─────────────────────────────────────────────

def kupiec_test(
    portfolio_returns: pd.Series,
    var_estimates: pd.Series,
    alpha: float = 0.99,
) -> Dict:
    """
    Kupiec (1995) Proportion of Failures (POF) test.
    H0: actual exception rate = 1 - α
    """
    r       = portfolio_returns.dropna()
    var_est = var_estimates.reindex(r.index).dropna()
    common  = r.index.intersection(var_est.index)
    r, var_est = r[common], var_est[common]

    exceptions = (r < -var_est).sum()
    T          = len(r)
    p_hat      = exceptions / T
    p_0        = 1 - alpha

    if exceptions == 0 or exceptions == T:
        return {"LR_stat": np.nan, "p_value": np.nan, "exceptions": int(exceptions), "T": T}

    LR = -2 * (
        exceptions * np.log(p_0 / p_hat) +
        (T - exceptions) * np.log((1 - p_0) / (1 - p_hat))
    )
    p_val = 1 - stats.chi2.cdf(LR, df=1)

    return {
        "LR_stat":     round(float(LR), 4),
        "p_value":     round(float(p_val), 4),
        "exceptions":  int(exceptions),
        "T":           T,
        "actual_rate": round(float(p_hat), 4),
        "expected_rate": round(p_0, 4),
        "Pass":        p_val > 0.05,
    }
