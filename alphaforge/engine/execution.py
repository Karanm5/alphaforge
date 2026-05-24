"""
AlphaForge Engine 4: Almgren-Chriss Optimal Execution
──────────────────────────────────────────────────────
Optimal trajectory minimising expected implementation shortfall + variance.

Mathematics:
  Inventory:    x_j  = initial shares remaining at step j
  Trade rate:   v_j  = shares traded per unit time at step j
  Trajectory:   x_{j+1} = x_j - v_j τ

  Cost model:
    Permanent impact:   g(v) = γ v          (price shift persists)
    Temporary impact:   h(v) = η v           (price recovers)
    (Almgren 2003 power-law: h(v) = η |v|^(1/2) v)

  Objective:  E[C] + λ Var[C]
    E[C]   = permanent + temporary costs
    Var[C] = σ² τ Σ x_j²

  Optimal trajectory (Almgren & Chriss 2000):
    x_j = X * sinh(κ(T-t_j)) / sinh(κT)
    where κ² = (λσ²)/(η̃)  and η̃ = η - γτ/2

  Efficient frontier parameterised by risk-aversion λ.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Market Impact Model
# ─────────────────────────────────────────────

@dataclass
class MarketParams:
    """
    Market microstructure parameters for Almgren-Chriss.
    All parameters refer to daily frequency unless stated.
    """
    sigma:      float   # daily return volatility (e.g. 0.02 = 2%)
    eta:        float   # temporary impact coefficient  (linear: price ∝ η*v/V)
    gamma:      float   # permanent impact coefficient  (linear: price ∝ γ*v/V)
    adv:        float   # average daily volume (shares)
    spread:     float   # bid-ask spread as fraction of price (e.g. 0.001)
    price:      float   # current mid price


def estimate_market_params(
    ohlcv: pd.DataFrame,
    pct_adv: float = 0.001,  # conservative: 0.1% of ADV
) -> MarketParams:
    """
    Estimate market params from OHLCV data.
    Uses Garman-Klass vol estimate and average volume.
    """
    close  = ohlcv["Close"].values
    hi     = ohlcv["High"].values
    lo     = ohlcv["Low"].values
    op     = ohlcv["Open"].values
    vol    = ohlcv["Volume"].values

    # Daily returns
    ret    = np.diff(np.log(close))
    sigma  = ret.std()

    # ADV (recent 20 days)
    adv    = vol[-20:].mean()
    price  = close[-1]

    # Typical impact coefficients (calibrated to equity microstructure)
    # Temporary impact: η such that 1% of ADV moves price by ~5 bps
    # Permanent impact: γ ≈ η/2 (symmetric decay)
    eta   = 0.142 * sigma / (adv ** 0.5)   # square-root impact scaling
    gamma = eta * 0.5

    # Spread estimate from HL data
    hl_spread = np.mean((hi - lo) / close)
    spread    = max(hl_spread * 0.2, 0.0001)   # ~20% of HL as effective spread

    return MarketParams(
        sigma=sigma, eta=eta, gamma=gamma,
        adv=adv, spread=spread, price=price,
    )


# ─────────────────────────────────────────────
#  Almgren-Chriss Trajectory
# ─────────────────────────────────────────────

@dataclass
class ExecutionResult:
    # Input
    X:              float    # initial inventory (shares)
    T:              float    # total time horizon (days)
    n_steps:        int      # number of trading intervals
    risk_aversion:  float    # λ

    # Trajectory
    times:          np.ndarray   # t_j
    inventory:      np.ndarray   # x_j (shares remaining)
    trade_list:     np.ndarray   # n_j = x_j - x_{j+1} (shares per interval)
    trade_rate:     np.ndarray   # v_j = n_j / τ (shares per day)

    # Costs (in $)
    permanent_cost:  float
    temporary_cost:  float
    spread_cost:     float
    total_cost:      float
    implementation_shortfall: float   # total IS / (X * P) in bps

    # Risk
    execution_variance: float
    expected_shortfall: float   # E[C] + λ * Var[C]

    # Benchmarks
    twap_cost:  float
    vwap_cost:  float   # approximated


class AlmgrenChriss:
    """
    Almgren-Chriss (2000) optimal liquidation model.
    Finds the trading trajectory that minimises E[C] + λ Var[C].
    """

    def __init__(self, market: MarketParams):
        self.mkt = market

    def optimal_trajectory(
        self,
        X: float,              # shares to liquidate (positive = sell, negative = buy)
        T: float,              # total time in days
        n: int = 20,           # number of trading intervals
        risk_aversion: float = 1e-6,  # λ (higher = more risk-averse, faster execution)
    ) -> ExecutionResult:
        """
        Compute Almgren-Chriss optimal trajectory.
        """
        mkt   = self.mkt
        tau   = T / n            # interval length (days)
        sigma = mkt.sigma
        eta   = mkt.eta
        gamma = mkt.gamma
        P     = mkt.price

        # Effective temporary impact coefficient η̃ = η - γτ/2
        eta_tilde = eta - 0.5 * gamma * tau

        # κ = sqrt(λσ²/η̃)
        kappa_sq = (risk_aversion * sigma**2) / eta_tilde if eta_tilde > 0 else 0
        kappa    = np.sqrt(max(kappa_sq, 0))

        # Time grid
        j_arr = np.arange(0, n + 1)
        t_arr = j_arr * tau

        # Optimal inventory schedule: x(t) = X * sinh(κ(T-t)) / sinh(κT)
        if kappa * T < 1e-8:
            # Risk-neutral limit: TWAP (linear)
            x_arr = X * (1 - j_arr / n)
        else:
            sinh_kT = np.sinh(kappa * T)
            if abs(sinh_kT) < 1e-10:
                x_arr = X * (1 - j_arr / n)
            else:
                x_arr = X * np.sinh(kappa * (T - t_arr)) / sinh_kT

        x_arr = np.clip(x_arr, 0, abs(X)) * np.sign(X)

        # Trade list: n_j = x_{j-1} - x_j
        n_arr = np.diff(x_arr)      # length n (trades at each step)
        v_arr = n_arr / tau          # trade rate (shares per day)

        # ─── Cost Calculation ───
        # Permanent impact cost
        perm_cost = gamma * np.sum(x_arr[:-1] * n_arr) * P

        # Temporary impact cost (linear model)
        temp_cost = eta * np.sum((n_arr / tau) ** 2) * tau * P

        # Spread cost
        spread_cost = 0.5 * mkt.spread * P * abs(X)

        total_cost = perm_cost + temp_cost + spread_cost

        # IS in bps
        notional = abs(X) * P
        is_bps   = (total_cost / notional) * 1e4 if notional > 0 else 0

        # Variance of execution cost
        exec_var = (sigma * P) ** 2 * tau * np.sum(x_arr[1:] ** 2)
        obj_val  = total_cost + risk_aversion * exec_var

        # TWAP benchmark (uniform execution)
        v_twap     = X / n / tau
        twap_temp  = eta * (v_twap ** 2) * T * P
        twap_perm  = gamma * np.mean(np.linspace(X, 0, n)) * (X / n) * P * n
        twap_cost  = twap_temp + abs(twap_perm) + spread_cost

        # VWAP approximation (proportional to sqrt of uniform)
        vwap_cost  = twap_cost * 0.85   # rough: VWAP ≈ 85% of TWAP cost

        return ExecutionResult(
            X=X, T=T, n_steps=n, risk_aversion=risk_aversion,
            times=t_arr,
            inventory=x_arr,
            trade_list=n_arr,
            trade_rate=v_arr,
            permanent_cost=round(perm_cost, 2),
            temporary_cost=round(temp_cost, 2),
            spread_cost=round(spread_cost, 2),
            total_cost=round(total_cost, 2),
            implementation_shortfall=round(is_bps, 2),
            execution_variance=round(exec_var, 2),
            expected_shortfall=round(obj_val, 2),
            twap_cost=round(twap_cost, 2),
            vwap_cost=round(vwap_cost, 2),
        )

    def efficient_frontier(
        self,
        X: float,
        T: float,
        n: int = 20,
        n_lambda: int = 50,
    ) -> pd.DataFrame:
        """
        Compute execution efficient frontier across risk-aversion levels.
        Trace the E[C] vs sqrt(Var[C]) curve.
        """
        lambdas = np.logspace(-8, -3, n_lambda)
        rows    = []
        for lam in lambdas:
            res = self.optimal_trajectory(X, T, n, risk_aversion=lam)
            rows.append({
                "lambda":    lam,
                "E_cost":    res.total_cost,
                "std_cost":  np.sqrt(max(res.execution_variance, 0)),
                "IS_bps":    res.implementation_shortfall,
                "twap_cost": res.twap_cost,
            })
        return pd.DataFrame(rows)

    def sensitivity_analysis(
        self,
        X: float,
        T: float,
        n: int = 20,
        risk_aversion: float = 1e-6,
    ) -> Dict:
        """
        Sensitivity of IS to ±20% changes in each market param.
        """
        base = self.optimal_trajectory(X, T, n, risk_aversion)
        base_is = base.implementation_shortfall
        results = {"base_IS_bps": base_is}

        for param, factor in [("sigma", 1.2), ("eta", 1.2), ("gamma", 1.2)]:
            original = getattr(self.mkt, param)
            setattr(self.mkt, param, original * factor)
            shocked  = self.optimal_trajectory(X, T, n, risk_aversion)
            results[f"IS_+20%_{param}"] = shocked.implementation_shortfall
            setattr(self.mkt, param, original * 0.8)
            shocked2 = self.optimal_trajectory(X, T, n, risk_aversion)
            results[f"IS_-20%_{param}"] = shocked2.implementation_shortfall
            setattr(self.mkt, param, original)

        return results


# ─────────────────────────────────────────────
#  Participation Rate Model
# ─────────────────────────────────────────────

def participation_rate_schedule(
    result: ExecutionResult,
    adv: float,
) -> pd.DataFrame:
    """
    Express trade schedule as participation rate (% of ADV per interval).
    """
    tau = result.T / result.n_steps
    trades   = np.abs(result.trade_list)
    interval_vol = adv * tau   # shares traded by mkt in interval

    df = pd.DataFrame({
        "interval":       np.arange(1, result.n_steps + 1),
        "time_day":       result.times[1:],
        "shares_traded":  trades,
        "inventory":      result.inventory[1:],
        "pct_completed":  (result.X - result.inventory[1:]) / result.X * 100,
        "participation":  trades / interval_vol * 100,
    })
    return df
