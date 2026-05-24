"""
AlphaForge Engine 1: Statistical Arbitrage
─────────────────────────────────────────────
Kalman Filter dynamic hedge ratio + Ornstein-Uhlenbeck spread modelling.

Mathematics:
  Kalman Filter state equation:  β_t = β_{t-1} + w_t,   w_t ~ N(0, Q)
  Kalman Filter obs equation:    y_t = x_t β_t + v_t,   v_t ~ N(0, R)

  OU process:  dS_t = θ(μ - S_t)dt + σ dW_t
  Half-life:   τ = ln(2) / θ
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Ornstein-Uhlenbeck Parameter Estimation
# ─────────────────────────────────────────────

@dataclass
class OUParams:
    theta: float   # mean-reversion speed
    mu: float      # long-run mean
    sigma: float   # diffusion coefficient
    half_life: float  # ln(2)/theta in days
    sigma_eq: float   # equilibrium std = sigma / sqrt(2*theta)

    def __str__(self):
        return (
            f"OU Parameters:\n"
            f"  θ (mean-reversion speed): {self.theta:.4f}\n"
            f"  μ (long-run mean):        {self.mu:.4f}\n"
            f"  σ (diffusion):            {self.sigma:.4f}\n"
            f"  Half-life:                {self.half_life:.1f} days\n"
            f"  σ_eq (equil. std):        {self.sigma_eq:.4f}"
        )


def fit_ou(spread: pd.Series) -> OUParams:
    """
    Estimate OU parameters via discrete-time OLS (Vasicek approach).
    Regress: S_{t+1} - S_t = a + b*S_t + ε
    => θ = -log(1 + b),  μ = -a/b,  σ from residuals
    """
    s = spread.dropna().values
    s_lag = s[:-1]
    s_now = s[1:]

    # OLS regression on discrete increments
    X = np.column_stack([np.ones(len(s_lag)), s_lag])
    beta, residuals, _, _ = np.linalg.lstsq(X, s_now, rcond=None)
    a, b = beta[0], beta[1]

    # Recover continuous-time parameters
    # b = e^{-theta*dt} - 1  (dt=1 day)
    theta = max(-np.log(1 + b), 1e-6)
    mu    = -a / b if abs(b) > 1e-10 else s.mean()

    # Residual std -> sigma
    resid = s_now - (a + b * s_lag)
    sigma_disc = np.std(resid, ddof=2)
    sigma = sigma_disc * np.sqrt(2 * theta / (1 - np.exp(-2 * theta)))

    half_life = np.log(2) / theta
    sigma_eq  = sigma / np.sqrt(2 * theta)

    return OUParams(theta=theta, mu=mu, sigma=sigma,
                    half_life=half_life, sigma_eq=sigma_eq)


# ─────────────────────────────────────────────
#  Kalman Filter Pairs Trading
# ─────────────────────────────────────────────

class KalmanPairsFilter:
    """
    Online Kalman Filter estimating dynamic hedge ratio β for pairs trading.

    State:  x_t = [β_t, α_t]  (hedge ratio + intercept)
    Observation model:  y_t = [price_x_t, 1] @ x_t + v_t
    Transition model:   x_t = x_{t-1} + w_t  (random walk)
    """

    def __init__(self, delta: float = 1e-4, vt: float = 1e-3):
        """
        delta: process noise magnitude (controls how fast β can change)
        vt:    observation noise variance
        """
        self.delta = delta
        self.vt    = vt
        self._reset()

    def _reset(self):
        self.n     = 2  # state dimension [β, α]
        self.R     = np.zeros((self.n, self.n))  # error covariance
        self.x     = np.zeros(self.n)            # state estimate
        self.Wt    = self.delta / (1 - self.delta) * np.eye(self.n)  # process noise
        self._initialised = False

    def step(self, price_y: float, price_x: float) -> Tuple[float, float, float]:
        """
        Process a single observation.
        Returns: (spread, hedge_ratio, prediction_error)
        """
        F = np.array([price_x, 1.0])  # observation matrix

        if not self._initialised:
            self.x = np.array([1.0, 0.0])
            self.R = np.eye(self.n)
            self._initialised = True
        else:
            # Predict
            self.R = self.R + self.Wt

        # Innovation (prediction error)
        y_hat = F @ self.x
        e     = price_y - y_hat

        # Innovation covariance
        S = F @ self.R @ F + self.vt

        # Kalman gain
        K = (self.R @ F) / S

        # Update state
        self.x = self.x + K * e
        self.R = (np.eye(self.n) - np.outer(K, F)) @ self.R

        hedge_ratio = self.x[0]
        spread      = price_y - hedge_ratio * price_x - self.x[1]

        return spread, hedge_ratio, e

    def run(self, price_y: pd.Series, price_x: pd.Series) -> pd.DataFrame:
        """
        Run filter over full series. Returns DataFrame with spread, hedge ratio, etc.
        """
        self._reset()
        index  = price_y.index
        n      = len(price_y)
        spreads      = np.zeros(n)
        hedge_ratios = np.zeros(n)
        errors       = np.zeros(n)

        for i, (y, x) in enumerate(zip(price_y.values, price_x.values)):
            s, hr, e = self.step(y, x)
            spreads[i]      = s
            hedge_ratios[i] = hr
            errors[i]       = e

        return pd.DataFrame({
            "spread":      spreads,
            "hedge_ratio": hedge_ratios,
            "pred_error":  errors,
        }, index=index)


# ─────────────────────────────────────────────
#  Signal Generation
# ─────────────────────────────────────────────

class StatArbSignal:
    """
    Full statistical arbitrage signal pipeline:
    1. Kalman filter → dynamic spread
    2. OU fit on spread → entry/exit thresholds
    3. Z-score normalised signals
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z:  float = 0.5,
        delta:   float = 1e-4,
        lookback: int  = 60,    # rolling window for z-score
    ):
        self.entry_z  = entry_z
        self.exit_z   = exit_z
        self.kf       = KalmanPairsFilter(delta=delta)
        self.lookback = lookback

    def fit(self, price_y: pd.Series, price_x: pd.Series) -> pd.DataFrame:
        """
        Compute spread, z-score, OU params, and trading signals.
        """
        kf_result = self.kf.run(price_y, price_x)
        spread     = kf_result["spread"]

        # Z-score over rolling window
        roll_mean = spread.rolling(self.lookback).mean()
        roll_std  = spread.rolling(self.lookback).std()
        z_score   = (spread - roll_mean) / roll_std

        # OU fit on most recent data
        ou = fit_ou(spread.dropna().iloc[-252:])

        # Signals: +1 = long spread (y - β*x), -1 = short spread
        signal = pd.Series(0.0, index=spread.index)
        position = 0
        for i in range(len(z_score)):
            z = z_score.iloc[i]
            if np.isnan(z):
                continue
            if position == 0:
                if z < -self.entry_z:
                    position = 1   # spread too low, go long
                elif z > self.entry_z:
                    position = -1  # spread too high, go short
            elif position == 1 and z > -self.exit_z:
                position = 0
            elif position == -1 and z < self.exit_z:
                position = 0
            signal.iloc[i] = position

        # Pair returns
        log_ret_y = np.log(price_y / price_y.shift(1))
        log_ret_x = np.log(price_x / price_x.shift(1))
        pair_pnl  = signal.shift(1) * (log_ret_y - kf_result["hedge_ratio"].shift(1) * log_ret_x)

        results = pd.DataFrame({
            "price_y":     price_y,
            "price_x":     price_x,
            "hedge_ratio": kf_result["hedge_ratio"],
            "spread":      spread,
            "z_score":     z_score,
            "signal":      signal,
            "pair_pnl":    pair_pnl.fillna(0),
        })

        results.attrs["ou_params"] = ou
        return results

    def performance_summary(self, results: pd.DataFrame) -> Dict:
        """Compute strategy performance metrics."""
        pnl = results["pair_pnl"].dropna()
        ann = 252
        cum_ret    = (1 + pnl).cumprod()
        total_ret  = cum_ret.iloc[-1] - 1
        ann_ret    = pnl.mean() * ann
        ann_vol    = pnl.std() * np.sqrt(ann)
        sharpe     = ann_ret / ann_vol if ann_vol > 0 else 0
        peak       = cum_ret.expanding().max()
        drawdowns  = (cum_ret - peak) / peak
        max_dd     = drawdowns.min()
        n_trades   = (results["signal"].diff().abs() > 0).sum()
        win_rate   = (pnl[pnl != 0] > 0).mean()

        return {
            "Total Return":   f"{total_ret:.2%}",
            "Annual Return":  f"{ann_ret:.2%}",
            "Annual Vol":     f"{ann_vol:.2%}",
            "Sharpe Ratio":   f"{sharpe:.3f}",
            "Max Drawdown":   f"{max_dd:.2%}",
            "Number Trades":  int(n_trades),
            "Win Rate":       f"{win_rate:.2%}",
        }


# ─────────────────────────────────────────────
#  Cointegration Tests
# ─────────────────────────────────────────────

def engle_granger_test(price_y: pd.Series, price_x: pd.Series) -> Dict:
    """
    Engle-Granger two-step cointegration test.
    Step 1: OLS regression  y = α + β*x + ε
    Step 2: ADF test on residuals ε
    """
    from statsmodels.tsa.stattools import adfuller

    y = price_y.dropna().values
    x = price_x.dropna().values
    min_len = min(len(y), len(x))
    y, x = y[-min_len:], x[-min_len:]

    # OLS
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta

    # ADF on residuals
    adf_stat, p_val, _, _, crit, _ = adfuller(residuals, maxlags=1, regression="nc")

    return {
        "ADF Statistic": round(adf_stat, 4),
        "p-value":       round(p_val, 4),
        "Cointegrated":  p_val < 0.05,
        "Hedge Ratio β": round(beta[1], 4),
        "Intercept α":   round(beta[0], 4),
        "Crit 1%":       round(crit["1%"], 4),
        "Crit 5%":       round(crit["5%"], 4),
        "Crit 10%":      round(crit["10%"], 4),
    }
