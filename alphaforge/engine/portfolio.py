"""
AlphaForge Engine 3: Portfolio Construction
─────────────────────────────────────────────
Black-Litterman model blends market equilibrium with investor views.
CVaR (Expected Shortfall) optimisation via CVXPY.

Mathematics (Black-Litterman):
  Equilibrium excess returns:  Π = λ Σ w_mkt
  Posterior expected returns:  E[r] = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}Π + P'Ω^{-1}Q]
  Posterior covariance:        M   = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}

CVaR Optimisation:
  min_{w,ζ}   ζ + (1/(1-α)T) Σ max(-r_t'w - ζ, 0)
  s.t.        1'w = 1,  w ≥ 0  (or allow shorts)
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import minimize
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Black-Litterman
# ─────────────────────────────────────────────

@dataclass
class BLView:
    """Single investor view for Black-Litterman."""
    assets:     List[str]        # asset names
    weights:    List[float]      # portfolio weights in view (sum to 1 or 0 for relative)
    expected_return: float       # view on this portfolio's return (annualised)
    confidence:      float       # view confidence ∈ (0, 1); higher = tighter Ω diagonal


@dataclass
class BLResult:
    tickers:           List[str]
    prior_returns:     pd.Series       # Π (equilibrium)
    posterior_returns: pd.Series       # E[r] (BL adjusted)
    posterior_cov:     pd.DataFrame    # M + Σ
    bl_weights:        pd.Series       # MV weights from posterior
    market_weights:    pd.Series


class BlackLitterman:
    """
    Full Black-Litterman model implementation.
    Reference: He & Litterman (1999), Idzorek (2005).
    """

    def __init__(self, tau: float = 0.05, risk_aversion: float = 2.5):
        """
        tau:           Scaling factor for prior uncertainty (typically 0.01-0.10)
        risk_aversion: Market risk-aversion coefficient λ
        """
        self.tau           = tau
        self.risk_aversion = risk_aversion

    def equilibrium_returns(
        self,
        cov: pd.DataFrame,
        market_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        Compute implied equilibrium excess returns Π = λ Σ w_mkt.
        If market_weights not provided, uses equal weights.
        """
        if market_weights is None:
            n = len(cov)
            market_weights = pd.Series(np.ones(n) / n, index=cov.index)
        else:
            market_weights = market_weights.reindex(cov.index).fillna(0)
            market_weights = market_weights / market_weights.sum()

        sigma  = cov.values
        w_mkt  = market_weights.values
        pi     = self.risk_aversion * sigma @ w_mkt
        return pd.Series(pi, index=cov.index, name="equilibrium_returns")

    def fit(
        self,
        returns: pd.DataFrame,
        views: List[BLView],
        market_weights: Optional[pd.Series] = None,
    ) -> BLResult:
        """
        Run full BL model.
        returns: historical returns DataFrame (T × N)
        views:   list of BLView objects
        """
        tickers = list(returns.columns)
        n       = len(tickers)
        sigma   = returns.cov() * 252  # annualised covariance

        # Prior
        pi = self.equilibrium_returns(sigma, market_weights)

        if not views:
            # No views: return prior
            bl_weights = self._mv_weights(pi.values, sigma.values)
            return BLResult(
                tickers=tickers,
                prior_returns=pi,
                posterior_returns=pi,
                posterior_cov=sigma,
                bl_weights=pd.Series(bl_weights, index=tickers),
                market_weights=market_weights if market_weights is not None
                               else pd.Series(np.ones(n)/n, index=tickers),
            )

        # Build P (K×N), Q (K,), Ω (K×K) from views
        k     = len(views)
        P     = np.zeros((k, n))
        Q     = np.zeros(k)
        omega = np.zeros((k, k))

        for i, view in enumerate(views):
            for asset, wt in zip(view.assets, view.weights):
                if asset in tickers:
                    j = tickers.index(asset)
                    P[i, j] = wt
            Q[i] = view.expected_return

            # Ω_ii = (1/conf - 1) * P_i Σ P_i'  (Idzorek approach)
            var_view = float(P[i] @ sigma.values @ P[i]) * self.tau
            conf     = max(min(view.confidence, 0.9999), 0.0001)
            omega[i, i] = var_view * (1 / conf - 1)

        # BL posterior
        tau_sigma_inv = np.linalg.inv(self.tau * sigma.values)
        omega_inv     = np.linalg.inv(omega)

        # Posterior precision matrix
        M_inv = tau_sigma_inv + P.T @ omega_inv @ P

        try:
            M = np.linalg.inv(M_inv)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(M_inv)

        # Posterior mean
        mu_bl = M @ (tau_sigma_inv @ pi.values + P.T @ omega_inv @ Q)
        posterior_ret = pd.Series(mu_bl, index=tickers, name="posterior_returns")

        # Full posterior covariance (add back original Σ for total uncertainty)
        post_cov = sigma + pd.DataFrame(M, index=tickers, columns=tickers)

        # Optimal weights
        bl_weights_arr = self._mv_weights(mu_bl, post_cov.values)
        bl_weights     = pd.Series(bl_weights_arr, index=tickers, name="bl_weights")

        mkt_w = market_weights if market_weights is not None else pd.Series(np.ones(n)/n, index=tickers)

        return BLResult(
            tickers=tickers,
            prior_returns=pi,
            posterior_returns=posterior_ret,
            posterior_cov=post_cov,
            bl_weights=bl_weights,
            market_weights=mkt_w,
        )

    def _mv_weights(self, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Unconstrained mean-variance optimal weights w* = (1/λ) Σ^{-1} μ, normalised."""
        try:
            sigma_inv = np.linalg.inv(sigma + 1e-8 * np.eye(len(sigma)))
        except np.linalg.LinAlgError:
            sigma_inv = np.linalg.pinv(sigma)
        raw   = sigma_inv @ mu / self.risk_aversion
        denom = raw.sum()
        if abs(denom) < 1e-10:
            return np.ones(len(mu)) / len(mu)
        return raw / denom


# ─────────────────────────────────────────────
#  CVaR Optimisation
# ─────────────────────────────────────────────

@dataclass
class CVaRResult:
    weights:       pd.Series
    expected_return: float
    portfolio_var:   float      # VaR at level alpha
    portfolio_cvar:  float      # CVaR / ES at level alpha
    sharpe_ratio:    float
    diversification: float      # 1 - HHI of weights
    optimization_status: str


class CVaROptimiser:
    """
    Minimise portfolio CVaR (Conditional Value at Risk) at confidence level α.

    Linear programme formulation (Rockafellar & Uryasev 2000):
      min_{w,ζ,u}  ζ + (1/((1-α)T)) 1'u
      s.t.         u ≥ -r_t'w - ζ  ∀ t
                   u ≥ 0
                   1'w = 1
                   w ≥ lb  (can be negative for long/short)
    """

    def __init__(
        self,
        alpha: float = 0.95,        # confidence level
        min_weight: float = 0.0,    # set negative for short allowed
        max_weight: float = 1.0,
        target_return: Optional[float] = None,   # annualised
        risk_free_rate: float = 0.05,
    ):
        self.alpha          = alpha
        self.min_weight     = min_weight
        self.max_weight     = max_weight
        self.target_return  = target_return
        self.risk_free_rate = risk_free_rate

    def optimise(self, returns: pd.DataFrame) -> CVaRResult:
        """
        Optimise portfolio weights to minimise CVaR.
        returns: daily log returns, shape (T × N).
        """
        T, N = returns.shape
        R    = returns.values  # (T × N)
        ann  = 252

        # Decision variables
        w   = cp.Variable(N, name="weights")
        zeta = cp.Variable(name="zeta")        # VaR auxiliary
        u    = cp.Variable(T, name="aux_u")    # loss exceedance

        # Objective: minimise CVaR
        cvar = zeta + (1 / ((1 - self.alpha) * T)) * cp.sum(u)
        objective = cp.Minimize(cvar)

        # Constraints
        constraints = [
            u >= -R @ w - zeta,
            u >= 0,
            cp.sum(w) == 1,
            w >= self.min_weight,
            w <= self.max_weight,
        ]

        if self.target_return is not None:
            daily_target = self.target_return / ann
            constraints.append(cp.sum(R.mean(axis=0) @ w) >= daily_target)

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except Exception:
            prob.solve(solver=cp.SCS, verbose=False)

        if w.value is None:
            # Fall back to equal weight
            w_val = np.ones(N) / N
            status = "FAILED – equal weight fallback"
        else:
            w_val  = np.clip(w.value, 0, None)
            w_val /= w_val.sum()
            status = prob.status

        weights = pd.Series(w_val, index=returns.columns, name="cvar_weights")

        # Metrics
        port_ret    = R @ w_val
        var_level   = np.percentile(port_ret, (1 - self.alpha) * 100)
        cvar_level  = port_ret[port_ret <= var_level].mean()
        exp_ret     = port_ret.mean() * ann
        exp_vol     = port_ret.std() * np.sqrt(ann)
        sharpe      = (exp_ret - self.risk_free_rate) / exp_vol if exp_vol > 0 else 0
        hhi         = (w_val ** 2).sum()
        diversif    = 1 - hhi

        return CVaRResult(
            weights=weights,
            expected_return=round(exp_ret, 4),
            portfolio_var=round(abs(var_level) * np.sqrt(ann), 4),
            portfolio_cvar=round(abs(cvar_level) * np.sqrt(ann), 4),
            sharpe_ratio=round(sharpe, 4),
            diversification=round(diversif, 4),
            optimization_status=status,
        )

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 30,
    ) -> pd.DataFrame:
        """
        Compute CVaR-efficient frontier by sweeping target returns.
        Returns DataFrame with return / cvar / weights per point.
        """
        ann = 252
        R   = returns.values
        min_r = R.mean(axis=0).min() * ann
        max_r = R.mean(axis=0).max() * ann
        targets = np.linspace(min_r * 0.5, max_r * 0.95, n_points)

        frontier = []
        for tgt in targets:
            opt = CVaROptimiser(
                alpha=self.alpha,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
                target_return=tgt,
                risk_free_rate=self.risk_free_rate,
            )
            res = opt.optimise(returns)
            frontier.append({
                "target_return": tgt,
                "expected_return": res.expected_return,
                "cvar": res.portfolio_cvar,
                "sharpe": res.sharpe_ratio,
                **{f"w_{t}": w for t, w in res.weights.items()},
            })

        return pd.DataFrame(frontier)


# ─────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────

def portfolio_metrics(weights: pd.Series, returns: pd.DataFrame, rf: float = 0.05) -> Dict:
    """Compute standard portfolio metrics for given weights."""
    ann = 252
    w   = weights.reindex(returns.columns).fillna(0).values
    w  /= w.sum()
    R   = returns.values
    port_ret = R @ w

    cum_ret = (1 + port_ret).cumprod()
    peak    = np.maximum.accumulate(cum_ret)
    mdd     = ((cum_ret - peak) / peak).min()

    ann_ret = port_ret.mean() * ann
    ann_vol = port_ret.std() * np.sqrt(ann)
    sharpe  = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    calmar  = ann_ret / abs(mdd) if mdd != 0 else 0

    var_5  = np.percentile(port_ret, 5)
    cvar_5 = port_ret[port_ret <= var_5].mean()

    return {
        "Ann Return":     round(ann_ret * 100, 2),
        "Ann Volatility": round(ann_vol * 100, 2),
        "Sharpe Ratio":   round(sharpe, 3),
        "Max Drawdown":   round(mdd * 100, 2),
        "Calmar Ratio":   round(calmar, 3),
        "VaR 95%":        round(var_5 * 100, 3),
        "CVaR 95%":       round(cvar_5 * 100, 3),
    }
