"""
AlphaForge Engine 2: Volatility Regime Detection
──────────────────────────────────────────────────
Gaussian HMM identifies latent market regimes (bull / bear / crisis).
GARCH(1,1) models conditional volatility within each regime.

Mathematics:
  HMM:
    P(s_t | s_{t-1}) = A  (transition matrix)
    P(r_t | s_t)     = N(μ_k, σ_k²)  (Gaussian emission)
    Viterbi → most likely state sequence

  GARCH(1,1):
    r_t   = μ + ε_t,          ε_t = σ_t z_t,  z_t ~ N(0,1)
    σ_t²  = ω + α ε_{t-1}² + β σ_{t-1}²
    Persistence: α + β  (< 1 for stationarity)
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from arch import arch_model
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  Regime Labels
# ─────────────────────────────────────────────

REGIME_NAMES = {0: "Bull / Low-Vol", 1: "Neutral", 2: "Bear / Crisis"}
REGIME_COLORS = {0: "#22c55e", 1: "#f59e0b", 2: "#ef4444"}


def _label_regimes(means: np.ndarray, stds: np.ndarray, n_components: int) -> Dict[int, str]:
    """
    Auto-label regimes by their mean return and volatility.
    Lowest vol + highest mean → Bull, highest vol → Crisis.
    """
    if n_components == 2:
        order = np.argsort(stds)
        return {order[0]: "Bull / Low-Vol", order[1]: "Bear / Crisis"}
    elif n_components == 3:
        vol_order = np.argsort(stds)
        return {
            vol_order[0]: "Bull / Low-Vol",
            vol_order[1]: "Neutral",
            vol_order[2]: "Bear / Crisis",
        }
    else:
        return {i: f"Regime {i}" for i in range(n_components)}


# ─────────────────────────────────────────────
#  Gaussian HMM
# ─────────────────────────────────────────────

@dataclass
class RegimeModel:
    n_regimes: int
    regime_labels: Dict[int, str]
    transition_matrix: np.ndarray
    means: np.ndarray          # per-regime mean return
    stds: np.ndarray           # per-regime volatility
    states: pd.Series          # Viterbi decoded state sequence
    state_probs: pd.DataFrame  # smoothed posterior P(s_t | Y_{1:T})
    regime_stats: pd.DataFrame # summary stats per regime
    aic: float
    bic: float


class HMMRegimeDetector:
    """
    Gaussian HMM for equity return regime detection.
    Fits multiple model orders and selects by BIC.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 200,
        covariance_type: str = "full",
        random_state: int = 42,
        auto_select: bool = False,  # if True, try 2-4 regimes and pick by BIC
    ):
        self.n_components    = n_components
        self.n_iter          = n_iter
        self.covariance_type = covariance_type
        self.random_state    = random_state
        self.auto_select     = auto_select
        self._model: Optional[GaussianHMM] = None

    def fit(self, returns: pd.Series) -> RegimeModel:
        """Fit HMM on return series. Returns RegimeModel."""
        r = returns.dropna().values.reshape(-1, 1)

        if self.auto_select:
            best_bic = np.inf
            best_k   = self.n_components
            for k in [2, 3, 4]:
                m = GaussianHMM(n_components=k, covariance_type=self.covariance_type,
                                n_iter=self.n_iter, random_state=self.random_state)
                try:
                    m.fit(r)
                    log_lik = m.score(r) * len(r)
                    n_params = k * k + k * 2  # rough param count
                    bic = -2 * log_lik + n_params * np.log(len(r))
                    if bic < best_bic:
                        best_bic = bic
                        best_k   = k
                except Exception:
                    pass
            self.n_components = best_k

        model = GaussianHMM(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(r)
        self._model = model

        # Decode states
        states_raw = model.predict(r)
        state_probs_raw = model.predict_proba(r)

        # Regime means / stds
        means = model.means_.flatten()
        if self.covariance_type == "full":
            stds = np.array([np.sqrt(model.covars_[i][0, 0]) for i in range(self.n_components)])
        else:
            stds = np.sqrt(model.covars_.flatten())

        # Label regimes
        labels = _label_regimes(means, stds, self.n_components)

        idx    = returns.dropna().index
        states = pd.Series(states_raw, index=idx, name="regime")
        state_probs = pd.DataFrame(
            state_probs_raw,
            index=idx,
            columns=[f"P(regime_{i})" for i in range(self.n_components)],
        )

        # Per-regime stats
        ret_series = returns.dropna()
        regime_stats_list = []
        ann = 252
        for k in range(self.n_components):
            mask = states == k
            r_k  = ret_series[mask]
            regime_stats_list.append({
                "Regime":       labels.get(k, f"R{k}"),
                "Obs":          int(mask.sum()),
                "Freq (%)":     round(100 * mask.mean(), 1),
                "Ann Return":   round(r_k.mean() * ann * 100, 2),
                "Ann Vol":      round(r_k.std() * np.sqrt(ann) * 100, 2),
                "Sharpe":       round(r_k.mean() / r_k.std() * np.sqrt(ann), 3) if r_k.std() > 0 else 0,
                "Skew":         round(r_k.skew(), 3),
                "Kurt":         round(r_k.kurt(), 3),
            })
        regime_stats = pd.DataFrame(regime_stats_list)

        # IC
        log_lik = model.score(r) * len(r)
        n_p     = self.n_components * self.n_components + self.n_components * 2
        aic     = -2 * log_lik + 2 * n_p
        bic     = -2 * log_lik + n_p * np.log(len(r))

        return RegimeModel(
            n_regimes=self.n_components,
            regime_labels=labels,
            transition_matrix=model.transmat_,
            means=means,
            stds=stds,
            states=states,
            state_probs=state_probs,
            regime_stats=regime_stats,
            aic=aic,
            bic=bic,
        )

    def forecast_regime(self, current_state: int) -> np.ndarray:
        """One-step-ahead regime transition probabilities."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.transmat_[current_state]


# ─────────────────────────────────────────────
#  GARCH(1,1) Per-Regime Volatility
# ─────────────────────────────────────────────

@dataclass
class GARCHResult:
    regime: int
    regime_label: str
    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_vol: float        # omega / (1 - alpha - beta)
    conditional_vol: pd.Series      # σ_t
    vol_forecast_1d: float          # 1-day ahead σ
    vol_forecast_5d: float          # 5-day ahead σ (annualised)
    aic: float
    bic: float
    n_obs: int


class RegimeGARCH:
    """
    Fit GARCH(1,1) models per HMM regime.
    Allows regime-conditional volatility forecasting.
    """

    def __init__(self, p: int = 1, q: int = 1, dist: str = "Normal"):
        self.p = p
        self.q = q
        self.dist = dist

    def fit_all(
        self,
        returns: pd.Series,
        regime_model: RegimeModel,
    ) -> Dict[int, GARCHResult]:
        """Fit GARCH on each regime subset. Returns dict keyed by regime index."""
        results = {}
        r = returns.dropna() * 100  # scale to percent (GARCH numerical stability)

        for k in range(regime_model.n_regimes):
            mask = regime_model.states == k
            r_k  = r[mask]

            if len(r_k) < 30:
                continue

            try:
                model = arch_model(
                    r_k,
                    mean="Constant",
                    vol="GARCH",
                    p=self.p,
                    q=self.q,
                    dist=self.dist,
                )
                res = model.fit(disp="off", show_warning=False)

                params      = res.params
                omega       = params.get("omega", params.iloc[1])
                alpha       = params.get("alpha[1]", params.iloc[2])
                beta        = params.get("beta[1]",  params.iloc[3])
                persistence = alpha + beta

                # Unconditional variance
                if persistence < 1:
                    uncond_var = omega / (1 - persistence)
                else:
                    uncond_var = res.conditional_volatility.mean() ** 2
                uncond_vol = np.sqrt(uncond_var) / 100  # back to decimal

                # Conditional volatility series (decimal)
                cond_vol = res.conditional_volatility / 100
                cond_vol.name = f"garch_vol_regime_{k}"

                # Forecasts
                fcast = res.forecast(horizon=5, reindex=False)
                var_1d = fcast.variance.iloc[-1, 0] / 1e4
                var_5d = fcast.variance.iloc[-1, 4] / 1e4
                vol_1d = np.sqrt(var_1d)
                vol_5d = np.sqrt(var_5d * 252 / 5)  # annualise 5-day

                results[k] = GARCHResult(
                    regime=k,
                    regime_label=regime_model.regime_labels.get(k, f"R{k}"),
                    omega=round(omega, 6),
                    alpha=round(alpha, 4),
                    beta=round(beta, 4),
                    persistence=round(persistence, 4),
                    unconditional_vol=round(uncond_vol * np.sqrt(252) * 100, 2),
                    conditional_vol=cond_vol,
                    vol_forecast_1d=round(vol_1d * np.sqrt(252) * 100, 2),
                    vol_forecast_5d=round(vol_5d * 100, 2),
                    aic=round(res.aic, 2),
                    bic=round(res.bic, 2),
                    n_obs=len(r_k),
                )
            except Exception as e:
                print(f"[GARCH] Regime {k} failed: {e}")

        return results

    def fit_unconditional(self, returns: pd.Series) -> GARCHResult:
        """Fit GARCH on full series (no regime conditioning)."""
        r = returns.dropna() * 100
        model = arch_model(r, mean="Constant", vol="GARCH", p=self.p, q=self.q, dist=self.dist)
        res   = model.fit(disp="off", show_warning=False)

        params      = res.params
        omega       = float(params.iloc[1])
        alpha       = float(params.iloc[2])
        beta        = float(params.iloc[3])
        persistence = alpha + beta
        uncond_var  = omega / (1 - persistence) if persistence < 1 else res.conditional_volatility.mean() ** 2
        uncond_vol  = np.sqrt(uncond_var) / 100

        fcast  = res.forecast(horizon=5, reindex=False)
        var_1d = fcast.variance.iloc[-1, 0] / 1e4
        var_5d = fcast.variance.iloc[-1, 4] / 1e4

        return GARCHResult(
            regime=-1,
            regime_label="Full Series",
            omega=round(omega, 6),
            alpha=round(alpha, 4),
            beta=round(beta, 4),
            persistence=round(persistence, 4),
            unconditional_vol=round(uncond_vol * np.sqrt(252) * 100, 2),
            conditional_vol=res.conditional_volatility / 100,
            vol_forecast_1d=round(np.sqrt(var_1d) * np.sqrt(252) * 100, 2),
            vol_forecast_5d=round(np.sqrt(var_5d * 252 / 5) * 100, 2),
            aic=round(res.aic, 2),
            bic=round(res.bic, 2),
            n_obs=len(r),
        )


# ─────────────────────────────────────────────
#  Volatility Surface Utilities
# ─────────────────────────────────────────────

def realised_vol(returns: pd.Series, window: int = 21, ann: int = 252) -> pd.Series:
    """Compute rolling realised volatility."""
    return returns.rolling(window).std() * np.sqrt(ann)


def vol_of_vol(returns: pd.Series, window: int = 63, ann: int = 252) -> pd.Series:
    """Rolling vol-of-vol: std of rolling vol estimates."""
    rv = realised_vol(returns, window=21, ann=ann)
    return rv.rolling(window).std()


def gk_vol(ohlcv: pd.DataFrame, ann: int = 252) -> pd.Series:
    """
    Garman-Klass volatility estimator using OHLC data.
    GK = sqrt(0.5*(ln(H/L))² - (2ln2-1)*(ln(C/O))²) * sqrt(ann)
    """
    lo = np.log(ohlcv["High"] / ohlcv["Low"])
    co = np.log(ohlcv["Close"] / ohlcv["Open"])
    gk = np.sqrt(0.5 * lo**2 - (2 * np.log(2) - 1) * co**2) * np.sqrt(ann)
    return gk
