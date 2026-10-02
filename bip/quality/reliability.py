"""Weibull reliability analysis of cycle life.

Failure = capacity falling to 80% of nominal. In a live test program many cells
have not failed yet when the analysis is run; they are right-censored ("survived
at least N cycles"). Dropping them biases life estimates low; treating their
current cycle count as a failure biases them low too. Maximum likelihood with
censoring uses them correctly:

    log L = sum over failures  [ log f(t) ]  +  sum over survivors [ log S(t) ]
    S(t)  = exp(-(t / eta)^beta)
    f(t)  = (beta / eta) (t / eta)^(beta - 1) S(t)

beta (shape) describes the failure pattern: beta > 1 means wear-out, which is
what capacity fade looks like. eta (scale) is the cycle count by which 63.2% of
cells have failed.

Also provided: B-life (the cycle count by which X% have failed), warranty
failure fraction at a cycle count, confidence intervals from the observed
Fisher information, a likelihood-ratio test of whether groups (batches,
charging policies) differ, and the Kaplan-Meier estimate as a model-free check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats


@dataclass
class WeibullFit:
    beta: float
    eta: float
    n: int
    failures: int
    log_likelihood: float
    cov_log: np.ndarray  # covariance of (log beta, log eta)

    def survival(self, t) -> np.ndarray:
        return np.exp(-(np.asarray(t, dtype=float) / self.eta) ** self.beta)

    def cdf(self, t) -> np.ndarray:
        return 1.0 - self.survival(t)

    def b_life(self, fraction: float = 0.10) -> float:
        """Cycles by which `fraction` of cells have failed (B10 for 0.10)."""
        return float(self.eta * (-np.log(1.0 - fraction)) ** (1.0 / self.beta))

    def b_life_interval(self, fraction: float = 0.10, level: float = 0.95) -> tuple[float, float]:
        """Delta-method interval for B-life, computed on the log scale."""
        k = -np.log(1.0 - fraction)
        # log B = log eta + (1/beta) log k ; gradient w.r.t. (log beta, log eta)
        grad = np.array([-np.log(k) / self.beta, 1.0])
        se = float(np.sqrt(grad @ self.cov_log @ grad))
        z = stats.norm.ppf(0.5 + level / 2)
        b = self.b_life(fraction)
        return b * np.exp(-z * se), b * np.exp(z * se)

    def param_interval(self, name: str, level: float = 0.95) -> tuple[float, float]:
        i = {"beta": 0, "eta": 1}[name]
        z = stats.norm.ppf(0.5 + level / 2)
        value = getattr(self, name)
        se = float(np.sqrt(self.cov_log[i, i]))
        return value * np.exp(-z * se), value * np.exp(z * se)


def _neg_log_lik(params, t, failed):
    log_beta, log_eta = params
    beta, eta = np.exp(log_beta), np.exp(log_eta)
    z = t / eta
    log_s = -(z**beta)
    log_f = np.log(beta) - np.log(eta) + (beta - 1) * np.log(z) + log_s
    return -(np.sum(log_f[failed]) + np.sum(log_s[~failed]))


def fit(times, failed=None) -> WeibullFit:
    """Fit a 2-parameter Weibull by maximum likelihood.

    times:  cycle count at failure, or at the last observation for survivors
    failed: True for failures, False for right-censored survivors (default: all failed)
    """
    t = np.asarray(times, dtype=float)
    d = np.ones_like(t, dtype=bool) if failed is None else np.asarray(failed, dtype=bool)
    if (t <= 0).any():
        raise ValueError("times must be positive")
    if d.sum() < 2:
        raise ValueError("need at least two failures to fit a Weibull")

    # start from a median-rank regression on the failures
    tf = np.sort(t[d])
    ranks = (np.arange(1, tf.size + 1) - 0.3) / (tf.size + 0.4)
    slope, intercept = np.polyfit(np.log(tf), np.log(-np.log(1 - ranks)), 1)
    x0 = [np.log(max(slope, 0.1)), -intercept / max(slope, 0.1)]

    res = optimize.minimize(_neg_log_lik, x0, args=(t, d), method="BFGS")
    if not res.success and not np.isfinite(res.fun):
        raise RuntimeError(f"Weibull fit failed: {res.message}")
    hess = _hessian(lambda p: _neg_log_lik(p, t, d), res.x)
    cov = np.linalg.inv(hess)
    beta, eta = np.exp(res.x)
    return WeibullFit(float(beta), float(eta), int(t.size), int(d.sum()), float(-res.fun), cov)


def _hessian(f, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    n = x.size
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.eye(n)[i] * eps, np.eye(n)[j] * eps
            h[i, j] = (f(x + e_i + e_j) - f(x + e_i - e_j) - f(x - e_i + e_j) + f(x - e_i - e_j)) / (4 * eps**2)
    return h


def compare_groups(groups: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Likelihood-ratio test: one Weibull for all groups vs a separate one per group.

    groups: name -> (times, failed). Returns the statistic, degrees of freedom and p-value.
    """
    fits = {g: fit(t, d) for g, (t, d) in groups.items()}
    t_all = np.concatenate([t for t, _ in groups.values()])
    d_all = np.concatenate([d for _, d in groups.values()])
    pooled = fit(t_all, d_all)
    stat = 2 * (sum(f.log_likelihood for f in fits.values()) - pooled.log_likelihood)
    dof = 2 * (len(groups) - 1)
    return {"statistic": float(stat), "dof": dof, "p_value": float(stats.chi2.sf(stat, dof)),
            "fits": fits, "pooled": pooled}


def kaplan_meier(times, failed=None) -> tuple[np.ndarray, np.ndarray]:
    """Model-free survival curve. Returns (event times, survival just after each)."""
    t = np.asarray(times, dtype=float)
    d = np.ones_like(t, dtype=bool) if failed is None else np.asarray(failed, dtype=bool)
    order = np.argsort(t)
    t, d = t[order], d[order]
    at_risk = t.size
    event_times, surv = [], []
    s = 1.0
    for ti in np.unique(t):
        here = t == ti
        deaths = int(d[here].sum())
        if deaths:
            s *= 1 - deaths / at_risk
            event_times.append(ti)
            surv.append(s)
        at_risk -= int(here.sum())
    return np.array(event_times), np.array(surv)


def censor_at(cycle_life: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """What an analysis at `cutoff` cycles would see: failures before it, survivors censored at it."""
    life = np.asarray(cycle_life, dtype=float)
    failed = life <= cutoff
    return np.where(failed, life, cutoff), failed
