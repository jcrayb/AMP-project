"""Block Coordinate Descent for the max-Sharpe put-protection model.

This reconciles the two block solves so that *every* code path (the numpy
objective evaluator, the w-block, and the Gurobi delta-block) builds the
**identical** model. See ``algo_design.md`` for the full theory.

Key pieces
----------
- ``FuncLib``               one config-driven library for Phi, Phi^{-1}, phi with
                            ``numpy`` and ``gurobi`` backends -> both blocks agree.
- ``build_moments``         shared construction of (net_mu, Sigma_d, P).
- ``evaluate`` / ``sharpe`` the single ground-truth objective.
- ``solve_w_given_delta``   w-block: a genuine convex SOCP (Charnes-Cooper).
- ``solve_delta_given_w``   delta-block: Gurobi non-convex, warm-started.
- ``run_bcd``               driver with a monotone safeguard.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
import gurobipy as gp
from gurobipy import GRB, nlfunc

SQRT2PI = math.sqrt(2 * math.pi)

# 101-point grid for the (negated) normal CDF, exactly as used in dancing.ipynb.
# The negation is part of the existing Black-Scholes put expression and is
# preserved verbatim -- we centralise the math, we do not change it.
X_PTS_CDF = np.linspace(-5, 5, 101)
Y_PTS_CDF = -np.array([math.erf(x / math.sqrt(2)) / 2 + 1 / 2 for x in X_PTS_CDF])

DEFAULT_APPROX = dict(pdf="exact", cdf="exact", inv_cdf="exact")


# --------------------------------------------------------------------------- #
# Problem container
# --------------------------------------------------------------------------- #
@dataclass
class Problem:
    """Fixed (Delta-independent) market data."""

    mu: np.ndarray       # (n,) base expected returns
    Sigma: np.ndarray    # (n, n) base covariance
    S: np.ndarray        # (n,) spot prices
    r: float             # per-period risk-free rate
    T: float             # option horizon

    @property
    def n(self) -> int:
        return len(self.mu)

    @property
    def sigma(self) -> np.ndarray:
        return np.sqrt(np.diag(self.Sigma))


# --------------------------------------------------------------------------- #
# 1. Single source of truth: config-driven function library
# --------------------------------------------------------------------------- #
class FuncLib:
    """Phi, Phi^{-1}, phi with exact/approx forms and numpy/gurobi backends.

    Same ``approx`` config + same input => same value in both backends, which is
    what forces the two blocks to solve an identical model.
    """

    def __init__(self, approx=None, backend="numpy", model=None):
        self.approx = {**DEFAULT_APPROX, **(approx or {})}
        if backend not in ("numpy", "gurobi"):
            raise ValueError("backend must be 'numpy' or 'gurobi'")
        self.backend = backend
        self.m = model
        if backend == "gurobi" and model is None:
            raise ValueError("gurobi backend requires a model")
        self._log = np.log if backend == "numpy" else nlfunc.log
        self._exp = np.exp if backend == "numpy" else nlfunc.exp

    # turn an expression into a named variable (gurobi) or pass through (numpy)
    def as_var(self, expr, lb=-GRB.INFINITY):
        if self.backend == "numpy":
            return expr
        v = self.m.addVar(lb=lb)
        self.m.addConstr(v == expr)
        return v

    def inv_cdf(self, delta):
        """Phi^{-1}(delta)."""
        if self.approx["inv_cdf"] == "exact":          # logistic approx to probit
            return -1 / 1.7 * self._log(1 / delta - 1)
        return 4 / 1.7 * (delta - 0.5)                  # paper eq:phi_inv_approx

    def pdf(self, x):
        """phi(x)."""
        if self.approx["pdf"] == "exact":
            return 1 / SQRT2PI * self._exp(-0.5 * x * x)
        return 1 / SQRT2PI * (1 - x * x / 2)            # paper eq:phi_approx

    def cdf(self, x):
        """(Negated) normal CDF as used inside the Black-Scholes put expression.

        ``x`` must be a *variable* in the gurobi backend (PWL requirement).
        """
        if self.approx["cdf"] == "approx":              # paper eq:Phi_approx, negated
            return -(0.5 + x / SQRT2PI)
        if self.backend == "numpy":
            return np.interp(x, X_PTS_CDF, Y_PTS_CDF)
        y = self.m.addVar(lb=-GRB.INFINITY)
        self.m.addGenConstrPWL(x, y, list(X_PTS_CDF), list(Y_PTS_CDF))
        return y


# --------------------------------------------------------------------------- #
# 2. Shared moment construction
# --------------------------------------------------------------------------- #
def build_moments(lib: FuncLib, prob: Problem, Delta):
    """Return (net_mu, Sigma_d, P).

    Works for both backends: with ``numpy`` everything is floats; with
    ``gurobi`` the entries are Vars/expressions and the needed auxiliary
    variables/constraints are added to ``lib.m``.

    ``net_mu[i] = mu^Delta_i - P^Delta_i / S_i``  (the per-share net return).
    """
    n, sigma, S, mu, Sigma, r, T = (
        prob.n, prob.sigma, prob.S, prob.mu, prob.Sigma, prob.r, prob.T,
    )
    gurobi = lib.backend == "gurobi"

    net_mu = [None] * n
    P = [None] * n
    Sigma_d = np.zeros((n, n), dtype=object if gurobi else float)

    for i in range(n):
        alpha = lib.as_var(lib.inv_cdf(Delta[i]))                       # Phi^{-1}(Delta_i)
        K = (sigma[i] * alpha + mu[i] + 1) * S[i]                       # strike, eq:strike_price

        d1 = lib.as_var((1 - K / S[i] + T * (r + sigma[i] ** 2 / 2)) / (sigma[i] * math.sqrt(T)))
        d2 = lib.as_var(d1 - sigma[i] * math.sqrt(T))

        cdf_d1 = lib.cdf(d1)
        cdf_d2 = lib.cdf(d2)
        phi = lib.pdf(alpha)

        P_i = lib.as_var(K * math.exp(-r * T) * cdf_d2 - S[i] * cdf_d1)  # Black-Scholes put
        P[i] = P_i

        mu_d = lib.as_var(mu[i] + sigma[i] * (Delta[i] * alpha + phi))   # eq:model_parameters_delta
        net_mu[i] = mu_d - P_i / S[i]

        var_i = sigma[i] ** 2 * (
            (1 - Delta[i]) * (1 + alpha ** 2 * Delta[i])
            + alpha * phi * (1 - 2 * Delta[i])
            - phi * phi
        )
        Sigma_d[i, i] = lib.as_var(var_i, lb=0) if gurobi else var_i

        for j in range(i + 1, n):
            rho = Sigma[i, j] / math.sqrt(Sigma[i, i] * Sigma[j, j])
            cov = sigma[i] * sigma[j] * rho * (1 - Delta[i]) * (1 - Delta[j])
            cov = lib.as_var(cov) if gurobi else cov
            Sigma_d[i, j] = Sigma_d[j, i] = cov

    return net_mu, Sigma_d, P


# --------------------------------------------------------------------------- #
# 3. Ground-truth objective (numpy)
# --------------------------------------------------------------------------- #
def moments_numpy(prob: Problem, Delta, approx=None):
    lib = FuncLib(approx, backend="numpy")
    net_mu, Sigma_d, P = build_moments(lib, prob, Delta)
    return np.asarray(net_mu, float), np.asarray(Sigma_d, float), np.asarray(P, float)


def sharpe(prob: Problem, w, Delta, approx=None) -> float:
    net_mu, Sigma_d, _ = moments_numpy(prob, Delta, approx)
    num = net_mu @ w - prob.r
    var = float(w @ Sigma_d @ w)
    return num / math.sqrt(var)


def evaluate(prob: Problem, mode, w, Delta, approx=None) -> float:
    """Single objective used for reporting and the monotone guard."""
    net_mu, Sigma_d, _ = moments_numpy(prob, Delta, approx)
    if mode == "max_sharpe":
        return (net_mu @ w - prob.r) / math.sqrt(float(w @ Sigma_d @ w))
    if mode == "max_mu":
        return float(net_mu @ w - prob.r)
    if mode == "min_var":
        return math.sqrt(float(w @ Sigma_d @ w))
    raise ValueError(f"unknown mode {mode!r}")


def _improves(mode, new, cur, tol=1e-9) -> bool:
    if mode == "min_var":
        return new <= cur - tol
    return new >= cur + tol


# --------------------------------------------------------------------------- #
# 4. w-block: convex SOCP (Charnes-Cooper homogenisation of the Sharpe ratio)
# --------------------------------------------------------------------------- #
def _new_model(time_limit, mip_gap):
    m = gp.Model()
    m.Params.OutputFlag = 0
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    return m


def _require_solution(m, where):
    """Ensure a usable incumbent exists before reading ``.X``."""
    if m.Status == GRB.INTERRUPTED:
        # Gurobi catches Ctrl-C during optimize() and returns INTERRUPTED rather
        # than letting Python raise; re-raise so the driver can stop cleanly.
        raise KeyboardInterrupt
    if m.SolCount > 0:
        return
    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        raise RuntimeError(
            f"{where}: model infeasible (status {m.Status}). For max_sharpe this "
            "usually means no portfolio has return above the risk-free rate at the "
            "current Delta, so the Charnes-Cooper normalisation is infeasible.")
    raise RuntimeError(f"{where}: no solution found (status {m.Status}).")


def solve_w_given_delta(prob: Problem, Delta, mode="max_sharpe", approx=None,
                        time_limit=None, mip_gap=None):
    """Optimise w with Delta fixed. Pure convex SOCP -> global optimum, fast.

    Returns ``(w, obj)`` where ``obj`` is the mode objective.
    """
    net_mu, Sigma_d, _ = moments_numpy(prob, Delta, approx)
    a = net_mu
    L = np.linalg.cholesky(Sigma_d)          # L @ L.T == Sigma_d (Sigma_d PD, paper l.282-284)
    n, r = prob.n, prob.r

    m = _new_model(time_limit, mip_gap)

    if mode == "max_sharpe":
        # max (a^T w - r)/||L^T w||, w in simplex.  Charnes-Cooper: y = kappa w,
        # normalise numerator a^T y - r kappa = 1, then minimise ||L^T y||.
        # Assumes the optimal numerator is positive (true for this data).
        y = m.addMVar(n, lb=0, name="y")
        kappa = m.addVar(lb=0, name="kappa")
        s = m.addVar(lb=0, name="s")
        g = m.addMVar(n, lb=-GRB.INFINITY, name="g")
        m.addConstr(g == L.T @ y)
        m.addConstr(g @ g <= s * s)                 # SOC ||L^T y|| <= s  (convex)
        m.addConstr(a @ y - r * kappa == 1)
        m.addConstr(y.sum() == kappa)
        m.setObjective(s, GRB.MINIMIZE)
        m.optimize()
        _require_solution(m, "solve_w_given_delta[max_sharpe]")
        w = y.X / kappa.X
        return w, 1.0 / s.X

    w = m.addMVar(n, lb=0, ub=1, name="w")
    m.addConstr(w.sum() == 1)
    if mode == "max_mu":
        m.setObjective(a @ w - r, GRB.MAXIMIZE)
        m.optimize()
        _require_solution(m, "solve_w_given_delta[max_mu]")
        return w.X, m.ObjVal
    if mode == "min_var":
        s = m.addVar(lb=0, name="s")
        g = m.addMVar(n, lb=-GRB.INFINITY, name="g")
        m.addConstr(g == L.T @ w)
        m.addConstr(g @ g <= s * s)
        m.setObjective(s, GRB.MINIMIZE)
        m.optimize()
        _require_solution(m, "solve_w_given_delta[min_var]")
        return w.X, m.ObjVal
    raise ValueError(f"unknown mode {mode!r}")


# --------------------------------------------------------------------------- #
# 5. delta-block: Gurobi non-convex, warm-started, same model as the evaluator
# --------------------------------------------------------------------------- #
def solve_delta_given_w(prob: Problem, w, Delta0, mode="max_sharpe", approx=None,
                        time_limit=100, mip_gap=1e-4):
    """Optimise Delta with w fixed. Intrinsically non-convex -> Gurobi global."""
    n, r = prob.n, prob.r
    w = np.asarray(w, float)

    m = _new_model(time_limit, mip_gap)
    m.Params.NonConvex = 2

    Delta = m.addMVar(n, lb=0.1, ub=0.5, name="Delta")
    Delta.Start = np.asarray(Delta0, float)            # warm start: incoming point feasible

    lib = FuncLib(approx, backend="gurobi", model=m)
    net_mu, Sigma_d, _ = build_moments(lib, prob, Delta)

    var_norm_sq = m.addVar(lb=0, name="var_norm_sq")
    m.addConstr(
        var_norm_sq
        == gp.quicksum(Sigma_d[i, i] * w[i] ** 2 for i in range(n))
        + 2 * gp.quicksum(Sigma_d[i, j] * w[i] * w[j]
                          for i in range(n) for j in range(i + 1, n))
    )
    num = gp.quicksum(net_mu[i] * w[i] for i in range(n)) - r

    if mode == "max_sharpe":
        var_norm = m.addVar(lb=0, name="var_norm")
        m.addConstr(var_norm_sq == var_norm ** 2)
        t = m.addVar(lb=-GRB.INFINITY, name="t")
        m.addConstr(num >= var_norm * t)
        m.setObjective(t, GRB.MAXIMIZE)
    elif mode == "max_mu":
        m.setObjective(num, GRB.MAXIMIZE)              # aligned to net_mu (incl. -P/S)
    elif mode == "min_var":
        var_norm = m.addVar(lb=0, name="var_norm")
        m.addConstr(var_norm_sq == var_norm ** 2)
        m.setObjective(var_norm, GRB.MINIMIZE)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    m.optimize()
    _require_solution(m, "solve_delta_given_w")
    return Delta.X, m.ObjVal


# --------------------------------------------------------------------------- #
# 6. Dual bound (always-valid, converging)
# --------------------------------------------------------------------------- #
def _box_spectral_bounds(prob: Problem, approx=None, n_grid=401, safety=1.0):
    """Rigorous bounds over the Delta box, exploiting separability.

    Returns a dict with per-asset numerator extremes and global spectral bounds on
    ``Sigma_d(Delta)``:
      - ``num_max[i] = max_{Delta_i} (a_i(Delta_i) - r)``  (numerator is separable)
      - ``lam_min_lo`` : positive lower bound on ``lambda_min(Sigma_d)`` over the box.
        Uses the paper's structure ``Sigma_d = Sigma o p p^T + D`` (PSD + PD diagonal,
        lines 282-284), so ``Sigma_d >= D`` and ``lambda_min(Sigma_d) >= min_i D_ii``
        with ``D_ii(Delta_i) = Var_i(Delta_i) - sigma_i^2 (1-Delta_i)^2`` -- separable.
      - ``lam_max_hi`` : valid upper bound on ``lambda_max(Sigma_d)`` (Gershgorin).
    All 1-D sweeps carry a Lipschitz safety margin from the max grid slope.
    """
    n = prob.n
    r = prob.r
    sig = prob.sigma
    grid = np.linspace(0.1, 0.5, n_grid)
    h = grid[1] - grid[0]
    base = np.full(n, 0.3)

    a_lo = np.empty(n)
    a_hi = np.empty(n)
    var_lo = np.empty(n)
    var_hi = np.empty(n)
    dcorr_lo = np.empty(n)          # min over box of the PD diagonal correction D_ii

    for i in range(n):
        ai = np.empty(n_grid)
        vi = np.empty(n_grid)
        di = np.empty(n_grid)
        for k, d in enumerate(grid):
            D = base.copy()
            D[i] = d
            nm, Sd, _ = moments_numpy(prob, D, approx)
            ai[k] = nm[i]                                    # a_i depends only on Delta_i
            vi[k] = Sd[i, i]                                 # Var_i depends only on Delta_i
            di[k] = Sd[i, i] - sig[i] ** 2 * (1 - d) ** 2    # D_ii = Var_i - sigma_i^2 p_i^2
        La = safety * (np.max(np.abs(np.diff(ai))) / h) * h / 2
        Lv = safety * (np.max(np.abs(np.diff(vi))) / h) * h / 2
        Ld = safety * (np.max(np.abs(np.diff(di))) / h) * h / 2
        a_hi[i] = ai.max() + La
        a_lo[i] = ai.min() - La
        var_hi[i] = vi.max() + Lv
        var_lo[i] = max(vi.min() - Lv, 1e-12)
        dcorr_lo[i] = di.min() - Ld

    # lambda_min lower bound (positive by the paper's PD-correction argument)
    lam_min_lo = float(dcorr_lo.min())
    # lambda_max upper bound via Gershgorin on the box (|Sigma_d_ij| <= |Sigma_ij|*0.81)
    absSig = np.abs(prob.Sigma)
    offrow = 0.81 * (absSig.sum(axis=1) - np.diag(absSig))
    lam_max_hi = float(np.max(var_hi + offrow))

    num_lo = a_lo - r
    num_hi = a_hi - r
    return dict(num_lo=num_lo, num_hi=num_hi, var_lo=var_lo, var_hi=var_hi,
                lam_min_lo=lam_min_lo, lam_max_hi=lam_max_hi)


def cheap_dual_bound(prob: Problem, mode="max_sharpe", approx=None, n_grid=401,
                     safety=1.0, bounds=None):
    """A rigorous, finite, always-valid bound on the global optimum (cheap).

    ``max_sharpe`` -> upper bound ``t_max`` via the tangency-Sharpe inequality
        ``Sharpe <= sqrt(a~^T Sigma_d^{-1} a~) <= sqrt(||a~||^2 / lambda_min)``.
    ``max_mu``     -> upper bound on net portfolio return.
    ``min_var``    -> lower bound on portfolio std.
    """
    b = bounds or _box_spectral_bounds(prob, approx, n_grid, safety)
    lam_min_lo = b["lam_min_lo"]
    if lam_min_lo <= 0:
        warnings.warn(
            "lambda_min lower bound is non-positive; the structural PD correction did "
            "not certify positivity numerically. The Sharpe cap may be loose/invalid -- "
            "pass an explicit t_max.", RuntimeWarning)
        lam_min_lo = max(lam_min_lo, 1e-9)

    if mode == "max_sharpe":
        num2 = np.maximum(b["num_hi"] ** 2, b["num_lo"] ** 2)   # max a~_i^2 over box
        return math.sqrt(float(num2.sum()) / lam_min_lo)
    if mode == "max_mu":
        return float(b["num_hi"].max())                          # best single asset, net of r
    if mode == "min_var":
        return math.sqrt(lam_min_lo / prob.n)                    # std lower bound on simplex
    raise ValueError(f"unknown mode {mode!r}")


def dual_bound(prob: Problem, incumbent, mode="max_sharpe", approx=None,
               time_limit=600, mip_gap=1e-4, t_max=None, n_grid=401, verbose=False):
    """Converging Gurobi dual bound on the joint (w, Delta) problem.

    Caps the relaxation so Gurobi's ``ObjBound`` is finite, always valid, and tightens
    toward the optimum. Returns ``dict(UB, LB, gap, t_max, status)`` with a non-negative
    gap.  ``incumbent`` is a ``dict(obj, w, Delta)`` (e.g. ``run_bcd``'s ``best``).
    """
    n, r = prob.n, prob.r
    bnds = _box_spectral_bounds(prob, approx, n_grid)
    if t_max is None:
        t_max = cheap_dual_bound(prob, mode, approx, bounds=bnds)
    var_norm_hi = math.sqrt(bnds["lam_max_hi"])     # finite bound -> finite McCormick env.

    m = _new_model(time_limit, mip_gap)
    m.Params.NonConvex = 2
    if verbose:
        m.Params.OutputFlag = 1

    w = m.addMVar(n, lb=0, ub=1, name="w")
    m.addConstr(w.sum() == 1)
    Delta = m.addMVar(n, lb=0.1, ub=0.5, name="Delta")
    if incumbent.get("w") is not None:
        w.Start = np.asarray(incumbent["w"], float)
        Delta.Start = np.asarray(incumbent["Delta"], float)

    lib = FuncLib(approx, backend="gurobi", model=m)
    net_mu, Sigma_d, _ = build_moments(lib, prob, Delta)

    var_norm_sq = m.addVar(lb=0, ub=bnds["lam_max_hi"], name="var_norm_sq")
    m.addConstr(
        var_norm_sq
        == gp.quicksum(Sigma_d[i, i] * w[i] * w[i] for i in range(n))
        + 2 * gp.quicksum(Sigma_d[i, j] * w[i] * w[j]
                          for i in range(n) for j in range(i + 1, n))
    )
    num = gp.quicksum(net_mu[i] * w[i] for i in range(n)) - r
    inc = incumbent["obj"]

    if mode == "max_sharpe":
        var_norm = m.addVar(lb=0, ub=var_norm_hi, name="var_norm")
        m.addConstr(var_norm_sq == var_norm ** 2)
        t = m.addVar(lb=-t_max, ub=t_max, name="t")          # finite cap (valid: |Sharpe|<=t_max)
        m.addConstr(num >= var_norm * t)
        m.setObjective(t, GRB.MAXIMIZE)
        m.optimize()
        UB = m.ObjBound
        LB = max(m.ObjVal, inc) if m.SolCount else inc
    elif mode == "max_mu":
        m.setObjective(num, GRB.MAXIMIZE)
        m.optimize()
        UB = m.ObjBound
        LB = max(m.ObjVal, inc) if m.SolCount else inc
    elif mode == "min_var":
        var_norm = m.addVar(lb=0, ub=var_norm_hi, name="var_norm")
        m.addConstr(var_norm_sq == var_norm ** 2)
        m.setObjective(var_norm, GRB.MINIMIZE)
        m.optimize()
        LB = m.ObjBound                                       # valid lower bound (minimisation)
        UB = min(m.ObjVal, inc) if m.SolCount else inc
        gap = (UB - LB) / abs(UB) if UB else float("inf")
        return dict(UB=UB, LB=LB, gap=gap, t_max=t_max, status=m.Status)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    gap = (UB - LB) / abs(LB) if LB else float("inf")
    return dict(UB=UB, LB=LB, gap=gap, t_max=t_max, status=m.Status)


# --------------------------------------------------------------------------- #
# 7. Driver: run until a time limit, a small gap, or the user stops it
# --------------------------------------------------------------------------- #
def _inc(best):
    """The incumbent objective, or ``None`` if no start has produced one yet."""
    return best["obj"] if best["w"] is not None else None


def _gap(mode, inc, bound):
    """Relative gap between the incumbent and the always-valid dual bound.

    For the maximisation modes ``bound`` is an *upper* bound (>= optimum); for
    ``min_var`` it is a *lower* bound (<= optimum). Returns ``inf`` until an
    incumbent exists. The gap is non-negative because ``cheap_dual_bound`` is
    provably superoptimal, so stopping on ``gap <= gap_tol`` is sound.
    """
    if inc is None or not np.isfinite(inc):
        return float("inf")
    if mode == "min_var":
        UB, LB = inc, bound
        return (UB - LB) / abs(UB) if UB else float("inf")
    UB, LB = bound, inc
    return (UB - LB) / abs(LB) if LB else float("inf")


def _one_start(prob, mode, approx, rng, num_iters, block_time_limit, mip_gap,
               best, bound, verbose, s, deadline):
    """One BCD start (monotone by construction). Updates ``best`` in place."""
    n = prob.n
    Delta = rng.random(n) * 0.4 + 0.1
    w = rng.random(n)
    w /= w.sum()
    cur = evaluate(prob, mode, w, Delta, approx)
    hist = [cur]

    for k in range(num_iters):
        # --- w-block (convex, exact) ---
        w_new, _ = solve_w_given_delta(prob, Delta, mode, approx,
                                       time_limit=block_time_limit, mip_gap=mip_gap)
        f_new = evaluate(prob, mode, w_new, Delta, approx)
        if _improves(mode, f_new, cur):
            w, cur = w_new, f_new
        hist.append(cur)
        _update_best(best, mode, cur, w, Delta)
        if verbose:
            print(f"{s + 1:<7}{k + 1:<6}{'w':<8}{cur:<14.6f}{best['obj']:<14.6f}"
                  f"{_gap(mode, _inc(best), bound) * 100:<10.2f}")

        # --- delta-block (non-convex, guarded) ---
        D_new, _ = solve_delta_given_w(prob, w, Delta, mode, approx,
                                       time_limit=block_time_limit, mip_gap=mip_gap)
        f_new = evaluate(prob, mode, w, D_new, approx)
        if _improves(mode, f_new, cur):
            Delta, cur = D_new, f_new
        hist.append(cur)
        _update_best(best, mode, cur, w, Delta)
        if verbose:
            print(f"{s + 1:<7}{k + 1:<6}{'Delta':<8}{cur:<14.6f}{best['obj']:<14.6f}"
                  f"{_gap(mode, _inc(best), bound) * 100:<10.2f}")

        if deadline is not None and time.monotonic() >= deadline:
            break          # stay responsive to the wall-clock limit mid-start
    return hist


def _write_solution(path, mode, best, bound, gap, stop_reason, elapsed, starts):
    """Persist the incumbent (w, Delta, objective) and the dual bound/gap."""
    inc = _inc(best)
    sol = dict(
        mode=mode,
        objective=(None if inc is None else float(inc)),
        w=(None if best["w"] is None else [float(x) for x in best["w"]]),
        Delta=(None if best["Delta"] is None else [float(x) for x in best["Delta"]]),
        bound=float(bound),
        gap=(None if not np.isfinite(gap) else float(gap)),
        stop_reason=stop_reason,
        starts=starts,
        elapsed_sec=round(elapsed, 3),
        timestamp=dt.datetime.now().isoformat(timespec="seconds"),
    )
    with open(path, "w") as f:
        json.dump(sol, f, indent=2)
    return sol


def run_bcd(prob: Problem, mode="max_sharpe", approx=None, num_iters=4, seed=0,
            block_time_limit=100, mip_gap=1e-4,
            time_limit=None, gap_tol=None, max_starts=None,
            output_file="solution.json", refine_bound=False, bound_time_limit=600,
            verbose=True):
    """Multi-start BCD that keeps drawing random starts until told to stop.

    A fresh random ``(w, Delta)`` start is drawn on every pass (exactly as before);
    each start's objective sequence is monotone by construction. The loop stops on
    the first of:

      * ``time_limit`` -- wall-clock seconds (``None`` => no time limit);
      * ``gap_tol``    -- relative duality gap vs the always-valid dual bound
                          (``None`` => no gap stop);
      * ``KeyboardInterrupt`` (Ctrl-C) -- user termination;
      * ``max_starts`` -- optional hard cap on the number of starts (``None`` => none).

    With ``time_limit`` and ``gap_tol`` both ``None`` (and no ``max_starts``) the loop
    runs indefinitely until Ctrl-C. **Whenever** it stops -- limit, gap, or user --
    the incumbent (w, Delta, objective) and the dual bound/gap are written to
    ``output_file`` (JSON).

    The live gap uses ``cheap_dual_bound`` -- an instant, always-superoptimal bound --
    so the reported gap is a conservative over-estimate (stopping on it is sound).
    Set ``refine_bound=True`` to tighten the final bound with the converging Gurobi
    ``dual_bound`` (<= ``bound_time_limit`` s) before writing the file.

    Returns ``(best, histories)`` where ``best`` is ``dict(obj, w, Delta, bound, gap)``
    and ``histories`` is a list (one per start) of per-step objective values.
    """
    rng = np.random.default_rng(seed)
    sense_best = -np.inf if mode != "min_var" else np.inf
    best = dict(obj=sense_best, w=None, Delta=None)
    histories = []

    # always-valid, instant bound (independent of the incumbent)
    bound = cheap_dual_bound(prob, mode, approx)

    if verbose:
        lim = "inf" if time_limit is None else f"{time_limit:g}s"
        gtol = "off" if gap_tol is None else f"{gap_tol * 100:g}%"
        print(f"BCD running: time_limit={lim}  gap_tol={gtol}  "
              f"dual bound={bound:.6f}  (Ctrl-C to stop)")
        print(f"{'start':<7}{'iter':<6}{'block':<8}{'obj':<14}{'best':<14}{'gap%':<10}")

    t0 = time.monotonic()
    deadline = None if time_limit is None else t0 + time_limit
    stop_reason = "completed"
    s = 0
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                stop_reason = "time_limit"
                break
            if gap_tol is not None and _gap(mode, _inc(best), bound) <= gap_tol:
                stop_reason = "gap"
                break
            if max_starts is not None and s >= max_starts:
                stop_reason = "max_starts"
                break

            hist = _one_start(prob, mode, approx, rng, num_iters,
                              block_time_limit, mip_gap, best, bound,
                              verbose, s, deadline)
            histories.append(hist)
            s += 1
    except KeyboardInterrupt:
        stop_reason = "user"
        if verbose:
            print("\n[interrupted] writing solution...")

    elapsed = time.monotonic() - t0

    if refine_bound and best["w"] is not None:
        if verbose:
            print(f"\nRefining dual bound (<= {bound_time_limit}s, Ctrl-C to skip)...")
        try:
            bnd = dual_bound(prob, best, mode, approx, time_limit=bound_time_limit)
            bound = bnd["UB"] if mode != "min_var" else bnd["LB"]
        except KeyboardInterrupt:
            if verbose:
                print("[interrupted] keeping the cheap dual bound")

    gap = _gap(mode, _inc(best), bound)
    best["bound"] = bound
    best["gap"] = gap
    _write_solution(output_file, mode, best, bound, gap, stop_reason, elapsed, s)

    if verbose:
        inc = _inc(best)
        print(f"\nstopped: {stop_reason}  starts={s}  elapsed={elapsed:.1f}s")
        if inc is not None:
            print(f"objective={inc:.6f}  bound={bound:.6f}  gap={gap * 100:.2f}%")
        print(f"solution written to {output_file}")

    return best, histories


def bcd_with_gap(prob: Problem, mode="max_sharpe", approx=None, bound_time_limit=600,
                 **kwargs):
    """Convenience wrapper: run BCD and refine the dual bound at the end."""
    return run_bcd(prob, mode=mode, approx=approx, refine_bound=True,
                   bound_time_limit=bound_time_limit, **kwargs)


def _update_best(best, mode, obj, w, Delta):
    better = obj < best["obj"] if mode == "min_var" else obj > best["obj"]
    if better:
        best.update(obj=obj, w=np.array(w), Delta=np.array(Delta))


# --------------------------------------------------------------------------- #
# Problem construction from market data (mirrors dancing.ipynb cells 0-2)
# --------------------------------------------------------------------------- #
def build_problem(all_stocks, start, end, a):
    """Fetch data and assemble a ``Problem`` (requires network: yfinance)."""
    from utils.data import (
        yf, generate_mu, generate_Sigma, add_covariates_to_covar,
        add_covariates_to_mu, conditional_moments, load_pce, load_ffr,
    )

    T = 1 / 4
    r = (1 + a[1] / 100) ** (1 / 4) - 1
    S = np.array([float(yf.Ticker(s).history(period="1d").Open.iloc[0]) for s in all_stocks])

    pce, effr = load_pce(), load_ffr()
    mu = generate_mu(all_stocks, start, end)
    Sigma = generate_Sigma(all_stocks, start, end)
    _, sep_Sigma = add_covariates_to_covar(Sigma, all_stocks, [pce, effr], start, end)
    _, sep_mu = add_covariates_to_mu(mu, [pce, effr])
    mu, Sigma = conditional_moments(sep_mu, sep_Sigma, a)

    return Problem(mu=np.asarray(mu, float), Sigma=np.asarray(Sigma, float), S=S, r=r, T=T)


if __name__ == "__main__":
    all_stocks = ["NVDA", "AAPL", "GOOG", "AVGO", "WMT", "JPM", "LLY", "XOM", "KO", "LMT", "F"]
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2023, 1, 1)
    a = (0.00742, 5)

    # Stop on whichever comes first; leave both None to run until Ctrl-C.
    TIME_LIMIT = None      # wall-clock seconds (e.g. 3600), or None for no limit
    GAP_TOL = None         # relative duality gap (e.g. 0.01 = 1%), or None to disable
    OUTPUT_FILE = "solution.json"

    prob = build_problem(all_stocks, start, end, a)

    best, histories = run_bcd(prob, mode="max_sharpe", num_iters=5,
                              time_limit=TIME_LIMIT, gap_tol=GAP_TOL,
                              output_file=OUTPUT_FILE, refine_bound=True)

    if best["w"] is not None:
        print("\nBest Sharpe:", best["obj"])
        print("w*    :", np.round(best["w"], 4))
        print("Delta*:", np.round(best["Delta"], 3))
        print(f"Dual bound: {best['bound']:.6f}   gap: {best['gap'] * 100:.2f}%")

    # Monotonicity check: every per-start objective sequence is non-decreasing.
    bad = [i for i, h in enumerate(histories)
           if any(h[j + 1] < h[j] - 1e-6 for j in range(len(h) - 1))]
    print("non-monotone starts:", bad if bad else "none")
