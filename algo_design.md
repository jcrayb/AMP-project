# Reconciling the two BCD block solves in `dancing.ipynb`

## 1. The model (from `main(1).tex`)

The paper's final model is the max-Sharpe portfolio with per-asset put protection
(objective `eq:objective`, line 349):

$$
\max_{w,\Delta}\quad
\frac{\sum_i w_i\!\left(\mu^\Delta_i - \tfrac{P^\Delta_i}{S_i}\right) - r_f}
     {\big\|(\Sigma^\Delta)^{1/2} w\big\|_2}
\qquad\text{s.t.}\quad
w\in\mathcal W=\{w\ge 0,\ \mathbf 1^\top w=1\},\ \ \Delta\in[0.1,0.5]^n .
$$

The censored-return moments are functions of $\Delta$ (`eq:model_parameters_delta`,
lines 324–331):

$$
\mu^\Delta_i = \mu_i + \sigma_i\big[\Delta_i\,\Phi^{-1}(\Delta_i) + \phi(\Phi^{-1}(\Delta_i))\big],
$$
$$
\Sigma^\Delta_{ii} = \sigma_i^2\big[\Delta_i(\Phi^{-1}(\Delta_i))^2(1-\Delta_i)
+ \phi(\cdot)\big(-2\Delta_i\Phi^{-1}(\Delta_i)+\Phi^{-1}(\Delta_i)-\phi(\cdot)\big) -\Delta_i+1\big],
$$
$$
\Sigma^\Delta_{ij} = \sigma_i\sigma_j\rho_{ij}(1-\Delta_i)(1-\Delta_j),\quad i\neq j,
$$

with the put price $P^\Delta_i$ from Black–Scholes (`eq:black_scholes_delta`) and strike
$K_i = S_i(1+\mu_i+\sigma_i\Phi^{-1}(\Delta_i))$ (`eq:strike_price`). $\Sigma^\Delta\succ 0$
is proven in lines 282–284, so $\sqrt{w^\top\Sigma^\Delta w}$ is well-defined and the
Cholesky factor exists.

The paper notes (line 317, line 602) that **for fixed $\Delta$ the problem is an SOCP /
convex-quadratic-representable**, which is exactly what makes Block Coordinate Descent
(BCD) attractive: alternately optimize $w$ (fixing $\Delta$) and $\Delta$ (fixing $w$).

## 2. What `dancing.ipynb` does and why it oscillates

The notebook alternates two Gurobi solves:

- `solve_w_given_delta(Δ, w₀, mode)` — fix $\Delta$, optimize $w$. Precomputes
  $\mu^\Delta, P, \Sigma^\Delta$ in **numpy**, takes a **Cholesky** factor
  $L=\mathrm{chol}(\Sigma^\Delta)$, sets $z=L^\top w$, and constrains
  $\texttt{var\_norm}^2 \ge z^\top z$.
- `solve_delta_given_w(Δ₀, w)` — fix $w$, optimize $\Delta$. Builds
  $\mu^\Delta, P, \Sigma^\Delta$ as **Gurobi expressions** (no Cholesky possible since
  $\Sigma^\Delta$ is variable), with $\texttt{var\_norm\_sq}=w^\top\Sigma^\Delta w$ and
  $\texttt{var\_norm\_sq}=\texttt{var\_norm}^2$.

By BCD theory the shared objective should be **monotone non-decreasing**; instead it goes
up/down/up/down after a few iterations.

### The Cholesky difference is *not* the cause

`Lᵀw` (with $LL^\top=\Sigma^\Delta$) and the direct quadratic $w^\top\Sigma^\Delta w$
describe the **exact same set** — Cholesky is just a numerically nicer SOC representation.
Moreover **both blocks already use identical nonlinear formulas**: the logit inverse-CDF
$\alpha=-(1/1.7)\log(1/\Delta-1)$, the exact density $\phi=\exp(-\alpha^2/2)/\sqrt{2\pi}$,
and the same 101-point PWL normal CDF. So at the formula level the two blocks agree.

### The actual root causes

1. **The w-block is not actually a pure SOCP.** Its Sharpe epigraph
   $\texttt{net\_mu}\cdot w - r \ge \texttt{var\_norm}\cdot t$ contains the **bilinear
   product** $\texttt{var\_norm}\cdot t$ (two decision variables). Gurobi therefore runs
   **non-convex spatial branch-and-bound**, not the fast convex SOCP the formulation is
   supposed to be.
2. **Both blocks are non-convex spatial-B&B solves truncated by `MIPGap=1e-4` and
   `TimeLimit=100s`.** Neither is certified globally optimal, so a block can return a
   point whose objective is **below** the previous iterate. BCD's "objective only
   increases" guarantee requires each subproblem solved to the global optimum of the
   *same* objective; truncated non-convex solves break exactly that, producing the
   observed oscillation.
3. **The reported objective is two separate `m.ObjVal`s** from two independently built
   Gurobi models, never reconciled against a single ground-truth evaluator — so even
   tiny modeling/solver discrepancies surface as non-monotone numbers.

## 3. Planned changes

Confirmed scope: **pure SOCP only for the w-block**; **keep Gurobi's non-convex global
solver for the Δ-block** (it is intrinsically non-convex — log/exp/products of $\Delta$).
Approximations become a **config dict**, default exact.

### 3.1 Single source of truth: a config-driven function library

One library every code path routes through (numpy scorer, w-block numeric precompute,
Gurobi Δ-block). Each function has an **exact** and an **approx** implementation, with a
**numpy backend** (floats) and a **Gurobi backend** (`Var`/`MLinExpr`/`nlfunc`). Same
config + same input ⇒ same value in both backends — this is the mechanism that forces the
two blocks to solve an identical model.

| function | `"exact"` | `"approx"` (paper) |
|---|---|---|
| `inv_cdf` $\Phi^{-1}$ | $-(1/1.7)\log(1/\Delta-1)$ (logit; `np.log` / `nlfunc.log`) | $(4/1.7)(\Delta-0.5)$ — `eq:phi_inv_approx` |
| `pdf` $\phi$ | $\exp(-x^2/2)/\sqrt{2\pi}$ (`np.exp` / `nlfunc.exp`) | $(1-x^2/2)/\sqrt{2\pi}$ — `eq:phi_approx` |
| `cdf` $\Phi$ | PWL on 101-pt erf grid (`np.interp` / `addGenConstrPWL`) | $1/2 + x/\sqrt{2\pi}$ — `eq:Phi_approx` |

Default config: `dict(pdf="exact", cdf="exact", inv_cdf="exact")`. Existing sign/shift
conventions (e.g. the negated CDF grid $y=-(\mathrm{erf}(x/\sqrt2)/2+1/2)$ and how it
feeds the Black–Scholes put) are **preserved verbatim** — this centralizes the math, it
does not change it. With the default, results are identical to the current notebook.

### 3.2 Shared moment + objective functions

Refactor the existing `compute_mu_d` / `compute_sigma_d` (cell 7) to take `approx` and
call the library:

- `compute_moments(Δ, approx) → (net_mu, Σ_d)`, with $\texttt{net\_mu}_i = \mu^\Delta_i - P^\Delta_i/S_i$.
- `sharpe(w, Δ, approx) = (net_mu·w − r) / sqrt(w·Σ_d·w)` — **the one ground-truth
  objective** used for all printing/comparison and the monotone guard.

### 3.3 w-block → genuine convex SOCP (Charnes–Cooper homogenization)

The fractional Sharpe program over the simplex is exactly SOCP-representable. Fix $\Delta$,
precompute $a=\texttt{net\_mu}(\Delta)$, $\Sigma^\Delta$, $L=\mathrm{chol}(\Sigma^\Delta)$.
Introduce $y\in\mathbb R^n_+$, $\kappa\ge 0$:

$$
\min\ s \quad\text{s.t.}\quad
\|L^\top y\|_2 \le s,\quad
a^\top y - r\kappa = 1,\quad
\mathbf 1^\top y = \kappa,\quad
y\ge 0,\ \kappa\ge 0,
$$

then recover $w=y/\kappa$ and Sharpe $=1/s$. This **eliminates the bilinear
$\texttt{var\_norm}\cdot t$**, is a **pure convex SOCP** (single second-order cone),
solved to global optimum and fast. (`max_mu`: maximize $a^\top w$ on the simplex;
`min_var`: minimize $\|L^\top w\|$ — both convex, same library.)

### 3.4 Δ-block → keep Gurobi non-convex, but identical model

- Build $\mu^\Delta, P, \Sigma^\Delta$ via the **Gurobi backend of the same library** with
  the same `approx` dict ⇒ the same model the scorer uses.
- Keep variance as $\texttt{var\_norm\_sq}=w^\top\Sigma^\Delta w$ (w fixed ⇒ linear in the
  $\Sigma^\Delta$ vars), $\texttt{var\_norm\_sq}=\texttt{var\_norm}^2$, epigraph
  $\texttt{net\_mu}\cdot w - r \ge \texttt{var\_norm}\cdot t$, maximize $t$
  (non-convex, expected; `NonConvex=2`).
- **Warm-start $\Delta$ at the incoming value** so the current point is a feasible
  incumbent (Gurobi cannot return worse than the entry Sharpe, modulo gap/time).
- Keep `MIPGap`/`TimeLimit` configurable; tighten the gap for reliability.
- Align `max_mu` mode to maximize $\texttt{net\_mu}\cdot w$ (currently it maximizes
  $\mu^\Delta\cdot w$ without $-P/S$) so the model is identical across all modes.

### 3.5 Monotone safeguard + consistent reporting

- After **each** block solve, recompute the true Sharpe via `sharpe(w, Δ, approx)` and use
  that for printing/comparison — never the two raw `m.ObjVal`s.
- Accept a block's new point only if `sharpe` does not decrease (tiny tolerance);
  otherwise keep the previous point. Each block's feasible set contains the incoming
  point (w-block: current $w$; Δ-block: warm-started $\Delta$), so an exact solver always
  weakly improves and the guard only catches solver inexactness. **Result: the inner BCD
  sequence is provably monotone non-decreasing** — the property expected from theory.
- Keep the 50-point multi-start outer loop unchanged.

## 4. Why this fixes it (theory)

BCD/alternating maximization is monotone iff each block is maximized exactly over a
feasible set containing the current iterate, using one common objective $f$. The redesign
secures all three conditions: (i) **one** objective $f=$`sharpe(w,Δ,approx)` shared by
both blocks and the reporting; (ii) the w-block is now a convex SOCP solved to global
optimum; (iii) the Δ-block is warm-started (current point feasible) and any residual
solver inexactness is absorbed by the monotone guard. Hence
$f(w^{k+1},\Delta^k)\ge f(w^k,\Delta^k)$ and $f(w^{k+1},\Delta^{k+1})\ge f(w^{k+1},\Delta^k)$,
giving a monotone non-decreasing, convergent sequence.

## 5. Files touched

- `BCD.py` — new standalone module implementing the whole redesign:
  - `FuncLib` config-driven function library (§3.1),
  - `build_moments` / `moments_numpy` / `sharpe` / `evaluate` shared objective (§3.2),
  - `solve_w_given_delta` — Charnes–Cooper convex SOCP (§3.3),
  - `solve_delta_given_w` — Gurobi non-convex, warm-started, same library (§3.4),
  - `run_bcd` — multi-start driver with the monotone safeguard (§3.5),
  - `build_problem` / `__main__` — data setup mirroring `dancing.ipynb` cells 0–2.
  `dancing.ipynb` is left unchanged.

## 6. Verification plan

- Assert the per-iteration `sharpe()` sequence is non-decreasing across several starts
  (original failure mode gone).
- Confirm the w-block solves as a convex SOCP (no `NonConvex`, sub-second) and its
  returned Sharpe equals `sharpe(w, Δ)`.
- Confirm both blocks report the **same** `sharpe` for the same hand-off $(w,\Delta)$.
- Sanity-check final Sharpe ≈ prior runs / `model.py` (~2.4) under all-exact config.
- Toggle `approx` entries (e.g. `inv_cdf="approx"`) and confirm blocks still agree and
  stay monotone.
