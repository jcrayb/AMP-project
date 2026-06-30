# Exact covariance of two censored-normal returns

**Goal.** Replace the *approximation* in the Covariance section of `main(1).tex` (label `eq:covar`),

$$
\operatorname{Covar}[X_i,X_j]\;\approx\;\sigma_i\sigma_j\rho_{ij}\,[1-\Phi(\alpha_i)][1-\Phi(\alpha_j)],
$$

with the **exact** covariance of the two censored (Tobit) returns, using the truncated‑normal
moments precomputed by Kan & Robotti (2018). The result below passes the three required tests:
$\alpha_{i},\alpha_{j}\to+\infty\Rightarrow 0$; $\;\alpha_{i},\alpha_{j}\to-\infty\Rightarrow\rho_{ij}$;
$\;\rho_{ij}=0\Rightarrow 0$.

---

## 1. Main result

Keep the manuscript's notation. $X_i\sim\mathcal N(0,1)$ are the standardized returns with
$\operatorname{corr}(X_i,X_j)=\rho_{ij}$, and the censored return is

$$
R_i \;=\; r(X_i,\alpha_i)\;=\;\max(X_i,\alpha_i)\qquad(\text{manuscript }\texttt{eq:returns}).
$$

Write $\rho\equiv\rho_{ij}$ and $s\equiv\sqrt{1-\rho_{ij}^{\,2}}$. Let $\phi,\Phi$ be the standard
normal pdf/cdf, and let

$$
\phi_2(h,k;\rho)=\frac{1}{2\pi\sqrt{1-\rho^2}}\exp\!\Big(-\frac{h^2-2\rho hk+k^2}{2(1-\rho^2)}\Big),
\qquad
\Phi_2(h,k;\rho)=\Pr(Z_1\le h,\,Z_2\le k)
$$

be the bivariate normal pdf/cdf for a standardized pair with correlation $\rho$.

**Second cross‑moment of the censored pair:**

$$
\boxed{\;
\mathbb E[R_iR_j]
=\rho_{ij}\,\Phi_2(-\alpha_i,-\alpha_j;\rho_{ij})
+\alpha_i\alpha_j\,\Phi_2(\alpha_i,\alpha_j;\rho_{ij})
+\alpha_i\phi(\alpha_j)\,\Phi\!\Big(\tfrac{\alpha_i-\rho_{ij}\alpha_j}{s}\Big)
+\alpha_j\phi(\alpha_i)\,\Phi\!\Big(\tfrac{\alpha_j-\rho_{ij}\alpha_i}{s}\Big)
+(1-\rho_{ij}^{\,2})\,\phi_2(\alpha_i,\alpha_j;\rho_{ij})
\;}
$$

**Exact covariance** (off‑diagonal entry of $\Sigma^{\alpha}$). With the censored mean
$\mu^{\alpha_i}=\mathbb E[R_i]=\alpha_i\Phi(\alpha_i)+\phi(\alpha_i)$ (manuscript `eq:censored_mu`),

$$
\boxed{\;
\Sigma^{\alpha}_{ij}
=\sigma_i\sigma_j\Big(\mathbb E[R_iR_j]-\mu^{\alpha_i}\mu^{\alpha_j}\Big),
\qquad i\neq j,
\;}
$$

and the diagonal is unchanged, $\Sigma^{\alpha}_{ii}=\sigma_i^2\,(\sigma^{\alpha_i})^2$ (manuscript `eq:censored_var`).
The factor $\sigma_i\sigma_j$ rescales the standardized result to assets $Y_i=\mu_i+\sigma_iX_i$,
exactly as in the manuscript's scale/shift section.

This is **not** an approximation: a covariance depends only on the *bivariate* marginal of
$(X_i,X_j)$, so — as the task assumes — it can be computed per‑pair, independently of the other
$n-2$ dimensions. The matrix $\Sigma^{\alpha}$ assembled from these entries is the genuine
covariance matrix of the random vector $R=(R_1,\dots,R_n)$, hence automatically positive
semidefinite (strictly positive definite unless some $|\rho_{ij}|=1$).

An equivalent fully‑expanded form, convenient for reading off the limits, is
($\bar\Phi_i\equiv 1-\Phi(\alpha_i)$, $L\equiv\Phi_2(-\alpha_i,-\alpha_j;\rho)$):

$$
\frac{\Sigma^{\alpha}_{ij}}{\sigma_i\sigma_j}
=\rho L
+\alpha_i\alpha_j\big(L-\bar\Phi_i\bar\Phi_j\big)
+\alpha_j\phi(\alpha_i)\Big(\bar\Phi_j-\Phi(\tfrac{\rho\alpha_i-\alpha_j}{s})\Big)
+\alpha_i\phi(\alpha_j)\Big(\bar\Phi_i-\Phi(\tfrac{\rho\alpha_j-\alpha_i}{s})\Big)
+(1-\rho^2)\phi_2(\alpha_i,\alpha_j;\rho)-\phi(\alpha_i)\phi(\alpha_j).
\tag{$\star$}
$$

---

## 2. Derivation

### 2.1 Truncated‑normal building blocks (Kan & Robotti, 2018)

> Raymond Kan and Cesare Robotti, *On Moments of Folded and Truncated Multivariate Normal
> Distributions*, March 12, 2018. Section 4.1, the $n=2$ **lower‑truncated** case.

For $X\sim\mathcal N(\mu,\Sigma)$ truncated to $\{X\ge a\}$, with $\sigma_i=1$, KR write the
moments of the truncated vector $Z$ using $\eta=\mu-a$ and
$w_{i\cdot j}=(\eta_i-\rho_{ij}\eta_j)/\sqrt{1-\rho_{ij}^2}$:

$$
\mathbb E[Z_1]=\mu_1+\frac{\phi(\eta_1)\Phi(w_{2\cdot1})+\rho_{12}\phi(\eta_2)\Phi(w_{1\cdot2})}{\Phi_2(\eta_1,\eta_2;\rho_{12})},
$$
$$
\mathbb E[Z_1Z_2]=\mu_1\mu_2+\rho_{12}
+\frac{(\mu_2+\rho_{12}a_1)\phi(\eta_1)\Phi(w_{2\cdot1})+(\mu_1+\rho_{12}a_2)\phi(\eta_2)\Phi(w_{1\cdot2})
+(1-\rho_{12}^2)\phi_2(\eta_1,\eta_2;\rho_{12})}{\Phi_2(\eta_1,\eta_2;\rho_{12})}.
$$

**Mapping to our problem** (standardized, lower‑truncation at the breakpoint $\alpha$):

| Kan–Robotti | here |
|---|---|
| $\mu_i$ | $0$ |
| $a_i$ (lower limit) | $\alpha_i$ |
| $\eta_i=\mu_i-a_i$ | $-\alpha_i$ |
| $w_{2\cdot1}=(\eta_2-\rho\eta_1)/s$ | $(\rho\alpha_1-\alpha_2)/s$ |
| $w_{1\cdot2}=(\eta_1-\rho\eta_2)/s$ | $(\rho\alpha_2-\alpha_1)/s$ |
| $\Phi_2(\eta_1,\eta_2;\rho)=\Pr(\text{cell})$ | $\Phi_2(-\alpha_1,-\alpha_2;\rho)=\Pr(X_1>\alpha_1,X_2>\alpha_2)$ |
| $\phi(\eta_i),\,\phi_2(\eta_1,\eta_2;\rho)$ | $\phi(\alpha_i),\,\phi_2(\alpha_1,\alpha_2;\rho)$ (both even) |

Multiplying KR's *conditional* moments by the cell probability $\Phi_2(-\alpha_1,-\alpha_2;\rho)$ gives
the **partial (unconditional) moments** over the upper‑right cell
$\mathcal U=\{X_1>\alpha_1,\,X_2>\alpha_2\}$. Abbreviate
$\Phi^{\!*}_1=\Phi(\tfrac{\rho\alpha_1-\alpha_2}{s})$, $\Phi^{\!*}_2=\Phi(\tfrac{\rho\alpha_2-\alpha_1}{s})$,
$\phi_i=\phi(\alpha_i)$, $L=\Phi_2(-\alpha_1,-\alpha_2;\rho)$:

$$
\begin{aligned}
\Pr(\mathcal U)&=L,\\
\mathbb E[X_1\mathbf 1_{\mathcal U}]&=\phi_1\Phi^{\!*}_1+\rho\,\phi_2\Phi^{\!*}_2,\\
\mathbb E[X_2\mathbf 1_{\mathcal U}]&=\phi_2\Phi^{\!*}_2+\rho\,\phi_1\Phi^{\!*}_1,\\
\mathbb E[X_1X_2\mathbf 1_{\mathcal U}]&=\rho L+\rho\alpha_1\phi_1\Phi^{\!*}_1+\rho\alpha_2\phi_2\Phi^{\!*}_2+(1-\rho^2)\phi_2(\alpha_1,\alpha_2;\rho).
\end{aligned}
$$

(These four are exactly what one obtains by direct integration of the bivariate density; KR provides
them in closed form, as required by the task.)

### 2.2 From truncation to censoring (law of total expectation)

Censoring keeps the left tail as a point mass at $\alpha$ instead of discarding it, so $R_i$ equals
$X_i$ or $\alpha_i$ depending on the cell. Partition the plane by the signs of $X_1-\alpha_1$ and
$X_2-\alpha_2$ into four cells and apply the law of total expectation:

$$
\mathbb E[R_1R_2]
=\underbrace{\mathbb E[X_1X_2\mathbf 1_{\mathcal U}]}_{X_1>\alpha_1,X_2>\alpha_2}
+\alpha_2\,\underbrace{\mathbb E[X_1\mathbf 1_{X_1>\alpha_1,X_2\le\alpha_2}]}_{X_2\text{ censored}}
+\alpha_1\,\underbrace{\mathbb E[X_2\mathbf 1_{X_1\le\alpha_1,X_2>\alpha_2}]}_{X_1\text{ censored}}
+\alpha_1\alpha_2\,\underbrace{\Pr(X_1\le\alpha_1,X_2\le\alpha_2)}_{\text{both censored}} .
$$

The side cells follow from the marginal partial moment $\mathbb E[X_i\mathbf 1_{X_i>\alpha_i}]=\phi_i$:

$$
\mathbb E[X_1\mathbf 1_{X_1>\alpha_1,X_2\le\alpha_2}]=\phi_1-\mathbb E[X_1\mathbf 1_{\mathcal U}],\qquad
\Pr(X_1\le\alpha_1,X_2\le\alpha_2)=\Phi_2(\alpha_1,\alpha_2;\rho).
$$

Substituting and cancelling the $\rho$‑weighted cross terms (the coefficients of $\phi_1\Phi^{\!*}_1$
and $\phi_2\Phi^{\!*}_2$ collapse to $-\alpha_2$ and $-\alpha_1$, and $\Phi^{\!*}\to1-\Phi^{\!*}=\Phi(-\cdot)$)
yields the boxed $\mathbb E[R_iR_j]$ of §1. Finally
$\operatorname{Cov}(R_1,R_2)=\mathbb E[R_1R_2]-\mu^{\alpha_1}\mu^{\alpha_2}$, and scaling by $\sigma_i\sigma_j$
gives $\Sigma^{\alpha}_{ij}$. Form $(\star)$ is the same expression regrouped (verified algebraically
identical to machine precision, §4).

*Cross‑check:* taking $\rho\to1^-$ with $\alpha_1=\alpha_2=\alpha$ makes $X_1=X_2$, so the formula must
reduce to the manuscript's censored variance $(\sigma^{\alpha})^2$ (`eq:censored_var`) — it does (§4).

---

## 3. The three required limit tests

**T1 — $\alpha_i,\alpha_j\to+\infty$ (full censoring).** Each $R_i\to\alpha_i$ becomes deterministic, so
$\operatorname{Var}(R_i)\to0$ and hence $\operatorname{Cov}\to0$. (In $(\star)$ every term vanishes, and in
the boxed form the two $\to+\infty$ pieces $\alpha_i\alpha_j\Phi_2(\alpha_i,\alpha_j;\rho)$ and
$\mu^{\alpha_i}\mu^{\alpha_j}$ cancel.) ✔

**T2 — $\alpha_i,\alpha_j\to-\infty$ (no censoring).** Then $R_i\to X_i$, so
$\operatorname{Cov}\to\sigma_i\sigma_j\operatorname{Cov}(X_i,X_j)=\sigma_i\sigma_j\rho_{ij}$ — i.e. the
original covariance $\Sigma_{ij}$ ($=\rho_{ij}$ in standardized units). In $(\star)$: $L\to1$,
$\bar\Phi_i\to1$, $\phi_i\to0$, $\phi_2\to0$, every product with a Gaussian factor vanishes, leaving
$\rho$. ✔

**T3 — $\rho_{ij}=0$ (independence).** Then $L=\bar\Phi_i\bar\Phi_j$,
$\phi_2(\alpha_i,\alpha_j;0)=\phi_i\phi_j$, $\Phi_2(\alpha_i,\alpha_j;0)=\Phi(\alpha_i)\Phi(\alpha_j)$, and the
boxed $\mathbb E[R_iR_j]$ factors exactly as
$\big(\alpha_i\Phi(\alpha_i)+\phi_i\big)\big(\alpha_j\Phi(\alpha_j)+\phi_j\big)=\mu^{\alpha_i}\mu^{\alpha_j}$,
so $\operatorname{Cov}=0$. ✔

---

## 4. Numerical verification

Independent checks of the closed form (`scipy`, standardized units; the `scipy` bivariate CDF was
replaced by a high‑accuracy 1‑D quadrature $\Phi_2(h,k;\rho)=\int_{-\infty}^{h}\phi(x)\Phi(\tfrac{k-\rho x}{s})dx$):

| Check | Method | Result |
|---|---|---|
| Two algebraic forms (boxed vs $(\star)$) | analytic identity, grid $49\times6$ | max diff **1.4e‑15** |
| Closed form vs 2‑D quadrature of $\mathbb E[R_iR_j]$ | kink‑aware region split (ground truth) | match to **9.5e‑8** at the worst point |
| Closed form vs Monte Carlo, $N=4\times10^{7}$ | 4 spot points | match to **≤1.8e‑4** (= MC std. error) |
| **T1** $\alpha_i{=}\alpha_j{=}{+}8,\rho{=}0.6$ | formula | $-2.2\times10^{-21}\approx0$ |
| **T2** $\alpha_i{=}\alpha_j{=}{-}8$ | formula | $\rho$ to $10^{-12}$ ($0.600000$, and $-0.300000$) |
| **T3** $\rho{=}0$, all $\alpha$ | formula | $\max|\!\operatorname{Cov}|=4.5\times10^{-16}\approx0$ |
| $\rho\to1^-$ diagonal vs manuscript $(\sigma^{\alpha})^2$ (`eq:censored_var`) | formula at $\rho=1-10^{-6}$ | matches; residual $\to0$ as $\rho\to1$ |

(With `scipy`'s default approximate `multivariate_normal.cdf`, the grid‑vs‑quadrature gap was a
harmless $2.4\times10^{-5}$, traced entirely to that routine, not the formula.)

---

## 5. Drop‑in replacement for the model

In the manuscript's matrix form, replace the approximation (`eq:covar`,
$\Sigma^{\alpha}=\Sigma\circ\mathbf p\mathbf p^{\!\top}+D$) by the exact

$$
\Sigma^{\alpha}_{ij}=\sigma_i\sigma_j\big(\mathbb E[R_iR_j]-\mu^{\alpha_i}\mu^{\alpha_j}\big)\quad(i\ne j),
\qquad
\Sigma^{\alpha}_{ii}=\sigma_i^2(\sigma^{\alpha_i})^2 .
$$

Under the manuscript's unifying variable $\alpha_i=\Phi^{-1}(\Delta_i)$ this is a function of
$(\Delta_i,\Delta_j,\rho_{ij})$. Note it is **exact** and PSD by construction; the old approximation
$\sigma_i\sigma_j\rho_{ij}\bar\Phi_i\bar\Phi_j$ is recovered only in the independence‑like regime
$L\approx\bar\Phi_i\bar\Phi_j$ and drops the $\alpha$‑ and $\phi_2$‑terms above. Because it contains the
bivariate normal CDF $\Phi_2$, it is not directly Gurobi‑representable — for the optimization model it
would enter through the same kind of polynomial surrogate already used for $\Phi,\phi,\Phi^{-1}$, or via
a fixed‑$\Delta$ (block‑coordinate) outer loop as suggested in the manuscript's conclusion.
