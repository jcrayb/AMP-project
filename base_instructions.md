Find attached the manuscript for our project in `main(1).tex`. For, now we are using an approximation to the true covariance of the multivariate censored normal distribution. I want you to find the true formula. Since part of this exercise is comprised of the moments of the truncated normal distribution, find those precomputed in this paper: https://www-2.rotman.utoronto.ca/~kan/papers/kr.pdf. Assume that even in an n-dimensional MVN, the covariance of two elements can be computed independently of all the other dimensions. Maintain the same nomenclature as the original paper (eg. \alpha for the truncation level, etc.), and check your output vs the following tests:
- \alpha_1 & 2 = +\infty =>  covar = 0
- \alpha_1 & 2 = -\infty =>  covar = \rho_{12}
- \rho = 0 => covar = 0
