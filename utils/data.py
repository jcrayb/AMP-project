import numpy as np
import yfinance as yf
import sympy as sp
import datetime as dt
import pandas as pd
import py_vollib.ref_python.black_scholes.greeks.analytical as pyv
import py_vollib.ref_python.black as black
from dateutil.relativedelta import relativedelta

def returns(hist):
    array = [(row.Close - row.Open)/row.Open for row in hist.iloc]
    return array

def expected_returns(hist):
    array = returns(hist)
    return np.mean(array)

def stock_covar(hist1, hist2):
    common = hist1.merge(hist2, 'inner', left_index=True, right_index=True)
    common_dates = common.index
    
    hist1 = hist1.loc[(hist1.index >= common_dates[0]) & (hist1.index <= common_dates[-1])]
    hist2 = hist2.loc[(hist2.index >= common_dates[0]) & (hist2.index <= common_dates[-1])]
    
    array1 = returns(hist1)
    array2 = returns(hist2)

    return np.cov(array1, array2)

def covariates_covar(hist1, hist2, covariates):
    common = hist1.merge(hist2, 'inner', left_index=True, right_index=True)
    common_dates = common.index
    
    hist1 = hist1.loc[(hist1.index >= common_dates[0]) & (hist1.index <= common_dates[-1])]
    hist2 = hist2.loc[(hist2.index >= common_dates[0]) & (hist2.index <= common_dates[-1])]
    
    array1 = returns(hist1) if not covariates[0] else hist1.Open
    array2 = returns(hist2) if not covariates[1] else hist2.Open

    return np.cov(array1, array2)

def get_hist(ticker, start, end):
    hist = yf.Ticker(ticker).history(start=start,end=end, interval="3mo")
    new_index = hist.reset_index().Date.apply(lambda x: x.replace(tzinfo=None))
    
    if new_index[0] > start :
        first_trading_date = yf.Ticker(ticker).history(period="max", interval="1d").index[0]
        quarter_start_mo = (int((first_trading_date.month-1)/3+1)*3+1)%12
        new_start = dt.datetime(year=first_trading_date.year, month=quarter_start_mo, day=1)

        hist = yf.Ticker(ticker).history(start=new_start,end=end, interval="3mo")

    return hist

def generate_mu(all_stocks, start, end):
    return np.array([expected_returns(get_hist(stock, start, end)) for stock in all_stocks])

def generate_Sigma(all_stocks, start, end):
    n = len(all_stocks)
    covar = np.zeros((n, n))

    combs = []

    for i, t1 in enumerate(all_stocks):
        for j, t2 in enumerate(all_stocks):
            if ((t1, t2) in combs) or ((t2, t1) in combs) or t1 == t2:
                continue
            combs.append((t1, t2))
            c = stock_covar(get_hist(t1, start, end), get_hist(t2, start, end))

            covar[i, i] = c[0, 0]
            covar[i, j] = c[0, 1]
            covar[j, i] = c[1, 0]
            covar[j, j] = c[1, 1]

    return covar

def benchmark(stocks, weights, benchmark_ticker, start, end):
    dates_array = yf.Ticker(stocks[0]).history(start=start,end=end, interval="1d").index

    df = pd.DataFrame(0, index=dates_array, columns=["portfolio", "benchmark"])
    
    for i, ticker in enumerate(stocks):
        hist = yf.Ticker(ticker).history(start=start,end=end, interval="1d")

        df.portfolio += weights[i]*hist.Open/hist.iloc[0].Open*100

    df.benchmark = yf.Ticker(benchmark_ticker).history(start=start, end=end, interval="1d").Open
    df.benchmark /= df.iloc[0].benchmark
    df.benchmark *= 100

    return df

def add_covariates_to_covar(Sigma, all_stocks, covars:list, start, end):
    n1 = Sigma.shape[0] 
    n2 = n1 + len(covars)
    new_Sigma = np.zeros((n2, n2))

    new_Sigma[:n1, :n1] = Sigma
    for j, c in enumerate(covars):
        c_idx1 = j + n1

        for i, t1 in enumerate(all_stocks):
            hist1 = get_hist(t1, start, end).reset_index()
            hist1.Date = hist1.Date.apply(lambda x: x.replace(tzinfo=None))
            hist1 = hist1.set_index("Date")

            cov = covariates_covar(hist1, c, covariates=(False, True))


            new_Sigma[i, c_idx1] = cov[0, 1]
            new_Sigma[c_idx1, i] = cov[1, 0]
            new_Sigma[c_idx1, c_idx1] = cov[1, 1]

        for j2, c2 in enumerate(covars):
            if j2 > j:
                c_idx2 = j2 + n1
                cov2 = covariates_covar(c, c2, covariates=(True, True))

                new_Sigma[c_idx1, c_idx1] = cov2[0, 0]
                new_Sigma[c_idx2, c_idx1] = cov2[0, 1]
                new_Sigma[c_idx1, c_idx2] = cov2[1, 0]
                new_Sigma[c_idx2, c_idx2] = cov2[1, 1]
                                                 
    Sxx = Sigma
    Sxy = new_Sigma[:n1, n1:]
    Syx = Sxy.T
    Syy = new_Sigma[n1:, n1:]

    return new_Sigma, (Sxx, Sxy, Syx, Syy)

def add_covariates_to_mu(mu, covars:list):
    n1 = mu.shape[0] 
    n2 = n1 + len(covars)
    new_mu = np.zeros((n2))

    new_mu[:n1] = mu
    for j, c in enumerate(covars):
        c_idx1 = j + n1

        new_mu[c_idx1] = c.Open.mean()
    return new_mu, (mu, new_mu[n1:])

def conditional_moments(mu, Sigma, a):
    (Sxx, Sxy, Syx, Syy) = Sigma
    (mu_x, mu_y) = mu
    mu_a = mu_x + Sxy @ np.linalg.inv(Syy)@(a-mu_y)
    Sigma_a = Sxx - Sxy@np.linalg.inv(Syy)@Syx

    return mu_a, Sigma_a

def load_pce():
    pce = pd.read_csv('data/pce.csv')
    shifted = np.zeros(len(pce.index))
    shifted[:-1] = pce.PCECTPI[1:]
    #returns = np.zeros(len(pce.index))
    #returns[:-1] = -(pce.PCECTPI[:-1].to_numpy()-pce.PCECTPI[1:].to_numpy())/pce.PCECTPI[:-1].to_numpy()

    pce['Close'] = shifted
    pce['Open'] = pce.PCECTPI

    pce.observation_date = pce.observation_date.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
    pce = pce.set_index('observation_date')
    pce = pce[['Close', 'Open']]
    pce = pce.iloc[:-1]
    
    pce.Open = returns(pce)

    return pce[['Open']]

def load_ffr():
    df = pd.read_csv('data/effr.csv')
    df.observation_date = df.observation_date.apply(lambda x: dt.datetime.strptime(x, "%Y-%m-%d"))
    return df.set_index('observation_date')

def getGreeks(date, expiry, stockPrice, r, sigma, strike):
    flag = 'p'
    dateDt = dt.datetime.strptime(date, "%Y-%m-%d")
    expiryDt = dt.datetime.strptime(expiry, "%Y-%m-%d")

    t = (expiryDt-dateDt).days/365.25

    delta = pyv.delta(flag, stockPrice, strike, t, r, sigma)
    gamma = pyv.gamma(flag, stockPrice, strike, t, r, sigma)
    theta = pyv.theta(flag, stockPrice, strike, t, r, sigma)
    vega = pyv.vega(flag, stockPrice, strike, t, r, sigma)
    rho  = pyv.rho(flag, stockPrice, strike, t, r, sigma)
    return delta, gamma, vega, theta, rho

def get_delta(date, expiry, stockPrice, r, sigma, strike):
    flag = 'p'
    dateDt = dt.datetime.strptime(date, "%Y-%m-%d")
    expiryDt = dt.datetime.strptime(expiry, "%Y-%m-%d")

    t = (expiryDt-dateDt).days/365.25
    return pyv.delta(flag, stockPrice, strike, t, r, sigma)

def quarter_start(start):
    return yf.Ticker("AAPL").history(start=start, interval="1d").index[0]

def quarter_end(start):
    end = start+relativedelta(months=3)
    return yf.Ticker("AAPL").history(end=end, interval="1d").index[-1]

def get_strike_from_delta(target_delta, initial_guess, qs, qe, r, stock_price, sigma):
    x1, x2 = (0, initial_guess)
    finished = False
    
    tol = 1e-3
    while not finished:
        midpoint = (x1+x2)/2
        
        delta = -get_delta(qs, qe, stock_price, r, sigma, midpoint)
        
        if target_delta-delta > 0:
            x1 = midpoint
        else:
            x2 = midpoint
    
        if abs(target_delta - delta) < tol: finished= True
    return x1/2+x2/2