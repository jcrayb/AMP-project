import numpy as np
import yfinance as yf
import sympy as sp
import datetime as dt
import pandas as pd

def returns(hist):
    array = [(row.Close - row.Open)/row.Open for row in hist.iloc]
    return array

def expected_returns(hist):
    array = returns(hist)
    return np.mean(array)

def stock_covar(hist1, hist2):
    common = hist1.merge(hist2, 'inner', left_on=hist1.index, right_on=hist2.index)
    common_dates = common.key_0
    
    hist1 = hist1.loc[hist1.index >= common_dates[0]]
    hist2 = hist2.loc[hist2.index >= common_dates[0]]
    
    array1 = returns(hist1)
    array2 = returns(hist2)

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
