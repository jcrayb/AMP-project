import gurobipy as gp
from gurobipy import GRB
from utils.data import *
import math
import warnings
import plotly.express as px
import tqdm
import datetime as dt

start_str = dt.datetime.now().strftime("%Y%m%d%H%M%S")
start = dt.datetime(year=2000, month=1, day=1)
end = dt.datetime(year=2023, month=1, day=1)

all_stocks = ["NVDA", "AAPL", 'GOOG', "AVGO", "WMT", "JPM", "LLY", "XOM", "KO", "LMT", "F"]

# CHOOSE OBJECTIVE:
# -`min_var`    : Minimize expected variance
# -`max_mu`     : Maximize expected returns
# -`max_sharpe` : Maximize expected ratio of returns to variance

mode = "max_sharpe"

n = len(all_stocks)

# COVARIATES (Inflation, risk-free rate)
#a = (0.005, 0.25)  # 2% yearly inflation, 0.25% risk free rate (represents circa 2015 "the good times")
#a = (0.022, 4.5)  # 9% yearly inflation, 4.5% risk free rate (represents post-covid inflation period, start of 2022)
a = (0.00742, 5) # 3% yearly inflation, 5% risk free rate (represents start of 2023 conditions)
#a = (0.005, 3)  # 2% yearly inflation, 3% risk free rate (represents ideal federal reserve conditions)


T = 1/4
r = (1+a[1]/100)**(1/4)-1
S = [float(yf.Ticker(s).history(period="1d").Open.iloc[0]) for s in all_stocks]

pce = load_pce()
effr = load_ffr()


mu = generate_mu(all_stocks, start, end)
Sigma = generate_Sigma(all_stocks, start, end)

new_sigma, sep_Sigma = add_covariates_to_covar(Sigma, all_stocks, [pce, effr], start, end)
new_mu, sep_mu = add_covariates_to_mu(mu, [pce, effr])

mu, Sigma = conditional_moments(sep_mu, sep_Sigma, a)

m = gp.Model()

m.setParam("MIPGap", 0.01)
m.setParam("TimeLimit", 72000)

S = [yf.Ticker(s).history(period="1d").Close.to_list()[0] for s in all_stocks]
sigma = [np.sqrt(Sigma[i, i]) for i in range(n)]

w = m.addMVar(n, lb=0, name="weight")

mu_d = m.addMVar(n, lb=-GRB.INFINITY, name="expectation")
Delta = m.addMVar(n, lb=0.1, ub=0.5, name="Delta")
P = m.addMVar(n, lb=-GRB.INFINITY, name="put price")

x_pts_CDF = np.linspace(-5, 5, 101)
y_pts_CDF = np.array([math.erf(x/np.sqrt(2))*1/2+1/2 for x in x_pts_CDF])

x_pts_PDF = np.linspace(-5, 5, 101)
y_pts_PDF = np.array([math.exp(-x**2/2)*1/np.sqrt(2*np.pi) for x in x_pts_CDF])

# COMPUTING PRICE OF PUT OPTION & TRUNCATED EXPECTATION
for i in range(n):
    Phi_inv_expr = 4/1.7*(Delta[i] - 0.5)
    
    K_expr = (sigma[i]*Phi_inv_expr + mu[i] + 1)*S[i]

    d1 = m.addVar(lb=-GRB.INFINITY, name=f"d1_{i}")
    d2 = m.addVar(lb=-GRB.INFINITY, name=f"d2_{i}")

    m.addConstr(d1 == (1 - K_expr/S[i] + T*(r + sigma[i]**2/2))/(sigma[i]*np.sqrt(T)))
    m.addConstr(d2 == d1 - sigma[i]*np.sqrt(T))

    cdf_d1_expr = m.addVar(lb=-GRB.INFINITY, name=f"cdf_d1_{i}")
    cdf_d2_expr = m.addVar(lb=-GRB.INFINITY, name=f"cdf_d2_{i}")

    cdf_d1 = m.addGenConstrPWL(d1, cdf_d1_expr, x_pts_CDF,\
                                                y_pts_CDF)
    cdf_d2 = m.addGenConstrPWL(d2, cdf_d2_expr, x_pts_CDF,\
                                                y_pts_CDF)

    d1_expr = (1 - K_expr/S[i] + T*(r + sigma[i]**2/2))/(sigma[i]*np.sqrt(T))
    d2_expr =  d1_expr - sigma[i]*np.sqrt(T)
    cdf_d1_expr = (0.5 - (1/np.sqrt(2*np.pi))*d1_expr)
    cdf_d2_expr = 0.5 - (1/np.sqrt(2*np.pi))*d2_expr
    
    phi = 1/(np.sqrt(2*np.pi))*(1-Phi_inv_expr**2/2)

    P_expr = K_expr*np.exp(-r*T)*cdf_d2_expr - S[i]*cdf_d1_expr

    m.addConstr(P[i] == P_expr)
    m.addConstr(mu_d[i] == mu[i] + sigma[i]*(Delta[i]*Phi_inv_expr + phi))

# PORTFOLIO MUST BE INVESTED
m.addConstr(gp.quicksum([wi for wi in w]) == 1)

Sigma_d = np.zeros((n, n), dtype=gp.Var)

# alpha_i = Phi^{-1}(Delta_i) via PWL on Delta_i
# x_pts_invCDF, y_pts_invCDF: PWL approximation of Phi^{-1} on (0,1)

from math import pi, sqrt

C_a = 4 / 1.7  # constant for alpha approximation
C_p = 1 / sqrt(2 * pi)  # constant for phi approximation

for i in range(n):
    alpha_i = C_a * (Delta[i] - 0.5)           # linear expression, not a variable
    phi_ai = m.addVar(lb=0, name=f"phi_{i}") # phi(alpha_i), quadratic in Delta_i
    m.addConstr(phi_ai == C_p * (1 - alpha_i * alpha_i / 2),
                name=f"phi_def_{i}")          # quadratic constraint
    
    var_i = m.addVar(lb=0, name=f"var_{i}")
    m.addConstr(
        var_i == sigma[i]**2 * (
            (1-Delta[i]) * (1 + alpha_i**2 * Delta[i])
          + alpha_i * phi_ai * (1 - 2*Delta[i])
          - phi_ai * phi_ai
        ),
        name=f"var_def_{i}"
    )
    Sigma_d[i, i] = var_i

    for j in range(i+1, n):
        rho_ij = Sigma[i, j] / np.sqrt(Sigma[i, i] * Sigma[j, j])
        sqrt_1mrho2 = np.sqrt(1 - rho_ij**2)

        alpha_j = C_a * (Delta[j] - 0.5)

        covar_ij = m.addVar(lb=-GRB.INFINITY, name=f"covar_{i}_{j}")
        m.addConstr(
            covar_ij == sigma[i] * sigma[j] * rho_ij*(1-Delta[i])*(1-Delta[j])
            , name=f"covar_def_{i}_{j}"
        )
        
        Sigma_d[i, j] = covar_ij
        Sigma_d[j, i] = covar_ij

# CONSTRAINTS TO COMPUTE RETURN/RISK RATIO EFFICIENTLY
t = m.addVar(lb=-GRB.INFINITY)
var_norm = m.addVar()
var_norm_sq = m.addVar()

m.addConstr(var_norm_sq == gp.quicksum([Sigma_d[i, i] * w[i]**2 for i in range(n)]) + \
                        2*gp.quicksum([Sigma_d[i, j] * w[i]*w[j] for i in range(n) for j in range(n) if j > i]))
m.addConstr(var_norm_sq == var_norm**2)

m.addConstr(gp.quicksum([(mu_d[i]-P[i]/S[i])*w[i] for i in range(n)]) -r >= var_norm * t)

if mode == "max_mu":   
    m.setObjective(gp.quicksum(mu_d[i]*w[i] for i in range(n)), GRB.MAXIMIZE)
elif mode == "max_sharpe":
    m.setObjective(t, GRB.MAXIMIZE)
elif mode == "min_var":
    #m.addConstr(gp.quicksum([mu_d[i]*w[i] for i in range(n)]) >=0)
    m.setObjective(var_norm, GRB.MINIMIZE)
else:
    print("CHOOSE VALID MODE")
    raise ValueError

m.optimize()
m.write(f'sol/{a}_{mode}_{start_str}.sol')

var = m.getVars()

weights = [float(v.X) for v in var if "weight" in v.VarName]
deltas = [float(v.X) for v in var if "Delta" in v.VarName]

# REMOVING WEIGHTS BELOW 0.1%
w = np.array([w if w > 0.001 else 0 for w in weights ])
w /= w.sum()

print("""
\\begin{table}[!h]
    \\centering
    \\begin{tabular}{|c|c|c|}""")

print("\\hline Ticker & Weight & Delta \\\\ \\hline")

for i, ticker in enumerate(all_stocks):
    if w[i]:
        print(f"{ticker} &  {round(w[i]*100, 2)}\\% & {round(deltas[i], 2)} \\\\ \\hline")
print("""
    \\end{tabular}
\\end{table}
""")

def is_trading_day(date):
    hist = yf.Ticker('AAPL').history(start=date, end=date + dt.timedelta(days=1))
    return True if len(hist.index) else False

dfs = []
num_quarters = 14

for i, ticker in tqdm.tqdm(enumerate(all_stocks)):
    if not weights[i]: continue
    

    df = pd.DataFrame(columns =['Open', 'price_with_put'])
    for q in range(num_quarters):
        period_start = (end + relativedelta(months=3*q))
        period_end = (end + relativedelta(months=3*(q+1), days=1))

        if period_end.weekday() >= 5:
            period_end += relativedelta(days=7-period_end.weekday())

        hist = yf.Ticker(ticker).history(start=period_start, end=period_end, interval="1d")[['Open']]

        
        hist["price_with_put"] = hist.Open - hist.Open.iloc[0] + previous_value if q else hist.Open

        
        strike = get_strike_from_delta(deltas[i], hist.Open.iloc[0]*2, period_start.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d"), r, hist.Open.iloc[0], np.sqrt(Sigma[i,i])*2)
        
        num_days = len(hist.index)

        if deltas[i] > 0.001:
            put_value = np.array([black.black_put(hist.Open.iloc[d], strike, (period_end-hist.reset_index().Date.apply(lambda x: x.replace(tzinfo=None)).iloc[d]).days/365.25, r, np.sqrt(Sigma[i,i])*2) for d in range(num_days)])
        else:
            print("NO PUT")
            put_value = np.array([0 for d in range(num_days)])

        put_value -= put_value[0]
        hist.price_with_put += put_value
        
        previous_value = hist.price_with_put.iloc[-1]
        df = hist if not len(df.index) else pd.concat([df, hist])

    df.Open *= w[i]/df.iloc[0].Open
    df.price_with_put *= w[i]/df.iloc[0].price_with_put
    dfs.append(df)

init_df = dfs[0]

for i, d in enumerate(dfs):
    if i:
        init_df = init_df.add(d, fill_value=0)
        
benchmark_ticker = 'SPY'

init_df[benchmark_ticker] = yf.Ticker(benchmark_ticker) \
                            .history(start = end, 
                                     end = end+relativedelta(months=3*(num_quarters+1))).Open

init_df /= init_df.iloc[0]/100

def get_metric(price_df, metric, start, num_quarters, r):
    #returns =  np.array([(price_df.iloc[i+1] - price_df.iloc[i])/price_df.iloc[i] for i in range(len(price_df.index)-1) if (i+1) <= len(price_df.index)])
    
    returns = np.zeros(num_quarters)
    for q in range(num_quarters):
        period_start = (start + relativedelta(months=3*q))
        period_end = (start + relativedelta(months=3*(q+1)))

        slice_df = price_df.loc[(price_df.index >= period_start) & (price_df.index < period_end)]

        returns[q] = (slice_df.iloc[-1]-slice_df.iloc[0])/slice_df.iloc[0]

    if metric == "max_mu":   
        return returns.mean()
    elif metric == "max_sharpe":
        return (returns.mean()-r)/np.std(returns)
    elif metric == "min_var":
        return np.std(returns)
    else:
        print("CHOOSE VALID MODE")
        raise ValueError
    

def remove_tz(df):
    df = df.reset_index()
    df.Date = df.Date.apply(lambda x: x.replace(tzinfo=None))
    return df.set_index("Date")


names = {
    "Open": "Portfolio",
    "price_with_put": "Put port.",
    "SPY": "Benchmark"
}

print(f"Mode: {mode}")

print("""
\\begin{table}[!h]
    \\centering
    \\begin{tabular}{|c|c|c|c|}""")

print("\\hline Asset & $\\mu$ & $\\sigma$ & SR \\\\ \\hline")

for c in init_df.columns:
    metrics = []
    for metric in ['max_mu', 'min_var', 'max_sharpe']:
        metrics.append(get_metric(remove_tz(init_df)[c], metric, end, num_quarters, r))

    print(f"{names[c]} & {' & '.join([str(round(m, 4)) for m in metrics])} \\\\ \\hline ")
print("""   \\end{tabular}
\\end{table}
""")

import plotly.graph_objects as go

red = '#f8766d'
blue = '#619cff'

style_dict = {
    'layout.plot_bgcolor': 'rgba(0, 0, 0, 0)',
    'layout.font.family': 'Times New Roman',
    'layout.xaxis.linecolor': 'black',
    'layout.xaxis.ticks': 'inside',
    'layout.xaxis.mirror': True,
    'layout.xaxis.showline': True,
    'layout.yaxis.linecolor': 'black',
    'layout.yaxis.ticks': 'inside',
    'layout.yaxis.mirror': True,
    'layout.yaxis.showline': True,
    'layout.autosize': False,
    'layout.legend.bgcolor': 'rgba(0, 0, 0, 0)',
    'layout.legend.xanchor': 'right',
    'layout.legend.x': 0.25,
    'layout.legend.font.family': 'monospace',
}

fig = go.Figure()

fig.update(**style_dict)

fig.update_layout(margin=dict(r=5, l=5, t=5, b=5))

fig.add_trace(go.Scatter(x = init_df.index, y = init_df.Open, name="Portfolio", mode='lines'))
fig.add_trace(go.Scatter(x = init_df.index, y = init_df.price_with_put, name="Portfolio w/ puts", mode='lines'))
fig.add_trace(go.Scatter(x = init_df.index, y = init_df.SPY, name="Benchmark", mode='lines'))

fig.write_image(f"graphs/{a}_{mode}.png", scale=4)
fig.show()