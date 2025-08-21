import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd
import numpy as np




# Question 1 
data = pd.read_csv('Individual_Project_Data.csv', 
                  parse_dates=['Date'], 
                  dayfirst=True)
data.set_index('Date', inplace=True)

start_date = pd.to_datetime('16/11/2000', dayfirst=True)
end_date = pd.to_datetime('16/11/2023', dayfirst=True)
data = data[(data.index >= start_date) & (data.index <= end_date)]



def OLShac(y, X, L=None):
    y = np.asarray(y).flatten()
    X = np.asarray(X)
    
    if len(y) != X.shape[0]:
        raise ValueError('Number of observations in y and X must match')

    T, K = X.shape
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    yhat = X @ beta
    res = y - yhat
    sigeps2 = (res.T @ res) / (T - K)

    
    se = np.sqrt(np.diag(sigeps2 * np.linalg.inv(X.T @ X)))

    
    Xe = X * res.reshape(-1, 1)  
    Var_white = np.linalg.inv(X.T @ X) @ (Xe.T @ Xe) @ np.linalg.inv(X.T @ X)
    se_white = np.sqrt(np.diag(Var_white))

    
    if L is None:
        L = int(4 * (T/100)**(2/9))  

    V = np.zeros((K, K))
    for j in range(L + 1):
        Gamma = np.zeros((K, K))
        for t in range(j, T):
            Gamma += res[t] * res[t-j] * np.outer(X[t], X[t-j])
        if j > 0:
            V += (1 - j/(L+1)) * (Gamma + Gamma.T)
        else:
            V += Gamma

    Var_NW = np.linalg.inv(X.T @ X) @ V @ np.linalg.inv(X.T @ X)
    se_nw = np.sqrt(np.diag(Var_NW))

    # Test statistics
    tstat = beta / se
    tstat_white = beta / se_white
    tstat_nw = beta / se_nw

    pval_t = 2 * (1 - stats.t.cdf(np.abs(tstat), T-K))
    pval_wh = 2 * (1 - stats.t.cdf(np.abs(tstat_white), T-K))
    pval_nw = 2 * (1 - stats.t.cdf(np.abs(tstat_nw), T-K))

    # Goodness-of-fit
    rmse = np.linalg.norm(res) / np.sqrt(T-K)
    s2 = rmse**2
    RSS = np.linalg.norm(yhat - np.mean(y))**2
    F = (RSS / (K-1)) / s2
    PvalueF = 1 - stats.f.cdf(F, K-1, T-K)

    yc = y - np.mean(y)
    R2 = 1.0 - (res.T @ res) / (yc.T @ yc)
    R2adj = 1.0 - (((1-R2)*(T-1)) / (T-K))
    
    loglik = -T/2 * np.log(2*np.pi*s2) - np.sum(res**2) / (2*s2)

    return {
        'beta': beta, 'yhat': yhat, 'res': res,
        'se': se, 'se_white': se_white, 'se_nw': se_nw,
        'tstat': tstat, 'tstat_wh': tstat_white, 'tstat_nw': tstat_nw,
        'pval': pval_t, 'pval_wh': pval_wh, 'pval_nw': pval_nw,
        'Fstat': F, 'PvalueF': PvalueF,
        'R2': R2, 'R2adj': R2adj,
        'loglik': loglik, 'sig2res': s2,
        'vcv_nw': Var_NW  # For Wald tests
    }

def test_unbiasedness(rv, vix2, transformation=None):

    rv = np.asarray(rv)
    vix2 = np.asarray(vix2)
    
    if transformation == 'sqrt':
        y = np.sqrt(rv)
        x = np.sqrt(vix2)
    elif transformation == 'log':
        y = np.log(rv + 1e-6)  
        x = np.log(vix2 + 1e-6)
    else:
        y = rv
        x = vix2
    
   
    mask = np.isfinite(y) & np.isfinite(x)
    y = y[mask]
    x = x[mask]
    
    # Regression
    X = np.column_stack((np.ones(len(x)), x))
    results = OLShac(y, X, 10)
    
    # results
    alpha = results['beta'][0]
    beta = results['beta'][1]
    alpha_se = results['se_nw'][0]
    beta_se = results['se_nw'][1]
    vcv = results['vcv_nw']
    
    # Hypothesis tests
    t_alpha = (alpha - 0)/alpha_se
    t_beta = (beta - 1)/beta_se
    pval_alpha = 2*(1 - stats.norm.cdf(abs(t_alpha)))
    pval_beta = 2*(1 - stats.norm.cdf(abs(t_beta)))
    
    # Joint test
    R = np.array([[1, 0], [0, 1]])
    r = np.array([0, 1])
    diff = R @ results['beta'] - r
    W = diff.T @ np.linalg.inv(R @ vcv @ R.T) @ diff
    pval_joint = 1 - stats.chi2.cdf(W, 2)
    
    return {
        'alpha': alpha, 'beta': beta,
        't_alpha': t_alpha, 't_beta': t_beta,
        'pval_alpha': pval_alpha, 'pval_beta': pval_beta,
        'Wald_stat': W, 'pval_joint': pval_joint,
        'transformation': transformation
    }

def main():
    data = pd.read_csv('Individual_Project_Data.csv', 
                      parse_dates=['Date'], 
                      dayfirst=True)
    data.set_index('Date', inplace=True)
    
    
    start_date = pd.to_datetime('16/11/2000', dayfirst=True)
    end_date = pd.to_datetime('16/11/2023', dayfirst=True)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    
    
    transformations = [None, 'sqrt', 'log']
    results = []
    
    for trans in transformations:
        res = test_unbiasedness(data['RV'].values, data['VIX2'].values, trans)
        results.append(res)
    
    # Display results
    print("\nUnbiasedness Test Results (16/11/2000 - 16/11/2023)")
    print("="*70)
    for res in results:
        trans = res['transformation'] if res['transformation'] else 'None'
        print(f"\nTransformation: {trans}")
        print("-"*50)
        print(f"α estimate: {res['alpha']:.4f} (t-test for α=0: {res['t_alpha']:.2f}, p-value: {res['pval_alpha']:.4f})")
        print(f"β estimate: {res['beta']:.4f} (t-test for β=1: {res['t_beta']:.2f}, p-value: {res['pval_beta']:.4f})")
        print(f"Joint test (α=0 & β=1): Wald stat = {res['Wald_stat']:.2f}, p-value = {res['pval_joint']:.4f}")
        
        # Significance interpretation
        print("\nSignificance at:")
        for level in [0.01, 0.05]:
            sig_alpha = "✓" if res['pval_alpha'] < level else "✗"
            sig_beta = "✓" if res['pval_beta'] < level else "✗"
            sig_joint = "✓" if res['pval_joint'] < level else "✗"
            print(f"{level*100:.0f}% level: α=0 {sig_alpha} | β=1 {sig_beta} | Joint {sig_joint}")

if __name__ == "__main__":
    main()

log_vix2 = np.log(data['VIX2'] + 1e-6)
log_rv = np.log(data['RV'] + 1e-6)


slope, intercept = np.polyfit(log_vix2, log_rv, 1)
fit_line = slope * log_vix2 + intercept

plt.figure(figsize=(8, 5))
plt.scatter(log_vix2, log_rv, alpha=0.5, label='Log-Transformed Data', color='skyblue')
plt.plot(log_vix2, fit_line, color='red', label=fr'Fit: $y = {slope:.2f}x + {intercept:.2f}$')

plt.title('Q1: Scatterplot of log(VIX²) vs log(RV) with Regression Line')
plt.xlabel('log(VIX²)')
plt.ylabel('log(RV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

    
    

    
    
    
#Question 2


def calculate_summary_stats_and_normality(data, name):
    """Calculate summary statistics and Jarque-Bera test for normality"""
    # Summary statistics
    stats_dict = {
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std Dev': np.std(data),
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data, fisher=False), 
        'JB Stat': None,
        'JB p-value': None,
        'Normal (5%)': None
    }
    
    # Jarque-Bera test 
    jb_stat, jb_pvalue = stats.jarque_bera(data)
    stats_dict['JB Stat'] = jb_stat
    stats_dict['JB p-value'] = jb_pvalue
    stats_dict['Normal (5%)'] = 'Yes' if jb_pvalue > 0.05 else 'No'
    
    return pd.DataFrame(stats_dict, index=[name])

def main():
    
    data = pd.read_csv('Individual_Project_Data.csv', 
                      parse_dates=['Date'], 
                      dayfirst=True)
    data.set_index('Date', inplace=True)
    
    
    start_date = pd.to_datetime('16/11/2000', dayfirst=True)
    end_date = pd.to_datetime('16/11/2023', dayfirst=True)
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    

    results = []
    
    
    results.append(calculate_summary_stats_and_normality(data['RV'], 'RV'))
    results.append(calculate_summary_stats_and_normality(data['VIX2'], 'VIX2'))
    
    # Square root transformed
    results.append(calculate_summary_stats_and_normality(np.sqrt(data['RV'] + 1e-6), '√RV'))
    results.append(calculate_summary_stats_and_normality(np.sqrt(data['VIX2'] + 1e-6), '√VIX2'))
    
    # Log transformed )
    results.append(calculate_summary_stats_and_normality(np.log(data['RV'] + 1e-6), 'log(RV)'))
    results.append(calculate_summary_stats_and_normality(np.log(data['VIX2'] + 1e-6), 'log(VIX2)'))
    
    # Combine results
    summary_table = pd.concat(results)
    
    # results
    print("="*70)
    print("Summary Statistics and Normality Tests (16/11/2000 - 16/11/2023)")
    print("="*70)
    print(summary_table.round(4).to_string())

    


if __name__ == "__main__":
    main()
    


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(log_rv, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of log(RV)')
plt.xlabel('log(RV)')
plt.ylabel('Frequency')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(log_vix2, bins=50, color='salmon', edgecolor='black')
plt.title('Distribution of log(VIX²)')
plt.xlabel('log(VIX²)')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()








 
    
#Questions 3
import numpy as np
import pandas as pd
from scipy import stats

def OLShac(y, X, L):
    y = np.asarray(y).flatten()
    X = np.asarray(X)
    T, K = X.shape
    
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    res = y - X @ beta
    
    V = np.zeros((K, K))
    for j in range(L + 1):
        Gamma = np.zeros((K, K))
        for t in range(j, T):
            Gamma += res[t] * res[t-j] * np.outer(X[t], X[t-j])
        V += (1 - j/(L+1))*(Gamma + Gamma.T) if j > 0 else Gamma
    
    vcv = np.linalg.inv(X.T @ X) @ V @ np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(vcv))
    tstat = beta / se
    
    # R-squared calculation
    yhat = X @ beta
    R2 = 1 - (res.T @ res)/((y - y.mean()).T @ (y - y.mean()))
    R2adj = 1 - (1-R2)*(T-1)/(T-K)
    
    return {'beta': beta, 'se': se, 'tstat': tstat, 'R2adj': R2adj, 'vcv': vcv}
def estimate_har(data, transformation=None):
    
    if transformation == 'sqrt':
        transformed = lambda x: np.sqrt(data[x] + 1e-6)
    elif transformation == 'log':
        transformed = lambda x: np.log(data[x] + 1e-6)
    else:
        transformed = lambda x: data[x]


    df = pd.DataFrame({
        'y': transformed('RV').shift(-1),  
        'd': transformed('RV').shift(0),
        'w': transformed('RV').rolling(5).mean().shift(1),
        'm': transformed('RV').rolling(22).mean().shift(1),
        'rsp': transformed('RSP').shift(1),
        'rsn': transformed('RSN').shift(1),
    })

    df = df.dropna()

    # HAR model
    X_har = df[['d', 'w', 'm']].values
    y = df['y'].values

    # SHAR model
    X_shar = df[['rsp', 'rsn', 'w', 'm']].values

    models = {
        'HAR': X_har,
        'SHAR': X_shar
    }

    results = []
    for name, X in models.items():
        X_full = np.column_stack([np.ones(len(X)), X])
        res = OLShac(y, X_full, 10)

        
        coeffs = list(res['beta']) + [np.nan] * (4 - len(res['beta']) + 1)
        tstats = list(res['tstat']) + [np.nan] * (4 - len(res['tstat']) + 1)

        results.append({
            'Model': f"{name}-{transformation}" if transformation else name,
            **dict(zip(['β0','βd','βw','βm'], coeffs)),
            **dict(zip(['t_β0','t_βd','t_βw','t_βm'], tstats)),
            'R2adj': res['R2adj']
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    data = pd.read_csv('Individual_Project_Data.csv', parse_dates=['Date'], dayfirst=True)
    data = data.set_index('Date')['2000-11-16':'2023-11-16']

   
    har_results = estimate_har(data)

   
    har_sqrt = estimate_har(data, 'sqrt')

   
    har_log = estimate_har(data, 'log')

    # Combine  results
    full_results = pd.concat([har_results, har_sqrt, har_log])
    

    print("\nFinal Results:\n")
    print(full_results.round(4))
    
    
    
    
# Visualisation
rv_series = data['RV']
har_fitted = rv_series.rolling(5).mean().shift(1)

plt.figure(figsize=(12, 5))
plt.plot(data.index, data['RV'], label='Realised Variance (RV)', color='steelblue')
plt.plot(data.index, har_fitted, label='HAR-log Fitted (approx)', linestyle='--', color='darkorange')

plt.title('Q3: In-Sample Realised Variance vs Fitted (HAR-log)')
plt.xlabel('Date')
plt.ylabel('Variance (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

 
    




# Question 4 

#  MZ_Test 
def MZ_Test(y, yhat, Lags, alpha, Plot):
    T = len(y)
    X = np.column_stack((np.ones(T), yhat))
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    res = y - (X @ beta)
    V = np.zeros((2, 2))
    for j in range(Lags + 1):
        Gamma = np.zeros((2, 2))
        for t in range(j, T):
            Gamma += res[t] * res[t - j] * np.outer(X[t], X[t - j])
        V += (1 - j / (Lags + 1)) * (Gamma + Gamma.T) if j > 0 else Gamma
    Var_NW = np.linalg.inv(X.T @ X) @ V @ np.linalg.inv(X.T @ X)
    se_nw = np.sqrt(np.diag(Var_NW))
    tstat_nw = (beta - np.array([0, 1])) / se_nw
    pval_nw = 2 * (1 - stats.t.cdf(np.abs(tstat_nw), T - 2))
    SSR = np.sum(res ** 2)
    SSRr = np.sum((y - yhat) ** 2)
    Ftest = ((SSRr - SSR) / 2) / (SSR / (T - 2))
    PvalueF = 1 - stats.f.cdf(Ftest, 2, T - 2)
    R2 = 1.0 - np.sum(res ** 2) / np.sum((y - np.mean(y)) ** 2)
    return beta[1], pval_nw[1], SSR / T, R2

#  VaRStats 
def VaRStats(Rt, VaR, alpha):
    T = len(Rt)
    Failures = np.sum(Rt < VaR)
    VaR_level = 1 - alpha
    Expected = T * alpha
    return {
        'Total Obs': T,
        'VaR Level': VaR_level,
        'Observed Level': (T - Failures) / T,
        'Failures': Failures,
        'Expected Failures': Expected,
        'Failure Rate': Failures / T,
        'Excess Failures': Failures - Expected,
        'Excess Ratio': Failures / Expected
    }

#  PoFTest 
def PoFTest(Rt, VaR, VaRsigLevel, alpha):
    p = VaRsigLevel
    T = len(VaR)
    x = np.sum((Rt < VaR))
    if x == 0:
        loglikelihood_null = T * np.log(1 - p)
        loglikelihood_alt = 0
    elif x == T:
        loglikelihood_null = T * np.log(p)
        loglikelihood_alt = 0
    else:
        observed_ratio = x / T
        loglikelihood_null = (T - x) * np.log(1 - p) + x * np.log(p)
        loglikelihood_alt = (T - x) * np.log(1 - observed_ratio) + x * np.log(observed_ratio)
    LR = -2 * (loglikelihood_null - loglikelihood_alt)
    p_value = 1 - stats.chi2.cdf(LR, 1)
    return {'PoF LR': LR, 'PoF p-value': p_value, 'Violations': x, 'Rate': x / T}

#  TL test
def TL(Rt, VaR, alpha):
    x = np.sum(Rt < VaR)
    T = len(Rt)
    TL_val = stats.binom.cdf(x, T, alpha)
    if TL_val <= 0.95:
        TrafficLight = 'Green'
    elif TL_val <= 0.9999:
        TrafficLight = 'Yellow'
    else:
        TrafficLight = 'Red'
    return {'TL Statistic': TL_val, 'TrafficLight': TrafficLight}

#  Rolling Forecast  
def optimized_rolling_forecast(data, model_type, transformation, window=1500):
    forecasts, actuals = [], []
    df = data.copy()
    if transformation == 'sqrt':
        df['RV_t'] = np.sqrt(df['RV'] + 1e-6)
        df['RSP_t'] = np.sqrt(df['RSP'] + 1e-6)
        df['RSN_t'] = np.sqrt(df['RSN'] + 1e-6)
        inv = lambda x: x**2
    elif transformation == 'log':
        df['RV_t'] = np.log(df['RV'] + 1e-6)
        df['RSP_t'] = np.log(df['RSP'] + 1e-6)
        df['RSN_t'] = np.log(df['RSN'] + 1e-6)
        inv = lambda x: np.exp(x)
    else:
        df['RV_t'] = df['RV']
        df['RSP_t'] = df['RSP']
        df['RSN_t'] = df['RSN']
        inv = lambda x: x

    df['RV_w'] = df['RV_t'].rolling(5).mean().shift(1)
    df['RV_m'] = df['RV_t'].rolling(22).mean().shift(1)
    df['RSP_lag'] = df['RSP_t'].shift(1)
    df['RSN_lag'] = df['RSN_t'].shift(1)
    df['y'] = df['RV_t'].shift(-1)

    for i in range(window, len(df) - 1):
        train = df.iloc[i - window:i].dropna()
        test = df.iloc[i]

        if model_type == 'HAR':
            X = train[['RV_t', 'RV_w', 'RV_m']].values
            xt = np.array([1, test['RV_t'], test['RV_w'], test['RV_m']])
        elif model_type == 'SHAR':
            X = train[['RSP_lag', 'RSN_lag', 'RV_w', 'RV_m']].values
            xt = np.array([1, test['RSP_lag'], test['RSN_lag'], test['RV_w'], test['RV_m']])
        else:
            raise ValueError("Invalid model type")

        X = np.column_stack([np.ones(len(X)), X])
        y = train['y'].values

        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            forecast = np.dot(beta, xt)
            forecasts.append(inv(forecast))
            actuals.append(df['RV'].iloc[i + 1])
        except np.linalg.LinAlgError:
            continue

    return np.array(forecasts), np.array(actuals)

#  Forecasts & Backtests 
data = pd.read_csv('Individual_Project_Data.csv', parse_dates=['Date'], dayfirst=True)
data.set_index('Date', inplace=True)
data = data['2000-11-16':'2023-11-16']
har_log_forecast, actuals = optimized_rolling_forecast(data, 'HAR', 'log')
shar_sqrt_forecast, _ = optimized_rolling_forecast(data, 'SHAR', 'sqrt')
model_avg_forecast = (har_log_forecast + shar_sqrt_forecast) / 2
returns = data['Rt'].iloc[-len(actuals):].values

# Evaluate Forecasts 
def evaluate_forecasts(actuals, forecasts, labels):
    results = []
    for fcast, label in zip(forecasts, labels):
        beta, pval, mse, r2 = MZ_Test(actuals, fcast, Lags=10, alpha=0.05, Plot=False)
        qlike = np.mean(np.log(fcast + 1e-6) + actuals / (fcast + 1e-6))
        results.append([label, beta, pval, mse, qlike, r2])
    df = pd.DataFrame(results, columns=["Model", "MZ Beta", "MZ p-value", "MSE", "QLIKE", "R²"])
    print("\n=== Q4 Forecast Evaluation Summary ===")
    print(df.round(4).to_string(index=False))
    return df

forecasts = [har_log_forecast, shar_sqrt_forecast, model_avg_forecast]
labels = ['HAR-log', 'SHAR-sqrt', 'Model Avg']
eval_df = evaluate_forecasts(actuals, forecasts, labels)


plt.figure(figsize=(14, 5))
plt.plot(actuals, label='Actual RV', color='black')
plt.plot(har_log_forecast, label='Forecast: HAR-log', linestyle='--', color='steelblue')
plt.plot(shar_sqrt_forecast, label='Forecast: √SHAR', linestyle='--', color='darkorange')
plt.plot(model_avg_forecast, label='Forecast: Model Avg', linestyle='--', color='green')

plt.title('Q4: Out-of-Sample Forecast vs Actual Realised Variance')
plt.xlabel('Days (Rolling Forecast Period)')
plt.ylabel('RV (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()







#  Backtest VaR Forecasts (Q5) 
def construct_var(rv_forecast, alpha):
    z = stats.norm.ppf(alpha)
    return z * np.sqrt(rv_forecast)

print("\n=== Q5 VaR Backtest Summary ===")
var_1_list = [construct_var(f, 0.01) for f in forecasts]
var_5_list = [construct_var(f, 0.05) for f in forecasts]
var_all = var_1_list + var_5_list
var_labels = ['1% HAR-log', '1% SHAR-sqrt', '1% Avg', '5% HAR-log', '5% SHAR-sqrt', '5% Avg']
var_levels = [0.01]*3 + [0.05]*3

for VaR, label, alpha in zip(var_all, var_labels, var_levels):
    stats_result = VaRStats(returns, VaR, alpha)
    pof_result = PoFTest(returns, VaR, alpha, 0.05)
    tl_result = TL(returns, VaR, alpha)
    print(f"\nBacktesting {label}:")
    for k, v in stats_result.items(): print(f"{k}: {v}")
    for k, v in pof_result.items(): print(f"{k}: {v}")
    for k, v in tl_result.items(): print(f"{k}: {v}")


#  Visualisation
from scipy.stats import norm

returns = data['Rt'].iloc[-4276:].values  
z = norm.ppf(0.01)
var_1 = z * np.sqrt(har_fitted.iloc[-4276:])

plt.figure(figsize=(12, 5))
plt.plot(data.index[-len(returns):], returns, color='black', label='Returns')
plt.plot(data.index[-len(returns):], var_1_list[0], linestyle='--', color='red', label='VaR 1% (HAR-log)')

plt.title('Q5: Returns and 1% VaR Boundary (HAR-log)')
plt.xlabel('Date')
plt.ylabel('Daily Return (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

