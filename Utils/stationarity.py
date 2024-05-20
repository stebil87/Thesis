from statsmodels.tsa.stattools import kpss

def check_combined_stationarity_kpss(df):
    signal_data = df.drop(columns='y', errors='ignore')
    combined_signal = signal_data.values.flatten()
    result = kpss(combined_signal, regression='c', nlags='auto')
    results = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Critical Values': result[3]
    }
    
    if result[1] > 0.05:
        results['Stationarity'] = 'Stationary'
    else:
        results['Stationarity'] = 'Non-stationary'
    
    return results
