from statsmodels.tsa.stattools import adfuller

def check_stationarity(df):
    results = {}
    for column in df.columns:
        if column != 'y':  
            result = adfuller(df[column], autolag='AIC')
            results[column] = {
                'Test Statistic': result[0],
                'p-value': result[1],
                'Lags Used': result[2],
                'Number of Observations Used': result[3],
                'Critical Values': result[4]
            }
            if result[1] < 0.05:
                results[column]['Stationarity'] = 'Stationary'
            else:
                results[column]['Stationarity'] = 'Non-stationary'
    return results

