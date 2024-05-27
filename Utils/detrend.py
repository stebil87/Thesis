import pywt
import pandas as pd
import numpy as np

def detrend_signal_wavelet_cont(df, wavelet='db4', level=1):
    signal_data = df.drop(columns='y', errors='ignore')
    detrended_signals = []
    
    for row in signal_data.values:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        coeffs[0] = np.zeros_like(coeffs[0])  
        detrended_row = pywt.waverec(coeffs, wavelet) 
        detrended_signals.append(detrended_row[:len(row)])  
    detrended_df = pd.DataFrame(detrended_signals, columns=signal_data.columns, index=signal_data.index)
    detrended_df['y'] = df['y']
    
    return detrended_df

def detrend_signal_wavelet_linear(df, wavelet='db4', level=1):
    signal_data = df.drop(columns='y', errors='ignore')
    detrended_signals = []
    
    for row in signal_data.values:
        coeffs = pywt.wavedec(row, wavelet, level=level)
        trend = coeffs[-1]
        trend_mean = np.mean(trend)
        detrended_trend = np.linspace(trend_mean, trend_mean, len(trend))
        coeffs[-1] = detrended_trend  
        detrended_row = pywt.waverec(coeffs, wavelet)
        detrended_signals.append(detrended_row[:len(row)])
        
    detrended_df = pd.DataFrame(detrended_signals, columns=signal_data.columns, index=signal_data.index)
    detrended_df['y'] = df['y']
    
    return detrended_df
