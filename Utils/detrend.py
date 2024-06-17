import pywt
import pandas as pd
import numpy as np

# Function for continuous wavelet transform detrending with batch processing
def detrend_signal_wavelet_cont(df, wavelet='cmor1.5-1.0', scales=None, batch_size=100):
    signal_data = df.drop(columns='y', errors='ignore')
    detrended_signals = []

    for start in range(0, len(signal_data), batch_size):
        end = min(start + batch_size, len(signal_data))
        batch = signal_data.values[start:end]
        for row in batch:
            coefs, _ = pywt.cwt(row, scales, wavelet)
            detrended_row = coefs.mean(axis=0)
            detrended_signals.append(detrended_row[:len(row)])  # Adjust length if needed
    
    detrended_df = pd.DataFrame(detrended_signals, columns=signal_data.columns, index=signal_data.index)
    detrended_df['y'] = df['y']

    return detrended_df

# Function for discrete wavelet transform detrending
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

# Function to apply the wavelet transform to all dictionaries
def detrend_all_dataframes(cleaned_dataframes, wavelet_type, scales=None, batch_size=100):
    detrended_dataframes = {}
    for name, df in cleaned_dataframes.items():
        if wavelet_type == 'continuous':
            detrended_dataframes[name] = {}
            for scale in scales:
                try:
                    detrended_dataframes[name][f'scale_{scale}'] = detrend_signal_wavelet_cont(df, wavelet='cmor1.5-1.0', scales=[scale], batch_size=batch_size)
                except ValueError as e:
                    print(f"Skipping scale {scale} for {name} due to error: {e}")
        elif wavelet_type == 'discrete':
            detrended_dataframes[name] = detrend_signal_wavelet_linear(df)
    return detrended_dataframes

# Defining scales for continuous wavelet transform
frequencies = [0.1, 0.3, 0.5]  # Define frequencies
scales = pywt.scale2frequency('cmor1.5-1.0', frequencies) 