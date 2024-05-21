import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft
from scipy.signal import find_peaks, hilbert

def feature_generation(row):
    features = {}
    
    # Convert row to numpy array to handle calculations
    data = np.asarray(row)
    
    # Basic statistical features
    features['mean'] = np.mean(data)
    features['std'] = np.std(data)
    features['var'] = np.var(data)
    features['min'] = np.min(data)
    features['max'] = np.max(data)
    features['range'] = np.max(data) - np.min(data)
    features['median'] = np.median(data)
    features['skew'] = skew(data)
    features['kurtosis'] = kurtosis(data)
    
    # Histogram entropy
    hist, _ = np.histogram(data, bins=10, density=True)
    features['entropy'] = entropy(hist)

    # FFT and energy features
    fft_vals = np.abs(fft(data))
    freqs = np.fft.fftfreq(len(data), d=1)
    features['total_energy'] = np.sum(fft_vals**2)
    low_freq_power = np.sum(fft_vals[freqs < 0.1]**2)
    high_freq_power = np.sum(fft_vals[freqs >= 0.1]**2)
    features['power_ratio_low_high'] = low_freq_power / high_freq_power

    # Peak frequencies and zero-crossing rate
    features['peak_freq_1'], features['peak_freq_2'], features['peak_freq_3'] = np.sort(fft_vals)[-3:]
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(data)

    # Mean-crossing rate
    mean_level = np.mean(data)
    mean_crossings = np.where(np.diff(np.sign(data - mean_level)))[0]
    features['mean_crossing_rate'] = len(mean_crossings) / len(data)

    # Local extrema: maxima and minima
    peaks, _ = find_peaks(data)
    troughs, _ = find_peaks(-data)
    features['local_maxima_rate'] = len(peaks) / len(data)
    features['local_minima_rate'] = len(troughs) / len(data)

    # Quantiles and interquantile ranges
    features['quantile_25'] = np.quantile(data, 0.25)
    features['quantile_50'] = np.quantile(data, 0.5)
    features['quantile_75'] = np.quantile(data, 0.75)
    features['iqr_25_75'] = features['quantile_75'] - features['quantile_25']
    features['iqr_10_90'] = np.quantile(data, 0.90) - np.quantile(data, 0.10)

    # Derivative-based features
    if len(data) > 1:
        features['first_derivative_mean'] = np.mean(np.diff(data, n=1))
        features['second_derivative_mean'] = np.mean(np.diff(data, n=2))
    else:
        features['first_derivative_mean'] = np.nan
        features['second_derivative_mean'] = np.nan

    # Signal envelope using Hilbert transform
    analytical_signal = hilbert(data)
    envelope = np.abs(analytical_signal)
    features['envelope_mean'] = np.mean(envelope)
    features['envelope_std'] = np.std(envelope)
    features['envelope_max'] = np.max(envelope)
    features['envelope_min'] = np.min(envelope)
    features['envelope_skew'] = skew(envelope)
    features['envelope_kurtosis'] = kurtosis(envelope)
    
    return features

def extract_features(df):
    signal_data = df.drop(columns='y', errors='ignore')
    all_features = []
    for _, row in signal_data.iterrows():
        row_features = feature_generation(row)
        all_features.append(row_features)
    features_df = pd.DataFrame(all_features)
    features_df['y'] = df['y'].values
    
    return features_df

