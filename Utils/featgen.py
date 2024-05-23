import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft
from scipy.signal import find_peaks, hilbert

def calculate_features(data, prefix=""):
    features = {}
    # Basic statistical features
    features[prefix + 'mean'] = np.mean(data)
    features[prefix + 'std'] = np.std(data)
    features[prefix + 'var'] = np.var(data)
    features[prefix + 'min'] = np.min(data)
    features[prefix + 'max'] = np.max(data)
    features[prefix + 'range'] = np.max(data) - np.min(data)
    features[prefix + 'median'] = np.median(data)
    features[prefix + 'skew'] = skew(data)
    features[prefix + 'kurtosis'] = kurtosis(data)
    
    # Histogram entropy
    hist, _ = np.histogram(data, bins=10, density=True)
    features[prefix + 'entropy'] = entropy(hist)

    # FFT and energy features
    fft_vals = np.abs(fft(data))
    freqs = np.fft.fftfreq(len(data), d=1)
    features[prefix + 'total_energy'] = np.sum(fft_vals**2)
    low_freq_power = np.sum(fft_vals[freqs < 0.1]**2)
    high_freq_power = np.sum(fft_vals[freqs >= 0.1]**2)
    features[prefix + 'power_ratio_low_high'] = low_freq_power / high_freq_power

    # Peak frequencies and zero-crossing rate
    features[prefix + 'peak_freq_1'], features[prefix + 'peak_freq_2'], features[prefix + 'peak_freq_3'] = np.sort(fft_vals)[-3:]
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    features[prefix + 'zero_crossing_rate'] = len(zero_crossings) / len(data)

    # Mean-crossing rate
    mean_level = np.mean(data)
    mean_crossings = np.where(np.diff(np.sign(data - mean_level)))[0]
    features[prefix + 'mean_crossing_rate'] = len(mean_crossings) / len(data)

    # Local extrema: maxima and minima
    peaks, _ = find_peaks(data)
    troughs, _ = find_peaks(-data)
    features[prefix + 'local_maxima_rate'] = len(peaks) / len(data)
    features[prefix + 'local_minima_rate'] = len(troughs) / len(data)

    # Quantiles and interquantile ranges
    features[prefix + 'quantile_25'] = np.quantile(data, 0.25)
    features[prefix + 'quantile_50'] = np.quantile(data, 0.5)
    features[prefix + 'quantile_75'] = np.quantile(data, 0.75)
    features[prefix + 'iqr_25_75'] = features[prefix + 'quantile_75'] - features[prefix + 'quantile_25']
    features[prefix + 'iqr_10_90'] = np.quantile(data, 0.90) - np.quantile(data, 0.10)

    # Signal envelope using Hilbert transform
    analytical_signal = hilbert(data)
    envelope = np.abs(analytical_signal)
    features[prefix + 'envelope_mean'] = np.mean(envelope)
    features[prefix + 'envelope_std'] = np.std(envelope)
    features[prefix + 'envelope_max'] = np.max(envelope)
    features[prefix + 'envelope_min'] = np.min(envelope)
    features[prefix + 'envelope_skew'] = skew(envelope)
    features[prefix + 'envelope_kurtosis'] = kurtosis(envelope)

    return features

def feature_generation(row, window_size=20):
    data = np.asarray(row)
    features = calculate_features(data, prefix="global_")
    
    windowed_features = []
    for start in range(0, len(data), window_size):
        end = start + window_size
        if end > len(data):  
            end = len(data)
        window_data = data[start:end]
        window_features = calculate_features(window_data, prefix=f"win_{start}_{end}_")
        windowed_features.append(window_features)

    
    for window_feature in windowed_features:
        for key, value in window_feature.items():
            features[key] = value
    
    return features

def extract_features(df, window_size=20):
    signal_data = df.drop(columns='y', errors='ignore')
    all_features = []
    for _, row in signal_data.iterrows():
        row_features = feature_generation(row, window_size)
        all_features.append(row_features)
    features_df = pd.DataFrame(all_features)
    features_df['y'] = df['y'].values
    
    return features_df




