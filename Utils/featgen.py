import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft
from scipy.signal import find_peaks, hilbert

def calculate_features(data, prefix=""):
    features = {}
    
    def cap_extreme_values(value, lower_bound=-1e10, upper_bound=1e10):
        if np.isinf(value) or np.isnan(value):
            return 0
        return max(min(value, upper_bound), lower_bound)
    
    def safe_stat(func, data, default=0):
        try:
            return func(data)
        except Exception:
            return default

    # Basic statistical features
    features[prefix + 'mean'] = cap_extreme_values(safe_stat(np.mean, data))
    features[prefix + 'std'] = cap_extreme_values(safe_stat(np.std, data))
    features[prefix + 'var'] = cap_extreme_values(safe_stat(np.var, data))
    features[prefix + 'min'] = cap_extreme_values(safe_stat(np.min, data))
    features[prefix + 'max'] = cap_extreme_values(safe_stat(np.max, data))
    features[prefix + 'range'] = cap_extreme_values(safe_stat(lambda d: np.max(d) - np.min(d), data))
    features[prefix + 'median'] = cap_extreme_values(safe_stat(np.median, data))
    features[prefix + 'skew'] = cap_extreme_values(safe_stat(skew, data))
    features[prefix + 'kurtosis'] = cap_extreme_values(safe_stat(kurtosis, data))
    
    # FFT and energy features
    fft_vals = np.abs(fft(data))
    freqs = np.fft.fftfreq(len(data), d=1)
    features[prefix + 'total_energy'] = cap_extreme_values(np.sum(fft_vals**2))
    low_freq_power = np.sum(fft_vals[freqs < 0.1]**2)
    high_freq_power = np.sum(fft_vals[freqs >= 0.1]**2)
    features[prefix + 'power_ratio_low_high'] = cap_extreme_values(low_freq_power / high_freq_power)

    # Peak frequencies and zero-crossing rate
    sorted_fft_vals = np.sort(fft_vals)
    features[prefix + 'peak_freq_1'] = cap_extreme_values(sorted_fft_vals[-1])
    features[prefix + 'peak_freq_2'] = cap_extreme_values(sorted_fft_vals[-2])
    features[prefix + 'peak_freq_3'] = cap_extreme_values(sorted_fft_vals[-3])
    zero_crossings = np.where(np.diff(np.sign(data)))[0]
    features[prefix + 'zero_crossing_rate'] = cap_extreme_values(len(zero_crossings) / len(data))

    # Mean-crossing rate
    mean_level = np.mean(data)
    mean_crossings = np.where(np.diff(np.sign(data - mean_level)))[0]
    features[prefix + 'mean_crossing_rate'] = cap_extreme_values(len(mean_crossings) / len(data))

    # Local extrema: maxima and minima
    peaks, _ = find_peaks(data)
    troughs, _ = find_peaks(-data)
    features[prefix + 'local_maxima_rate'] = cap_extreme_values(len(peaks) / len(data))
    features[prefix + 'local_minima_rate'] = cap_extreme_values(len(troughs) / len(data))

    # Quantiles and interquantile ranges
    features[prefix + 'quantile_25'] = cap_extreme_values(np.quantile(data, 0.25))
    features[prefix + 'quantile_50'] = cap_extreme_values(np.quantile(data, 0.5))
    features[prefix + 'quantile_75'] = cap_extreme_values(np.quantile(data, 0.75))
    features[prefix + 'iqr_25_75'] = cap_extreme_values(features[prefix + 'quantile_75'] - features[prefix + 'quantile_25'])
    features[prefix + 'iqr_10_90'] = cap_extreme_values(np.quantile(data, 0.90) - np.quantile(data, 0.10))

    # Signal envelope using Hilbert transform
    analytical_signal = hilbert(data)
    envelope = np.abs(analytical_signal)
    features[prefix + 'envelope_mean'] = cap_extreme_values(np.mean(envelope))
    features[prefix + 'envelope_std'] = cap_extreme_values(np.std(envelope))
    features[prefix + 'envelope_max'] = cap_extreme_values(np.max(envelope))
    features[prefix + 'envelope_min'] = cap_extreme_values(np.min(envelope))
    features[prefix + 'envelope_skew'] = cap_extreme_values(safe_stat(skew, envelope))
    features[prefix + 'envelope_kurtosis'] = cap_extreme_values(safe_stat(kurtosis, envelope))

    return features

def extract_features(df):
    signal_data = df.drop(columns='y', errors='ignore')
    all_features = []
    for _, row in signal_data.iterrows():
        row_features = calculate_features(np.asarray(row), prefix="global_")
        all_features.append(row_features)
    features_df = pd.DataFrame(all_features)
    features_df['y'] = df['y'].values
    
    return features_df

