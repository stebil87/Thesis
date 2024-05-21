import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft

def feature_generation(row):
    features = {}
    
    # Basic statistical features
    features['mean'] = row.mean()
    features['std'] = row.std()
    features['var'] = row.var()
    features['min'] = row.min()
    features['max'] = row.max()
    features['range'] = row.max() - row.min()
    features['median'] = np.median(row)
    features['skew'] = skew(row)
    features['kurtosis'] = kurtosis(row)
    
    # entropy
    hist, bin_edges = np.histogram(row, bins=10, density=True)
    features['entropy'] = entropy(hist)

    # FFT and energy features
    fft_vals = np.abs(fft(row.to_numpy()))
    features['total_energy'] = np.sum(fft_vals**2)
    
    # Frequency bands energy
    n = len(row)
    freqs = np.fft.fftfreq(n)
    band_energies = [0, 0, 0, 0]
    band_edges = [0.25, 0.5, 0.75, 1.0]
    for i, edge in enumerate(band_edges):
        mask = (freqs >= (edge - 0.25)) & (freqs < edge)
        band_energies[i] = np.sum(fft_vals[mask]**2)
    for i in range(4):
        features[f'band_energy_{i+1}'] = band_energies[i]

    # Peak frequencies
    peaks_indices = np.argsort(fft_vals)[-3:]
    peaks_values = fft_vals[peaks_indices]
    for i in range(3):
        features[f'peak_freq_{i+1}'] = peaks_values[i]

    # Zero-crossing rate
    zero_crossings = np.where(np.diff(np.sign(row)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(row)

    # Quantiles and interquantile ranges
    features['quantile_25'] = np.quantile(row, 0.25)
    features['quantile_50'] = np.quantile(row, 0.5)
    features['quantile_75'] = np.quantile(row, 0.75)
    features['iqr_25_75'] = features['quantile_75'] - features['quantile_25']
    features['iqr_10_90'] = np.quantile(row, 0.90) - np.quantile(row, 0.10)

    # Derivative-based features
    features['first_derivative_mean'] = np.mean(np.diff(row, n=1))
    features['second_derivative_mean'] = np.mean(np.diff(row, n=2))

    # Sound-like features
    magnitude_spectrum = np.abs(np.fft.rfft(row))
    frequencies = np.fft.rfftfreq(len(row), d=1)
    features['spectral_centroid'] = np.sum(frequencies * magnitude_spectrum) / np.sum(magnitude_spectrum)
    spectral_rolloff_threshold = 0.85 * np.sum(magnitude_spectrum)
    cumulative_spectrum = np.cumsum(magnitude_spectrum)
    spectral_rolloff_frequency = frequencies[np.where(cumulative_spectrum >= spectral_rolloff_threshold)[0][0]]
    features['spectral_rolloff'] = spectral_rolloff_frequency
    spectrum_prev = np.abs(np.fft.rfft(np.roll(row, 1)))
    spectral_flux = np.sqrt(np.sum((magnitude_spectrum - spectrum_prev) ** 2))
    features['spectral_flux'] = spectral_flux
    features['rms_energy'] = np.sqrt(np.mean(row**2))
    features['autocorrelation'] = np.correlate(row, row, mode='full')[len(row)-1] / row.var() / len(row)
    
    return features

def extract_features(df):
    signal_data = df.drop(columns='y', errors='ignore')
    all_features = []
    for i, row in signal_data.iterrows():
        row_features = feature_generation(row)
        all_features.append(row_features)
    features_df = pd.DataFrame(all_features)
    features_df['y'] = df['y'].values
    cols = [col for col in features_df.columns if col != 'y'] + ['y']
    features_df = features_df[cols]
    
    return features_df

