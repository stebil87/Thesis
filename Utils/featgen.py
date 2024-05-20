import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft

def feature_generation(row):
    features = {}
    
    features['mean'] = row.mean()
    features['std'] = row.std()
    features['var'] = row.var()
    features['min'] = row.min()
    features['max'] = row.max()
    features['range'] = row.max() - row.min()
    features['median'] = np.median(row)
    features['skew'] = skew(row)
    features['kurtosis'] = kurtosis(row)
    
    hist, bin_edges = np.histogram(row, bins=10, density=True)
    features['entropy'] = entropy(hist)

    fft_vals = np.abs(fft(row.to_numpy())) 
    features['total_energy'] = np.sum(fft_vals**2)

    n = len(row)
    freqs = np.fft.fftfreq(n)
    band_energies = [0, 0, 0, 0]
    band_edges = [0.25, 0.5, 0.75, 1.0]
    
    for i, edge in enumerate(band_edges):
        mask = (freqs >= (edge - 0.25)) & (freqs < edge)
        band_energies[i] = np.sum(fft_vals[mask]**2)
    
    features['band_energy_1'] = band_energies[0]
    features['band_energy_2'] = band_energies[1]
    features['band_energy_3'] = band_energies[2]
    features['band_energy_4'] = band_energies[3]
    peaks_indices = np.argsort(fft_vals)[-3:]
    peaks_values = fft_vals[peaks_indices]
    features['peak_freq_1'] = peaks_values[0]
    features['peak_freq_2'] = peaks_values[1]
    features['peak_freq_3'] = peaks_values[2]
    zero_crossings = np.where(np.diff(np.sign(row)))[0]
    features['zero_crossing_rate'] = len(zero_crossings) / len(row)
    features['quantile_25'] = np.quantile(row, 0.25)
    features['quantile_50'] = np.quantile(row, 0.5)
    features['quantile_75'] = np.quantile(row, 0.75)
    
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
