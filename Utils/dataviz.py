import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import seaborn as sns

def plot_bio_signal(df, label):
    germination_row = df[df['y'] == 0].index[0]
    start = max(germination_row - 500, 0)
    end = min(germination_row + 500, len(df) - 1)
    signal_data = df.iloc[start:end+1].drop(columns='y')
    plt.figure(figsize=(15, 5))
    for column in signal_data.columns:
        plt.plot(signal_data.index, signal_data[column], label=column)
    
    plt.axvline(x=germination_row, color='r', linestyle='--', label='Sprouting')
    plt.title(f'Bio Signal around Sprouting in {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.show()

def plot_time_series(df, title):
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        if column != 'y':
            plt.plot(df.index, df[column], label=column)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.show()


def plot_spectrogram(df, title):
    plt.figure(figsize=(14, 7))
    for column in df.columns:
        if column != 'y':
            f, t, Sxx = spectrogram(df[column], fs=1)
            plt.pcolormesh(t, f, Sxx, shading='gouraud')
            plt.title(f'Spectrogram for {column}')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar(label='Intensity')
            plt.show()
            
def plot_heatmap_with_y(df, title):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.show()



