import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import seaborn as sns

def plot_bio_signal(df, label):
    germination_row = df[df['y'] == 0].index[0]
    if germination_row < 500 or germination_row + 500 >= len(df):
        print(f"Not enough data to plot 500 steps before and after germination for {label}")
        return
    
    start = germination_row - 500
    end = germination_row + 500
    signal_data = df.iloc[start:end+1].drop(columns='y')
    combined_signal = signal_data.values.flatten()
    germination_position = 500
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(combined_signal)), combined_signal, label='Combined Signal')
    plt.axvline(x=germination_position * signal_data.shape[1], color='r', linestyle='--', label='Sprouting')
    plt.title(f'Bio Signal around Sprouting in {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()


def plot_time_series(df, title):
    signal_data = df.drop(columns='y', errors='ignore')
    combined_signal = signal_data.values.flatten()
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(combined_signal)), combined_signal, label='Combined Signal')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()

