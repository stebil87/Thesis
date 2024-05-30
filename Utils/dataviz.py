import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    signal_data = df.drop(columns='y', errors='ignore')  # Remove the 'y' column
    combined_signal = signal_data.values.flatten()  # Flatten all rows into a single continuous array
    
    # Calculate the total number of seconds
    total_seconds = len(combined_signal)
    
    # Create a time array from 0 to total_seconds - 1
    time = np.arange(total_seconds)

    plt.figure(figsize=(14, 7))
    plt.plot(time, combined_signal, label='Combined Signal')
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()
    
def plot_detrended_signal(df, label):
    signal_data = df.drop(columns='y', errors='ignore')
    combined_signal = signal_data.values.flatten()
    time = np.arange(len(combined_signal)) / signal_data.shape[1]
    plt.figure(figsize=(15, 5))
    plt.plot(time, combined_signal, label='Detrended Combined Signal')
    plt.title(f'Detrended Bio Signal for {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()

def downsample_signal(signal, factor):
    return signal[::factor]

def plot_acf_pacf(signal, title):
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(signal, ax=plt.gca(), lags=50) 
    plt.title(f'ACF - {title}')
    plt.subplot(122)
    plot_pacf(signal, ax=plt.gca(), lags=50)  
    plt.title(f'PACF - {title}')
    plt.tight_layout()
    plt.show()
    
