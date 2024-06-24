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
    signal_data = df.drop(columns='y', errors='ignore')  
    combined_signal = signal_data.values.flatten() 
    num_samples_per_row = signal_data.shape[1]
    total_seconds = num_samples_per_row * len(df)
    time = np.arange(total_seconds) / num_samples_per_row

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
    
def plot_results(results):
    for dict_name, dict_results in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [maes for model_name, maes in dict_results.items()]
        ax.boxplot(data_to_plot, labels=dict_results.keys())
        ax.set_title(f'MAE Boxplot for {dict_name}')
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE')
        plt.show()



def create_boxplots(individual_maes):
    for dict_name, model_results in individual_maes.items():
        plt.figure(figsize=(10, 6))
        data = []
        labels = []
        
        for model_name, maes in model_results.items():
            data.append(maes)
            labels.append(model_name)
        
        plt.boxplot(data, notch=True, patch_artist=True, labels=labels,
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    whiskerprops=dict(color='red'),
                    medianprops=dict(color='green'),
                    capprops=dict(color='purple'))
        plt.title(f'Boxplot of MAEs for {dict_name}')
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.show()
        

def plot_lines(individual_maes):
    for dict_name, model_results in individual_maes.items():
        plt.figure(figsize=(10, 6))
        for model_name, maes in model_results.items():
            plt.plot(maes, marker='o', linestyle='-', label=model_name)
        plt.title(f'Individual MAEs for {dict_name}')
        plt.xlabel('Fold')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        plt.show()


import pandas as pd
import matplotlib.pyplot as plt

def comparison(df_reference, dfs_to_compare, labels):
    plt.figure(figsize=(15, 15))
    min_val = float('inf')
    max_val = float('-inf')
    
    all_signals = [df_reference] + dfs_to_compare
    for df in all_signals:
        signal_data = df.drop(columns=['y']).values.flatten()
        min_val = min(min_val, signal_data.min())
        max_val = max(max_val, signal_data.max())
    
    reference_signal = df_reference.drop(columns=['y']).values.flatten()
    normalized_reference_signal = (reference_signal - min_val) / (max_val - min_val)
    time_vector = range(len(normalized_reference_signal))
    
    for i, (df, label) in enumerate(zip(dfs_to_compare, labels), start=1):
        plt.subplot(3, 1, i)
        compare_signal = df.drop(columns=['y']).values.flatten()
        normalized_compare_signal = (compare_signal - min_val) / (max_val - min_val)
        compare_time_vector = range(len(normalized_compare_signal))
        
        plt.plot(time_vector, normalized_reference_signal, label='df9')
        plt.plot(compare_time_vector, normalized_compare_signal, label=label)
        
        plt.title(f'Signal for df9 and {label}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Mvs')
        plt.legend()
    
    plt.tight_layout()
    plt.show()