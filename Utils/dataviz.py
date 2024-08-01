import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def sprouting_500(df, label):
    row = df[df['y'] == 0].index[0]
    prev_values = df.iloc[row, -501:-1].values
    next_values = df.iloc[row + 1, 0:500].values
    combined_signal = np.concatenate((prev_values, next_values))

    plt.figure(figsize=(15, 5))
    plt.plot(range(len(combined_signal)), combined_signal, label='Signal')
    plt.title(f'Signal around Sprouting in {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()
    
def sprouting_100(df, label):
    row = df[df['y'] == 0].index[0]
    prev_values = df.iloc[row, -101:-1].values
    next_values = df.iloc[row + 1, 0:100].values
    combined_signal = np.concatenate((prev_values, next_values))

    plt.figure(figsize=(15, 5))
    plt.plot(range(len(combined_signal)), combined_signal, label='Signal')
    plt.title(f'Signal around Sprouting in {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()   
    
def sprouting_1000(df, label):
    row = df[df['y'] == 0].index[0]
    prev_values = df.iloc[row, -2001:-1].values
    next_values = df.iloc[row + 1, 0:2000].values
    combined_signal = np.concatenate((prev_values, next_values))

    plt.figure(figsize=(15, 5))
    plt.plot(range(len(combined_signal)), combined_signal, label='Signal')
    plt.title(f'Signal around Sprouting in {label}')
    plt.xlabel('Time (s)')
    plt.ylabel('Millivolts')
    plt.legend()
    plt.show()   

def plot_time_series(df, title):
    signal_data = df.drop(columns='y', errors='ignore')  
    combined_signal = signal_data.values.flatten()
    total_samples = len(combined_signal)
    time = np.arange(total_samples)

    plt.figure(figsize=(14, 7))
    plt.plot(time, combined_signal, label='Signal')
    plt.title(title)
    plt.xlabel('Sample Number')
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
    

def plot_signal_and_target(dataframes):
    for df_name, df in dataframes.items():
        df = df.sort_values(by='timestamp')
        
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df['mV'], label='mV Signal', color='blue')
        
        plt.plot(df.index, df['y'], label='y Target', color='red', linestyle='--')
        
        plt.title(f'Signal and Target for {df_name}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        

def plot_delta_days_percentiles(delta_days):
    for dict_name, model_deltas in delta_days.items():
        plt.figure(figsize=(10, 6))
        for model_name, deltas in model_deltas.items():
            if not isinstance(deltas, list):
                print(f"Skipping {model_name} in {dict_name} because it does not contain a list of deltas.")
                continue
            deltas = np.array(deltas)
            percentiles = np.linspace(0, 100, len(deltas))
            sorted_deltas = np.sort(deltas)

            plt.plot(percentiles, sorted_deltas, label=model_name)

        plt.xlabel('Percentile')
        plt.ylabel('Delta Days')
        plt.title(f'Error Distribution Percentiles for {dict_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
        