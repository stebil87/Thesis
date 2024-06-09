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

def plot_trends(predictions):
    for dict_name, dict_preds in predictions.items():
        fig, axs = plt.subplots(len(dict_preds), 1, figsize=(10, 6 * len(dict_preds)))
        fig.suptitle(f'Prediction Trends for {dict_name}', fontsize=16)
        
        if len(dict_preds) == 1:
            axs = [axs]
        
        for ax, (model_name, preds) in zip(axs, dict_preds.items()):
            sample_size = len(preds)
            num_samples_to_plot = min(sample_size, 10) 

            for y_test, y_pred in preds[:num_samples_to_plot]:
                ax.plot(y_test, label='True', alpha=0.6)
                ax.plot(y_pred, label='Predicted', alpha=0.6)

            y_true_all = np.concatenate([y_test for y_test, y_pred in preds])
            y_pred_all = np.concatenate([y_pred for y_test, y_pred in preds])
            
            y_pred_mean = np.mean(y_pred_all, axis=0)
            y_pred_std = np.std(y_pred_all, axis=0)
            y_pred_upper = y_pred_mean + y_pred_std
            y_pred_lower = y_pred_mean - y_pred_std
            
            ax.plot(y_pred_mean, label='Mean Prediction', color='black')
            ax.fill_between(range(len(y_pred_mean)), y_pred_lower, y_pred_upper, color='gray', alpha=0.3)

            ax.set_title(f'Model: {model_name}')
            ax.set_xlabel('Sample')
            ax.set_ylabel('Value')
            ax.legend()
        plt.tight_layout()
        plt.show()
