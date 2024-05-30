import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
import warnings
import matplotlib.pyplot as plt

# Corrected warning filter
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

def perform_regression_and_cv(dictionaries):
    results = {}
    predictions = {}
    errors = {}
    loo = LeaveOneOut()

    for dict_name, datasets in dictionaries.items():
        print(f"Processing dictionary: {dict_name}")
        dataframes = list(datasets.values())
        results[dict_name] = {}
        predictions[dict_name] = {}
        errors[dict_name] = {}

        for train_index, test_index in loo.split(dataframes):
            train_dfs = [dataframes[i] for i in train_index]
            test_df = dataframes[test_index[0]]

            X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])
            y_train = pd.concat([df['y'] for df in train_dfs])
            X_test = test_df.drop(columns='y', errors='ignore')
            y_test = test_df['y']

            models = {
                'XGBoost': XGBRegressor(),
                'LightGBM': LGBMRegressor(),
                'AdaBoost': AdaBoostRegressor()
            }

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                if model_name not in results[dict_name]:
                    results[dict_name][model_name] = []
                results[dict_name][model_name].append(mae)
                
                if model_name not in predictions[dict_name]:
                    predictions[dict_name][model_name] = []
                predictions[dict_name][model_name].append(y_pred)
                
                if model_name not in errors[dict_name]:
                    errors[dict_name][model_name] = []
                errors[dict_name][model_name].append(y_test - y_pred)

    return results, predictions, errors

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items():
            avg_mae = np.mean(maes)
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")

def plot_results(results):
    for dict_name, dict_results in results.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        data_to_plot = [maes for model_name, maes in dict_results.items()]
        ax.boxplot(data_to_plot, labels=dict_results.keys())
        ax.set_title(f'MAE Boxplot for {dict_name}')
        ax.set_xlabel('Model')
        ax.set_ylabel('MAE')
        plt.show()

def plot_trends(predictions, errors):
    for dict_name, model_predictions in predictions.items():
        for model_name, preds in model_predictions.items():
            fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax[0].plot(preds, label=f'Predictions ({model_name})')
            ax[0].set_ylabel('Predictions')
            ax[0].legend()
            ax[0].set_title(f'Predictions and Errors for {dict_name} - {model_name}')
            
            ax[1].plot(errors[dict_name][model_name], label=f'Errors ({model_name})', color='r')
            ax[1].set_ylabel('Errors')
            ax[1].set_xlabel('Fold')
            ax[1].legend()
            
            plt.show()