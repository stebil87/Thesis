import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

def perform_regression_and_cv(dictionaries):
    results = {}
    predictions = {}
    loo = LeaveOneOut()

    for dict_name, datasets in dictionaries.items():
        print(f"Processing dictionary: {dict_name}")
        dataframes = list(datasets.values())
        results[dict_name] = {}
        predictions[dict_name] = {}

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
                predictions[dict_name][model_name].append((y_test.values, y_pred))

    return results, predictions

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items():
            avg_mae = np.mean(maes)
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")

def plot_box_plots(results):
    for dict_name, dict_results in results.items():
        plt.figure(figsize=(10, 6))
        data = [maes for model_name, maes in dict_results.items()]
        plt.boxplot(data, labels=dict_results.keys())
        plt.title(f'Box Plot of MAE for {dict_name}')
        plt.xlabel('Model')
        plt.ylabel('Mean Absolute Error')
        plt.show()

def plot_predictions_vs_actuals(predictions):
    for dict_name, dict_preds in predictions.items():
        plt.figure(figsize=(10, 6))
        for model_name, preds in dict_preds.items():
            y_test, y_pred = preds[0]
            plt.plot(y_test, label=f'Actual {model_name}')
            plt.plot(y_pred, label=f'Predicted {model_name}')
        plt.title(f'{dict_name} - Actual vs Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()


