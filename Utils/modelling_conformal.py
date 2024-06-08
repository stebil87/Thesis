import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

def conformal_prediction(X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    y_pred_cal = model.predict(X_cal)
    residuals = np.abs(y_cal - y_pred_cal)
    q = np.quantile(residuals, 1 - alpha)
    
    y_pred_test = model.predict(X_test)
    lower_bound = y_pred_test - q
    upper_bound = y_pred_test + q
    
    return y_pred_test, lower_bound, upper_bound

def perform_conformal_prediction(dictionaries, alpha=0.1):
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

            X_train_val = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])
            y_train_val = pd.concat([df['y'] for df in train_dfs])
            X_test = test_df.drop(columns='y', errors='ignore')
            y_test = test_df['y']
            
            X_train, X_cal, y_train, y_cal = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

            y_pred, lower_bound, upper_bound = conformal_prediction(X_train, y_train, X_cal, y_cal, X_test, alpha)

            uncertainty = upper_bound - lower_bound
            accepted_prediction = y_pred.copy()
            accepted_prediction[uncertainty > (2 * np.quantile(uncertainty, 1 - alpha))] = np.nan

            accepted_indices = ~np.isnan(accepted_prediction)
            if np.any(accepted_indices):
                mae = mean_absolute_error(y_test[accepted_indices], accepted_prediction[accepted_indices])
            else:
                mae = np.nan

            if 'XGBoost' not in results[dict_name]:
                results[dict_name]['XGBoost'] = []
            results[dict_name]['XGBoost'].append(mae)

            if 'XGBoost' not in predictions[dict_name]:
                predictions[dict_name]['XGBoost'] = []
            predictions[dict_name]['XGBoost'].append((y_test.values, y_pred, lower_bound, upper_bound))

    return results, predictions

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items():
            avg_mae = np.nanmean(maes)
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
            y_test, y_pred, lower_bound, upper_bound = preds[0]
            plt.plot(y_test, label=f'Actual - {model_name}')
            plt.plot(y_pred, label=f'Predicted - {model_name}')
            plt.fill_between(range(len(y_test)), lower_bound, upper_bound, alpha=0.2, label=f'Uncertainty - {model_name}')
        plt.title(f'{dict_name} - Actual vs Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

