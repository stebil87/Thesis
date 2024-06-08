import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from xgboost import XGBRegressor
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

def train_bootstrap_models(X_train, y_train, n_models=10, random_state=None, params=None):
    models = []
    for i in range(n_models):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state)
        model = XGBRegressor(objective='reg:squarederror', random_state=random_state, **params)
        model.fit(X_resampled, y_resampled)
        models.append(model)
    return models

def compute_uncertainty(models, X_test):
    predictions = np.array([model.predict(X_test) for model in models])
    mean_prediction = np.mean(predictions, axis=0)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
    return mean_prediction, lower_bound, upper_bound

def optimize_xgb(X_train, y_train):
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    model = XGBRegressor(objective='reg:squarederror')
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)
    return random_search.best_params_

def perform_regression_and_cv_with_bootstrap(dictionaries, n_models=10, uncertainty_threshold=10, n_splits=5):
    results = {}
    predictions = {}
    loo = LeaveOneOut()
    kf = KFold(n_splits=n_splits)

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

            print("Optimizing XGBoost parameters...")
            best_params = optimize_xgb(X_train, y_train)

            models = train_bootstrap_models(X_train, y_train, n_models, params=best_params)
            mean_prediction, lower_bound, upper_bound = compute_uncertainty(models, X_test)
            uncertainty = upper_bound - lower_bound
            accepted_prediction = mean_prediction.copy()
            accepted_prediction[uncertainty > uncertainty_threshold] = np.nan  
            
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
            predictions[dict_name]['XGBoost'].append((y_test.values, mean_prediction, lower_bound, upper_bound))

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
            y_test, mean_pred, lower_bound, upper_bound = preds[0]
            plt.plot(y_test, label=f'Actual - {model_name}')
            plt.plot(mean_pred, label=f'Predicted - {model_name}')
            plt.fill_between(range(len(y_test)), lower_bound, upper_bound, alpha=0.2, label=f'Uncertainty - {model_name}')
        plt.title(f'{dict_name} - Actual vs Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
