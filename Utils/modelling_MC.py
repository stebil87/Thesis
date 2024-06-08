import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

class MCDropoutXGBRegressor(XGBRegressor):
    def __init__(self, dropout_rate=0.1, n_estimators=100, random_state=None, **kwargs):
        super().__init__(n_estimators=n_estimators, random_state=random_state, **kwargs)
        self.dropout_rate = dropout_rate
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        self._train_data = X
        self._train_labels = y
        super().fit(X, y, **kwargs)

    def predict(self, X, n_iter=100):
        predictions = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(n_iter):
            dropout_mask = rng.binomial(1, 1 - self.dropout_rate, size=X.shape)
            X_dropped = X * dropout_mask
            predictions.append(super().predict(X_dropped))
        return np.mean(predictions, axis=0), np.percentile(predictions, 2.5, axis=0), np.percentile(predictions, 97.5, axis=0)

def mc_dropout(dictionaries, dropout_rate=0.1, n_iter=100, uncertainty_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], random_state=None):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    parameter_combinations = list(ParameterGrid(param_grid))

    results = {}
    predictions = {}
    loo = LeaveOneOut()

    for dict_name, datasets in dictionaries.items():
        print(f"Processing dictionary: {dict_name}")
        dataframes = list(datasets.values())
        results[dict_name] = {threshold: [] for threshold in uncertainty_thresholds}
        predictions[dict_name] = {threshold: [] for threshold in uncertainty_thresholds}

        for params in parameter_combinations:
            print(f"Testing parameters: {params}")
            for train_index, test_index in loo.split(dataframes):
                train_dfs = [dataframes[i] for i in train_index]
                test_df = dataframes[test_index[0]]

                X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])
                y_train = pd.concat([df['y'] for df in train_dfs])
                X_test = test_df.drop(columns='y', errors='ignore')
                y_test = test_df['y']

                model = MCDropoutXGBRegressor(dropout_rate=dropout_rate, random_state=random_state, **params)
                model.fit(X_train, y_train)
                mean_prediction, lower_bound, upper_bound = model.predict(X_test, n_iter=n_iter)

                for threshold in uncertainty_thresholds:
                    uncertainty = upper_bound - lower_bound
                    accepted_prediction = mean_prediction.copy()
                    accepted_prediction[uncertainty > threshold] = np.nan

                    accepted_indices = ~np.isnan(accepted_prediction)
                    if np.any(accepted_indices):
                        mae = mean_absolute_error(y_test[accepted_indices], accepted_prediction[accepted_indices])
                    else:
                        mae = np.nan

                    results[dict_name][threshold].append(mae)
                    predictions[dict_name][threshold].append((y_test.values, mean_prediction, lower_bound, upper_bound, uncertainty))

    return results, predictions

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for threshold, maes in dict_results.items():
            valid_maes = [mae for mae in maes if not np.isnan(mae)]
            if valid_maes:
                avg_mae = np.nanmean(valid_maes)
                print(f"  Threshold: {threshold}, MAE: {avg_mae:.4f}")
            else:
                print(f"  Threshold: {threshold}, MAE: No valid predictions")

def calculate_unreliable_predictions(predictions, uncertainty_thresholds):
    unreliable_counts = {}

    for dict_name, dict_preds in predictions.items():
        unreliable_counts[dict_name] = {}

        for threshold in uncertainty_thresholds:
            total_unreliable = 0
            total_predictions = 0

            for model_name, preds in dict_preds.items():
                for y_test, mean_pred, lower_bound, upper_bound, uncertainty in preds:
                    unreliable = np.sum(uncertainty > threshold)
                    total_unreliable += unreliable
                    total_predictions += len(y_test)

            unreliable_counts[dict_name][threshold] = {
                'total_unreliable': total_unreliable,
                'total_predictions': total_predictions,
                'percentage_unreliable': (total_unreliable / total_predictions) * 100 if total_predictions > 0 else 0
            }

    return unreliable_counts

def plot_box_plots(results):
    for dict_name, dict_results in results.items():
        plt.figure(figsize=(10, 6))
        data = [maes for threshold, maes in dict_results.items()]
        plt.boxplot(data, labels=dict_results.keys())
        plt.title(f'Box Plot of MAE for {dict_name}')
        plt.xlabel('Threshold')
        plt.ylabel('Mean Absolute Error')
        plt.show()

def plot_predictions_vs_actuals(predictions):
    for dict_name, dict_preds in predictions.items():
        plt.figure(figsize=(10, 6))
        for threshold, preds in dict_preds.items():
            y_test, mean_pred, lower_bound, upper_bound, uncertainty = preds[0]
            plt.plot(y_test, label=f'Actual Threshold: {threshold}')
            plt.plot(mean_pred, label=f'Predicted Threshold: {threshold}')
        plt.title(f'{dict_name} - Actual vs Predicted')
        plt.xlabel('Sample')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

# Example usage
results, predictions = mc_dropout(scaled_dictionaries)
print_results(results, "MAE")
plot_box_plots(results)
plot_predictions_vs_actuals(predictions)
