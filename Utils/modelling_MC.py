import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

class MCDropoutXGBRegressor(XGBRegressor):  # Define a class for XGBoost with MC Dropout
    def __init__(self, dropout_rate=0.1, n_estimators=100, random_state=None, **kwargs):
        super().__init__(n_estimators=n_estimators, random_state=random_state, **kwargs)
        self.dropout_rate = dropout_rate  # Set dropout rate
        self.random_state = random_state  # Set random state

    def fit(self, X, y, **kwargs):
        self._train_data = X  # Store training data
        self._train_labels = y  # Store training labels
        super().fit(X, y, **kwargs)  # Call the fit method of the base class

    def predict(self, X, n_iter=100):
        predictions = []  # Initialize list to store predictions
        rng = np.random.default_rng(self.random_state)  # Initialize random number generator
        for _ in range(n_iter):  # Perform MC dropout iterations
            dropout_mask = rng.binomial(1, 1 - self.dropout_rate, size=X.shape)  # Create dropout mask
            X_dropped = X * dropout_mask  # Apply dropout mask
            predictions.append(super().predict(X_dropped))  # Predict with dropout
        return np.mean(predictions, axis=0), np.percentile(predictions, 2.5, axis=0), np.percentile(predictions, 97.5, axis=0)  # Return mean and percentiles

def mc_dropout(dictionaries, dropout_rate=0.1, n_iter=100, uncertainty_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5], random_state=None):
    param_grid = {  # Define grid of parameters for XGBoost
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    parameter_combinations = list(ParameterGrid(param_grid))  # Get all parameter combinations

    results = {}  # Initialize dictionary to store results
    predictions = {}  # Initialize dictionary to store predictions
    loo = LeaveOneOut()  # Initialize Leave-One-Out cross-validation

    for dict_name, datasets in dictionaries.items():  # Loop through each dictionary of datasets
        print(f"Processing dictionary: {dict_name}")  # Print dictionary name
        dataframes = list(datasets.values())  # Get list of dataframes
        results[dict_name] = {threshold: [] for threshold in uncertainty_thresholds}  # Initialize results for each threshold
        predictions[dict_name] = {threshold: [] for threshold in uncertainty_thresholds}  # Initialize predictions for each threshold

        for params in parameter_combinations:  # Loop through each parameter combination
            print(f"Testing parameters: {params}")  # Print parameter combination
            for train_index, test_index in loo.split(dataframes):  # Loop through each Leave-One-Out split
                train_dfs = [dataframes[i] for i in train_index]  # Get training dataframes
                test_df = dataframes[test_index[0]]  # Get test dataframe

                X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])  # Concatenate training features
                y_train = pd.concat([df['y'] for df in train_dfs])  # Concatenate training labels
                X_test = test_df.drop(columns='y', errors='ignore')  # Get test features
                y_test = test_df['y']  # Get test labels

                model = MCDropoutXGBRegressor(dropout_rate=dropout_rate, random_state=random_state, **params)  # Initialize model
                model.fit(X_train, y_train)  # Fit model
                mean_prediction, lower_bound, upper_bound = model.predict(X_test, n_iter=n_iter)  # Predict with uncertainty

                for threshold in uncertainty_thresholds:  # Loop through each uncertainty threshold
                    uncertainty = upper_bound - lower_bound  # Calculate uncertainty
                    accepted_prediction = mean_prediction.copy()  # Copy mean prediction
                    accepted_prediction[uncertainty > threshold] = np.nan  # Invalidate high uncertainty predictions

                    accepted_indices = ~np.isnan(accepted_prediction)  # Get indices of accepted predictions
                    if np.any(accepted_indices):  # Check if there are accepted predictions
                        mae = mean_absolute_error(y_test[accepted_indices], accepted_prediction[accepted_indices])  # Calculate MAE
                    else:
                        mae = np.nan  # Set MAE to NaN if no predictions are accepted

                    results[dict_name][threshold].append(mae)  # Append MAE to results
                    predictions[dict_name][threshold].append((y_test.values, mean_prediction, lower_bound, upper_bound, uncertainty))  # Append predictions

    return results, predictions  # Return results and predictions

def print_results(results, description):
    print(f"\n{description} Results:")  # Print description
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        print(f"\nDictionary: {dict_name}")  # Print dictionary name
        for threshold, maes in dict_results.items():  # Loop through each threshold's results
            valid_maes = [mae for mae in maes if not np.isnan(mae)]  # Get valid MAEs
            if valid_maes:
                avg_mae = np.nanmean(valid_maes)  # Calculate average MAE
                print(f"  Threshold: {threshold}, MAE: {avg_mae:.4f}")  # Print threshold and average MAE
            else:
                print(f"  Threshold: {threshold}, MAE: No valid predictions")  # Print no valid predictions if applicable

def calculate_unreliable_predictions(predictions, uncertainty_thresholds):
    unreliable_counts = {}  # Initialize dictionary to store counts

    for dict_name, dict_preds in predictions.items():  # Loop through each dictionary of predictions
        unreliable_counts[dict_name] = {}  # Initialize counts for current dictionary

        for threshold in uncertainty_thresholds:  # Loop through each uncertainty threshold
            total_unreliable = 0  # Initialize count of unreliable predictions
            total_predictions = 0  # Initialize count of total predictions

            for model_name, preds in dict_preds.items():  # Loop through each model's predictions
                for y_test, mean_pred, lower_bound, upper_bound, uncertainty in preds:  # Loop through each prediction
                    unreliable = np.sum(uncertainty > threshold)  # Count unreliable predictions
                    total_unreliable += unreliable  # Update total unreliable count
                    total_predictions += len(y_test)  # Update total predictions count

            unreliable_counts[dict_name][threshold] = {  # Store counts in dictionary
                'total_unreliable': total_unreliable,
                'total_predictions': total_predictions,
                'percentage_unreliable': (total_unreliable / total_predictions) * 100 if total_predictions > 0 else 0
            }

    return unreliable_counts  # Return unreliable counts

def plot_box_plots(results):
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        plt.figure(figsize=(10, 6))  # Create new figure
        data = [maes for threshold, maes in dict_results.items()]  # Get list of MAEs
        plt.boxplot(data, labels=dict_results.keys())  # Create box plot
        plt.title(f'Box Plot of MAE for {dict_name}')  # Set plot title
        plt.xlabel('Threshold')  # Set x-axis label
        plt.ylabel('Mean Absolute Error')  # Set y-axis label
        plt.show()  # Display plot

def plot_predictions_vs_actuals(predictions):
    for dict_name, dict_preds in predictions.items():  # Loop through each dictionary of predictions
        y_test_avg = np.mean([pred[0] for preds in dict_preds.values() for pred in preds], axis=0)  # Calculate average actual values
        y_pred_avg = np.mean([pred[1] for preds in dict_preds.values() for pred in preds], axis=0)  # Calculate average predicted values

        plt.figure(figsize=(10, 6))  # Create new figure
        plt.plot(y_test_avg, label='Actual')  # Plot average actual values
        plt.plot(y_pred_avg, label='Predicted')  # Plot average predicted values
        plt.title(f'{dict_name} - Actual vs Predicted')  # Set plot title
        plt.xlabel('Sample')  # Set x-axis label
        plt.ylabel('Value')  # Set y-axis label
        plt.legend()  # Show legend
        plt.show()  # Display plot
