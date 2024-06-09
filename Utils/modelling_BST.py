import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.utils import resample
from xgboost import XGBRegressor
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

def train_bootstrap_models(X_train, y_train, n_models=10, random_state=None, params=None):  # Define function to train multiple bootstrap models
    models = []  # Initialize an empty list to store models
    for i in range(n_models):  # Loop to create bootstrap samples
        X_resampled, y_resampled = resample(X_train, y_train, random_state=random_state)  # Resample the training data
        model = XGBRegressor(objective='reg:squarederror', random_state=random_state, **params)  # Initialize XGBoost model
        model.fit(X_resampled, y_resampled)  # Fit the model on resampled data
        models.append(model)  # Append the trained model to the list
    return models  # Return the list of trained models

def compute_uncertainty(models, X_test):  # Define function to compute prediction uncertainty
    predictions = np.array([model.predict(X_test) for model in models])  # Get predictions from all models
    mean_prediction = np.mean(predictions, axis=0)  # Calculate mean of predictions
    lower_bound = np.percentile(predictions, 2.5, axis=0)  # Calculate 2.5th percentile
    upper_bound = np.percentile(predictions, 97.5, axis=0)  # Calculate 97.5th percentile
    return mean_prediction, lower_bound, upper_bound  # Return mean prediction and uncertainty bounds

def optimize_xgb(X_train, y_train):  # Define function to optimize XGBoost parameters
    param_distributions = {  # Define parameter grid for Bayesian optimization
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    model = XGBRegressor(objective='reg:squarederror')  # Initialize XGBoost model
    random_search = RandomizedSearchCV(  # Initialize RandomizedSearchCV for hyperparameter optimization
        estimator=model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    random_search.fit(X_train, y_train)  # Fit the RandomizedSearchCV on training data
    return random_search.best_params_  # Return best parameters

def perform_regression_and_cv_with_bootstrap(dictionaries, n_models=10, uncertainty_threshold=10, n_splits=5):  # Define function to perform regression with bootstrap and cross-validation
    results = {}  # Initialize dictionary to store results
    predictions = {}  # Initialize dictionary to store predictions
    loo = LeaveOneOut()  # Initialize Leave-One-Out cross-validation
    kf = KFold(n_splits=n_splits)  # Initialize K-Fold cross-validation

    for dict_name, datasets in dictionaries.items():  # Loop through each dictionary of datasets
        print(f"Processing dictionary: {dict_name}")  # Print the name of the current dictionary
        dataframes = list(datasets.values())  # Get list of dataframes in the current dictionary
        results[dict_name] = {}  # Initialize results for the current dictionary
        predictions[dict_name] = {}  # Initialize predictions for the current dictionary

        for train_index, test_index in loo.split(dataframes):  # Loop through each Leave-One-Out split
            train_dfs = [dataframes[i] for i in train_index]  # Get training dataframes
            test_df = dataframes[test_index[0]]  # Get test dataframe

            X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])  # Concatenate training features
            y_train = pd.concat([df['y'] for df in train_dfs])  # Concatenate training labels
            X_test = test_df.drop(columns='y', errors='ignore')  # Get test features
            y_test = test_df['y']  # Get test labels

            print("Optimizing XGBoost parameters...")  # Print optimization status
            best_params = optimize_xgb(X_train, y_train)  # Optimize XGBoost parameters

            models = train_bootstrap_models(X_train, y_train, n_models, params=best_params)  # Train bootstrap models
            mean_prediction, lower_bound, upper_bound = compute_uncertainty(models, X_test)  # Compute uncertainty of predictions
            uncertainty = upper_bound - lower_bound  # Calculate uncertainty
            accepted_prediction = mean_prediction.copy()  # Copy mean prediction
            accepted_prediction[uncertainty > uncertainty_threshold] = np.nan  # Invalidate predictions with high uncertainty

            accepted_indices = ~np.isnan(accepted_prediction)  # Get indices of accepted predictions
            if np.any(accepted_indices):  # Check if there are accepted predictions
                mae = mean_absolute_error(y_test[accepted_indices], accepted_prediction[accepted_indices])  # Calculate MAE for accepted predictions
            else:
                mae = np.nan  # Set MAE to NaN if no predictions are accepted

            if 'XGBoost' not in results[dict_name]:  # Initialize results for XGBoost if not already present
                results[dict_name]['XGBoost'] = []
            results[dict_name]['XGBoost'].append(mae)  # Append MAE to results

            if 'XGBoost' not in predictions[dict_name]:  # Initialize predictions for XGBoost if not already present
                predictions[dict_name]['XGBoost'] = []
            predictions[dict_name]['XGBoost'].append((y_test.values, mean_prediction, lower_bound, upper_bound))  # Append predictions and intervals

    return results, predictions  # Return results and predictions

def print_results(results, description):  # Define function to print results
    print(f"\n{description} Results:")  # Print description
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        print(f"\nDictionary: {dict_name}")  # Print dictionary name
        for model_name, maes in dict_results.items():  # Loop through each model's results
            avg_mae = np.nanmean(maes)  # Calculate average MAE
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")  # Print model name and average MAE

def plot_box_plots(results):  # Define function to plot box plots of MAE
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        plt.figure(figsize=(10, 6))  # Create a new figure
        data = [maes for model_name, maes in dict_results.items()]  # Get list of MAEs
        plt.boxplot(data, labels=dict_results.keys())  # Create box plot
        plt.title(f'Box Plot of MAE for {dict_name}')  # Set plot title
        plt.xlabel('Model')  # Set x-axis label
        plt.ylabel('Mean Absolute Error')  # Set y-axis label
        plt.show()  # Display plot

def plot_predictions_vs_actuals(predictions):  # Define function to plot actual vs. predicted values
    for dict_name, dict_preds in predictions.items():  # Loop through each dictionary of predictions
        plt.figure(figsize=(10, 6))  # Create a new figure
        for model_name, preds in dict_preds.items():  # Loop through each model's predictions
            y_test, mean_pred, lower_bound, upper_bound = preds[0]  # Get predictions and intervals
            plt.plot(y_test, label=f'Actual - {model_name}')  # Plot actual values
            plt.plot(mean_pred, label=f'Predicted - {model_name}')  # Plot predicted values
            plt.fill_between(range(len(y_test)), lower_bound, upper_bound, alpha=0.2, label=f'Uncertainty - {model_name}')  # Fill uncertainty interval
        plt.title(f'{dict_name} - Actual vs Predicted')  # Set plot title
        plt.xlabel('Sample')  # Set x-axis label
        plt.ylabel('Value')  # Set y-axis label
        plt.legend()  # Show legend
        plt.show()  # Display plot
