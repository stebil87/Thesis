import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')  # Ignore LightGBM warnings

def optimize_xgb(X_train, y_train):  # Define function to optimize XGBoost parameters
    param_distributions = {  # Define parameter search space
        'n_estimators': Integer(50, 200),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'max_depth': Integer(3, 7),
        'subsample': Real(0.6, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.6, 1.0, prior='uniform')
    }

    model = XGBRegressor()  # Initialize XGBoost model
    bayes_search = BayesSearchCV(  # Initialize Bayesian optimization
        estimator=model,
        search_spaces=param_distributions,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    bayes_search.fit(X_train, y_train)  # Fit Bayesian optimization
    return bayes_search.best_estimator_  # Return best model

def bayesian_optimization(datasets):  # Define function for Bayesian optimization and cross-validation
    results = {}  # Initialize results dictionary
    predictions = {}  # Initialize predictions dictionary

    dict_name = 'augmented_features_linear'  # Set dictionary name
    print(f"Processing dictionary: {dict_name}")  # Print dictionary name
    dataframes = list(datasets[dict_name].values())  # Get list of dataframes
    results[dict_name] = {}  # Initialize results for current dictionary
    predictions[dict_name] = {}  # Initialize predictions for current dictionary

    loo = LeaveOneOut()  # Initialize Leave-One-Out cross-validation
    for train_index, test_index in loo.split(dataframes):  # Loop through each Leave-One-Out split
        train_dfs = [dataframes[i] for i in train_index]  # Get training dataframes
        test_df = dataframes[test_index[0]]  # Get test dataframe

        X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])  # Concatenate training features
        y_train = pd.concat([df['y'] for df in train_dfs])  # Concatenate training labels
        X_test = test_df.drop(columns='y', errors='ignore')  # Get test features
        y_test = test_df['y']  # Get test labels

        print("Optimizing XGBoost...")  # Print optimization status
        best_model = optimize_xgb(X_train, y_train)  # Optimize and get best model
        best_model.fit(X_train, y_train)  # Fit best model on training data
        y_pred = best_model.predict(X_test)  # Predict on test data
        mae = mean_absolute_error(y_test, y_pred)  # Calculate MAE
        if 'XGBoost' not in results[dict_name]:  # Initialize results if not already
            results[dict_name]['XGBoost'] = []
        results[dict_name]['XGBoost'].append(mae)  # Append MAE to results

        if 'XGBoost' not in predictions[dict_name]:  # Initialize predictions if not already
            predictions[dict_name]['XGBoost'] = []
        predictions[dict_name]['XGBoost'].append((y_test.values, y_pred))  # Append predictions

    return results, predictions  # Return results and predictions

def print_results(results, description):  # Define function to print results
    print(f"\n{description} Results:")  # Print description
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        print(f"\nDictionary: {dict_name}")  # Print dictionary name
        for model_name, maes in dict_results.items():  # Loop through each model's results
            avg_mae = np.mean(maes)  # Calculate average MAE
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")  # Print model and average MAE

def plot_box_plots(results):  # Define function to plot box plots of results
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        plt.figure(figsize=(10, 6))  # Create new figure
        data = [maes for model_name, maes in dict_results.items()]  # Get list of MAEs
        plt.boxplot(data, labels=dict_results.keys())  # Create box plot
        plt.title(f'Box Plot of MAE for {dict_name}')  # Set plot title
        plt.xlabel('Model')  # Set x-axis label
        plt.ylabel('Mean Absolute Error')  # Set y-axis label
        plt.show()  # Display plot

def plot_predictions_vs_actuals(predictions):  # Define function to plot actual vs predicted values
    for dict_name, dict_preds in predictions.items():  # Loop through each dictionary of predictions
        plt.figure(figsize=(10, 6))  # Create new figure
        for model_name, preds in dict_preds.items():  # Loop through each model's predictions
            y_test, y_pred = preds[0]  # Get actual and predicted values
            plt.plot(y_test, label=f'Actual {model_name}')  # Plot actual values
            plt.plot(y_pred, label=f'Predicted {model_name}')  # Plot predicted values
        plt.title(f'{dict_name} - Actual vs Predicted')  # Set plot title
        plt.xlabel('Sample')  # Set x-axis label
        plt.ylabel('Value')  # Set y-axis label
        plt.legend()  # Show legend
        plt.show()  # Display plot
