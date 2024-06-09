import pandas as pd                          
import numpy as np                          
from sklearn.model_selection import train_test_split, LeaveOneOut  
from sklearn.metrics import mean_absolute_error 
from xgboost import XGBRegressor        
import matplotlib.pyplot as plt             

def conformal_prediction(X_train, y_train, X_cal, y_cal, X_test, alpha=0.1):  # Define function for conformal prediction
    model = XGBRegressor()                    # Initialize XGBoost regressor model
    model.fit(X_train, y_train)               # Fit the model on training data
    
    y_pred_cal = model.predict(X_cal)         # Predict on calibration data
    residuals = np.abs(y_cal - y_pred_cal)    # Calculate absolute residuals
    q = np.quantile(residuals, 1 - alpha)     # Calculate quantile of residuals for uncertainty estimation
    
    y_pred_test = model.predict(X_test)       # Predict on test data
    lower_bound = y_pred_test - q             # Calculate lower bound of prediction interval
    upper_bound = y_pred_test + q             # Calculate upper bound of prediction interval
    
    return y_pred_test, lower_bound, upper_bound  # Return predictions and prediction intervals

def perform_conformal_prediction(dictionaries, alpha=0.1):  # Define function to perform conformal prediction on multiple datasets
    results = {}                               # Initialize dictionary to store results
    predictions = {}                           # Initialize dictionary to store predictions
    loo = LeaveOneOut()                        # Initialize Leave-One-Out cross-validation

    for dict_name, datasets in dictionaries.items():  # Loop through each dictionary of datasets
        print(f"Processing dictionary: {dict_name}")  # Print the name of the current dictionary
        dataframes = list(datasets.values())   # Get list of dataframes in the current dictionary
        results[dict_name] = {}                # Initialize results for the current dictionary
        predictions[dict_name] = {}            # Initialize predictions for the current dictionary

        for train_index, test_index in loo.split(dataframes):  # Loop through each Leave-One-Out split
            train_dfs = [dataframes[i] for i in train_index]  # Get training dataframes
            test_df = dataframes[test_index[0]]               # Get test dataframe

            X_train_val = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])  # Concatenate training features
            y_train_val = pd.concat([df['y'] for df in train_dfs])  # Concatenate training labels
            X_test = test_df.drop(columns='y', errors='ignore')  # Get test features
            y_test = test_df['y']                             # Get test labels
            
            X_train, X_cal, y_train, y_cal = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)  # Split training data for calibration

            y_pred, lower_bound, upper_bound = conformal_prediction(X_train, y_train, X_cal, y_cal, X_test, alpha)  # Get predictions and intervals

            uncertainty = upper_bound - lower_bound  # Calculate uncertainty
            accepted_prediction = y_pred.copy()      # Copy predictions
            accepted_prediction[uncertainty > (2 * np.quantile(uncertainty, 1 - alpha))] = np.nan  # Invalidate predictions with high uncertainty

            accepted_indices = ~np.isnan(accepted_prediction)  # Get indices of accepted predictions
            if np.any(accepted_indices):             # Check if there are accepted predictions
                mae = mean_absolute_error(y_test[accepted_indices], accepted_prediction[accepted_indices])  # Calculate MAE for accepted predictions
            else:
                mae = np.nan                          # Set MAE to NaN if no predictions are accepted

            if 'XGBoost' not in results[dict_name]:  # Initialize results for XGBoost if not already present
                results[dict_name]['XGBoost'] = []
            results[dict_name]['XGBoost'].append(mae)  # Append MAE to results

            if 'XGBoost' not in predictions[dict_name]:  # Initialize predictions for XGBoost if not already present
                predictions[dict_name]['XGBoost'] = []
            predictions[dict_name]['XGBoost'].append((y_test.values, y_pred, lower_bound, upper_bound))  # Append predictions and intervals

    return results, predictions                  # Return results and predictions

def print_results(results, description):          # Define function to print results
    print(f"\n{description} Results:")            # Print description
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        print(f"\nDictionary: {dict_name}")       # Print dictionary name
        for model_name, maes in dict_results.items():  # Loop through each model's results
            avg_mae = np.nanmean(maes)            # Calculate average MAE
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")  # Print model name and average MAE

def plot_box_plots(results):                      # Define function to plot box plots of MAE
    for dict_name, dict_results in results.items():  # Loop through each dictionary of results
        plt.figure(figsize=(10, 6))               # Create a new figure
        data = [maes for model_name, maes in dict_results.items()]  # Get list of MAEs
        plt.boxplot(data, labels=dict_results.keys())  # Create box plot
        plt.title(f'Box Plot of MAE for {dict_name}')  # Set plot title
        plt.xlabel('Model')                       # Set x-axis label
        plt.ylabel('Mean Absolute Error')         # Set y-axis label
        plt.show()                                # Display plot

def plot_predictions_vs_actuals(predictions):     # Define function to plot actual vs. predicted values
    for dict_name, dict_preds in predictions.items():  # Loop through each dictionary of predictions
        plt.figure(figsize=(10, 6))               # Create a new figure
        for model_name, preds in dict_preds.items():  # Loop through each model's predictions
            y_test, y_pred, lower_bound, upper_bound = preds[0]  # Get predictions and intervals
            plt.plot(y_test, label=f'Actual - {model_name}')  # Plot actual values
            plt.plot(y_pred, label=f'Predicted - {model_name}')  # Plot predicted values
            plt.fill_between(range(len(y_test)), lower_bound, upper_bound, alpha=0.2, label=f'Uncertainty - {model_name}')  # Fill uncertainty interval
        plt.title(f'{dict_name} - Actual vs Predicted')  # Set plot title
        plt.xlabel('Sample')                       # Set x-axis label
        plt.ylabel('Value')                        # Set y-axis label
        plt.legend()                               # Show legend
        plt.show()                                 # Display plot
