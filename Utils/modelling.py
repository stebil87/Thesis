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
    results = {} # Dictionary to store the results
    predictions = {} # Dictionary to store the predictions
    loo = LeaveOneOut() # Leave-One-Out cross-validation

    for dict_name, datasets in dictionaries.items(): # Iterate over each dictionary
        print(f"Processing dictionary: {dict_name}")
        dataframes = list(datasets.values()) # List of dataframes in the current dictionary
        results[dict_name] = {} # Initialize results for the current dictionary
        predictions[dict_name] = {'y_test': [], 'y_pred': []} # Initialize predictions for the current dictionary

        for train_index, test_index in loo.split(dataframes): # Perform Leave-One-Out CV
            train_dfs = [dataframes[i] for i in train_index] # Training dataframes
            test_df = dataframes[test_index[0]] # Testing dataframe

            X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs]) # Combine training features
            y_train = pd.concat([df['y'] for df in train_dfs]) # Combine training targets
            X_test = test_df.drop(columns='y', errors='ignore') # Testing features
            y_test = test_df['y'] # Testing target

            models = { # Dictionary of models to be evaluated
                'XGBoost': XGBRegressor(),
                'LightGBM': LGBMRegressor(),
                'AdaBoost': AdaBoostRegressor()
            }

            for model_name, model in models.items(): # Iterate over each model
                model.fit(X_train, y_train) # Fit the model on the training data
                y_pred = model.predict(X_test) # Predict on the testing data
                mae = mean_absolute_error(y_test, y_pred) # Calculate mean absolute error

                if model_name not in results[dict_name]: # Initialize model results if not already
                    results[dict_name][model_name] = []
                results[dict_name][model_name].append(mae) # Append the MAE for the current model

                predictions[dict_name]['y_test'].append(y_test.values) # Append the actual values
                predictions[dict_name]['y_pred'].append(y_pred) # Append the predicted values

    return results, predictions # Return the results and predictions

def print_results(results, description):
    print(f"\n{description} Results:") # Print the description
    for dict_name, dict_results in results.items(): # Iterate over each dictionary in results
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items(): # Iterate over each model in the dictionary
            avg_mae = np.mean(maes) # Calculate the average MAE
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}") # Print the average MAE for the model

def plot_box_plots(results):
    for dict_name, dict_results in results.items(): # Iterate over each dictionary in results
        plt.figure(figsize=(10, 6)) # Create a new figure
        data = [maes for model_name, maes in dict_results.items()] # Extract MAE data for each model
        plt.boxplot(data, labels=dict_results.keys()) # Create a box plot
        plt.title(f'Box Plot of MAE for {dict_name}') # Set the title
        plt.xlabel('Model') # Set the x-axis label
        plt.ylabel('Mean Absolute Error') # Set the y-axis label
        plt.show() # Display the plot

def plot_predictions_vs_actuals(predictions):
    for dict_name, dict_preds in predictions.items(): # Iterate over each dictionary in predictions
        y_test_avg = np.mean(predictions[dict_name]['y_test'], axis=0) # Calculate the average actual values
        y_pred_avg = np.mean(predictions[dict_name]['y_pred'], axis=0) # Calculate the average predicted values

        plt.figure(figsize=(10, 6)) # Create a new figure
        plt.plot(y_test_avg, label='Actual') # Plot the average actual values
        plt.plot(y_pred_avg, label='Predicted') # Plot the average predicted values
        plt.title(f'{dict_name} - Actual vs Predicted') # Set the title
        plt.xlabel('Sample') # Set the x-axis label
        plt.ylabel('Value') # Set the y-axis label
        plt.legend() # Display the legend
        plt.show() # Display the plot

