from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor

def regression(models, dictionaries, sprouting_days):
    results = {}  # Initialize a dictionary to store MAE results
    individual_maes = {}  # Initialize a dictionary to store individual MAEs for each model and dataset
    individual_predictions = {}  # Initialize a dictionary to store individual predictions for each model and dataset
    delta_days = {}  # Initialize a dictionary to store average delta days
    trained_models = {}  # Initialize a dictionary to store trained models

    for dict_name, df_dict in dictionaries.items():  # Loop through each dictionary of dataframes
        print(f"Evaluating {dict_name}...")
        results[dict_name] = {}  # Initialize dictionary for each key in dictionaries
        individual_maes[dict_name] = {}  # Initialize dictionary for individual MAEs
        individual_predictions[dict_name] = {}  # Initialize dictionary for individual predictions
        delta_days[dict_name] = {}  # Initialize dictionary for delta days
        trained_models[dict_name] = {}  # Initialize dictionary for trained models

        for model_name, model in models.items():  # Loop through each model
            print(f"  Using model {model_name}...")
            loo = LeaveOneOut()  # Initialize Leave-One-Out cross-validator
            maes = []  # List to collect MAE values
            model_predictions = {}  # Dictionary to store predictions for each test set
            model_predicted_dates = {}  # Dictionary to store predicted dates for each test set

            df_names = list(df_dict.keys())  # List of dataframe names

            for train_index, test_index in loo.split(df_names):  # Loop through each split of dataframes
                train_df_names = [df_names[i] for i in train_index]  # Get training dataframe names
                test_df_name = df_names[test_index[0]]  # Get test dataframe name
                train_dfs = [df_dict[name] for name in train_df_names]  # Get training dataframes
                test_df = df_dict[test_df_name]  # Get test dataframe

                X_train_list = []  # List to store training features
                y_train_list = []  # List to store training labels

                for train_df in train_dfs:  # Loop through each training dataframe
                    X_train_list.append(train_df.drop(columns=['y', 'timestamp'], errors='ignore').values)
                    y_train_list.append(train_df['y'].values)

                # Concatenate all training features and labels
                X_train_array = np.concatenate(X_train_list, axis=0)
                y_train_array = np.concatenate(y_train_list, axis=0)

                # Get test features and labels
                X_test = test_df.drop(columns=['y', 'timestamp'], errors='ignore').values
                y_test = test_df['y'].values

                # Fit the model
                model.fit(X_train_array, y_train_array)
                if test_df_name not in trained_models[dict_name]:
                    trained_models[dict_name][test_df_name] = {}
                trained_models[dict_name][test_df_name][model_name] = model  # Store the trained model
                y_pred = model.predict(X_test)  # Make predictions

                # Calculate MAE and store it
                mae = mean_absolute_error(y_test, y_pred)
                maes.append(mae)

                if test_df_name not in model_predictions:
                    model_predictions[test_df_name] = []
                    model_predicted_dates[test_df_name] = []

                model_predictions[test_df_name].extend(y_pred)  # Store predictions

                for i, pred in enumerate(y_pred):
                    pred_float = float(pred)
                    predicted_date = test_df['timestamp'].iloc[i] + timedelta(days=abs(pred_float))
                    model_predicted_dates[test_df_name].append(predicted_date)  # Store predicted dates

            for df_name in model_predicted_dates.keys():
                predicted_dates = model_predicted_dates[df_name]
                if predicted_dates:
                    epoch = datetime.utcfromtimestamp(0)
                    date_nums = [(date - epoch).total_seconds() for date in predicted_dates]
                    avg_date_num = np.mean(date_nums)
                    avg_predicted_date = epoch + timedelta(seconds=avg_date_num)

                    ground_truth_date = sprouting_days.get(df_name, None)
                    if ground_truth_date is not None:
                        delta = abs((avg_predicted_date - ground_truth_date).days)
                        if model_name not in delta_days[dict_name]:
                            delta_days[dict_name][model_name] = []
                        delta_days[dict_name][model_name].append(delta)  # Store delta days

            results[dict_name][model_name] = np.mean(maes)  # Store average MAE for each model
            individual_maes[dict_name][model_name] = maes  # Store individual MAEs
            individual_predictions[dict_name][model_name] = model_predictions  # Store individual predictions
            print(f"    Average MAE: {np.mean(maes):.4f}")

        print(f"Trained models for {dict_name}: {trained_models[dict_name].keys()}")

    for dict_name, model_deltas in delta_days.items():  # Calculate average delta days
        for model_name, deltas in model_deltas.items():
            delta_days[dict_name][model_name] = np.mean(deltas)

    return results, individual_maes, individual_predictions, delta_days, trained_models  # Return results

def pretty_print(results, delta_days):
    for dict_name, model_results in results.items():  # Print average MAEs
        print(f"\nAverage MAEs for {dict_name}:")
        for model_name, avg_mae in model_results.items():
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")
        
        print(f"\nAverage Delta Days for {dict_name}:")  # Print average delta days
        for model_name, avg_delta in delta_days.get(dict_name, {}).items():
            print(f"  {model_name}: Average Delta Days = {avg_delta:.4f}")
