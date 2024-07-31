import shap
import numpy as np
import pandas as pd
import warnings
from shap import KernelExplainer, TreeExplainer
import matplotlib.pyplot as plt

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

def compute_shap_values_for_dfs(dictionaries, trained_models):
    shap_values_dict = {}  # Initialize a dictionary to store SHAP values

    for dict_name, df_dict in dictionaries.items():  # Loop through each dictionary of dataframes
        shap_values_dict[dict_name] = {}  # Initialize dictionary for each key in dictionaries

        for model_type in ['XGB', 'ADAB', 'LGBM']:  # Loop through each model type
            shap_values_dict[dict_name][model_type] = {}  # Initialize dictionary for each model type

            for df_name, df in df_dict.items():  # Loop through each dataframe in the dictionary
                model = trained_models.get(dict_name, {}).get(df_name, {}).get(model_type, None)  # Get the corresponding model
                if model is not None:
                    # Drop 'y' and 'timestamp' columns if they exist
                    X = df.drop(columns=['y', 'timestamp'], errors='ignore')
                    # Choose appropriate SHAP explainer based on model type
                    if model_type in ['XGB', 'LGBM']:
                        explainer = TreeExplainer(model)
                    elif model_type == 'ADAB':
                        explainer = KernelExplainer(model.predict, shap.kmeans(X, 10))
                    # Compute SHAP values for the dataframe
                    shap_values_for_df = explainer.shap_values(X)
                    shap_values_dict[dict_name][model_type][df_name] = (shap_values_for_df, X.columns)  # Store SHAP values and feature names

    return shap_values_dict  # Return the dictionary containing all SHAP values

def average_shap_values(shap_values_dict):
    averaged_shap_values = {}  # Initialize a dictionary to store averaged SHAP values

    for dict_name, model_dict in shap_values_dict.items():  # Loop through each dictionary of SHAP values
        averaged_shap_values[dict_name] = {}  # Initialize dictionary for each key in shap_values_dict

        for model_type, df_dict in model_dict.items():  # Loop through each model type
            all_shap_values = []  # List to collect SHAP values for all dataframes
            feature_names = None  # Variable to store feature names

            for df_name, (shap_values_for_df, feature_names_for_df) in df_dict.items():  # Loop through each dataframe's SHAP values
                if shap_values_for_df is not None:
                    all_shap_values.append(shap_values_for_df)  # Collect SHAP values
                    if feature_names is None:
                        feature_names = feature_names_for_df  # Store feature names

            if len(all_shap_values) > 0:
                all_shap_values = np.concatenate(all_shap_values, axis=0)  # Concatenate SHAP values across dataframes
                mean_abs_shap_values = np.mean(np.abs(all_shap_values), axis=0)  # Compute mean absolute SHAP values
            else:
                mean_abs_shap_values = np.array([])  # If no SHAP values, set as empty array

            averaged_shap_values[dict_name][model_type] = {
                'mean_abs_shap_values': mean_abs_shap_values,
                'feature_names': feature_names
            }  # Store averaged SHAP values and feature names

    return averaged_shap_values  # Return the dictionary containing averaged SHAP values

def plot_averaged_absolute_shap_values(averaged_shap_values):
    for dict_name, model_dict in averaged_shap_values.items():  # Loop through each dictionary of averaged SHAP values
        for model_type, data in model_dict.items():  # Loop through each model type
            mean_abs_shap_values = data['mean_abs_shap_values']
            feature_names = data['feature_names']
            
            if feature_names is not None and len(mean_abs_shap_values) > 0:  # Check if there are SHAP values to plot
                plt.figure(figsize=(15, 7))  # Create a new figure
                plt.barh(feature_names, mean_abs_shap_values)  # Create a horizontal bar plot
                plt.xlabel('Mean Absolute SHAP Value')  # Label the x-axis
                plt.ylabel('Feature Names')  # Label the y-axis
                plt.title(f"Averaged Absolute SHAP Values for {dict_name} - {model_type}")  # Title of the plot
                plt.show()  # Display the plot

def compute_and_plot_shap(dictionaries, trained_models):
    shap_values_dict = compute_shap_values_for_dfs(dictionaries, trained_models)  # Compute SHAP values for all dataframes
    averaged_shap_values = average_shap_values(shap_values_dict)  # Compute averaged SHAP values
    plot_averaged_absolute_shap_values(averaged_shap_values)  # Plot the averaged SHAP values
