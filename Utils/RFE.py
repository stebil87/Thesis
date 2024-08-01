import shap
import numpy as np
import pandas as pd
import warnings
from shap import KernelExplainer, TreeExplainer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


def perform_feature_selection(shap_values_dict, n_features_to_select=10):
    selected_features_dict = {}

    for dict_name, model_dict in shap_values_dict.items():
        selected_features_dict[dict_name] = {}

        for model_type, df_dict in model_dict.items():
            selected_features_dict[dict_name][model_type] = {}

            for df_name, (shap_values_for_df, feature_names) in df_dict.items():
                if shap_values_for_df is not None:
                    estimator = LinearRegression()
                    selector = RFE(estimator, n_features_to_select=n_features_to_select)
                    selector = selector.fit(shap_values_for_df, np.zeros(shap_values_for_df.shape[0]))

                    selected_features = feature_names[selector.support_]
                    selected_features_dict[dict_name][model_type][df_name] = list(selected_features)  

    return selected_features_dict

def create_dfs_with_selected_features(dictionaries, selected_features_dict):
    selected_dfs = {}

    for dict_name, model_dict in selected_features_dict.items():
        for model_type, df_features_dict in model_dict.items():
            for df_name, selected_features in df_features_dict.items():
                original_df = dictionaries[dict_name][df_name]
                
                print(f"Processing DataFrame: {df_name}")
                print(f"Selected features: {selected_features}")
                print(f"Original DataFrame columns: {original_df.columns}")

                if not isinstance(selected_features, list):
                    selected_features = list(selected_features)
                required_columns = selected_features + ['y', 'timestamp']

                for col in required_columns:
                    if col not in original_df.columns:
                        print(f"Column {col} not found in DataFrame {df_name}")
                        continue

                selected_df = original_df[required_columns].copy()
                selected_dfs[df_name] = selected_df

    return selected_dfs