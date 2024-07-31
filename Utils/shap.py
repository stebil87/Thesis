import shap
import numpy as np
import pandas as pd
import warnings
from shap import KernelExplainer, TreeExplainer
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

def compute_shap_values_for_dfs(dictionaries, trained_models):
    shap_values_dict = {}

    for dict_name, df_dict in dictionaries.items():
        shap_values_dict[dict_name] = {}

        for model_type in ['XGB', 'ADAB', 'LGBM']:
            shap_values_dict[dict_name][model_type] = {}

            for df_name, df in df_dict.items():
                model = trained_models.get(dict_name, {}).get(df_name, {}).get(model_type, None)
                if model is not None:
                    X = df.drop(columns=['y', 'timestamp'], errors='ignore')
                    if model_type in ['XGB', 'LGBM']:
                        explainer = TreeExplainer(model)
                    elif model_type == 'ADAB':
                        explainer = KernelExplainer(model.predict, shap.kmeans(X, 10))
                    shap_values_for_df = explainer.shap_values(X)
                    shap_values_dict[dict_name][model_type][df_name] = (shap_values_for_df, X.columns)

    return shap_values_dict

def average_shap_values(shap_values_dict):
    averaged_shap_values = {}

    for dict_name, model_dict in shap_values_dict.items():
        averaged_shap_values[dict_name] = {}

        for model_type, df_dict in model_dict.items():
            all_shap_values = []
            feature_names = None  

            for df_name, (shap_values_for_df, feature_names_for_df) in df_dict.items():
                if shap_values_for_df is not None:
                    all_shap_values.append(shap_values_for_df)
                    if feature_names is None:
                        feature_names = feature_names_for_df 

            if len(all_shap_values) > 0:
                all_shap_values = np.concatenate(all_shap_values, axis=0)
                mean_abs_shap_values = np.mean(np.abs(all_shap_values), axis=0)
            else:
                mean_abs_shap_values = np.array([])

            averaged_shap_values[dict_name][model_type] = {
                'mean_abs_shap_values': mean_abs_shap_values,
                'feature_names': feature_names
            }

    return averaged_shap_values

def plot_averaged_absolute_shap_values(averaged_shap_values):
    for dict_name, model_dict in averaged_shap_values.items():
        for model_type, data in model_dict.items():
            mean_abs_shap_values = data['mean_abs_shap_values']
            feature_names = data['feature_names']
            
            if feature_names is not None and len(mean_abs_shap_values) > 0:
                plt.figure(figsize=(15, 7))
                plt.barh(feature_names, mean_abs_shap_values)
                plt.xlabel('Mean Absolute SHAP Value')
                plt.ylabel('Feature Names')
                plt.title(f"Averaged Absolute SHAP Values for {dict_name} - {model_type}")
                plt.show()

def compute_and_plot_shap(dictionaries, trained_models):
    shap_values_dict = compute_shap_values_for_dfs(dictionaries, trained_models)
    averaged_shap_values = average_shap_values(shap_values_dict)
    plot_averaged_absolute_shap_values(averaged_shap_values)


