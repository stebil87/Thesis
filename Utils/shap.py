import shap
import numpy as np
import pandas as pd
import warnings
from shap import KernelExplainer, TreeExplainer
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

feature_groups = {
    'basic': ['mean', 'std', 'var', 'min', 'max', 'range', 'median', 'skew', 'kurtosis'],
    'fft': ['total_energy', 'power_ratio_low_high', 'peak_freq_1', 'peak_freq_2', 'peak_freq_3'],
    'peaks': ['zero_crossing_rate', 'mean_crossing_rate', 'local_maxima_rate', 'local_minima_rate'],
    'quantiles': ['quantile_25', 'quantile_50', 'quantile_75', 'iqr_25_75', 'iqr_10_90'],
    'envelope': ['envelope_mean', 'envelope_std', 'envelope_max', 'envelope_min', 'envelope_skew', 'envelope_kurtosis']
}

group_colors = {
    'basic': 'blue',
    'fft': 'orange',
    'peaks': 'green',
    'quantiles': 'red',
    'envelope': 'purple'
}

def compute_shap_values_for_dfs(dictionaries, trained_models, model_types):
    shap_values_dict = {}

    for dict_name, df_dict in dictionaries.items():
        shap_values_dict[dict_name] = {}

        for model_type in model_types:
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

            ordered_features = []
            ordered_shap_values = []
            colors = []

            for group, features in feature_groups.items():
                for feature in features:
                    matched_features = [fname for fname in feature_names if fname.endswith('_' + feature)]
                    for matched_feature in matched_features:
                        ordered_features.append(matched_feature)
                        idx = feature_names.tolist().index(matched_feature)
                        ordered_shap_values.append(mean_abs_shap_values[idx])
                        colors.append(group_colors[group])

            averaged_shap_values[dict_name][model_type] = {
                'mean_abs_shap_values': ordered_shap_values,
                'feature_names': ordered_features,
                'colors': colors
            }

    return averaged_shap_values

def plot_averaged_absolute_shap_values(averaged_shap_values):
    for dict_name, model_dict in averaged_shap_values.items():
        for model_type, data in model_dict.items():
            mean_abs_shap_values = data['mean_abs_shap_values']
            feature_names = data['feature_names']
            colors = data['colors']

            if feature_names is not None and len(mean_abs_shap_values) > 0:
                plt.figure(figsize=(15, 7))
                bars = plt.barh(feature_names, mean_abs_shap_values, color=colors)

                plt.xlabel('Mean Absolute SHAP Value')
                plt.ylabel('Feature Names')
                plt.title(f"Averaged Absolute SHAP Values for {dict_name} - {model_type}")

                legend_elements = [Line2D([0], [0], color=color, lw=4, label=group) for group, color in group_colors.items()]
                plt.legend(handles=legend_elements, title="Feature Groups")

                plt.show()

def plot_shap_summary(shap_values_dict):
    overall_shap_values = []
    overall_feature_names = None

    for dict_name, model_dict in shap_values_dict.items():
        for model_type, df_dict in model_dict.items():
            for df_name, (shap_values_for_df, feature_names_for_df) in df_dict.items():
                if shap_values_for_df is not None:
                    overall_shap_values.append(shap_values_for_df)
                    if overall_feature_names is None:
                        overall_feature_names = feature_names_for_df

    if len(overall_shap_values) > 0:
        overall_shap_values = np.concatenate(overall_shap_values, axis=0)
        shap.summary_plot(overall_shap_values, feature_names=overall_feature_names, title="Overall SHAP Summary Plot")

def compute_and_plot_shap(dictionaries, trained_models, model_types=['XGB', 'ADAB', 'LGBM']):
    shap_values_dict = compute_shap_values_for_dfs(dictionaries, trained_models, model_types)
    averaged_shap_values = average_shap_values(shap_values_dict)
    plot_averaged_absolute_shap_values(averaged_shap_values)
    plot_shap_summary(shap_values_dict)
    return shap_values_dict



