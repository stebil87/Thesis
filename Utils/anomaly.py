import numpy as np
import pandas as pd
from scipy.stats import zscore

def detect_replace_outliers_rowwise(df, target='y', z_thresh=4):
    df_cleaned = df.copy()
    
    # Handle both DataFrame and Series correctly
    if isinstance(df_cleaned, pd.Series):
        df_cleaned = df_cleaned.to_frame()

    numeric_df = df_cleaned.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number])

    def replace_outliers(row):
        mean = np.mean(row)
        std_dev = np.std(row)
        is_outlier = np.abs(row - mean) > z_thresh * std_dev

        for idx in range(len(row)):
            if is_outlier[idx]:
                non_outlier_idx = idx - 1
                while non_outlier_idx >= 0 and is_outlier[non_outlier_idx]:
                    non_outlier_idx -= 1
                if non_outlier_idx >= 0:
                    row[idx] = row[non_outlier_idx]
                else:
                    non_outlier_idx = idx + 1
                    while non_outlier_idx < len(row) and is_outlier[non_outlier_idx]:
                        non_outlier_idx += 1
                    if non_outlier_idx < len(row):
                        row[idx] = row[non_outlier_idx]
                    else:
                        row[idx] = np.nan
        return row

    numeric_df = numeric_df.apply(replace_outliers, axis=1)
    
    # Preserve the 'y' column if it exists
    if target in df_cleaned.columns:
        numeric_df[target] = df_cleaned[target]
    
    df_cleaned.update(numeric_df)
    
    return df_cleaned

def apply_outlier_detection_to_dictionaries(dicts):
    cleaned_dicts = {}
    for dict_name, dataframes in dicts.items():
        cleaned_dicts[dict_name] = {name: detect_replace_outliers_rowwise(df) for name, df in dataframes.items()}
    return cleaned_dicts

