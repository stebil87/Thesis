import numpy as np
import pandas as pd
from scipy.stats import boxcox, zscore
from scipy.special import boxcox1p


def detect_replace_outliers(df, target='y', z_thresh=4):
    df_cleaned = df.copy()
    for column in df_cleaned.drop(columns=[target], errors='ignore').columns:
        zs = zscore(df_cleaned[column])
        outliers = np.where(np.abs(zs) > z_thresh)[0]
        
        for idx in outliers:
            non_outlier_idx = idx - 1
            while non_outlier_idx >= 0 and non_outlier_idx in outliers:
                non_outlier_idx -= 1

            if non_outlier_idx < 0:
                non_outlier_idx = idx + 1
                while non_outlier_idx < len(df_cleaned) and non_outlier_idx in outliers:
                    non_outlier_idx += 1
            
            if 0 <= non_outlier_idx < len(df_cleaned):
                df_cleaned.at[idx, column] = df_cleaned.at[non_outlier_idx, column]
            else:
                df_cleaned.at[idx, column] = df_cleaned[column].mean()

    return df_cleaned