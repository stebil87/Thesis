import numpy as np
import pandas as pd
from scipy.stats import zscore

def detect_replace_outliers(df, target='y', z_thresh=4):
    df_cleaned = df.copy()
    numeric_df = df_cleaned.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number])
    combined_signal = numeric_df.values.flatten()
    zs = np.abs(zscore(combined_signal, nan_policy='omit'))
    
    outliers = np.where(zs > z_thresh)[0]
    
    for idx in outliers:
        non_outlier_idx = idx - 1
        while non_outlier_idx >= 0 and non_outlier_idx in outliers:
            non_outlier_idx -= 1

        if non_outlier_idx < 0:
            non_outlier_idx = idx + 1
            while non_outlier_idx < len(combined_signal) and non_outlier_idx in outliers:
                non_outlier_idx += 1
        
        if 0 <= non_outlier_idx < len(combined_signal):
            combined_signal[idx] = combined_signal[non_outlier_idx]
        else:
            combined_signal[idx] = np.nan

    reshaped_signal = combined_signal.reshape(numeric_df.shape)
    df_cleaned.update(pd.DataFrame(reshaped_signal, columns=numeric_df.columns, index=numeric_df.index))
    
    return df_cleaned