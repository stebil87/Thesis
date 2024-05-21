import numpy as np
import pandas as pd
from scipy.stats import boxcox, zscore
from scipy.special import boxcox1p

def apply_box_cox(df, target='y', shift=0.01):
    df_transformed = df.copy()
    for column in df_transformed.drop(columns=[target], errors='ignore').columns:
        min_value = df_transformed[column].min()
        if min_value <= 0:
            shift_value = shift - min_value
            df_transformed[column] = df_transformed[column] + shift_value
        df_transformed[column], _ = boxcox(df_transformed[column])
    return df_transformed