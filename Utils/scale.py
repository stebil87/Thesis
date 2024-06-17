import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def scale_features(df, target_column='y'):
    scaler = RobustScaler()

    if target_column in df.columns:
        features = df.drop(columns=[target_column], errors='ignore')
        target = df[target_column]
    else:
        features = df
        target = None

    if not features.empty and features.select_dtypes(include=[np.number]).shape[1] > 0:
        scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
        
        if target is not None:
            scaled_df = pd.concat([scaled_features, target], axis=1)
        else:
            scaled_df = scaled_features
        
        return scaled_df
    else:
        print(f"Skipping scaling because it is empty or does not contain numeric features.")
        return df

def scale(dictionaries, target_column='y'):
    scaled_dictionaries = {}
    for dict_name, df_dict in dictionaries.items():
        if isinstance(df_dict, dict):  
            scaled_dictionaries[dict_name] = {}
            for scale, df in df_dict.items():
                scaled_dictionaries[dict_name][scale] = scale_features(df, target_column=target_column)
        else:
            scaled_dictionaries[dict_name] = scale_features(df_dict, target_column=target_column)
    return scaled_dictionaries