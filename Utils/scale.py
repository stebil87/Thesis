import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

def scale_features(df, target_column='y', timestamp_column='timestamp'):
    scaler = RobustScaler()

    if target_column in df.columns and timestamp_column in df.columns:
        features = df.drop(columns=[target_column, timestamp_column], errors='ignore')
        target = df[target_column]
        timestamp = df[timestamp_column]
    elif target_column in df.columns:
        features = df.drop(columns=[target_column], errors='ignore')
        target = df[target_column]
        timestamp = None
    elif timestamp_column in df.columns:
        features = df.drop(columns=[timestamp_column], errors='ignore')
        target = None
        timestamp = df[timestamp_column]
    else:
        features = df
        target = None
        timestamp = None

    if not features.empty and features.select_dtypes(include=[np.number]).shape[1] > 0:
        scaled_features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
   
        if target is not None and timestamp is not None:
            scaled_df = pd.concat([timestamp, scaled_features, target], axis=1)
        elif target is not None:
            scaled_df = pd.concat([scaled_features, target], axis=1)
        elif timestamp is not None:
            scaled_df = pd.concat([timestamp, scaled_features], axis=1)
        else:
            scaled_df = scaled_features
        
        return scaled_df
    else:
        print(f"No numeric features to scale in the dataframe.")
        return df

def scale(dictionaries, target_column='y', timestamp_column='timestamp'):
    scaled_dictionaries = {}
    for dict_name, df_dict in dictionaries.items():
        if isinstance(df_dict, dict):  
            scaled_dictionaries[dict_name] = {}
            for scale, df in df_dict.items():
                scaled_dictionaries[dict_name][scale] = scale_features(df, target_column=target_column, timestamp_column=timestamp_column)
        else:
            scaled_dictionaries[dict_name] = scale_features(df_dict, target_column=target_column, timestamp_column=timestamp_column)
    return scaled_dictionaries