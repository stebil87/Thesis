import numpy as np
import pandas as pd

def check_infinity_and_large_values(df):
    infinity_mask = np.isinf(df)
    large_values_mask = np.abs(df) > 1e308 
    
    problematic_values = infinity_mask | large_values_mask
    
    if problematic_values.any().any():
        problematic_cols = df.columns[problematic_values.any()]
        problematic_rows = df.index[problematic_values.any(axis=1)]
        print(f"Columns with problematic values: {problematic_cols.tolist()}")
        print(f"Rows with problematic values: {problematic_rows.tolist()}")
        
        problematic_df = df[problematic_cols].loc[problematic_rows]
        return problematic_df
    else:
        return None

def replace_problematic_values_with_precedent(df):
    for column in df.columns:
        valid_indices = np.where(~np.isinf(df[column]) & (np.abs(df[column]) <= 1e308))[0]
        
        for idx in df.index:
            if np.isinf(df.loc[idx, column]) or np.abs(df.loc[idx, column]) > 1e308:
                valid_idx = valid_indices[valid_indices < idx]
                if len(valid_idx) > 0:
                    df.loc[idx, column] = df.loc[valid_idx[-1], column]
                else:
                    df.loc[idx, column] = np.nan  
                
    return df.ffill()

def clean_dictionaries(dicts):
    for name, df in dicts.items():
        print(f"Checking {name}:")
        problematic_df = check_infinity_and_large_values(df.drop('y', axis=1))
        if problematic_df is not None:
            print(problematic_df)
            df = replace_problematic_values_with_precedent(df.drop('y', axis=1))
            df['y'] = dicts[name]['y']
            dicts[name] = df
            print(f"Cleaned dataset: {name}")

def clean_augmented_features(augmented_features_continuous, augmented_features_linear, augmented_features_cleaned):
    print("Checking augmented_features_continuous:")
    clean_dictionaries(augmented_features_continuous)

    print("\nChecking augmented_features_linear:")
    clean_dictionaries(augmented_features_linear)

    print("\nChecking augmented_features_cleaned:")
    clean_dictionaries(augmented_features_cleaned)