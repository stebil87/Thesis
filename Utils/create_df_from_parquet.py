import os
import pandas as pd

def merge_parquets_to_dataframes(base_dir):
    dataframes = {}
    df_counter = 1  # Initialize a counter for DataFrame naming
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            parquet_files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith('.parquet')]
            if parquet_files:
                dfs = [pd.read_parquet(pf) for pf in parquet_files]
                full_df = pd.concat(dfs, ignore_index=True)
                
                # Drop the 'timezone' column if it exists
                if 'timezone' in full_df.columns:
                    full_df.drop(columns=['timezone'], inplace=True)
                
                # Sort by 'timestamp'
                full_df.sort_values('timestamp', inplace=True)
                
                # Reset index to ensure it starts fresh from 0
                full_df.reset_index(drop=True, inplace=True)
                
                # Store the DataFrame using a numeric key
                dataframes[f'df{df_counter}'] = full_df
                df_counter += 1  # Increment the counter
    return dataframes
