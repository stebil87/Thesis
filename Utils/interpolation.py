import pandas as pd

def interpolate_mv_column(dataframes_dict):
    for key in dataframes_dict.keys():
        print(f"Interpolating 'mV' column in dataframe {key}")
        df = dataframes_dict[key]
        nan_count_before = df['mV'].isna().sum()
        df['mV'] = df['mV'].interpolate(method='linear', limit_direction='forward', axis=0)
        last_valid_index = df['mV'].last_valid_index()
        df.loc[last_valid_index + 1:, 'mV'] = pd.NA
        nan_count_after = df['mV'].isna().sum()
        interpolations_count = nan_count_before - nan_count_after
        dataframes_dict[key] = df
        
        print(f"Updated dataframe {key} head after interpolation:")
        print(df.head())
        print(f"Number of interpolations in dataframe {key}: {interpolations_count}")
    return dataframes_dict