import pandas as pd
import matplotlib.pyplot as plt

def fill_gaps(df_dict):
    for key, df in df_dict.items():
        print(f"Processing dataframe: {key}")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        full_range = pd.date_range(start=start_time, end=end_time, freq='S')
        df_filled = pd.DataFrame({'timestamp': full_range})
        df_filled = df_filled.merge(df, on='timestamp', how='left')
        df_filled['mV'] = df_filled['mV'].fillna(pd.NA)
        df_dict[key] = df_filled
        
        print(f"Finished processing dataframe: {key}")
        print()

    return df_dict
