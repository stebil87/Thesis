import pandas as pd
import matplotlib.pyplot as plt

def fill_gaps(df_dict):
    for key, df in df_dict.items():
        print(f"Processing dataframe: {key}")
        
        # Convert timestamps to datetime and sort the dataframe
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
        
        # Create a full range of timestamps
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        full_range = pd.date_range(start=start_time, end=end_time, freq='S')
        
        # Create a new dataframe with the full range of timestamps
        df_filled = pd.DataFrame({'timestamp': full_range})
        
        # Merge the original dataframe with the full range dataframe
        df_filled = df_filled.merge(df, on='timestamp', how='left')
        
        # Fill missing mV values with 0
        df_filled['mV'] = df_filled['mV'].fillna(0)

        # Update the dictionary with the filled dataframe
        df_dict[key] = df_filled
        
        print(f"Finished processing dataframe: {key}")
        print()

    return df_dict