import pandas as pd
from datetime import timedelta

def check_temporal_integrity(df_dict):
    for key, df in df_dict.items():
        print(f"Checking temporal integrity for dataframe: {key}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
        time_diff = df['timestamp'].diff().dropna()
        gaps = time_diff[time_diff > timedelta(seconds=1)]
        
        if not gaps.empty:
            print(f"Gaps found in dataframe {key}:")
            for gap in gaps:
                print(f"  Gap of {gap} found")
        else:
            print(f"No gaps found in dataframe {key}")
        
        print()