import pandas as pd

def clean_timestamp(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def clean_all_dataframes(dataframes):
    for name, df in dataframes.items():
        print(f"Cleaning dataframe: {name}")
        dataframes[name] = clean_timestamp(df)
        print(f"Finished cleaning dataframe: {name}")
    return dataframes