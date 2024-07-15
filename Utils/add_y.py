import pandas as pd

match_data = {
    'date': [
        '2022.01.17', '2022.01.14', '2022.01.17', '2022.01.06', '2022.01.18', 
        '2022.02.01', '2022.01.21', '2022.01.22', '2022.01.31', '2022.02.08',
        '2022.01.07', '2022.01.30', '2022.01.04', '2022.01.26', '2022.01.24',
        '2022.01.17', '2022.03.27', '2022.03.30', '2022.04.10', '2022.03.09',
        '2022.05.05', '2022.04.04', '2022.03.09', '2022.04.08', '2022.04.17',
        '2022.06.15', '2022.05.22', '2022.04.15', '2022.03.18', '2022.05.23',
        '2022.05.17', '2022.04.03'
    ],
    'time': [
        '12:00', '23:30', '9:30', '3:30', '11:30', '1:00', '19:30', '1:30', '23:30',
        '5:30', '23:30', '5:30', '23:30', '11:30', '23:30', '23:30', '6:00', '19:00',
        '14:00', '17:00', '2:00', '20:00', '23:50', '14:00', '10:00', '14:00', '23:50',
        '16:00', '8:30', '14:00', '2:00', '2:00'
    ],
    'df': [
        25, 26, 27, 28, 29, 30, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21,
        22, 23, 24, 9, 10, 11, 12, 13, 14, 15, 16
    ]
}

match_table = pd.DataFrame(match_data)

def add_y_column(cleaned_dataframes, match_table):
    print("Keys in cleaned_dataframes:", cleaned_dataframes.keys())
    
    for _, row in match_table.iterrows():
        date_time_str = f"{row['date']} {row['time']}"
        target_datetime = pd.to_datetime(date_time_str, format='%Y.%m.%d %H:%M')
        df_key = f"df{row['df']}"  
        
        if df_key in cleaned_dataframes:
            print(f"Processing dataframe {df_key}")
            df = cleaned_dataframes[df_key]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
     
            df['y'] = (df['timestamp'] - target_datetime).dt.days

            match_indices = df['timestamp'] == target_datetime
            if match_indices.any():
                df.loc[match_indices, 'y'] = 0

            print(f"Updated dataframe {df_key} head with 'y' column:")
            print(df.head())
            
            cleaned_dataframes[df_key] = df
        else:
            print(f"Dataframe {df_key} not found in cleaned_dataframes.")

    return cleaned_dataframes


def drop_timestamp_column(added_dataframes):
    for key in added_dataframes.keys():
        print(f"Dropping 'timestamp' column from dataframe {key}")
        added_dataframes[key] = added_dataframes[key].drop(columns=['timestamp'])
        print(f"Updated dataframe {key} head:")
        print(added_dataframes[key].head())
    return added_dataframes
