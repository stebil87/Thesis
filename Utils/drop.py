def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')

import pandas as pd

def drop_rows_y_greater_than_zero(dataframes_dict):
    for key in dataframes_dict.keys():
        print(f"Processing dataframe: {key}")
        df = dataframes_dict[key]
        df = df[df['y'] <= 0]
        dataframes_dict[key] = df
        
        print(f"Finished processing dataframe: {key}")
        print(f"Remaining rows: {len(df)}")
        print()

    return dataframes_dict

import pandas as pd

def filter_dataframes(num_dataframes):
    filtered_dataframes = {}
    
    for i in range(1, num_dataframes + 1):
        df_name = f'df{i}'
        df = globals().get(df_name)
        if df is not None:
            filtered_df = df[df['y'] <= 0]
            filtered_dataframes[df_name] = filtered_df

    return filtered_dataframes