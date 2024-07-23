"""import numpy as np
import pandas as pd

def window(filtered_dataframes, hours):
    processed_dataframes = {}
    interval_to_rows = {
        6: 12,   # 6 hours - 12 rows
        12: 24,  # 12 hours - 24 rows
        18: 36,  # 18 hours - 36 rows
        24: 48   # 24 hours - 48 rows
    }
    rows_per_window = interval_to_rows[hours]

    for df_name, df in filtered_dataframes.items():
        new_data = []
        n_rows = len(df)
        
        for j in range(0, n_rows, rows_per_window):
            end_index = min(j + rows_per_window, n_rows) 
            group = df.iloc[j:end_index]

            if not group.empty:
                combined_features = group.iloc[:, :-1].values.flatten()
                y_value = group.iloc[-1, -1]  
                combined_row = np.append(combined_features, y_value)  
                new_data.append(combined_row)

        if new_data:
            total_features = len(new_data[0]) - 1 
            new_columns = [f'feature_{i}' for i in range(total_features)] + ['y']
            processed_df = pd.DataFrame(new_data, columns=new_columns)
            processed_df = processed_df.dropna(subset=['y'])
            processed_dataframes[df_name] = processed_df

    return processed_dataframes



def window_from_raw(dataframes, hours):
    processed_dataframes = {}
    interval_to_rows = {
        6: 21600,   
        12: 43200,  
        18: 64800, 
        24: 86400   
    }
    rows_per_window = interval_to_rows[hours]

    for df_name, df in dataframes.items():
        new_data = []
        n_rows = len(df)
        
        for j in range(0, n_rows, rows_per_window):
            end_index = min(j + rows_per_window, n_rows) 
            group = df.iloc[j:end_index]

            if not group.empty:
                combined_features = group.iloc[:, :-1].values.flatten()
                y_value = group.iloc[-1, -1]  
                combined_row = np.append(combined_features, y_value)  
                new_data.append(combined_row)

        if new_data:
            total_features = len(new_data[0]) - 1 
            new_columns = [f'feature_{i}' for i in range(total_features)] + ['y']
            processed_df = pd.DataFrame(new_data, columns=new_columns)
            processed_df = processed_df.dropna(subset=['y'])
            processed_dataframes[df_name] = processed_df

    return processed_dataframes"""
    
import numpy as np
import pandas as pd

def window(dataframes, hours):
    processed_dataframes = {}
    interval_to_rows = {
        6: 21600,   
        12: 43200,  
        18: 64800, 
        24: 86400   
    }
    rows_per_window = interval_to_rows[hours]

    for df_name, df in dataframes.items():
        new_data = []
        n_rows = len(df)
        
        for j in range(0, n_rows, rows_per_window):
            end_index = min(j + rows_per_window, n_rows)
            group = df.iloc[j:end_index]

            if not group.empty:
                timestamp = group.iloc[0, 0]  
                combined_features = group.iloc[:, 1:-1].values.flatten()  
                y_value = group.iloc[-1, -1]  
                combined_row = np.append(np.array([timestamp]), combined_features)  
                combined_row = np.append(combined_row, y_value)  
                new_data.append(combined_row)

        if new_data:
            total_features = len(new_data[0]) - 2  
            new_columns = ['timestamp'] + [f'feature_{i}' for i in range(total_features)] + ['y']
            processed_df = pd.DataFrame(new_data, columns=new_columns)
            processed_df = processed_df.dropna(subset=['y'])
            processed_dataframes[df_name] = processed_df

    return processed_dataframes