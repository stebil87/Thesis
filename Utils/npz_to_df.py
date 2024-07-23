import numpy as np
import pandas as pd
import os

def load_npz_files(folder_path):
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    npz_files.sort()  
    dataframes = {}  
    
    for i, file_name in enumerate(npz_files, start=1):
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path)
        x = data['x']
        y = data['y']
        df = pd.DataFrame(x, columns=[f'x{j+1}' for j in range(x.shape[1])])
        df['y'] = y
        new_df_name = f'df{i}'
        dataframes[new_df_name] = df  
        print(f"Original filename: {file_name} -> New dataframe name: {new_df_name}")
        
    return dataframes