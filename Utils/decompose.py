import pandas as pd
import numpy as np
import pywt

def cwt(dataframes, wavelet='morl', scales=None, sampling_period=1):


    transformed_dataframes = {}

    for key, df in dataframes.items():
        print(f"Processing dataframe: {key}")

        signal_columns = [col for col in df.columns if col != 'y']
        y_column = df['y'].values if 'y' in df.columns else None
        
        transformed_rows = []

        for i, row in df[signal_columns].iterrows():
            signal = row.values
            coefficients, _ = pywt.cwt(signal, scales, wavelet, sampling_period=sampling_period)
     
            transformed_row = coefficients.flatten()
            
            if y_column is not None:
                transformed_row = np.append(transformed_row, y_column[i])
            
            transformed_rows.append(transformed_row)

        num_features = len(transformed_rows[0]) - 1 if y_column is not None else len(transformed_rows[0])
        feature_columns = [f'feature_{i}' for i in range(num_features)]
        if y_column is not None:
            feature_columns.append('y')

        transformed_df = pd.DataFrame(transformed_rows, columns=feature_columns)
        transformed_dataframes[key] = transformed_df

        print(f"Finished processing dataframe: {key}")
        print()

    return transformed_dataframes
