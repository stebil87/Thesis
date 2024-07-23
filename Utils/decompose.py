import pandas as pd
import numpy as np
import pywt

def frequency_to_scale(wavelet, frequency, sampling_period):
    return pywt.scale2frequency(wavelet, 1.0 / frequency) * sampling_period

def cwt(dataframes, frequency, wavelet='morl', sampling_period=1):
    scale = frequency_to_scale(wavelet, frequency, sampling_period)
    print(f"Using scale: {scale} for frequency: {frequency}")

    transformed_dataframes = {}

    for key, df in dataframes.items():
        print(f"Processing dataframe: {key}")
        
        if 'timestamp' in df.columns:
            timestamp_column = df['timestamp']
        else:
            timestamp_column = None

        y_column = df['y'].values if 'y' in df.columns else None

    
        signal_columns = [col for col in df.columns if col not in ['timestamp', 'y']]
        
        transformed_rows = []

        for i, row in df[signal_columns].iterrows():
            signal = row.values
            coefficients, _ = pywt.cwt(signal, [scale], wavelet, sampling_period=sampling_period)
     
            transformed_row = coefficients.flatten()
            
            if y_column is not None:
                transformed_row = np.append(transformed_row, y_column[i])
            
            transformed_rows.append(transformed_row)

        num_features = len(transformed_rows[0]) - 1 if y_column is not None else len(transformed_rows[0])
        feature_columns = [f'feature_{i}' for i in range(num_features)]
        if y_column is not None:
            feature_columns.append('y')

        transformed_df = pd.DataFrame(transformed_rows, columns=feature_columns)

        if timestamp_column is not None:
            transformed_df.insert(0, 'timestamp', timestamp_column.values)

        transformed_dataframes[key] = transformed_df

        print(f"Finished processing dataframe: {key}")
        print()

    return transformed_dataframes
