import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

def analyze_gap(df_dict):
    results = {}

    for key, df in df_dict.items():
        print(f"Processing dataframe: {key}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').drop_duplicates(subset='timestamp', keep='first')
        
        timestamps = df['timestamp'].to_list()
        num_gaps = 0
        total_gap_seconds = 0
        gaps_list = []
        gaps_positions = []

        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i-1]).total_seconds()
            if diff > 1:
                gap_seconds = diff - 1
                num_gaps += 1
                total_gap_seconds += gap_seconds
                gaps_list.append(gap_seconds)
                gaps_positions.append(timestamps[i-1])
        
        total_rows = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() + 1
        gap_percentage = (total_gap_seconds / total_rows) * 100
        avg_gap_length = total_gap_seconds / num_gaps if num_gaps > 0 else 0

        results[key] = {
            'num_gaps': num_gaps,
            'total_gap_seconds': total_gap_seconds,
            'total_rows': total_rows,
            'gap_percentage': gap_percentage,
            'avg_gap_length': avg_gap_length
        }
        
        print(f"Finished processing dataframe: {key}")
        print(f"  Total rows: {total_rows}")
        print(f"  Total gap seconds: {total_gap_seconds}")
        print(f"  Gap percentage: {gap_percentage:.2f}%")
        print(f"  Average gap length: {avg_gap_length:.2f} seconds")
        print(f"  Number of gaps: {num_gaps}")
        print()
        
        if gaps_list:
            # Plot histogram and boxplot for the current dataframe
            plt.figure(figsize=(12, 5))
            plt.subplot(121)
            plt.hist(gaps_list, bins=30, edgecolor='black')
            plt.title(f'{key} - Gap Distribution - Histogram')
            plt.xlabel('Gap Length (seconds)')
            plt.ylabel('Frequency')

            plt.subplot(122)
            plt.boxplot(gaps_list)
            plt.title(f'{key} - Gap Length Boxplot')
            plt.ylabel('Gap Length (seconds)')
            plt.tight_layout()
            plt.show()
            
            # Plot the distribution of gaps over the time series
            plt.figure(figsize=(12, 5))
            plt.plot(df['timestamp'], df['mV'], label='Original Data', marker='o', linestyle='', markersize=2)
            for gap_pos in gaps_positions:
                plt.axvline(x=gap_pos, color='r', linestyle='--', linewidth=0.5)
            plt.title(f'{key} - Gaps in Time Series')
            plt.xlabel('Timestamp')
            plt.ylabel('mV')
            plt.legend()
            plt.show()
        
    return results

