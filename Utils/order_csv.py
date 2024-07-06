import os
import pandas as pd

def check_and_order_csvs(base_dir):

    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):  
            csv_path = os.path.join(subdir_path, f"{subdir}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                if not df['timestamp'].is_monotonic_increasing:
                    sorted_df = df.sort_values(by='timestamp').reset_index(drop=True)
                    df = df.reset_index(drop=True)
                    moved_rows = (df['timestamp'] != sorted_df['timestamp']).sum()
                    df = sorted_df
                    df.to_csv(csv_path, index=False)
                    print(f"Reordered {moved_rows} rows in {csv_path}")
                else:
                    print(f"The file {csv_path} was already in the correct order.")

