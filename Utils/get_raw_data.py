import numpy as np
from glob import iglob
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_mV_column(df, plant_id):
    # Extract the 'mV' column from the dataframe
    mV_column = df['mV']
    
    # Plot the 'mV' column
    plt.plot(mV_column)
    plt.xlabel('Index')
    plt.ylabel('mV')
    plt.title(plant_id)
    plt.show()

def load_parquet_files(parquet_files):
    dfs = [pd.read_parquet(file) for file in parquet_files]
    df = pd.concat(dfs, ignore_index=False)
    return df

# Define the output directory
output_folder = 'C:/Users/stebi/Desktop/potatoes/'
if not os.path.exists(output_folder):
    # If it doesn't exist, create it
    os.makedirs(output_folder)

all_dfs = {}
for dataset_folder in ['C:/Users/stebi/Desktop/potatoes/year_2021', 'C:/Users/stebi/Desktop/potatoes/year_2022']:

    print(dataset_folder, 'qui')
    dataset_id = os.path.basename(dataset_folder)  # Get the base name of the dataset folder

    items_per_plant = {}
    df_per_plant = {}
    for year_folder in iglob(os.path.join(dataset_folder, '*')):
        if 'README.txt' not in year_folder:
            for month_folder in iglob(os.path.join(year_folder, '*')):
                for day_folder in iglob(os.path.join(month_folder, '*')):
                    for plant_folder in iglob(os.path.join(day_folder, '*')):
                        plant_id = os.path.basename(plant_folder)  # Get the base name of the plant folder
                        if plant_id not in items_per_plant:
                            items_per_plant[plant_id] = []
                        for item in iglob(os.path.join(plant_folder, '*')):
                            items_per_plant[plant_id].append(item)

    for plant_id in items_per_plant:
        csv_filename = os.path.join(output_folder, f"{dataset_id}_{plant_id}.csv")
        if os.path.exists(csv_filename):
            print(f"Skipping {plant_id} as CSV file already exists.")
            continue

        print(f"Currently doing {plant_id}")
        all_curr_items = items_per_plant[plant_id]
        try:
            df = load_parquet_files(all_curr_items)
        except Exception as e:
            print(f"Cannot do {plant_id}: {e}")
            continue

        df = df.sort_values(by='timestamp')
        # plot_mV_column(df, plant_id)
        df_per_plant[plant_id] = df
        df.to_csv(csv_filename)

    all_dfs[dataset_folder] = df_per_plant

print("Processing complete.")
