import os
import shutil

def collect_parquet_files(source_dirs, target_dir):
    plant_dirs = {}
    
    for source_dir in source_dirs:
        print(f"Checking directory: {source_dir}")  
        for root, dirs, files in os.walk(source_dir):
            print(f"Entering root: {root}")  
            for file in files:
                if file.endswith('.parquet'):
                    print(f"Found Parquet file: {file} in {root}")  
                    plant_id = root.split(os.sep)[-1]
                    plant_path = os.path.join(target_dir, plant_id)
                    if plant_id not in plant_dirs:
                        os.makedirs(plant_path, exist_ok=True)
                        plant_dirs[plant_id] = plant_path
                    source_file_path = os.path.join(root, file)
                    dest_file_path = os.path.join(plant_path, file)
                    shutil.copy2(source_file_path, dest_file_path)