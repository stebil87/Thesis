import os

def rename_station_folders(base_dir):
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    dirs.sort()  

    for i, directory in enumerate(dirs, start=1):
        original_path = os.path.join(base_dir, directory)
        new_path = os.path.join(base_dir, str(i))
        os.rename(original_path, new_path)