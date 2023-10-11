import os
import threading
from concurrent.futures import ThreadPoolExecutor

def rename_file(file_path, new_file_path):
    try:
        os.rename(file_path, new_file_path)
        #print(f"Renamed '{file_path}' to '{new_file_path}'")
    except Exception as e:
        print(f"Error renaming '{file_path}' to '{new_file_path}': {e}")

def process_file(file_path):
    file_name = os.path.basename(file_path)
    parent_folder = os.path.dirname(file_path)
    new_file_name = 's' + file_name
    new_file_path = os.path.join(parent_folder, new_file_name)
    rename_file(file_path, new_file_path)

def main():
    folder_path = "frames0505/train640"
    if not os.path.exists(folder_path):
        print("The folder does not exist.")
        return

    with ThreadPoolExecutor() as executor:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                executor.submit(process_file, file_path)

if __name__ == "__main__":
    main()
    print("done")
