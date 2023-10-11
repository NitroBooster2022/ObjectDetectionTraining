import os
from pathlib import Path
import concurrent.futures

def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    with open(file_path, 'w') as file:
        for line in lines:
            parts = line.split()
            first_number = int(float(parts[0]))
            new_line = ' '.join([str(first_number)] + parts[1:])
            file.write(new_line + '\n')
    return 1

def process_all_txt_files(folder_path):
    folder_path = Path(folder_path)
    txt_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == '.txt']

    processed_files = 0
    # Use ThreadPoolExecutor to process files concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_file, txt_files))
        processed_files = sum(results)
        print("Processed files:", processed_files)

# Example usage:
folder_path = "datasets_a/coco128/labels/train"
process_all_txt_files(folder_path)
