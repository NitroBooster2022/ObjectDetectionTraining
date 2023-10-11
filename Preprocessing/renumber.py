import os
import glob
import uuid

# Set the folder path containing the images
folder_path = 'frames0505/rf9'
offset = 499

# Find all common image formats in the folder
image_files = glob.glob(os.path.join(folder_path, '*.jpg')) + \
              glob.glob(os.path.join(folder_path, '*.jpeg')) + \
              glob.glob(os.path.join(folder_path, '*.png')) + \
              glob.glob(os.path.join(folder_path, '*.gif'))

# Dictionary to store original file paths and their corresponding temporary file paths
temp_file_paths = {}

# First pass: Rename all images with a temporary unique name
for image_file in image_files:
    # Get the file extension
    file_extension = os.path.splitext(image_file)[1]
    
    # Create a temporary unique file name
    temp_file_name = f"{uuid.uuid4()}{file_extension}"
    
    # Set the temporary file path
    temp_file_path = os.path.join(folder_path, temp_file_name)
    
    # Rename the file with the temporary unique name
    os.rename(image_file, temp_file_path)
    
    # Store the original file path and its corresponding temporary file path
    temp_file_paths[temp_file_path] = image_file

# Second pass: Rename temporary files with the new numbering sequence
for i, (temp_file_path, original_file_path) in enumerate(temp_file_paths.items()):
    # Get the file extension from the original file path
    file_extension = os.path.splitext(original_file_path)[1]
    
    # Create the new file name
    new_file_name = f"{i+offset}{file_extension}"
    
    # Set the new file path
    new_file_path = os.path.join(folder_path, new_file_name)
    
    # Rename the temporary file with the new numbering sequence
    os.rename(temp_file_path, new_file_path)

print("Images renamed successfully.")
