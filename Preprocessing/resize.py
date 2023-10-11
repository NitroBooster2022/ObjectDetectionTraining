import os
import cv2
from concurrent.futures import ThreadPoolExecutor
import numpy as np 

# Set the folder path containing the images to be resized
folder_path = "images"
new_path = "images_resized"

# Set the new size for the images
new_size = (640, 480)
os.makedirs(new_path, exist_ok=True)

def resize_image(filename):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):
        image = cv2.imread(os.path.join(folder_path, filename))
        # Resize the image
        ratio = np.random.uniform(low = 0.1, high = 0.42)
        resized_image = cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))
        # Save the resized image with the same filename
        cv2.imwrite(os.path.join(new_path, filename), resized_image)
        return 1
    return 0

# Loop through all the files in the folder using multithreading
with ThreadPoolExecutor() as executor:
    results = list(executor.map(resize_image, os.listdir(folder_path)))

print(f"Processed {sum(results)} images")
