import os
import cv2
from concurrent.futures import ThreadPoolExecutor

try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("base path is " + base_path)
except:
    # Fallback: If __file__ is not defined, use the current working directory
    base_path = os.getcwd()
    print("base path is " + base_path)
    
# Set the folder path containing the images to be resized
folder_path = "datasets/coco128/images/train"
new_path = "datasets/train"

# Set the new size for the images
new_size = (352, 352)
os.makedirs(os.path.join(base_path,new_path), exist_ok=True)

def resize_image(filename):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):
        image = cv2.imread(os.path.join(folder_path, filename))
        # Resize the image
        resized_image = cv2.resize(image, new_size)
        # Save the resized image with the same filename
        cv2.imwrite(os.path.join(new_path, filename), resized_image)
        return 1
    return 0

# Loop through all the files in the folder using multithreading
with ThreadPoolExecutor() as executor:
    results = list(executor.map(resize_image, os.listdir(folder_path)))

print(f"Processed {sum(results)} images")

