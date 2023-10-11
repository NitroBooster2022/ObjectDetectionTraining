import os
import random
import cv2
import numpy as np
import shutil
from concurrent.futures import ThreadPoolExecutor
import time

# Set the base path to the image, label, and road_brightness folders
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("base path is " + base_path)
except:
    # Fallback: If __file__ is not defined, use the current working directory
    base_path = os.getcwd()
    print("base path is " + base_path)

road_brightness_path = os.path.join(base_path, "bg_new")

# Create new folders if they don't exist
os.makedirs(os.path.join(base_path, "images_padV"), exist_ok=True)
os.makedirs(os.path.join(base_path, "labels_padV"), exist_ok=True)

n = 30955  # number of images
target_height = 480  # Change target_width to target_height
num_road_brightness_images = 7000

def process_image(idx):
    # Prepare paths
    original_image_path = (base_path + "/images_resized/" + f"{idx}.jpg")
    original_label_path = (base_path + "/labels/" + f"{idx}.txt")
    new_image_path = (base_path + "/images_padV/" + f"{idx}.jpg")
    new_label_path = (base_path + "/labels_padV/" + f"{idx}.txt")

    # Read image and label
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Could not read image {original_image_path}")
        return 0

    img_height, img_width, _ = img.shape

    with open(original_label_path, "r") as label_file:
        label = label_file.readline().strip().split()
        class_id, x_center, y_center, width, height = [float(x) for x in label]

    # Calculate the padding needed to make the image have a height of target_height
    padding = target_height - img_height  # Change img_width to img_height
    pad_top = random.randint(0, padding)
    pad_bottom = padding - pad_top

    # Load a random image from road_brightness folder
    random_image_number = random.randint(1, 1335)
    random_image_path = os.path.join(road_brightness_path, f"{random_image_number}.jpg")
    random_image = cv2.imread(random_image_path)
    if random_image is None:
        print(random_image_path+" is None!!")
    while random_image is None:
        random_image_number = random.randint(1, 1335)
        random_image_path = os.path.join(road_brightness_path, f"{random_image_number}.jpg")
        random_image = cv2.imread(random_image_path)
    # Extract a random region from the random_image with required padding size
    rand_img_height, rand_img_width, _ = random_image.shape
    
    while rand_img_height < 480 or rand_img_width < 640:
        random_image_number = random.randint(1, 1335)
        random_image_path = os.path.join(road_brightness_path, f"{random_image_number}.jpg")
        random_image = cv2.imread(random_image_path)
        rand_img_height, rand_img_width, _ = random_image.shape
    x_offset = random.randint(0, rand_img_width - img_width)
    if rand_img_height - pad_top - pad_bottom <0:
        print(random_image_path, rand_img_height, pad_top, pad_bottom)
    y_offset = random.randint(0, rand_img_height - pad_top - pad_bottom)
    padding_top_region = random_image[y_offset:y_offset + pad_top, x_offset:x_offset + img_width]
    padding_bottom_region = random_image[y_offset + pad_top:y_offset + pad_top + pad_bottom, x_offset:x_offset + img_width]

    # Pad the image with the random regions
    img_square = np.vstack((padding_top_region, img, padding_bottom_region))

    # Update label to account for the padding
    new_y_center = (y_center * img_height + pad_top) / target_height  # Change x_center to y_center
    new_height = height * img_height / target_height  # Change width to height

    # Write the new label file
    with open(new_label_path, "w") as new_label_file:
        new_label_file.write(f"{int(class_id)} {x_center} {new_y_center} {width} {new_height}\n")

    # Save the new square image
    cv2.imwrite(new_image_path, img_square)
    return 1

t1 = time.time()
# Use ThreadPoolExecutor for multithreading
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, range(n)))

print(f"Processed {sum(results)} images")
print("time: ", time.time()-t1)
