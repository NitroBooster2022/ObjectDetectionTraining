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
road_brightness_path = os.path.join(base_path, "road_brightness")

# Create new folders if they don't exist
os.makedirs(os.path.join(base_path, "datasets/coco128/images/train0/"), exist_ok=True)
os.makedirs(os.path.join(base_path, "datasets/coco128/labels/train/"), exist_ok=True)

n = 30955  # number of images
target_width = 640
num_road_brightness_images = 1390
num_brightness = 3

def process_image(idx):
    # Prepare paths
    original_image_path = (base_path + "/images_padV/" + f"{idx}.jpg")
    original_label_path = (base_path + "/labels_padV/" + f"{idx}.txt")
    new_image_path = (base_path + "/datasets/coco128/images/train0/" + f"{idx}.jpg")
    new_label_path = (base_path + "/datasets/coco128/labels/train/" + f"{idx}.txt")

    # Read image and label
    img = cv2.imread(original_image_path)
    if img is None:
        print(f"Could not read image {original_image_path}")
        return 0

    img_height, img_width, _ = img.shape

    with open(original_label_path, "r") as label_file:
        label = label_file.readline().strip().split()
        class_id, x_center, y_center, width, height = [float(x) for x in label]

    # Calculate the padding needed to make the image have a width of target_width
    padding = target_width - img_width
    pad_left = random.randint(0, padding)
    pad_right = padding - pad_left

    # Load a random image from road_brightness folder
    # random_image_number = random.randint(0, num_road_brightness_images - 1)
    random_image_number = (idx%num_road_brightness_images)*num_brightness+random.randint(0, num_brightness)
    random_image_path = os.path.join(road_brightness_path, f"{random_image_number}.jpg")
    random_image = cv2.imread(random_image_path)
    if random_image is None:
        print(random_image_path +" is None")

    # Extract a random region from the random_image with required padding size
    rand_img_height, rand_img_width, _ = random_image.shape
    x_offset = random.randint(0, rand_img_width - pad_left - pad_right)
    y_offset = random.randint(0, rand_img_height - img_height)
    padding_left_region = random_image[y_offset:y_offset + img_height, x_offset:x_offset + pad_left]
    padding_right_region = random_image[y_offset:y_offset + img_height, x_offset + pad_left:x_offset + pad_left + pad_right]

    # Pad the image with the random regions
    img_square = np.hstack((padding_left_region, img, padding_right_region))

    # Update label to account for the padding
    new_x_center = (x_center * img_width + pad_left) / target_width
    new_width = width * img_width / target_width

    # Write the new label file
    with open(new_label_path, "w") as new_label_file:
        new_label_file.write(f"{int(class_id)} {new_x_center} {y_center} {new_width} {height}\n")

    # Save the new square image
    cv2.imwrite(new_image_path, img_square)
    return 1

t1 = time.time()
# Use ThreadPoolExecutor for multithreading
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, range(n)))

print(f"Processed {sum(results)} images")
print("time: ", time.time()-t1)
