import os
import random
import cv2

image_folder = "test_images"
label_folder = "test_labels"

# Get the list of label files
label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]

# Select a random label file
random_label_file = random.choice(label_files)
print(random_label_file)
# random_label_file = 'frame_0000.txt'
# Read the label file
with open(os.path.join(label_folder, random_label_file), "r") as label_file:
    label = label_file.readline().strip().split()
    class_id, x_center, y_center, width, height = [float(x) for x in label]

# Prepare the image file path
image_filename = os.path.splitext(random_label_file)[0] + ".jpg"
image_path = os.path.join(image_folder, image_filename)

# Read the image
img = cv2.imread(image_path)
if img is None:
    print(f"Could not read image {image_path}")
else:
    img_height, img_width, _ = img.shape

    # Calculate the bounding box coordinates
    x1 = int((x_center - width / 2) * img_width)
    y1 = int((y_center - height / 2) * img_height)
    x2 = int((x_center + width / 2) * img_width)
    y2 = int((y_center + height / 2) * img_height)

    # Draw the bounding box on the image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the image with the bounding box
    cv2.imshow("Image with Bounding Box", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
