import os
import math
import statistics
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
label_folder = os.path.join(base_path, "datasets/coco128/labels/train")

# Get the list of label files
label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]

def read_label(file):
    with open(os.path.join(label_folder, file), "r") as label_file:
        label = label_file.readline().strip().split()
        _, _, _, width, height = [float(x) for x in label]
        return width*640, height*480

t1 = time.time()
# Use ThreadPoolExecutor for multithreading
with ThreadPoolExecutor() as executor:
    dimensions = list(executor.map(read_label, label_files))

# Calculate the average width and height
widths = [w for w, _ in dimensions]
heights = [h for _, h in dimensions]

avg_width = sum(widths) / len(widths)
avg_height = sum(heights) / len(heights)

# Calculate the standard deviation for width and height
std_dev_width = statistics.stdev(widths)
std_dev_height = statistics.stdev(heights)

# Find the minimum and maximum width and height
min_width = min(widths)
max_width = max(widths)
min_height = min(heights)
max_height = max(heights)

print(f"Average width: {avg_width}")
print(f"Average height: {avg_height}")
print(f"Standard deviation of width: {std_dev_width}")
print(f"Standard deviation of height: {std_dev_height}")
print(f"Minimum width: {min_width}")
print(f"Maximum width: {max_width}")
print(f"Minimum height: {min_height}")
print(f"Maximum height: {max_height}")
print(f"Time: {time.time()-t1}")
