import os
print("hi")
type = "train"

# Set the base path to the image, label, and road_brightness folders
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("base path is " + base_path)
except:
    # Fallback: If __file__ is not defined, use the current working directory
    base_path = os.getcwd()
    print("base path is " + base_path)
# Set the folder path containing the images
folder_path =  os.path.join(base_path, "datasets", type)

i = 0
# Create a new file called train.txt
with open("datasets/"+ type+".txt", "w") as f:
    # Loop through all the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            # Get the absolute path of the file
            filepath = os.path.abspath(os.path.join(folder_path, filename))
            # Write the filepath to train.txt
            f.write(filepath + "\n")
            i +=1
            if i%100==0:
                print("i: ", i)
