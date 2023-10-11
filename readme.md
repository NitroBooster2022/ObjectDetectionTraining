
PART 1: PREPROCESSING

0) Download Preprocessing.zip and unzip it in the folder Preprocessing

1) Read and Run resize.py to resize all the car images randomly. 
    - examine the new folder "images_resized" created. What do you notice about the size of the images?
    - explain here what it does (3 points):

2) Read and Run pad_height_bg.py 
    - examine the new image folder created. Examine the new label folder created. Compare the new labels with the original ones.
    - explain here what it does (4 points):

3) Read and Run pad_width_bg.py 
    - examine the new image folder created. Examine the new label folder created. Compare the new labels with the original ones.
    - explain here what it does (3 points):

4) Read and Run all the python scripts that start with "process_"
    - examine the generated images.
    - explain here what they do (5 points):

5) Read and Run create_datasets2.py
    - explain here what it does (7 points):

6) Run the script analyze and examine the key statistics of car size.

7) Look at the label files:
    - you'll notice that except for the first line, there's a space at the beginning of each line.
    - Run remove_spaceMT.py to remove that.
    - Run remove_decimalMT.py to remove the decimals on the first number of each line, which is supposed to be an integer.

PART 2: ANNOTATION
7.5) Go to the folder Annotation

8) Read annotate_multiple.py and run once you've made sure you understand what it does.
    - press '0' to '9' to change the index pointer to 0 to 9. 'i', 'o', 'p' correspond to 10, 11, 12, respectively.
    - once you've set the correct index, you can draw a bounding box around the object corresponding to that index: 
    - if you can't remember which index corresponds to which object, refer to coco.NAMES. Index starts at 0.
    - I put it here to save you some time:
        ["oneway", "highwayexit", "stop", "roundabout", "parking", "crosswalk", "noentry", "highwayentrance", "prio", "light", "block", "girl", "car"]
    - Once you've drawn bounding boxes for all objects in an image, you can press 's' to save the label and move to the next one.
    - If you've drawn a bounding box that's not up to my standards, you can press 'r' to remove it so that nobody is disappointed. You then have to redraw.
    - If there's an image with no objects, you can press 'q' to skip it without saving a label file, or 's' to save an empty label file (think of why this could be useful)
    - If you are tired and would like to take a break (I suggest you not to), you can press 'x' and the program will exit. When you re-run the program, it will automatically continue at the place you left off last time.

9) Once you're done annotating, run test_labels.py to check them. 
    - After annotating hundreds of images, errors occur inevitably. 
    - Check all labeling carefully, and press 'd' to change to the next page.

10) Clean labels
    - There's a bug where sometime the bounding box values are negative.
    - Run remove_negative.py to fix this problem.

11) Rename the files
    - After checking, we have to merge these new data with what we created in part 1.
    - To avoid having files with the same name, write a script to renumber the images and labels so that it starts from 30955 to 30955+n (e.g. "30955.jpg", "30956.jpg", and so on).
    - Combine these files with the original dataset created in part 1.

PART 3: TRAINING:
For this part, you can refer to the following for more info: https://github.com/dog-qiuqiu/Yolo-FastestV2
Prereq: create a conda environment with python 3.8, name it xinlin.

11.5) open conda terminal, and enter conda activate xinlin. 

11.6) open requirements.txt inside Yolo-FastestV2, comment out torch==1.9.0 and torchvision==0.10.0

11.7) in conda terminal, cd to the yolo folder, then enter "pip install -r requirements.txt"

11.8) Open the following link: https://pytorch.org/get-started/previous-versions/
    - find the pytorch version specified in requirements.txt (1.9.0).
    - Inside the "Linux and Windows" box, copy the line under "# CPU only" (you can use the pip one or conda one, up to your preference).
    - paste that command into the conda terminal to install torch.


12) Move the datasets folder into Yolo-FastestV2.

13) Run resize.py to resize all images to 352x352, which is the default dimension for YoloFastestV2 training.

14) Manually move all the label files into datasets/train (same folder as the images).

15) Run create_val2.py to save a portion of the training data for validation purposes. 
    - You can modify the ratio of validation data inside the script. 
    - DO some quick research and answer the following: What is the usual train/val ratio for object detection? (2 points):

16) Run create_txt.py 
    - then change the type from "train" to "val" to create 1 for validation data.
    - What are the 2 generated txt files for? (3 points):

17) Read and Run genanchors.py
    - explain what it does by refering to the github page or doing some research (9 points):
    - Open the generated anchors6.txt, copy the first line and paste it into line 17 of data/coco.DATA (after "anchors=")

18) Read and Run train.py
    - since you don't have a gpu, running is pretty slow. You can go into coco.data and set the number of epochs to 1.
    - explain here what it does (4 points):

19) Once training is done, you can run test.py --weights ${name}. (replace ${name} by the path to the .pth file you trained).
    - explain here what it does (4 points):

20) If the results are good, follow instructions on the github to convert it to onnx file.

21) Follow instructions on how to deploy the model with ncnn.

22) If everything works fine, send me the .param, .bin, .onnx and .pth files.

PART 4: DOCUMENTATION

23) Congratulations! You're almost done. Now you just need to write a proper readme file explaining how everything you just did works.
