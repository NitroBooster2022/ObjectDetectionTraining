import cv2
import os

# Global variables
ix, iy = -1, -1
drawing = False
bounding_boxes = []
object_id = 0
filename = ''
grid_size = 50

def draw_grid(img, grid_size):
    for i in range(0, img.shape[1], grid_size):
        cv2.line(img, (i, 0), (i, img.shape[0]), (255, 255, 255), 1)
    for i in range(0, img.shape[0], grid_size):
        cv2.line(img, (0, i), (img.shape[1], i), (255, 255, 255), 1)

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bounding_boxes, object_id

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_temp = img.copy()
            cv2.rectangle(img_temp, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow(filename, img_temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bounding_boxes.append((object_id, ix, iy, x - ix, y - iy))
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)

def save_annotation(filename, img_width, img_height, bounding_boxes):
    with open(filename, 'w') as f:
        for bbox in bounding_boxes:
            obj_id, x, y, width, height = bbox
            x_center = abs((x + width / 2) / img_width)
            y_center = abs((y + height / 2) / img_height)
            width = abs(width / img_width)
            height = abs(height / img_height)
            print(f"{obj_id} {x_center} {y_center} {width} {height}\n")
            f.write(f"{obj_id} {x_center} {y_center} {width} {height}\n")


image_folder = 'track_images'
labels_folder = 'track_labels'
os.makedirs(labels_folder, exist_ok=True)
for file in os.listdir(image_folder):
    if file.endswith('.jpg') or file.endswith('.png'):
        img_path = os.path.join(image_folder, file)
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
#         draw_grid(img, grid_size)
        
        txt_filename = os.path.splitext(file)[0] + '.txt'
        txt_path = os.path.join(labels_folder, txt_filename)
        filename = file
        # Skip the image if the corresponding txt file already exists
        if os.path.exists(txt_path):
            continue
        
        bounding_boxes = []

        cv2.namedWindow(file)
        cv2.setMouseCallback(file, draw_rectangle)

        clean_img = img.copy()
        while True:
            cv2.imshow(file, img)
            key = cv2.waitKey(1) & 0xFF
            if '1' <= chr(key) <= '9':  # Press '1' to '9' to change the object_id
                print("key is: ", int(chr(key)))
                object_id = int(chr(key))
            if key  == ord('i'):
                print("key is: ", 10)
                object_id = 10
            if key  == ord('o'):
                print("key is: ", 11)
                object_id = 11
            if key  == ord('p'):
                print("key is: ", 12)
                object_id = 12
            if key == ord('s'):  # Press 's' to save the annotation
                txt_filename = os.path.splitext(file)[0] + '.txt'
                txt_path = os.path.join(labels_folder, txt_filename)
                save_annotation(txt_path, img_width, img_height, bounding_boxes)
                break
            elif key == ord('q'):  # Press 'q' to skip the image
                break
            elif key == ord('r'):  # Press 'r' to remove the last bounding box
                if bounding_boxes:
                    bounding_boxes.pop()
                    img_temp = img.copy()
                    for box in bounding_boxes:
                        cv2.rectangle(img_temp, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                    cv2.imshow(filename, img_temp)
                    img_temp = clean_img.copy()
                    img = img_temp.copy()
                    print("removing last item. id = ", object_id)
            elif key == ord('x'):  # Press 'x' to exit the program
                cv2.destroyAllWindows()
                exit()

        cv2.destroyAllWindows()

