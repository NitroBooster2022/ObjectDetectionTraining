import cv2
vidcap = cv2.VideoCapture('C:/Users/simon/Downloads/Yolo-FastestV2/images/tes3.mp4')
fps = vidcap.get(cv2.CAP_PROP_FPS) # get the original frame rate
success, image = vidcap.read()
count = 0
i=0
while success:
  if count % int(fps/2) == 0: # save one frame per second
    cv2.imwrite(f"frames/frame{i}.jpg", image) # save frame as jpg file
    i+=1
    if i%5==0:
        print("i: ", i)
  success, image = vidcap.read()
  count += 1