import cv2
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random
import os

def overlap(r1, r2, tolerance = 0.1):
    leftA, rightA, topA, botA = r1
    leftB, rightB, topB, botB = r2
    if rightB-leftB > rightA-leftA or botB-topB>botA-topA:
        left1, right1, top1, bot1 = leftA, rightA, topA, botA
        leftA, rightA, topA, botA = leftB, rightB, topB, botB
        leftB, rightB, topB, botB = left1, right1, top1, bot1
#     if ((left1 < right2) and (left2 < right1) and (bot1 < top2) and (bot2 < top1)):
#         return False
#     else:
#         return True
    if (leftB>(rightA-(rightA-leftA)*tolerance)) or (botB<topA+(botA-topA)*tolerance) or (leftA > rightB-(rightB-leftB)*tolerance) or (botA < topB+(botB-topB)*tolerance):
        return False
    else:
        return True
    #return ((left1 < right2) and (left2 < right1) and (bot1 < top2) and (bot2 < top1))
def insertImage(imgs, imgClasses, ratios, bg, imgType, number, index, testing = False):
    name2 = str(index)+".jpg"
    name_txt =("datasets/coco128/labels/train/"+str(index)+".txt")
    df1 =pd.read_csv(name_txt, sep=' ', header = None)
    a = df1.iloc[0,1]*1280
    b = df1.iloc[0,3]*680
    c = df1.iloc[0,2]*960
    d = df1.iloc[0,4]*480
    left1 = int(0.5*(a-b))
    right1 = int(0.5*(a+b))
    bot1 = int(0.5*(c+d))
    top1 = int(0.5*(c-d))
    ds = np.array([int(left1), int(right1), int(top1), int(bot1)])
    #print("ds: ", ds)
    f = open(name_txt, 'a')
    occupied = []
    occupied.append(ds)
    for i in range(len(imgs)):
        stuck = False
        t1 = time.time()+10
        if testing:
            print("i: ",i)
        img1 = imgs[i]
        flag = True
        while flag:
            if time.time()>t1:
                stuck = True
                print("stuck")
                break
            #print("occupied: ",occupied)
            flag = False
            sz = max(img1.shape)
#             if sz > 270:
#                 ratio = np.random.uniform(low=0.75, high=0.85)
#             elif sz > 240:
#                 ratio = np.random.uniform(low=0.75, high=0.9)
#             elif sz > 210:
#                 ratio = np.random.uniform(low=0.8, high=0.95)
#             elif sz > 180:
#                 ratio = np.random.uniform(low=0.8, high=1)
#             else:
#                 ratio = 1
# 
#             img = cv2.resize(img1, (int(img1.shape[1]*ratio),int(img1.shape[0]*ratio)))
            img = img1
            #img = img1
            #print("shapes: ", bg.shape, img.shape)
            maxOffsetX = bg.shape[1] - img.shape[1]
            maxOffsetY = bg.shape[0] - img.shape[0]

            offsetX = np.random.randint(0, high=maxOffsetX)
            offsetY = np.random.randint(0, high=maxOffsetY)
            left,right,top,bot=offsetX,offsetX+img.shape[1],offsetY,offsetY+img.shape[0]
            dimensions = np.array([left, right, top, bot])
            #print("dimensions: ", dimensions)
            for o in occupied:
                if overlap(dimensions, o):
                    flag = True
                    #print("overlapped, try again")
        if stuck:
            continue
        #print(dimensions)
        d = dimensions.copy()
        occupied.append(d)
        bg[top:bot,left:right] = img
        txt_param1 = str((left+right)/2/bg.shape[1])
        txt_param2 = str((top+bot)/2/bg.shape[0])
        txt_param3= str(img.shape[1]/bg.shape[1])
        txt_param4= str(img.shape[0]/bg.shape[0])        
        txt_line = (' '+str(imgClasses[i])+' '+str((left+right)/2/bg.shape[1])+' '+str((top+bot)/2/bg.shape[0])
                +' '+str(img.shape[1]/bg.shape[1])+' '+str(img.shape[0]/bg.shape[0])+'\n')
        if testing:
            print('bg, img: ', bg.shape,img.shape)
            print('maxOffsetX,maxOffsetY,boundaryX: ',maxOffsetX,maxOffsetY,bg.shape[1]/3*(i+1))
            print('offsetX,offsetY: ',offsetX,offsetY)
            print('top,bot,left,right :',top,bot,left,right)
            print('left,right,img.shape[1]: ',left,right,bg.shape[1])
            cv2.rectangle(bg, (left, top), (right, bot), (0, 0, 255), thickness=2)
            print('txt_line: '+txt_line)
            print("saving...")
        f.write(txt_line)
    cv2.imwrite('datasets/coco128/images/train/'+name2,bg)
    if testing:
        cv2.imshow('1', bg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    f.close()
    return bg
def create(imgClasses, imgIndices, number ,imgType, flag, index):
    roadNum = number
    name = str(index)+'.jpg'
    bg = cv2.imread('datasets/coco128/images/train0/'+name)
    names = []
    imgs = []
    for i in range(num):
        names.append('./'+str(int(imgClasses[i]))+'/'+str(int(imgIndices[i]))+'.jpg')
        img = cv2.imread(names[i])
        imgs.append(img)
    isNone = False
    for i in range(num):
        if imgs[i] is None:
            print("None: ", names[i])
            isNone = True
    if isNone:
        exit()
    ratios = np.random.uniform(0.10,0.73, 4)
    bg=insertImage(imgs,imgClasses, ratios, bg, imgType, number, index, testing=flag)
    return bg

def process_image(i):
#     number = np.random.randint(low=0, high=total, size=num)
    global unique_numbers, unique_numbers_lock
    number = []
    with unique_numbers_lock:
#         if i+num>len(unique_numbers):
#             unique_numbers = np.arange(total)
#             unique_numbers = np.random.permutation(unique_numbers)
#         number = unique_numbers[i]
        for c1 in range(num):
            if len(unique_numbers)==0:
                unique_numbers = random.sample(range(total), total)
            number.append(unique_numbers.pop())
    imgClasses = np.zeros(num)
    imgIndices = np.zeros(num)
    for v in range(num):
        for w in range(numClasses):
            if number[v] < classIndices[w]:
                imgClasses[v] = w
                imgIndices[v] = np.random.randint(low=1, high=classCounts[int(imgClasses[v])]+1)
                break
    for j in imgClasses:
        IdxCount[int(j)]+=1
    create(imgClasses, imgIndices, i, 'train', False, i)
    return i

classCounts = np.array([10272, 6696, 13344, 10500, 11232, 14378, 8544, 6840, 12705, 14742, 5346, 12922, 10008])
numClasses = 13
classIndices = np.zeros(numClasses)
for a in range(numClasses):
    for b in range(a+1):
        classIndices[a] += classCounts[b]
total = np.sum(classCounts)
unique_numbers = np.load('unique_numbers.npy').tolist() #random.sample(range(total), 16185)
unique_numbers_lock = threading.Lock()
print("classIndices: ", classIndices)
print("classCounts: ", classCounts)
IdxCount = np.zeros(numClasses)
filename = 'annotations.txt'
# Set the base path to the image, label, and road_brightness folders
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    print("base path is " + base_path)
except:
    # Fallback: If __file__ is not defined, use the current working directory
    base_path = os.getcwd()
    print("base path is " + base_path)
os.makedirs(os.path.join(base_path, "datasets/coco128/images/train"), exist_ok=True)
os.makedirs(os.path.join(base_path, "datasets/coco128/labels/train"), exist_ok=True)

# df = pd.read_csv(filename, sep='\t', header = None)
num = 3
num_threads = 7
# Create the ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit tasks to the executor
    futures = [executor.submit(process_image, i) for i in range(30955)]

    # Print the results as they complete
    for future in as_completed(futures):
        i = future.result()
        if i % 100 == 0:
            print(f"i: {i}")

print("IdxCount: ", IdxCount)

    
