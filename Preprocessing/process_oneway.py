from transformations import *
from augmentations import *
import cv2
import numpy as np
from numpy.random import uniform
from numpy.random import randint
import os
'''
1)perspective_transform(img, ratio) 1, 0.9, 0.75, 0.6,  add_fog, add_rain, add_snow
2)sharpen(img, kdim=5, sigma=15.0, amount=15.0, threshold=0) sigma [0,5,10] 3
3)gaussian_noise(img, var=100, mean=0) var [0, 40, 80] 3
4)gaussian_blur(img, kdim=3, var=5) var [0 7 15] 3
5)rotate(img, angle=0) [0 10 20] 3
6)cv2.convertScaleAbs(img, alpha=0.5, beta=10) [0.5 1 1.5] 3
'''

def generate_dataset(indices):
    count = 1
    for index in indices: 
        img=cv2.imread(name+'/'+str(index)+'.JPG')
        # if img is None:
        #     img=cv2.imread(indices.lower())
        if img is None:
            print("none: ," + name+'/'+str(index)+'.JPG')
        s = max(img.shape)
        # print("s, thresh: ", s, thresh)
        if s >= 450:
            img = cv2.resize(img, (int(img.shape[1]/s*357),int(img.shape[0]/s*357)))
        imgs=[] 
        imgs.append(img)
#         imgs.append(add_rain(img))
        # imgs.append(add_fog(img))
        # if s>thresh:
        #     #imgs.append(add_rain(img))
        #     imgs.append(add_fog(img))
        # else:
        #     #imgs.append(img)
        #     imgs.append(img)
        #imgs.append(Hist_Eq(img))
        for i in imgs:
            #print("i shape ", i.shape)
            perspect = [] #2
            perspect.append(i)
            #perspect.append(horizontal_flip(i))
            #perspect.append(perspective_transform(i, 0.9))
            #perspect.append(perspective_transform(i, 0.75))
            for s1 in perspect:
                gauss_noise = [] #1
                high = 37 if s>thresh else 2
                var = np.random.randint(low=0, high=high) 
                #gauss_noise.append(s)
                gauss_noise.append(gaussian_noise(s1, var=var, mean=0))
                #gauss_noise.append(gaussian_noise(s, var=30, mean=0))
                for g in gauss_noise:
                    gauss_blur = [] #1
                    gauss_blur.append(g)
                    gauss_blur.append(horizontal_flip(g))
                    kdim = np.random.randint(low=0, high=3)*16+1 if s>thresh else np.random.randint(low=0, high=3)*2+1
                    gauss_blur.append(cv2.GaussianBlur(g, (kdim, kdim), 5))
#                     kdim = np.random.randint(low=1, high=4)*16+1 if s>thresh else 1
#                     gauss_blur.append(cv2.GaussianBlur(g, (kdim, kdim), 5))
                    for gb in gauss_blur:
                        rotate1 = [] #3
                        rotate1.append(gb)
                        rotate1.append(rotate(gb, angle=randint(low=-15, high=15)))
                        shapeR = min(gb.shape[0], gb.shape[1])/max(gb.shape[0], gb.shape[1]) 
                        if shapeR > 0.88: 
                            ratio = np.random.uniform(low = 0.7, high = 0.9)
                            rotate1.append(perspective_transform(gb, ratio))
                            ratio = np.random.uniform(low = 0.75, high = 0.85)
                            rotate1.append(perspective_transform(gb, ratio))
                        # rotate1.append(rotate(gb, angle=randint(low=-15, high=0)))
                        #rotate1.append(perspective_transform(gb, 0.75))
                        #rotate1.append(perspective_transform(gb, 0.85))
                        for r in rotate1:
                            final = [] #8
                            final.append(r)
                            r1 = 0.37
                            r2 = 0.48
                            final.append(cv2.convertScaleAbs(r, alpha=0.37, beta=0))
                            #final.append(cv2.convertScaleAbs(r, alpha=1.5, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=1.2, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.57, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=r1, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.75, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.88, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.25, beta=10))
                            for image in final:
                                final2 = []
                                if s < thresh:
                                    final2.append(image)
                                if s > thresh and s<=150:
#                                     ratio = np.random.uniform(low=0.55, high=0.85)
#                                     final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.28, high=0.64)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 150 and s<=200:
                                    ratio = np.random.uniform(low=0.12, high=0.46)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.12, high=0.46)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 200 and s<=250:
#                                     ratio = np.random.uniform(low=0.4, high=0.6)
#                                     final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.085, high=0.347)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 250 and s<=300:
                                    ratio = np.random.uniform(low=0.088, high=0.29)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 300 and s<=350:
                                    ratio = np.random.uniform(low=0.073, high=0.24)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 350:
                                    ratio = np.random.uniform(low=0.062, high=0.22)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                for finalImage in final2:
                                    cv2.imwrite('0/'+str(count)+'.jpg', finalImage)
                                    count+=1
                                    if count%50==0:
                                        print('count: ', count)

name = 'oneway'
os.makedirs('0', exist_ok=True)
num = 94
threshold = 19
thresh = 75
indices = []
for i in range(num):
    indices.append(i)
generate_dataset(indices)

