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

def generate_dataset(images):
    count = 1
    for image in images: #15
        img=cv2.imread(image)
        if img is None:
            img=cv2.imread(image.lower())
        if img is None:
            print("none")
        s = max(img.shape)
#         if s >= 450:
#             img = cv2.resize(img, (int(img.shape[1]/s*450),int(img.shape[0]/s*450)))
        imgs=[] #3
        imgs.append(img)
        imgs.append(horizontal_flip(img))
        #imgs.append(add_fog(img))
        #imgs.append(add_rain(img))
        #imgs.append(LAB(img))
        #imgs.append(Discrete_Wavelet(img))
#         imgs.append(HSV(img))
        #imgs.append(CLAHE(img))
        #imgs.append(add_snow(img))
#         imgs.append(horizontal_flip(img))
#         imgs.append(gaussian_noise(img, var=115, mean=3))
#         imgs.append(gaussian_noise(gaussian_blur(img, kdim=3, var=7), var=75, mean=37))
        #imgs.append(Hist_Eq(img))
        for i in imgs:
            #print("i shape ", i.shape)
            perspect = [] #2
            perspect.append(i)
            shapeR = min(i.shape[0], i.shape[1])/max(i.shape[0], i.shape[1]) 
            if shapeR > 0.83: 
                ratio = np.random.uniform(low = 0.8, high = 0.9)
                perspect.append(perspective_transform(gb, ratio))
            #perspect.append(perspective_transform(i, 0.9))
            #perspect.append(horizontal_flip(i))
            #perspect.append(perspective_transform(i, 0.75))
            perspect.append(cv2.convertScaleAbs(i, alpha=0.57, beta=19))
            #perspect.append(cv2.convertScaleAbs(i, alpha=1.5, beta=37))
            #perspect.append(cv2.convertScaleAbs(i, alpha=1.2, beta=19))
            perspect.append(cv2.convertScaleAbs(i, alpha=0.75, beta=75))
            perspect.append(cv2.convertScaleAbs(i, alpha=0.25, beta=10))
            perspect.append(cv2.convertScaleAbs(i, alpha=0.13, beta=19))
            #perspect.append(cv2.convertScaleAbs(i, alpha=0.37, beta=19))
            for s1 in perspect:
                gauss_noise = [] #1
#                 gauss_noise.append(s)
                var = np.random.randint(low=0, high=150)
                gauss_noise.append(gaussian_noise(s1, var=var, mean=0))
                for g in gauss_noise:
                    gauss_blur = [] #1
                    gauss_blur.append(g)
                    gauss_blur.append(gaussian_blur(g, kdim=31, var=7))
                    gauss_blur.append(gaussian_blur(g, kdim=21, var=15))
                    #gauss_blur.append(horizontal_flip(g))
                    for gb in gauss_blur:
                        rotate1 = [] #3
                        rotate1.append(gb)
#                         rotate1.append(rotate(gb, angle=randint(low=0, high=20)))
#                         rotate1.append(rotate(gb, angle=randint(low=-20, high=20)))
                        for r in rotate1:
                            final = [] #4
                            final.append(r)
                            for image in final:
                                final2 = []
                                if s < 120:
                                    final2.append(image)
                                if s > 120 and s<=230:
                                    ratio = np.random.uniform(low=0.29, high=0.82)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.27, high=0.86)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 230 and s<=280:
                                    ratio = np.random.uniform(low=0.18, high=0.73)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.17, high=0.75)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 280 and s<=330:
                                    ratio = np.random.uniform(low=0.13, high=0.67)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.15, high=0.73)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 330 and s<=380:
                                    ratio = np.random.uniform(low=0.12, high=0.52)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.1, high=0.57)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 380 and s< 420:
                                    ratio = np.random.uniform(low=0.08, high=0.47)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.08, high=0.5)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 420:
                                    ratio = np.random.uniform(low=0.07, high=0.41)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.07, high=0.45)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                for finalImage in final2:
                                    cv2.imwrite('12/'+str(count)+'.jpg', finalImage)
                                    count+=1
                                    if count%50==0:
                                        print('count: ', count)
def main():
    os.makedirs('12', exist_ok=True)
    images = []
    for i in range(164):
        images.append('cars2/'+str(i)+'.JPG')
    generate_dataset(images)

main()

# print('img size: ', img.shape)
# 
# img1 = cv2.convertScaleAbs(img, alpha=0.5, beta=10)
# cv2.imshow('img1', img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()