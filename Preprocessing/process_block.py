from transformations import *
from augmentations import *
import cv2
import numpy as np
from numpy.random import uniform
from numpy.random import randint
import os
def generate_dataset(indices):
    count = 1
    for index in indices: 
        img=cv2.imread(name+'/'+str(index)+'.JPG')
        if img is None:
            print("none: ," + name+'/'+str(index)+'.JPG')
        s = min(img.shape[0],img.shape[1])
        if s >= 357:
            img = cv2.resize(img, (int(img.shape[1]/s*357),int(img.shape[0]/s*357)))
        s = min(img.shape[0],img.shape[1])
        imgs=[] 
        imgs.append(img)
        imgs.append(add_rain(img))
        imgs.append(add_fog(img))
#         imgs.append(crop(img, 0.9))
        imgs = [random_saturation(i) for i in imgs]
        imgs = [random_hue(i) for i in imgs]
        imgs = [random_contrast(i) for i in imgs]
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
            perspect.append(horizontal_flip(i))
            perspect.append(random_zoom(i))
#             ratio = np.random.uniform(low = 0.8, high = 0.9)
#             perspect.append(perspective_transform(i, ratio))
            #perspect.append(perspective_transform(i, 0.75))
            for s1 in perspect:
                gauss_noise = [] #1
                high = 100
                var = np.random.randint(low=0, high=high) 
                gauss_noise.append(s1)
                gauss_noise.append(gaussian_noise(s1, var=var, mean=0))
                gauss_noise.append(random_erasing(s1,  True, 15, 0, 15))
                #gauss_noise.append(gaussian_noise(s1, var=30, mean=0))
                for g in gauss_noise:
                    gauss_blur = [] #1
                    gauss_blur.append(g)
                    kdim = np.random.randint(low=0, high=3)*16+1 if s>thresh else np.random.randint(low=0, high=3)*2+1
                    gauss_blur.append(cv2.GaussianBlur(g, (kdim, kdim), 5))
                    kdim = np.random.randint(low=1, high=4)*16+1 if s>thresh else np.random.randint(low=0, high=3)*2+1
                    gauss_blur.append(cv2.GaussianBlur(g, (kdim, kdim), 5))
                    for gb in gauss_blur:
                        rotate1 = [] #3
                        rotate1.append(gb)
                        rotate1.append(random_shear(gb, shear_range=0.3))
#                         rotate1.append(random_zoom(gb))
#                         rotate1.append(random_elastic_transform(gb, alpha=34, sigma=4))
                        rotate1.append(rotate(gb, angle=randint(low=-15, high=15)))
#                         rotate1.append(rotate(gb, angle=randint(low=-15, high=0)))
                        shapeR = min(gb.shape[0], gb.shape[1])/max(gb.shape[0], gb.shape[1]) 
                        if shapeR > 0.88: 
                            ratio = np.random.uniform(low = 0.8, high = 0.9)
                            rotate1.append(perspective_transform(gb, ratio))
#                         else:
#                             print("shapeR, shape: ", shapeR, gb.shape)
                        #rotate1.append(perspective_transform(gb, 0.85))
                        for r in rotate1:
                            final = [] #8
                            final.append(r)
                            r1 = 0.37
                            r2 = 0.48
                            final.append(cv2.convertScaleAbs(r, alpha=r2, beta=0))
                            final.append(cv2.convertScaleAbs(r, alpha=1.5, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=1.2, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.57, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=r1, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.75, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.31, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.88, beta=10))
                            final.append(cv2.convertScaleAbs(r, alpha=0.25, beta=10))
                            final.append(Hist_Eq(r))
                            for image in final:
                                final2 = []
                                if s < thresh:
                                    final2.append(image)
                                if s > thresh and s<=150:
                                    ratio = np.random.uniform(low=0.38, high=0.62)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.38, high=0.62)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 150 and s<=200:
#                                     ratio = np.random.uniform(low=0.45, high=0.7)
#                                     final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.192, high=0.46)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 200 and s<=250:
#                                     ratio = np.random.uniform(low=0.4, high=0.6)
#                                     final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                    ratio = np.random.uniform(low=0.14, high=0.368)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 250 and s<=300:
                                    ratio = np.random.uniform(low=0.0115, high=0.307)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 300 and s<=350:
                                    ratio = np.random.uniform(low=0.092, high=0.267)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                if s > 350:
                                    ratio = np.random.uniform(low=0.07, high=0.233)
                                    final2.append(cv2.resize(image,(int(image.shape[1]*ratio), int(image.shape[0]*ratio))))
                                for finalImage in final2:
                                    cv2.imwrite('10/'+str(count)+'.jpg', finalImage)
                                    count+=1
                                    if count%50==0:
                                        print('count: ', count)

name = 'roadblock'
num = 2
os.makedirs('10', exist_ok=True)
threshold = 19
thresh = 75
indices = []
for i in range(num):
    indices.append(i)
generate_dataset(indices)

