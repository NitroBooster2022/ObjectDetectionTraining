import cv2
import numpy as np
from flask import current_app
import random

def random_shear(image, shear_range):
    rows, cols, _ = image.shape
    shear_factor = np.random.uniform(-shear_range, shear_range)
    M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
    return cv2.warpAffine(image, M, (cols, rows))

def random_contrast(image, lower=0.5, upper=1.5):
    alpha = np.random.uniform(lower, upper)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def random_saturation(image, lower=0.5, upper=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(lower, upper), 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_hue(image, delta=18):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-delta, delta)) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_crop(image, crop_size):
    h, w, _ = image.shape
    if h <= crop_size[0] or w <= crop_size[1]:
        return image

    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    return image[top:top + crop_size[0], left:left + crop_size[1]]

def random_zoom(image, zoom_range=(0.9, 1.1)):
    h, w, _ = image.shape
    scale = np.random.uniform(zoom_range[0], zoom_range[1])
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    if scale > 1:
        return random_crop(resized_image, (h, w))
    else:
        return cv2.resize(resized_image, (w, h))

def random_elastic_transform(image, alpha, sigma, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = image.shape
    dx = random_state.rand(*shape[:2]) * 2 - 1
    dy = random_state.rand(*shape[:2]) * 2 - 1
    dx = cv2.GaussianBlur(dx, (sigma | 1, sigma | 1), 0) * alpha
    dy = cv2.GaussianBlur(dy, (sigma | 1, sigma | 1), 0) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return cv2.remap(image, indices[1].astype(np.float32), indices[0].astype(np.float32), cv2.INTER_LINEAR)
def a1(img):
    c = np.ones(img.shape)*10
    img1 = img+c
    #img1 = cv2.addWeighted(img, 0.5, c, 0.5, 0)
    return img1
def increase_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    img1 = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img1

def rotate(img, angle=0):
    """
    Applies angular Rotationn to the input image
    
    Args:
        img: Input image to be augmented
        angle(float): Angle of Rotation for image
    Output:
        timg: Roatated Image
    
    Source:
        https://docs.opencv.org/master/
    
    """
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    timg = cv2.warpAffine(img, M, (cols,rows))
    return timg

def average_blur(img, kdim=8):
    """
    Applies Average Blur to the input image
    
    Args:
        img: Input image to be augmented
        kdim(int): Dimension of Kernel to do Blur
    Output:
        timg: Average Blured Image
    
    Source:
        https://docs.opencv.org/master/
    
    """
    timg = cv2.blur(img, (kdim, kdim))
    return timg




def gaussian_blur(img, kdim=3, var=5):
    """
    Applies Gaussian Blur to the input image
    
    Args:
        img: Input image to be augmented
        kdim(int): 
            Dimension of Kernel to do Blur.
            Default: 8
        var(float):
            Variance for gaussian Blur
            Default: 5   
    Output:
        timg: Gaussian Blured Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    try:
        timg = cv2.GaussianBlur(img, (kdim, kdim), var)
        return timg
    except:
        if (kdim[0] % 2 == 0):
            print("kernel dimension cannot be even for gaussian blur.")




def gaussian_noise(img, var=100, mean=0):
    """
    Applies Gaussian Noise to the input image
    
    Args:
        img: Input image to be augmented
        var(float):
            Variance for gaussian noise
            Default: 10 
        mean(float):
            Mean for gaussian noise
            Default: 0
    Output:
        timg: Image with gaussian noise
    
    Source: 
        https://docs.opencv.org/master/
        https://numpy.org/doc/   
    
    """
    row, col, _ = img.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean,sigma,(row, col))
    timg = np.zeros(img.shape, np.float32)
    timg[:, :, 0] = img[:, :, 0] + gaussian
    timg[:, :, 1] = img[:, :, 1] + gaussian
    timg[:, :, 2] = img[:, :, 2] + gaussian
    cv2.normalize(timg, timg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    timg = timg.astype(np.uint8)
    return timg





def sharpen(img, kdim=5, sigma=15.0, amount=15.0, threshold=0):
    """
    Applies sharpen to the input image
    
    Args:
        img: Input image to be augmented
        kdim(int): 
            Dimension of Kernel to do sharpening.
            Default: 8
        sigma(float):
            standard deviation for sharpening
            Default: 1.0
        amount(float):
            Amount of sharpening Required
            Default: 1.0
        threshold(float):
            threshold for sharpening
            Default: 0
    Output:
        timg: Image with sharpening
    
    Source: 
        https://docs.opencv.org/master/
        https://numpy.org/doc/   
    
    """
    blurred = cv2.GaussianBlur(img, (kdim, kdim), sigma)
    timg = float(amount + 1) * img - float(amount) * blurred
    timg = np.maximum(timg, np.zeros(timg.shape))
    timg = np.minimum(timg, 255 * np.ones(timg.shape))
    timg = timg.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(img - blurred) < threshold
        np.copyto(timg, img, where=low_contrast_mask)
    return timg

def horizontal_flip(img):
    """
    Applies horizontal flip to the input image
    
    Args:
        img: Input image to be flipped
    Output:
        timg: Horizontal Flipped Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    timg = cv2.flip(img, 1)
    return timg




def vertical_flip(img):
    """
    Applies Vertical flip to the input image
    
    Args:
        img: Input image to be flipped
    Output:
        timg: Vertically Flipped Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    timg = cv2.flip(img, 0)
    return timg



def perspective_transform(img, ratio):
    """
    Applies Prespective Transform to the input image
    
    Args:
        img: Input image to be Transformed
        input_pts(nparray): NumPy array of points to transform
    Shape:
        input_pts :maths: '(4,2)'
    Output:
        timg: Prespective Transformed Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    input_pts=np.float32([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
    width = int(round(img.shape[0]*ratio))
    #print("width: ", width)
    row, col, _ = img.shape
    output_pts=np.float32([[0, 0], [width, 0], [0, img.shape[1]], [width, img.shape[1]]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    timg = cv2.warpPerspective(img, M, (img.shape[0], img.shape[1]))
    timg = timg[:, 0:width]
    return timg




def crop(img, ratio):
    """
    Crops Input Image
    
    Args:
        img: Input image to be Cropped
        input_pts(nparray): NumPy array of points to transform
    Shape:
        input_pts :maths: '(4,2)'
    Output:
        timg: Cropped Image
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    input_pts=np.float32([[0, 0], [img.shape[0], 0], [0, img.shape[1]], [img.shape[0], img.shape[1]]])
    width = int(round(img.shape[0]*ratio))
    row, col, _ = img.shape
    output_pts=np.float32([[0, 0], [width, 0], [0, img.shape[1]], [width, img.shape[1]]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    timg = cv2.warpPerspective(img, M, (img.shape[0], img.shape[1]))
    return timg




def random_erasing(img,  randomize, grayIndex, mean, var, region=np.array([[12, 12], [20, 12], [12, 20], [20, 20]])):
    
    """
    Applies Random Erasing to the input image
    
    Args:
        img: Input image to be Transformed
        randomize(bool): Option to randomize fill or not
        grayIndex(float): Index to grayscale fill in void
        mean(float): mean of randomize fill
        var(float): variance of randomize fill
        region(nparray): Coordinates of random erase region
    Shape:
        region :maths: '(4,2)'
    Output:
        timg: Image with erasing in given region
    
    Source: 
        https://docs.opencv.org/master/
    
    """
    row, col, _ = img.shape
    sigma = var ** 0.5
    timg = img
    a = int(region[0, 0])
    b = int(region[1, 0])
    c = int(region[0, 1])
    d = int(region[2, 1])
    if randomize:
        gaussian = np.random.normal(mean, sigma, (b-a, d-c))
        timg[a:b, c:d, 0] = gaussian
        timg[a:b, c:d, 1] = gaussian
        timg[a:b, c:d, 2] = gaussian
        cv2.normalize(timg, timg, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    else:
        patch = grayIndex*np.ones((b-a, d-c))
        timg[a:b, c:d, 0] = patch
        timg[a:b, c:d, 1] = patch
        timg[a:b, c:d, 2] = patch
    return timg