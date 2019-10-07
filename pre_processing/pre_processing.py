import cv2
import numpy as np
from image_segmentation.main import segmentation

k = 256

getStructuringElement = lambda n : cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(n,n))
opening = lambda img_sample, kernel : cv2.morphologyEx(img_sample, cv2.MORPH_OPEN, kernel)
closing = lambda img_sample, kernel : cv2.morphologyEx(img_sample, cv2.MORPH_CLOSE, kernel)

def lut_gamma_correction(gamma):
    global k

    lut = np.array([i**gamma for i in range(k)]).astype(np.float64)
    lut /= lut.max()
    lut *= k-1

    return lut

def otsu_threshold(img_sample): 
    global k
    return cv2.threshold(img_sample, 0,  k-1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def laplacian(img_sample, uint8=True):
    global k
    laplacian_img = cv2.Laplacian(img_sample, cv2.CV_64F)

    if uint8:
        return np.clip(img_sample + laplacian_img, 0, k-1).astype('uint8')
		#return np.uint8(np.absolute(img_sample + laplacian_img))

    return img_sample + laplacian_img

def sobel(img_sample, kernel_size=3):
		sobelx = cv2.Sobel(img_sample, cv2.CV_64F,1,0,ksize=kernel_size)
		sobely = cv2.Sobel(img_sample, cv2.CV_64F,0,1,ksize=kernel_size)

		return img_sample + sobelx + sobely

def histogram(img_sample):
    global k

    return np.array([len(np.where(img_sample == i)[0]) for i in np.arange(k)])

'''
    Proposed approach:
    img_sample -> laplacian -> gamma_correction ->  otsu_threshold -> morphology -> segmentation -> histogram_cut

'''
def pipeline(img_sample, output_path="./img_samples/", gamma=2):
    global k

    img_sample = lut_gamma_correction(gamma)[img_sample.astype('uint8')]
    cv2.imwrite('./img_samples/2_gamma.png', img_sample)

    img_sample = sobel(img_sample)
    cv2.imwrite('./img_samples/1_sobel.png', img_sample)    

    #img_sample = laplacian(img_sample)
    #cv2.imwrite('./img_samples/1_1_laplacian.png', img_sample)

    img_sample = otsu_threshold(img_sample.astype('uint8'))
    cv2.imwrite('./img_samples/3_otsu.png', img_sample)

    img_sample = opening(img_sample, getStructuringElement(5))
    cv2.imwrite('./img_samples/4_opening.png', img_sample)

    img_sample = closing(img_sample, getStructuringElement(5))
    cv2.imwrite('./img_samples/5_closing.png', img_sample)

    seg_img, hist = segmentation(img_sample, output_path, plot_histogram=True)
    cv2.imwrite('./img_samples/6_seg.png', seg_img)