from pre_processing import pipeline
import cv2
import numpy as np

#input_img = cv2.imread('./img_samples/01_Diabetes&Healthy&00&32_c1.jpg', 0)
input_img = cv2.imread('./img_samples/01_Diabetes&Healthy&00&32_c1.jpg', 1)
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
pipeline(input_img[:, :, 2])

'''
mask = cv2.inRange(input_image_hsv, (345, 0, 0), (15, 255,255))
imask = mask>0
red = np.zeros_like(input_image_hsv, np.uint8)
red[imask] = input_image_hsv[imask]
cv2.imwrite("red.png", red)
'''
cv2.imwrite("0.png", input_img[:, :, 0])
cv2.imwrite("1.png", input_img[:, :, 1])
cv2.imwrite("2.png", input_img[:, :, 2])