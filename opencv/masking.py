import cv2
import numpy as np

img = cv2.imread('resources/face.jpg')

# NOTE: Dimension of the mask must be the same size as the picture
blank = np.zeros(img.shape[:2], dtype='uint8')

# Mask circle with center point of the middle of the img
mask = cv2.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)

masked = cv2.bitwise_and(img, img, mask=mask) # Give intersection

cv2.imshow('Original', img)
cv2.imshow('Blank', blank)
cv2.imshow('Mask', mask)
cv2.imshow('Masked_Image', masked)

cv2.waitKey(0)