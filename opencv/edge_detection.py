import cv2
import numpy as np

# Gradients - Edgelike regions in an image
img = cv2.imread('resources/face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Laplacian edges - A1: Data depth
lap = cv2.Laplacian(gray, cv2.CV_64F)
lap = np.uint8(np.absolute(lap))

# Sobel edges - Compute gradient into directions (x, y)
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
combine_sobel = cv2.bitwise_or(sobelx, sobely) # bitwise OR takes the union

cv2.imshow('Gray', gray)
cv2.imshow('Laplacian', lap)
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Combine Sobel', combine_sobel)
cv2.waitKey(0)