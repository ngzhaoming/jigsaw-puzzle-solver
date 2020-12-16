import cv2

# Binarization of an image (pixels are either 0 or 255)

img = cv2.imread('resources/face.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Simple Thresholding
# thresh - Binarized output image
threshold, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# Inverse Threshold
threshold, thresh_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Adaptive Thresholding - Allow system to find the optimal thresholding value instead
# A2: Adaptive threshold method, A3: Threshold type, A4: Blocksize, A5: C-value - Integer substracted from the mean to finetune the threshold
adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)

cv2.imshow('Gray', gray)
cv2.imshow('Simple Thresholded', thresh)
cv2.imshow('Inverse Thresholded', thresh_inv)
cv2.imshow('Adaptive Thresholded', adaptive_thresh)
cv2.waitKey(0)