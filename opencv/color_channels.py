import cv2
import numpy as np

img = cv2.imread('resources/hearts.jpg')
blank = np.zeros(img.shape[:2], dtype='uint8')

# Return grayscale img with lighter areas representing higher concentration of that color
b, g, r = cv2.split(img)

blue = cv2.merge([b, blank, blank]) # Only extract the blue color
green = cv2.merge([blank, g, blank]) # Only extract the green color
red = cv2.merge([blank, blank, r]) # Only extract the red color

merged = cv2.merge([b, g, r])

cv2.imshow("Original", img)
cv2.imshow("Blue", blue)
cv2.imshow("Green", green)
cv2.imshow("Red", red)
cv2.imshow("Merged", merged)
cv2.waitKey(0)