import cv2
import numpy as np

img = cv2.imread("resources/hearts.jpg")
print(img.shape)

width, height = 250, 350

# Define the 4 corners of the part to crop - Use jspaint to identify corners
pts1 = np.float32([[989, 230], [863, 632], [1152, 724], [1275, 319]])
pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

# Change the perspective to another 4 corners
transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_output = cv2.warpPerspective(img, transform_matrix, (width, height))

cv2.imshow("Original", img)
cv2.imshow("Warped_Image", img_output)

cv2.waitKey(0)