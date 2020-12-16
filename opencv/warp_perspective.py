import cv2
import numpy as np

img = cv2.imread("resources/forest/forest_picture2.jpg")

width, height = 1050, 950

# Define the 4 corners of the part to crop - Use jspaint to identify corners
pts1 = np.float32([[139, 12], [109, 906], [1100, 924], [1100, 24]])
pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

# Change the perspective to another 4 corners
transform_matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_output = cv2.warpPerspective(img, transform_matrix, (width, height))

cv2.imshow("Original", img)
cv2.imshow("Warped_Image", img_output)
cv2.imwrite("Warped_Result.jpg", img_output)

cv2.waitKey(0)