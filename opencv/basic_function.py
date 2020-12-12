import cv2
import numpy as np

img = cv2.imread("../white/piece1.jpg")

# 5x5 matrix with all values set to 1, 8-bit unsigned integer
kernel = np.ones((5, 5), np.uint8)

# Convert color to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# kernel size is magnitude of the blur, then sigma x
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

# Canny edge detector to determine the edges in the image
# Other 2 parameters are threshold values
imgCanny = cv2.Canny(img, 100, 100)

# Increase the thickness of the edges
# Kernel is a matrix - Need to define the size and value
# Iteration defines the number of times the edges are thickened
imgDilation = cv2.dilate(imgCanny, kernel, iterations = 1)

# Thin the edges
imgEroded = cv2.erode(imgDilation, kernel, iterations = 1)

cv2.imshow("Gray_img", imgGray)
cv2.imshow("Blur_img", imgBlur)
cv2.imshow("Canny_img", imgCanny)
cv2.imshow("Dilated_img", imgDilation)
cv2.imshow("Eroded_img", imgEroded)
cv2.waitKey(0)