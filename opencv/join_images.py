import cv2
import numpy as np

img = cv2.imread('../white/piece1.jpg')

# Stack the images horizontally
# NOTE: Both images need to have the same number of channels
imgHor = np.hstack((img, img))
imgVec = np.vstack((img, img))

cv2.imshow("Horizontal", imgHor)
cv2.imshow("Vertical", imgVec)

cv2.waitKey(0)