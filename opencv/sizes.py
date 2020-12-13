import cv2
import numpy as np

img = cv2.imread("../white/piece1.jpg")

# NOTE: To resize, need to know the size of the image first
# Height, Width, Channel (BGR)
print(img.shape)

def rescaleImg(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

imgResizeFunc = rescaleImg(img)

# Tuple (Width, Height), note that opencv y-axis is pointing south
imgResize = cv2.resize(img, (300, 250))
print(imgResize.shape)

# Define the coordinates [height, width]
imgCropped = img[:50, 50:100]

cv2.imshow("Normal", img)
cv2.imshow("Resize", imgResize)
cv2.imshow("Resize Funtion", imgResizeFunc)
cv2.imshow("Cropped", imgCropped)
cv2.waitKey(0)