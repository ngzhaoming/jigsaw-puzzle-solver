import cv2
import numpy as np

path = '../white/piece1.jpg'

def empty(a):
    pass

# Trackbars to know the optimum max and min value of the colors
# Hue has a range of 0 - 179
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Saturation Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Saturation Max", "TrackBars", 255, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    img = cv2.imread(path)
    # Hue Saturation Value
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Saturation Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Saturation Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")

    # Lower and Upper is a numpy array
    # Mask outputs out the filtered out colors of the image
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    
    # Using the mask to create another new image with only the specific color
    # Bitwise and is a mask function
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)

    cv2.waitKey(1)
