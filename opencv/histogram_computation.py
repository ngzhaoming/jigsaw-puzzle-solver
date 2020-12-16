import cv2
import numpy as np
import matplotlib.pyplot as plt

# Visualize the pixel intensity distribution of an image

img = cv2.imread('resources/face.jpg')

# Include masking to find the historgram of a specific area
blank = np.zeros(img.shape[:2], dtype='uint8')
mask = cv2.circle(blank, (img.shape[1] // 2, img.shape[0] // 2), 100, 255, -1)
masked = cv2.bitwise_and(img, img, mask=mask)

"""
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
masked = cv2.bitwise_and(gray, gray, mask=mask)

# Grayscale histogram
# A0: List of images, A1: Channels, A2: Mask (Specific Portion of img), A3: Histogram Size (Number of bins)
gray_hist = cv2.calcHist([gray], [0], mask, [256], [0, 256])

# Using matplotlib to identify the intensity of the pixel distribution
plt.figure()
plt.title('Grayscale Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')
plt.plot(gray_hist)
plt.xlim([0, 256])
plt.show()

cv2.imshow("Gray Masked", masked)
"""

plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('# of pixels')

# Color Histogram
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.show()

cv2.imshow("Masked", masked)
cv2.waitKey(0)