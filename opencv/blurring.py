import cv2

img = cv2.imread('resources/face.jpg')

# Method 1: Averaging - Based on the kernel window, compute the middle pixel to surrouding pixel (average)
# Higher kernel size leads to more blur
average = cv2.blur(img, (3, 3))

# Method 2: Gaussian Blur - Surrounding pixel is given a weight (average) to the center pixel
# sigma x - Standard deviation along the x-axis
gauss = cv2.GaussianBlur(img, (3, 3), 0)

# Method 3: Median Blur - Similar to average blurring BUT use median instead of average
# More effective in lower kernel sizes
median = cv2.medianBlur(img, 3)

# Bilateral Blurring - Apply blurring but still retaining the edges
# A1: Diameter, A2: sigmaColor, A3: sigmaSpace
bilateral = cv2.bilateralFilter(img, 5, 15, 15)

cv2.imshow("Original", img)
cv2.imshow("Averaging", average)
cv2.imshow("Gaussian", gauss)
cv2.imshow("Median", median)
cv2.imshow("Bilateral", bilateral)

cv2.waitKey(0)