import cv2
import numpy as np

img = cv2.imread('resources/face.jpg')

# Translation: shifting the image along the axis
# -x value shift left
# -y value shift up
# x value shift right
# y value shift down

def translate(img, x, y):
    translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, translation_matrix, dimensions)

translated = translate(img, 100, 100)

# Rotation: Rotate the image

def rotate(img, angle, rotPoint = None):
    (height, width) = img.shape[:2]

    if rotPoint is None: # Rotate around the center
        rotPoint = (width // 2, height // 2)

    # A1: Angle of the rotation, A2: Scale of the rotation
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv2.warpAffine(img, rotMat, dimensions)

rotated = rotate(img, 45)

# Flipping an image 0: Vertically, 1: Horizontally, -1: Both vert and horizon

flip = cv2.flip(img, 1)

cv2.imshow("Original", img)
cv2.imshow("Tranlated", translated)
cv2.imshow("Rotated", rotated)
cv2.imshow("Flipped", flip)

cv2.waitKey(0)