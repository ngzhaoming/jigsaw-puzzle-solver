import cv2

img = cv2.imread('resources/underwater_uncropped.jpg')
img = img[380:640, 210:430]

path_out = 'resources/underwater/cropped.jpg'
cv2.imwrite(path_out, img)
