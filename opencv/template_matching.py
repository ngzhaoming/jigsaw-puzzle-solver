import cv2
import numpy as np

# Load images
img = cv2.imread('resources/pikachu.jpg', 0)
template = cv2.imread('resources/pikachu_crop.jpg', 0)

w, h = template.shape[::-1]

# Apply template matching
# availible methods are 
# cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR
# cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED

res = cv2.matchTemplate(template, img, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

# draw box on template
cv2.rectangle(img, top_left, bottom_right, 255, 2)
cv2.imshow("Template_Matching", img)

cv2.waitKey(0)
