import cv2
import os
from os.path import join

folder = 'resources/forest_scanned'

filenames = os.listdir(folder)

for filename in filenames:
    if filename == ".DS_Store":
        continue
    
    path_in = join(folder, filename)

    img = cv2.imread(path_in)
    # img = img[380:640, 190:420]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)

    threshold, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    path_out = 'resources/results'
    cv2.imwrite(join(path_out, filename), thresh)

# cv2.waitKey(0)