import cv2
import numpy as np

# AND, OR, XOR, NOT

blank = np.zeros((400, 400), dtype='uint8')

rectangle = cv2.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv2.circle(blank.copy(), (200, 200), 200, 255, -1)

# Bitwise AND - Returns the intersection of 2 images
bitwise_and = cv2.bitwise_and(rectangle, circle)

# Bitwise OR - Returns the union of all the images
bitwise_or = cv2.bitwise_or(rectangle, circle)

# Bitwise XOR - Returning the non-intersecting region (white)
bitwise_xor = cv2.bitwise_xor(rectangle, circle)

# Bitwise NOT - Negate/Invert the image
bitwise_not = cv2.bitwise_not(circle)

cv2.imshow("Rectangle", rectangle)
cv2.imshow("Circle", circle)
cv2.imshow("AND", bitwise_and)
cv2.imshow("OR", bitwise_or)
cv2.imshow("XOR", bitwise_xor)
cv2.imshow("NOT", bitwise_not)
cv2.waitKey(0)