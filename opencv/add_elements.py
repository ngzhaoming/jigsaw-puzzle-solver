import cv2
import numpy as np

# Create a 512x512 matrix with all values set to 0, with 3 channels
img = np.zeros((512, 512, 3), np.uint8)

# Iterate through the matrix and set that part to the BGR color
# img[:] = 255, 0, 0

# A1: Start, A2: End, A3: Color, A4: Thickness
cv2.line(img, (0, 0), (img.shape[1], 300), (0, 255, 255), 3)
cv2.rectangle(img, (0, 0), (img.shape[1] // 2, img.shape[0] // 2), (0, 0, 255), 2) # Fill rec

# A1: Center pt, A2: Radius, A3: Color, A4: Thickness
cv2.circle(img, (400, 50), 30, (255, 255, 0), cv2.FILLED)

# A3: Font family, A4: Scale, A5: Color, A6: Thickness, A7: Line Type
cv2.putText(img, "Hello World!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 150, 0), 2)

cv2.imshow("Black", img)
cv2.waitKey(0)