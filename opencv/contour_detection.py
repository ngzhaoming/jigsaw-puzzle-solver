import cv2

# Find contors of the image, taking img as argument
def get_contours(img):
    # A2: Retrieval methods - There are various retrieval methods
    # A3: Approximation - All/Compressed contour values
    # RETR_TREE: All the hierarchal contours
    # RETR_EXTERNAL: Only all the external contours
    # RETR_LIST: All the contour in the image
    # contours - Python list of all corners of contours
    # hierarchy - Hierarchal representation of the contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > 500: # Draw the contours if the shape area meet inequality
            # A2: Array of points in contour, A5: Thickness
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt, True)

            # A2: Resolution, A3: Boolean value to show all shapes are closed
            approx_corner = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            
            # Creating a bouding box around the detected objects using the corners
            objCor = len(approx_corner)
            x, y, w, h = cv2.boundingRect(approx_corner) # Bouding box (x, y) coordinates

            objectType = None

            if objCor == 3:
                objectType = "Triangle"
            elif objCor == 4:
                aspect_ratio = w / float(h) # Check square or rectangle
                if aspect_ratio > 0.95 and aspect_ratio < 1.05: # Deviation of 0.5
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circle"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgContour, objectType, (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)


path = 'resources/2d_shapes.jpg'
img = cv2.imread(path)
imgContour = img.copy() # Make a new copy to make changes here

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Method 1: Using Canny Edge detector
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgCanny = cv2.Canny(imgBlur, 50, 50) # Threshold is arbitary
get_contours(imgCanny)

# Method 2: Using threshold method - Binarize the image to 0 and 1 based on the inequality
# Threshold set to 200 <= x <= 255 at this range, pixels are set to 1 (white)
ret, thresh = cv2.threshold(imgGray, 200, 255, cv2.THRESH_BINARY)
contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.imshow("Original", img)
cv2.imshow("Gray", imgGray)
cv2.imshow("Blur", imgBlur)
cv2.imshow("Contour", imgContour)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)