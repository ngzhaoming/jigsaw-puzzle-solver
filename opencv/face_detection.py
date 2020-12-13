import cv2

# Opencv have cascade templates
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

img = cv2.imread('resources/face.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# A2: scale factor, A3: Minimum neighbours, threshold value
faces = face_cascade.detectMultiScale(imgGray, 1.1, 10)

# Create a bouding box around the faces detected
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Result", img)
cv2.waitKey(0)