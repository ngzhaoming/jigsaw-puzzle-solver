import cv2

"""
Reading a video:

1) Import the video using VideoCapture - Give parameter of id for webcam 0 for in-built
2) Video is a sequence of images (frames), use a while loop
    success is a bool whether img is read properly
3) If block to check whether 'q' key is pressed to quit video

NOTE: set(id, size): id == 3 is width, id == 4 is height, id == 10 is brightness

vid = cv2.VideoCapture("resources/mya.mp4")

while True:
    success, img = vid.read()
    cv2.imshow("VID_WINDOW", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 640)
cam.set(10, 100)

while True:
    success, img = cam.read()
    cv2.imshow("VID_WINDOW", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

"""
Reading an image:

1) Import the image using imread
2) Display the image using imshow
3) Use a delay so displayed image does not close immediately

img = cv2.imread("../white/piece1.jpg")
cv2.imshow("WINDOW_NAME", img)
cv2.waitKey(0)
"""

