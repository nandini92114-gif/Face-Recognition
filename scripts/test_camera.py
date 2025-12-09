import cv2

cap = cv2.VideoCapture(0)
print("Opened:", cap.isOpened())

for i in range(10):
    ret, frame = cap.read()
    print("Frame", i, "ok:", ret)

cap.release() 