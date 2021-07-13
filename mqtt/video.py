import cv2
import time

cap = cv2.VideoCapture("http://192.168.60.103:8080/?action=stream")

count = 1
start = time.time()
while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    end = time.time()
    if end - start > 1:
        cv2.imwrite("image/" + str(count) + ".jpg", frame)
        count = count + 1
        start = time.time()
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()