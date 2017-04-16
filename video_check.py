import cv2
import numpy as np

cap = cv2.VideoCapture("./project_video_test.mp4")
while(True):
    currentFrameCandidates = []
    ret, frame2 = cap.read()
    if  ret == -1 :
        break
    cv2.imshow('frame2',frame2) #paint frame

    k = cv2.waitKey(1000)
    if k==32:    # Esc key to stop
        cv2.destroyAllWindows()
        break
cv2.destroyAllWindows()
