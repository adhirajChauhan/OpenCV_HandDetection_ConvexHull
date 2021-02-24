# OpenCV_HandDetection_ConvexHull

import cv2
import numpy as np

video = cv2.VideoCapture(0)
kernel = np.ones((2, 2), np.uint8)

while True:
    
    (ret, frame) = video.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.medianBlur(gray, 3)
    #crcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    retval, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    #newFrame = np.array(thresh)

    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    ###############################################################################
    hull_list = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i])
        hull_list.append(hull)

        color = (255, 0, 0)
        cv2.drawContours(frame, hull_list, i, color)
    ###############################################################################

        
    if len(contours) > 0:
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 5)
        # find the biggest countour (c) by the area
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Capturing", frame)
    #cv2.imshow("CrCb",crcb)
    #cv2.imshow("gray", gray)
    cv2.imshow("thresh", thresh)
    #cv2.waitKey(0)
        
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
