import cv2
from cv2 import VideoCapture
from cv2 import waitKey
import numpy as np
import wx
from pynput.mouse import Button, Controller

mouse = Controller()

app = wx.App(False)
(screenx, screeny) = wx.GetDisplaySize()
(capturex, capturey) = (700, 500)

cap = cv2.VideoCapture(0)
cap.set(3, capturex)
cap.set(4, capturey)

# Noise in Yellow Area
kernelOpen = np.ones(5, 5)
kernelClose = np((20, 20))

# Detectable Colour Range
lowerBound = np.array([20,100,100])
upperBound = np.array([120,255,255])

cd = 0

while True:
    ret, frame = cap.read()

    imgSeg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masking and Filtering Yellow
    mask = cv2.inRange(imgSeg,lowerBound,upperBound)

    # Morphology
    maskOpen =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernelClose)

    final = maskClose
    _, conts, h = cv2.findContours(maskClose,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(conts) != 0: # Contour of Highest Object
        b = max(conts, key=cv2.contourArea)
        west = tuple(b[b[:,:,0].argmin()][0]) #Left Extreme Contour
        east = tuple(b[b[:,:,0].argmax()][0])
        north = tuple(b[b[:, :, 1].argmin()][0])
        south = tuple(b[b[:,:,1].argmax()][0])

        centre_x = (west[0]+east[0])/2
        centre_y = (north[0]+south[0])/2

        cv2.drawContours(frame,b,-1,(0,255,0),3)
        cv2.circle(frame,west, 6, (0,0,255), -1)
        cv2.circle(frame,east, 6, (0,0,255), -1)
        cv2.circle(frame,north, 6, (0,0,255), -1)
        cv2.circle(frame,south, 6, (0,0,255), -1)
        cv2.circle(frame, (int(centre_x), int(centre_y)),6,(0,0,255),-1) #Plots Centre

        bint = int(cv2.contourArea(b))

        if bint in range (8000,18000): # Open Hand
            mouse.release(Button.left)
            cv2.circle(frame, (int(centre_x),int(centre_y)), 6, (255,0,0), -1)
            mouse.position = (screenx-(centre_x*screenx/capturex),screeny-(centre_y*screeny/capturey))

        elif bint in range(2000,7000): # Closed Hand
            cv2.circle(frame, (int(centre_x),int(centre_y)), 10, (255,255,255), -1)#plots centre of the area
            mouse.position = (screenx-(centre_x*screenx/capturex), screeny-(centre_y*screeny/capturey))
            mouse.press(Button.left)

        cv2.imshow('video',frame)
        if cv2.waitKey(1) & 0xFF == ord(' '): #Exit
            break

cap.release()
cv2.destroyAllWindows()