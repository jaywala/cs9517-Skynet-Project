import cv2
import numpy as np

cam = cv2.VideoCapture('69.mp4')

ch1min = 0.795*180
ch1max = 0.539*180

ch2min = 0.552*255.0
ch2max = 0.247*255.0

ch3min = 0.851*255.0
ch3max = 0.629*255.0

cv2.namedWindow("Window")
channel1Min = 255
channel1Max =  151

channel2Min = 0
channel2Max = 1*255.0

channel3Min = 0*255.0
channel3Max = 1*255.0



while(cam.isOpened()):
    ret, frame = cam.read()
    frame = np.asarray(frame)
    frame = cv2.resize(frame, (720,720))
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fra = frame.copy()
    BW = (lab[...,0] >= ch1min ) | (lab[...,0] <= ch1max) & (lab[...,1] >= ch2min ) & (lab[...,1] <= ch2max) & (lab[...,2] >= ch3min ) & (lab[...,2] <= ch3max)
        #Convert true and false to 1 and 0 respectively
    BW.dtype='uint8'
    #Scale 1/0 to 255/0
    BW = BW*255

    BW12 = (fra[...,0] >= channel1Min ) | (fra[...,0] <= channel1Max) & (fra[...,1] >= channel2Min ) & (fra[...,1] <= channel2Max) & (fra[...,2] >= channel3Min ) & (fra[...,2] <= channel3Max)
    BW12.dtype='uint8'
    #Scale 1/0 to 255/0
    BW12 = BW12*255
    BW12 = ~BW12
    BW = BW - BW12

    # To show the detected orange parts:
    im_orange = frame.copy()
    im_hsv = BW
    im_orange[im_hsv==0] = 0

    # Perform opening to remove smaller elements
    element = np.ones((9,9)).astype(np.uint8)
    im_hsv = cv2.erode(im_hsv, element)
    element = np.ones((3,3)).astype(np.uint8)
    im_hsv = cv2.dilate(im_hsv, element)
    print (np.max(im_hsv))
    print (np.min(im_hsv))
    cv2.imshow('im_orange',im_hsv)

    points = (np.where(im_hsv>254))

    x = np.average(points[0][:])
    y = np.average(points[1][:])
    x = round(x)
    y = round(y)
    
    print('-'*30)

    if not np.isnan(x) and not np.isnan(y):
        cv2.circle(frame, (int(y), int(x)), int(3), (0,0,255), thickness=5)

    cv2.imshow('Window',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
