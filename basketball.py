import numpy as np
import sys, cv2,re,math
from PIL import Image
import argparse
import imutils
from skimage.measure import compare_ssim
import time
from math import sqrt
import EKF
import PedestrianDetection
import tracker
import perspectiveT


trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']



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

def detectBall(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    fra = frame.copy()
    BW = (lab[...,0] >= ch1min ) | (lab[...,0] <= ch1max) & (lab[...,1] >= ch2min ) & (lab[...,1] <= ch2max) & (lab[...,2] >= ch3min ) & (lab[...,2] <= ch3max)
    BW.dtype='uint8'
    BW = BW*255
    
    BW12 = (fra[...,0] >= channel1Min ) | (fra[...,0] <= channel1Max) & (fra[...,1] >= channel2Min ) & (fra[...,1] <= channel2Max) & (fra[...,2] >= channel3Min ) & (fra[...,2] <= channel3Max)
    BW12.dtype='uint8'
    BW12 = BW12*255
    BW12 = ~BW12
    BW = BW - BW12
    
    # To show the detected orange parts:
    im_orange = frame.copy()
    im_hsv = BW
    im_orange[im_hsv==0] = 0
    
    # Perform opening to remove smaller elements
    element = np.ones((11,11)).astype(np.uint8)
    im_hsv = cv2.erode(im_hsv, element)
    element = np.ones((3,3)).astype(np.uint8)
    im_hsv = cv2.dilate(im_hsv, element)
    
    #    (96, 274, 388, 147)

    im_hsv[275:470, 96: 500] =0
    
#    cv2.imshow('bw',im_hsv)

    points = (np.where(im_hsv>254))
    x = np.average(points[0][:])
    y = np.average(points[1][:])
    x = round(x)
    y = round(y)
    return x,y


def houghMethod(frame,flag,R_avg):
    ret, box1 = tracker1.update(frame.copy())
    
    x_box1 = (int(box1[0][1]+box1[0][3]))/2
    y_box1 = (int(box1[0][0]+box1[0][2]))/2
#    B = frame.copy()
#    B[:,:,1]=0
#    B[:,:,2] = 0
#
#    G = frame.copy()
#    G[:,:,0]=0
#    G[:,:,2] = 0
#
#    R = frame.copy()
#    R[:,:,1]=0
#    R[:,:,0] = 0

    region1 = (frame[int(box1[0][1]): int(box1[0][1]+box[3]),
                                int(box1[0][0]): int(box1[0][0]+box[2])])
                                
    region = cv2.cvtColor(frame[int(box1[0][1]): int(box1[0][1]+box[3]),
                                int(box1[0][0]): int(box1[0][0]+box[2])], cv2.COLOR_BGR2GRAY)
    
    lab = cv2.cvtColor(region1, cv2.COLOR_BGR2HSV)
    BW = (lab[...,0] >= ch1min ) | (lab[...,0] <= ch1max) & (lab[...,1] >= ch2min ) & (lab[...,1] <= ch2max) & (lab[...,2] >= ch3min ) & (lab[...,2] <= ch3max)
    BW.dtype='uint8'
    BW = BW*255

    BW[272:422,77:357]
    cv2.imshow('region',region1)
    
    if flag == 7:
        R_avg = np.average(region1[:,:,2])
    
    (score, diff) = compare_ssim(ref_region, region, full=True)
    scored =0
    if score < 0.40 and np.average(region1[:,:,2]) > R_avg + 8:
        time.sleep(0.3)
#        print(score)
#        print("B:", np.average(region1[:,:,0]))
#        print("G:", np.average(region1[:,:,1]))
#        print("R:", np.average(region1[:,:,2]))
        scored= 1
#        print('-'*30)
    return R_avg,scored


if __name__ == "__main__":
    #(384, 118, 43, 40)
    score = 0
    not_detect = 0
    cam = cv2.VideoCapture('123.mp4')
    flag =7
    tracker1 = cv2.MultiTracker_create()
#    out = cv2.VideoWriter('output1.mp4', -1, 20.0, (1200,720))
    count = 1
    detectFreq = 10
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    colour = (238,130,238)
    while(cam.isOpened()):
        count -=1
        ret, frame = cam.read()
        frame = np.asarray(frame)
        frame = cv2.resize(frame, (1200,720))
        frame1 = frame.copy()
#        frame = imutils.resize(frame, width=min(1200, frame.shape[1]))
        if count == 0:
            rects,frame = PedestrianDetection.PedestrianDetection(hog, frame)
#            print("length is:",len(rects))
            if (len(rects) == 0):
                # skip 10 frames if no people are found
                count = 1
#                print("Null returned")
            else:
#                print("initalized")
                multiTracker = tracker.initialiseMultiTracker(trackerTypes[7], frame, rects)
                count = detectFreq
        else:
#            print(count)
            success, boxes = multiTracker.update(frame)
            if (len(boxes) < len(rects)) :
                count = 1
            
            if success:
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, colour, 2, 1)
        
        if flag == 7:
            box = cv2.selectROI('Select hoop area', frame, fromCenter=False,showCrosshair=True)
            print(box)
            ref_region = frame1[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
            ref_region = cv2.cvtColor(ref_region ,cv2.COLOR_BGR2GRAY)
            ret = tracker1.add(cv2.TrackerCSRT_create(), frame, box)
            R_avg = 0
            R_avg, scored = houghMethod(frame1,flag, R_avg)
#        print(flag)

        
        y,x = detectBall(frame)
#        print(x)
#        print(y)
        if flag == 7:
            Xe = np.array([y,x,0.,0.])[np.newaxis,:].T
            P = np.zeros((4,4))
            dt = 1/30

        
        if flag % 7 == 0:
            prev_x = x
            prev_y = y
        flag = flag + 1


#        print('-'*30)
#        if( flag > 50):
#            time.sleep(2)

        if not np.isnan(x) and not np.isnan(y):
            if not np.isnan(prev_x) and not np.isnan(prev_y):
                dist = math.hypot(x - prev_x, y - prev_y)
                if (not_detect < 5 and dist >60):
#                    print("wrong")
#                    print('\/'*30)

#                    prev_x = Xe[1]
#                    x = Xe[1]
#                    prev_y = y = Xe[0]

                    (Xe,P) = EKF.predict(Xe,P,dt)
                elif(not_detect > 5 or dist <60):
                    (Xe,P) = EKF.ApplyEKF(Xe,P,dt,y,x)
                    not_detect = 0
#            print("something special:", Xe[2])
            elif(not_detect> 5):
                print("else: ",not_detect)
                (Xe,P) = EKF.ApplyEKF(Xe,P,dt,y,x)
            if Xe[2] > 0:
                R_avg, scored = houghMethod(frame1,flag, R_avg)
                
                if scored==1:
                    score = score +1
                    for j in range(7):
                        ret, frame = cam.read()
#                        cv2.imshow('video',frame)
                        frame[:,:,:] = 0
                        cv2.putText(frame, "Scored!!!", (300,360), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 2)
                        cv2.imshow('video',frame)
                    print("socred")
            
            cv2.circle(frame, (int(x), int(y)), int(3), (255,255,255), thickness=5)
        else:
#            print("not detected")
#            print("flag: ",flag)
#            print(not_detect)
            not_detect += 1
            (Xe,P) = EKF.predict(Xe,P,dt)
        cv2.putText(frame, ("Score: " +str(score)), (25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.circle(frame, (int(Xe[1]), int(Xe[0])), int(3), (112,255,75), thickness=5)
        cv2.imshow('video',frame)
        cv2.waitKey(2)
#        out.write(frame)
#        if flag == 490:
#            out.release()
#            break
        if cv2.waitKey(1) & 0xFF == ord('/'):
            r = cv2.selectROI(frame)
            print(r)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
