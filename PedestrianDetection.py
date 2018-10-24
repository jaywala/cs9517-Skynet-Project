# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import tracker
import numpy as np
import argparse
import imutils
import cv2
from random import randint
import perspectiveT

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def PedestrianDetection(hog, image):

    # image = frame

    orig = image.copy()

    # # Histrogram equalisation
    # img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # # equalize the histogram of the Y channel
    # img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # # convert the YUV image back to RGB format
    # image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

	# detect people in the image
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),padding=(8, 8), scale=1.05)
 
	# draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
 
	# show some information on the number of bounding boxes
	# filename = imagePath[imagePath.rfind("/") + 1:]
    # print("[INFO] {} original boxes, {} after suppression".format(len(rects), len(pick)))
 
	# show the output images
    # cv2.imshow("Before NMS", orig)
    cv2.imshow("video", image)
    cv2.waitKey(4)
    return rects

if __name__ == "__main__":

    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    colour = (238,130,238)

    # Read video
    video = cv2.VideoCapture('/media/sf_VirtualBox/cs9517/assigntments/main_project/cs9517-Skynet-Project/data/CompressedVideos/20181022_142951_compressed.mp4')
    # fps = video.get(cv2.CV_CAP_PROP_FPS)
    # print(fps)
    # Check every n number of frames
    detectFreq = 50
    count = 1

    success, first_frame = video.read()
    if not success:
        print("video read error")
        exit

    # Set up for PerspectiveT
    ref_court = cv2.imread('half_court_ref.PNG')
    H = perspectiveT.setupTransform(first_frame)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        count -= 1
        frame = imutils.resize(frame, width=min(800, frame.shape[1]))
        ref_court_copy = ref_court.copy()

        if count == 0:
            rects = PedestrianDetection(hog, frame)
            if (len(rects) == 0):
                # skip 10 frames if no people are found
                count = 10
                print("Null returned")
            else:
                # print(rects[0])
                multiTracker = tracker.initialiseMultiTracker(trackerTypes[6], frame, rects)
                count = detectFreq
        else:
            success, boxes = multiTracker.update(frame)
            if success:
                for i, newbox in enumerate(boxes):
                    p1 = (int(newbox[0]), int(newbox[1]))
                    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                    cv2.rectangle(frame, p1, p2, colour, 2, 1)
                    ref_court = perspectiveT.draw_ref_point(newbox[1]+newbox[3]/2,newbox[0]+newbox[2],H, ref_court_copy,frame)
                # print("showing image")
                cv2.imshow('video', frame)
                cv2.waitKey(1)
        

    

    video.release()
    cv2.destroyAllWindows()