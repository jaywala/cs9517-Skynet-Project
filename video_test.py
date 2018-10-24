# import the necessary packages
#from __future__ import print_function
#from imutils.object_detection import non_max_suppression
#from imutils import paths
#import tracker
import numpy as np
#import argparse
import imutils
import cv2
#from random import randint
import perspectiveT

if __name__ == "__main__":

    # initialize the HOG descriptor/person detector
    #hog = cv2.HOGDescriptor()
    #hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #colour = (238,130,238)

    # Read video
	video = cv2.VideoCapture('/media/sf_VirtualBox/cs9517/assigntments/main_project/cs9517-Skynet-Project/data/CompressedVideos/20181022_143252_compressed.mp4')
    # fps = video.get(cv2.CV_CAP_PROP_FPS)
    # print(fps)
    # Check every n number of frames
    #detectFreq = 50
    #count = 1

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
        #count -= 1
		frame = imutils.resize(frame, width=min(800, frame.shape[1]))
		ref_court_copy = ref_court.copy()
        
		cv2.imshow('current frame', frame)
		if cv2.waitKey(20) == 27:
			break

		ref_court = perspectiveT.draw_ref_point(550,360,H,ref_court_copy,frame)
                # print("showing image")
		cv2.imshow('video', ref_court)
		cv2.waitKey(1)

	video.release()
	cv2.destroyAllWindows()
