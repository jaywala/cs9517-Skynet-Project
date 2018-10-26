import numpy as np
# from __future__ import print_function
import sys
import cv2
from random import randint

def funcname(self, parameter_list):
    pass

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
 
def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
        
    return tracker

def initialiseMultiTracker(trackerType, frame, rects):
    # Create MultiTracker object
    multiTracker = cv2.MultiTracker_create()
    # Initialize MultiTracker 
    for rect in rects:
            bbox = (rect[0],rect[1], rect[2]-rect[0],rect[3]-rect[1])
            multiTracker.add(createTrackerByName(trackerType), frame, bbox)
    return multiTracker

