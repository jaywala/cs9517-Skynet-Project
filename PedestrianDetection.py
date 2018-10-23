# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2


def PedestrianDetection(hog, frame):

    # image = frame
    image = imutils.resize(frame, width=min(800, frame.shape[1]))
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
    cv2.imshow("Before NMS", orig)
    cv2.imshow("After NMS", image)
    cv2.waitKey(4)
    return rects

if __name__ == "__main__":
    # initialize the HOG descriptor/person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Read video
    video = cv2.VideoCapture('/media/sf_VirtualBox/cs9517/assigntments/main_project/cs9517-Skynet-Project/data/new/20181022_150153.mp4')
    # Check every n number of frames
    checkfreq = 30
    count = checkfreq
    while video.isOpened():
        count -= 1
        success, frame = video.read()
        if not success:
            break
        

        if count == 0:
            count = checkfreq
            if (len(PedestrianDetection(hog, frame)) == 0):
                # skip 10 frames if no people are found
                count = 10
                print("Null returned")
    

    video.release()
    cv2.destroyAllWindows()