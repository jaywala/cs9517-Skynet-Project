#!/usr/bin/python
#COMP9517 Assignment 2, Recognition of court
#By Mahima Mandal, z5113392

import numpy as np
import sys, cv2,re,math
from PIL import Image
import argparse
#from skimage.measure import compare_ssim
import imutils
from skimage.measure import compare_ssim


## =============== START OF PROGRAM ============== ##
if __name__ == "__main__":
	#Input error checking code
	if (len(sys.argv) != 2):
		print ('Usage: ./houghMethod.py <input video>')
		sys.exit(1)

	#use first image as reference image
	ref = cv2.imread('F_1.PNG')

	#Get user to select the hoop
	box = cv2.selectROI('Select hoop area', ref, fromCenter=False,showCrosshair=True)
	ref_region = ref[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])]
	ref_region = cv2.cvtColor(ref_region ,cv2.COLOR_BGR2GRAY)

	for i in range(1,7):
		fileName = 'F_'+ str(i)+'.PNG'
		img = cv2.imread(fileName)
		region = cv2.cvtColor(img[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2])], cv2.COLOR_BGR2GRAY)
		(score, diff) = compare_ssim(ref_region, region, full=True)
		print(score)
		cv2.imshow(fileName, region)
		cv2.waitKey(0)

	cv2.destroyAllWindows()
