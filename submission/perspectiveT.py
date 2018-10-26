#!/usr/bin/python
#COMP9517 Assignment 2, Perspective Tranform of the court
#By Mahima Mandal, z5113392

import numpy as np 
import sys, cv2
from PIL import Image
import math

#Create window for selecting corners
winName = 'Select court corners, going clockwise from top left corner'
cv2.namedWindow(winName)
#Selection frame needs to be global for mouseCallback
sel_frame = []
#Create global for recording the corners, for mouseCallback
corners = np.zeros((4,2), dtype="float32")
#And how many corners have been selected
corners_found = 0

#Mouse callback function
def get_corner(event, x,y, flags, param):
	global corners_found
	if event == cv2.EVENT_LBUTTONDBLCLK:
		#draw a small indication dot
		cv2.circle(sel_frame, (x,y),5, (0,255,0),-1)
		#add and update number of corners found
		corners[corners_found,:]= [x,y]
		corners_found +=1
		
#Link callback function to the window
cv2.setMouseCallback(winName,get_corner)

#Define the four point transform
#Input: original image, width of ref court, height of ref court, 4 points of court
def boundary_transform(img, width, height,pts):
	#Get each of the four corners separately
	(tl, tr, br, bl) = pts

	maxWidth = width
	maxHeight = height
	
	#Define destination points for the birds eye view
	dst = np.array([[0,0], [maxWidth -1,0], [maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
	
	#Compute perspective tranform matrix
	M = cv2.getPerspectiveTransform(pts, dst)
	#Get the newly warped image
	#print("passing in "+ str(maxHeight))
	#print("passing in "+ str(maxWidth))
	warped = cv2.warpPerspective(img, M, (maxWidth,maxHeight))
	
	return (warped, M)
	
#Transform the given point under the given perspective transform
#returns the transformed point
def transform_point(x,y,H,maxW,maxH):
	global sel_frame
	#get dimensions of the selection frame
	h, w = sel_frame.shape[:2]
	#print("Selection:" + str(h) +" "+str(w))
	#Make new frame into an image
	newFrame = np.zeros((h,w),dtype="float32")
	newFrame[y][x] = 255
	im_bit= np.stack([newFrame, newFrame, newFrame], axis=2) 
	im_bit = cv2.circle(im_bit, (int(x),int(y)),5, (255,255,255),-1)
	#cv2.imshow('im_bit', im_bit)
	#cv2.waitKey(0)
	wrp = cv2.warpPerspective(im_bit, H,(maxW,maxH))
	#cv2.imshow('gray', wrp)
	#cv2.waitKey(0)
	point = np.where(wrp[...,0]>0)
	#print(point)
	x_point = np.average(point[0][:])
	y_point = np.average(point[1][:])

	return x_point, y_point

#Given a point and the tranformation matrix
#return a circle where the point is transformed to
def draw_ref_point(x,y,H, ref_img, frame):
	ref_h, ref_w = ref_img.shape[:2]
	fr_h, fr_w = frame.shape[:2]
	global sel_frame
	sel_h, sel_w = sel_frame.shape[:2]
	#print("==============frame sizes===============")
	#print("Reference:" + str(ref_h) +" "+str(ref_w))
	#print(ref_img.shape)
	#print("Frame:" + str(fr_h) +" "+str(fr_w))
	#print("Selection:" + str(sel_h) +" "+str(sel_w))
	#print("transforming : "+ str(y*sel_h/fr_h)+"  "+str(x*sel_w/fr_w))
	newX, newY = transform_point(int(x*sel_w/fr_w),int(y*sel_h/fr_h),H, ref_w,ref_h)
	#print("returned value : " + str(newY) + " " + str(newX))
	if math.isnan(newY) != True:
		cv2.circle(ref_img, (int(newY),int(newX)),5, (0,255,0),-1)
	return ref_img

#Choose boundary parameters based on first frame
def choose_boundary(img):
	global sel_frame
	sel_frame = img.copy()
	sel_frame = cv2.resize(sel_frame,(898,503))

	sel_frame = np.pad(sel_frame, [(0,0), (140,140), (0, 0)], 'constant')

	while(corners_found != 4):
		cv2.imshow(winName, sel_frame)
		#Allow user to escape...
		if cv2.waitKey(20) == 27:
			break
	#End program if they didn't select the corners
	if corners_found !=4:
		sys.exit()

#set up the transformation matrix, returns the transformation matrix
def setupTransform(frame, ref_img):
	#set the dimensions of the reference court
	#ref_court = cv2.imread('half_court_ref.PNG')
	ref_court = ref_img.copy()
	ref_h, ref_w = ref_court.shape[:2]
	
	choose_boundary(frame)
	
	#Get the birds eye of the selected court
	#Warped is the selected boundary warped into a rectangle
	#H is the perspective transform matrix
	(warped, H)= boundary_transform(sel_frame,ref_w,ref_h,corners)
	
	return H
	
## =============== START OF PROGRAM ============== ##
#if __name__ == "__main__": 
	#Input error checking code
#	if (len(sys.argv) != 2):
#		print ('Usage: ./projection.py <input image> ')
#		sys.exit(1)
		
#	first_frame = cv2.resize(cv2.imread(sys.argv[1]), (0,0), fx=0.5 ,fy=0.5) 
#	ref_court = cv2.resize(cv2.imread('half_court_ref.PNG'), (0,0), fx=0.5 ,fy=0.5) 
	
#	H = setupTransform(first_frame)
	
#	x = 291 #point on the actual frame, no rescale
#	y = 388
#	ref_court = draw_ref_point(x,y,H, ref_court,first_frame)
#	toSave = Image.fromarray(ref_court)
#	toSave.save("cooourt.jpg")
	#The rest is for drawing the court
#	cv2.imshow('Reference court', ref_court)
#	cv2.waitKey(0)
#	cv2.circle(first_frame, (x,y),5,(0,255,0),-1)
#	cv2.imshow('point', first_frame)
#	cv2.waitKey(0)
#	cv2.destroyAllWindows()
