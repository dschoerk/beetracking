## http://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import pickle
 
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--images", required=True, help="path to images directory")
#args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#trainedmodel = pickle.load(open("svm.pickle"))
#hog.setSVMDetector(trainedmodel)

# loop over the image paths
cap = cv2.VideoCapture('1.tracking/count-hard.mp4')
outputVideo = None 
frame = 0
while(True):
    # Capture frame-by-frame
	ret, image = cap.read()
	if not ret:
		break
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	#image = cv2.imread(imagePath)
	image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
	orig = image.copy()
	
	if outputVideo is None:
		outputVideo = cv2.VideoWriter('personbeetracking.avi', -1, 15, (image.shape[1], image.shape[0]))

	# detect people in the image
	(rects, weights) = hog.detect(image, winStride=(2, 2))

	w = 64
	h = 128
	# draw the original bounding boxes
	for (x, y) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		
	cv2.imshow("Before NMS", orig)
	cv2.imshow("After NMS", image)
	cv2.waitKey(3)
	
	outputVideo.write(image)
	
	frame+=1
	print(frame)
	#if frame > 200:
#		break
	