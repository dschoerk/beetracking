from os import listdir
import os
from os.path import isfile, join
import re
import pickle
import imutils
from imutils.object_detection import non_max_suppression
from imutils import paths

import cv2
import numpy as np
#import xml.etree.ElementTree as ET

#from skimage.feature import hog
#from skimage import data, color, exposure

inputvids = ['1.tracking/count-hard.mp4', 
             '1.tracking/tracking-moderate.mp4', 
             '2.pollen-hubs/pollen-hub-blue.mp4',
             '2.pollen-hubs/pollen-hub-yellow.mp4',
             '2.pollen-hubs/pollen-hub-yellow2.mp4',
             '2.pollen-hubs/pollen-hub-yellow3.mp4',
             '3.parasites/mite1.mp4',
             '3.parasites/mite-walking-through.mp4',
             '4.other-insects/bug1.mp4',
             '5.dead-bee/bee1.mp4',
             '5.dead-bee/bee2.mp4',
             '5.dead-bee/bee3.mp4'
]
selectedVideoIdx = 0

# parameterize hog
winSize = (64,128)
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 16
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
cvhog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

if False: # should learn a new model

    n_features = cvhog.compute(np.zeros((128, 64), np.uint8)).shape[0]

    # read positive training data
    pathtoimg_positive = 'positive'
    onlyfiles_positive = [f for f in listdir(pathtoimg_positive) if isfile(join(pathtoimg_positive, f))]
    np.random.shuffle(onlyfiles_positive)
    num_files_positive = len(onlyfiles_positive)

    # read negative training data
    pathtoimg_negative = 'negative'
    onlyfiles_negative = [f for f in listdir(pathtoimg_negative) if isfile(join(pathtoimg_negative, f))]
    np.random.shuffle(onlyfiles_negative)
    num_files_negative = len(onlyfiles_negative)

    # set number of data used
    #num_files_positive = min(1000, num_files_positive) # limit number of training files
    #num_files_negative = min(1000, num_files_negative)
    trainData = np.zeros((num_files_positive + num_files_negative, n_features), np.float32)

    # set labels
    trainLabels = np.zeros((num_files_positive + num_files_negative,), np.int32)
    #trainLabels[0:num_files_positive] = 1
    #trainLabels[num_files_positive : num_files_positive + num_files_negative] = 2



    #svm_params = dict( kernel_type = cv2.ml.SVM_LINEAR, svm_type = cv2.ml.SVM_ONE_CLASS )

    #svm.setC(1.0); # penalty constant on margin optimization
    #svm.setType(cv2.ml.SVM_C_SVC); # multiple class (2 or more) classification
    #svm.setGamma(0.5); # used for SVM_RBF kernel only, otherwise has no effect
    #svm.setDegree(3);  # used for SVM_POLY kernel only, otherwise has no effect
    #svm.setNu(0.5);

    # set data negative
    print('set data negative ... %d' % num_files_negative)

    for idx, file in enumerate(onlyfiles_negative):
    
        if idx >= num_files_negative:
            break

        img = cv2.imread(pathtoimg_negative + "/" + file)
        hog_features = cvhog.compute(img)
        hog_features = hog_features.reshape(1, -1)
    
        trainData[idx, :] = hog_features
        trainLabels[idx] = 2

    # set data positive
    print('set data positive ... %d' % num_files_positive)
    for idx, file in enumerate(onlyfiles_positive):

        if idx >= num_files_positive:
            break

        idx = idx + num_files_negative

        img = cv2.imread(pathtoimg_positive + "/" + file)
        hog_features = cvhog.compute(img)
        hog_features = hog_features.reshape(1, -1)

        trainData[idx, :] = hog_features
        trainLabels[idx] = 1



    if False: # do validation
        validation_iterations = 5
        for n in range(validation_iterations):
            # init svm
            svm = cv2.ml.SVM_create()
            svm.setType(cv2.ml.SVM_C_SVC)
            svm.setKernel(cv2.ml.SVM_LINEAR)

            trainDataPerm = trainData.copy();
            trainLabelsPerm = trainLabels.copy();

            perm = np.random.permutation(trainData.shape[0])
            for idx, p in enumerate(perm):
                trainDataPerm[idx] = trainData[p]
                trainLabelsPerm[idx] = trainLabels[p]

            numValidate = int(trainDataPerm.shape[0] * 20 / 100)
            testSetData = trainDataPerm[0:numValidate, :]
            trainSetData = trainDataPerm[numValidate : -1, :]
            testSetLabel = trainLabelsPerm[0:numValidate]
            trainSetLabel = trainLabelsPerm[numValidate:-1]
            
            #print(testDataPerm.shape)

            print('train ... ');
            svm.train(trainSetData, cv2.ml.ROW_SAMPLE, trainSetLabel)
            #print('training done');
            #svm.train(training_attributes, cv2.ml.ROW_SAMPLE, training_class_labels);
            #save_svm_data(svm, 'svm.pickle')
            #svm.save("svm.xml")

            correctScore = 0
            for idx in range(testSetData.shape[0]):

                testsample = testSetData[idx, :].reshape(1, -1)

                _, result = svm.predict(testsample, cv2.ml.ROW_SAMPLE)
                if result == testSetLabel[idx] :
                   correctScore += 1

            print('score: %3.2f' % (correctScore / testSetData.shape[0]))  
    
    
    # compute the full model
    print('full model train ... ');
    svm = cv2.ml.SVM_create()
    svm.train(trainData, cv2.ml.ROW_SAMPLE, trainLabels)
    svm.save("svm.xml")
else:
    print('loading svm model ... ');
    svm = cv2.ml.SVM_load('svm.xml')

#########################
## test a video


# http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

cap = cv2.VideoCapture(inputvids[selectedVideoIdx])
outputVideo = None 
while(True):
    ret, image = cap.read()
    if not ret:
        break

    image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    image_out = image.copy()
    stepSize = 16
    winW, winH = 64, 128

    if outputVideo is None:
        outputVideo = cv2.VideoWriter('SVM_classified_nonmaximasupression.avi', -1, 10, (image.shape[1], image.shape[0]))

    rects = []

    for (x, y, window) in sliding_window(image, stepSize=8, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
           continue
        
        hog_features = cvhog.compute(window)
        hog_features = hog_features.reshape(1, -1)
        _, result = svm.predict(hog_features, cv2.ml.ROW_SAMPLE);

        if(result == 1):
            rects.append((x,y))
            cv2.rectangle(image_out, (x,y), (x+winW, y+winH), (0,0,255), 1)
            #print('not 2
        #else:
        #    print(result)
        # and for undocumented reasons take the first element of the resulting array
        # as the result

        #print("Test data example result =  {}".format(int(result[0])));
    
    rects = np.array([[x, y, x + winW, y + winH] for (x, y) in rects])
    #print(rects.shape)
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    #print(pick.shape)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image_out, (xA,yA), (xB, yB), (0,255,0), 2)

    cv2.imshow("result", image_out)
    cv2.waitKey(1)
    outputVideo.write(image_out)
