import numpy as np
import cv2
import os

print(cv2.__version__)

inputvids = ['1.tracking/tracking-moderate.mp4', 
             '1.tracking/count-hard.mp4', 
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

    # video 4 ghosts
for selectedVideoIdx in range(0,12):
    selectedVideo = inputvids[selectedVideoIdx]

    video_paused = False

    #fgbg = cv2.createBackgroundSubtractorKNN()
    #fgbg2 = cv2.createBackgroundSubtractorMOG2(500, 16, False)
    #kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    font = cv2.FONT_HERSHEY_SIMPLEX

    c,r,w,h = 1,1,1,1
    track_window = (c,r,w,h)
    # Create mask and normalized histogram
    #roi = None
    #hsv_roi = None
    #mask = None
    #roi_hist = None
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    track_window_w = 40
    track_window_h = 60

    bound_top = 45 # y coordinate where to stop tracking
    bound_bottom = 240 # 

    #n_hist_dims = 256

    tracker = None
    trackers = []
    positive_image_index = 0

    frame = 0

    ### Tracks one bee
    class BeeTracker:
        def __init__ (self, center, dimensions, image_hsv):

            self.n_hist_dims = 256 # could probably be lower
            self.color = (np.random.rand(3) * 255)#.astype(np.uint8)
        
            self.x = center[0]
            self.y = center[1]
            self.oldx = self.x+3
            self.oldy = self.y+3

            self.matching = 0
            self.val = 0

            self.dimensions = dimensions
            self.reposition(center, image_hsv)
            self.repositioned = frame

        def reposition(self, center, image_hsv):

            halfw = int(self.dimensions[0] / 2)
            halfh = int(self.dimensions[1] / 2)

            self.track_window = (center[0]-halfw,center[1]-halfh,self.dimensions[0],self.dimensions[1])
            hsv_roi = image_hsv[center[1]-halfh : center[1]+halfh, center[0]-halfw : center[0]+halfw] 
            self.roi_hist = cv2.calcHist([hsv_roi], [0], None, [self.n_hist_dims], [0, self.n_hist_dims])
            cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

            #self.oldx = self.x
            #self.oldy = self.y
            self.x = center[0]
            self.y = center[1]

            self.repositioned = frame
        
        def update(self,image_hsv, illustration_image, trajectory_image):
            dst = cv2.calcBackProject([image_hsv], [0], self.roi_hist, [0, self.n_hist_dims], 1)
            #print(dst.sum().sum())
            ret, self.track_window = cv2.meanShift(dst, self.track_window, term_crit)
            x,y,w,h = self.track_window

            font = cv2.FONT_HERSHEY_SIMPLEX
        
        

            #cv2.imshow('frame3', image_hsv)
            self.oldx = self.x
            self.oldy = self.y
            self.x = int(x + w / 2)
            self.y = int(y + h / 2)

        def draw(self, illustration_image, trajectory_image):

            halfw = int(self.dimensions[0] / 2)
            halfh = int(self.dimensions[1] / 2)

            if(illustration_image is not None):
                cv2.rectangle(illustration_image, (self.x - halfw, self.y - halfh), (self.x + halfw, self.y + halfh), self.color, 2)

            if(trajectory_image is not None):
                cv2.line(trajectories, (self.x, self.y), (self.oldx, self.oldy), self.color)

        def pos(self):
            return (self.x, self.y)

        def sanitize(self, mask, illustration_image, trackers):
            dst = np.linalg.norm([self.oldx - self.x, self.oldy - self.y])
            #print(dst)
            halfw = int(self.dimensions[0] / 2)
            halfh = int(self.dimensions[1] / 2)
        
            self.val = ( mask[self.y - halfh : self.y + halfh, self.x - halfw : self.x + halfh].sum() / 255 ) * 100 / (self.dimensions[0]*self.dimensions[1])
            cv2.putText(illustration_image,str(int(self.val)),(self.x,self.y), font, 0.7,(255,255,255),1,cv2.LINE_AA)

            for t in trackers:
                if t is self:
                    continue

                tdst = np.linalg.norm([self.x - t.x, self.y - t.y])
            
                if(tdst <= 15 and self.repositioned < t.repositioned):
                    #cv2.line(trajectories, (self.x, self.y), (t.x, t.y), self.color)
                    return False
        
            return dst < 30 and self.val > 25 #and self.matching < 8000000 # and dst > 0.5

        def blackout_window(self, img):
            halfw = int(self.dimensions[0] / 2) + 15
            halfh = int(self.dimensions[1] / 2)
            img[self.y - halfh : self.y + halfh, self.x - halfw : self.x + halfh] = 0

        def write_positive_image(self, index, image):

            if(self.x - 32 < 0 or self.x + 32 > image.shape[1] or self.y - 64 < 0 or self.y + 63 > image.shape[0]):
                return

            roi = image[self.y - 64 : self.y + 64, self.x - 32 : self.x + 32]
            if(self.val > 90):
                cv2.imwrite('positive/img_{}_{}.jpg' .format (selectedVideoIdx, index), roi);

        def __del__(self):
            pass    #print("bee lost")

    # The following code block was used to extract clean images of the background
    # negative samples for the generated dataset were sampled from the generated background image
    # see the report for more details
    """
    # background extraction
    cap = cv2.VideoCapture(selectedVideo)
    #fgbg = cv2.createBackgroundSubtractorMOG2(500,4,False)
    averageImage = None
    averageCount = None
    maskTemp = None
    num_frames = 0

    # os.system("pause")
    while(True):
        if not video_paused:
            ret, image = cap.read()
        if not ret:
            break

        image = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image_hsv_f = image_hsv.astype(np.float) / 255;

        if averageImage is None:
            averageImage = np.zeros(image.shape, np.float)
            averageCount = np.zeros(image.shape, np.float)
            maskTemp = np.zeros(image.shape, np.float)
    
   
        #fgmask = ~fgbg.apply(image)
        #bluemask = (image[:,:,0] < 128).astype(np.uint8)
        fgmask = (image_hsv[:,:,1] < 128).astype(np.uint8) * 255

        masked = cv2.bitwise_and(image, image, mask=fgmask)

        averageImage += masked

        maskTemp[:,:,0] = fgmask
        maskTemp[:,:,1] = fgmask
        maskTemp[:,:,2] = fgmask
        averageCount += maskTemp

        cv2.imshow('averageImage', averageImage / averageCount)
        cv2.waitKey(1)
    
        num_frames+=1

        #if(num_frames > 100):
        #    break
    
    averageImage = (averageImage / averageCount)
    averageHsv = cv2.cvtColor((averageImage * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
    equ = cv2.equalizeHist(averageHsv[:,:,0])
    image_hsv_equalized = averageHsv.copy()
    image_hsv_equalized[:,:,0] = equ
    #averageImage = cv2.cvtColor(image_hsv_equalized, cv2.COLOR_HSV2BGR)

    averageImage = (averageImage * 255).astype(np.uint8)
    cv2.imshow("averageImage", averageImage)

    def sliding_window(image, stepSize, windowSize):
	    # slide a window across the image
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
			    # yield the current window
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    stepSize = 8
    winW, winH = 64, 128
    for idx, (x, y, window) in enumerate(sliding_window(averageImage, stepSize=8, windowSize=(winW, winH))):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        cv2.imwrite('negative/img_{}_{}.jpg' . format(selectedVideoIdx, idx), window);

    #os.system("pause")
    """
    # end background extraction

    '''
    cap = cv2.VideoCapture(selectedVideo)
    summedImage = None
    while(True):
        if not video_paused:
            ret, image = cap.read()
        if not ret:
            break
        if summedImage is None:
            summedImage = np.zeros(image.shape, np.float)

        image_diff = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC).astype(np.float)
        image_diff = cv2.absdiff(image_diff, averageImage).astype(np.uint8);
        mask = (image_diff > 30).astype() # we see some activity on a pixel
    '''

    firstFrame = True
    trajectories = None
    cap = cv2.VideoCapture(selectedVideo)
    while(True):
        if not video_paused:
            ret, image = cap.read()

        if not ret:
            break

        frame += 1

        ## transform input image
        #image_diff = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC).astype(np.float)
        #image_diff = cv2.absdiff(image_diff, averageImage.astype(np.float)).astype(np.uint8);
        #image_diff_hsv = cv2.cvtColor(image_diff, cv2.COLOR_BGR2HSV)
        #cp = image_diff.copy();
        #cv2.reduce(cp, image_diff, dim=2, op=cv2.cv.CV_REDUCE_SUM)

        #diff_mask = ((image_diff[:,:,2] > 40))
        #cv2.imshow("diffr", image_diff[:,:,0])
        #cv2.imshow("diffg", image_diff[:,:,1])
        #cv2.imshow("diffb", image_diff[:,:,2])
        #cv2.imshow("diff", image_diff > 30)

        image_small = cv2.resize(image, None, fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
        image_intensity = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
        image_hsv = cv2.cvtColor(image_small, cv2.COLOR_BGR2HSV)
    
        #equ = cv2.equalizeHist(image_hsv[:,:,0])
        #image_hsv_equalized = image_hsv.copy()
        #image_hsv_equalized[:,:,0] = equ
        #equalized = cv2.cvtColor(image_hsv_equalized, cv2.COLOR_HSV2BGR)
        #cv2.imshow("equalized", (diff_mask * 255).astype(np.uint8))


    
        if firstFrame:
            trajectories = np.zeros(image_small.shape, np.uint8)
            firstFrame = False


        image_col = image_small / np.dstack([image_hsv[:,:,2],image_hsv[:,:,2],image_hsv[:,:,2]])
        image_hsv_f = image_hsv.astype(np.float) / 255;

        #bluemask = image_col[:,:,0] < 0.5 # unused but works too for thresholding
        redmask = image_hsv_f[:,:,1] > 0.6 # threshold on the saturation
        imgmask = ((redmask) * 255).astype(np.uint8)

        cv2.imshow("rawmask", imgmask)

        imgmask = cv2.morphologyEx(imgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))) # close the image

        # holefilling
        # floodfill the background # https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
        im_floodfill = imgmask.copy()
        h, w = imgmask.shape[:2]
        flood_mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(im_floodfill, flood_mask, (w-1,h-1), 255)
        cv2.floodFill(im_floodfill, flood_mask, (0,0), 255)
        cv2.floodFill(im_floodfill, flood_mask, (0,h-1), 255)
        cv2.floodFill(im_floodfill, flood_mask, (w-1,0), 255)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        
        imgmask_filled = imgmask | im_floodfill_inv
        too_small_blobs = np.ones((h, w), np.uint8) * 255
    
        
        #cv2.imshow("oldmask", imgmask)
    
        #print(image_hsv.shape)

        image_hsv = image_small # cv2.cvtColor(image_diff, cv2.COLOR_BGR2GRAY) # cv2.equalizeHist(.astype(np.float)) # 
        #cv2.imshow("lenewimg", image_hsv)
        #
        #diff_mask = (diff_mask * 255).astype(np.uint8) 
        #cv2.imshow("newmask", diff_mask)
    
        image_small_show = image_small.copy()

        #imgmask_cc = imgmask.copy();

        # update all the tracking windows
        for t in trackers:
            t.update(image_hsv, image_small_show, trajectories)
            #t.blackout_window(imgmask)


        ############
        ### Round 1

        imgmask_filled_blacked = imgmask_filled.copy();
        

        for t in trackers:
            # draws black boxes inside the binary image mask to remove already detected bees
            # this helps to better detect colliding bees
            t.blackout_window(imgmask_filled_blacked) 

        n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(imgmask_filled_blacked, 4, cv2.CV_8U)
        labeled_image = labels / np.amax(labels) # just for illustration

        for i in range(n_components):
            if(i == 0):
                continue

            area = stats[i, cv2.CC_STAT_AREA] # most important feature to classify the blobs is their area 
            centroid = (int(centroids[i][0]), int(centroids[i][1]))
        
            if(area > 800 and area < 3500): 

                # a blob with the correct size has been found 
                # potential new bee tracking

                cv2.circle(image_small_show, centroid, 10, (255,0,255), -1) # mark the detection with a pink circle

                # if no tracker is near, start a new tracker
                # if a tracking is near, update 
                smallest_distance = 999999999
                nearest_tracker = None
                for t in trackers:
                    distance = np.linalg.norm([t.x - centroid[0], t.y - centroid[1]])
                    if distance < smallest_distance:
                        smallest_distance = distance

                if smallest_distance > 60: # only create a new tracker if the nearest tracker is at least 60 pixels away
                    newtracker = BeeTracker(centroid, (track_window_w, track_window_h), image_hsv)
                    trackers.append(newtracker)

        ############
        ### Round 2 
        n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(imgmask_filled, 4, cv2.CV_8U)
        labeled_image = labels / np.amax(labels)

        for i in range(n_components):
            if(i == 0):
                continue

            area = stats[i, cv2.CC_STAT_AREA]
            centroid = (int(centroids[i][0]), int(centroids[i][1]))
        
            if(area > 500 and area < 3500 and centroid[1] > bound_top and centroid[1] < bound_bottom):

                # a blob with the correct size has been found 
                # potential new bee tracking

                cv2.circle(image_small_show, centroid, 10, (255,0,255), -1) # mark the detection with a pink circle

                # if no tracker is near, start a new tracker
                # if a tracking is near, update its 
                smallest_distance = 999999999
                nearest_tracker = None
                for t in trackers:
                    distance = np.linalg.norm([t.x - centroid[0], t.y - centroid[1]])
                    #print(distance)
                    if distance < smallest_distance:
                        smallest_distance = distance
                        if smallest_distance < 20:
                            nearest_tracker = t

                if(nearest_tracker is not None):
                    nearest_tracker.reposition(centroid, image_hsv); 
                    # snaps tracking window to the nearest tracking window, max 20 pixels aways
                    # that is more reliable than mean-shift tracking over a number of frames
          
            elif(area < 500):
                labeled_image[labels == i] = 0 # remove too small connected components, just for illustration
                too_small_blobs[labels == i] = 0

        cv2.imshow("finalmask", cv2.bitwise_and(too_small_blobs, imgmask_filled))
        

        for t in trackers:
            t.draw(image_small_show, trajectories)
            positive_image_index+=1
            #t.write_positive_image(positive_image_index, image_small)

        # remove colliding or jumping trackers
        trackers = [x for x in trackers if not ( ~x.sanitize(imgmask, image_small_show, trackers))]

        cv2.imshow('frame3', image_small_show)
        cv2.imshow('trajectories', trajectories)
    
        key = cv2.waitKey(1)
        if key & 0xFF == ord('p'): # pause video with p
            video_paused = not video_paused

        print(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()