import numpy as np
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This sample demonstrates the camshift algorithm. \
                                              The example file can be downloaded from: \
                                              https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4')
parser.add_argument('image', type=str, help='path to image file')
args = parser.parse_args()

cap = ncvslideio.VideoCapture(args.image)

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
x, y, w, h = 300, 200, 100, 50 # simply hardcoded the values
track_window = (x, y, w, h)

# set up the ROI for tracking
roi = frame[y:y+h, x:x+w]
hsv_roi =  ncvslideio.cvtColor(roi, ncvslideio.COLOR_BGR2HSV)
mask = ncvslideio.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = ncvslideio.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv.normalize(roi_hist,roi_hist,0,255,ncvslideio.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( ncvslideio.TERM_CRITERIA_EPS | ncvslideio.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = ncvslideio.cvtColor(frame, ncvslideio.COLOR_BGR2HSV)
        dst = ncvslideio.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply camshift to get the new location
        ret, track_window = ncvslideio.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = ncvslideio.boxPoints(ret)
        pts = np.int0(pts)
        img2 = ncvslideio.polylines(frame,[pts],True, 255,2)
        ncvslideio.imshow('img2',img2)

        k = ncvslideio.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
