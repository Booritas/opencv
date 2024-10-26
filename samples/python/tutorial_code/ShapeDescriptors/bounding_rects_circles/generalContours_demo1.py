from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng

rng.seed(12345)

def thresh_callback(val):
    threshold = val

    ## [Canny]
    # Detect edges using Canny
    canny_output = ncvslideio.Canny(src_gray, threshold, threshold * 2)
    ## [Canny]

    ## [findContours]
    # Find contours
    contours, _ = ncvslideio.findContours(canny_output, ncvslideio.RETR_TREE, ncvslideio.CHAIN_APPROX_SIMPLE)
    ## [findContours]

    ## [allthework]
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = ncvslideio.approxPolyDP(c, 3, True)
        boundRect[i] = ncvslideio.boundingRect(contours_poly[i])
        centers[i], radius[i] = ncvslideio.minEnclosingCircle(contours_poly[i])
    ## [allthework]

    ## [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]

    ## [forContour]
    # Draw polygonal contour + bonding rects + circles
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        ncvslideio.drawContours(drawing, contours_poly, i, color)
        ncvslideio.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
        ncvslideio.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    ## [forContour]

    ## [showDrawings]
    # Show in a window
    ncvslideio.imshow('Contours', drawing)
    ## [showDrawings]

## [setup]
# Load source image
parser = argparse.ArgumentParser(description='Code for Creating Bounding boxes and circles for contours tutorial.')
parser.add_argument('--input', help='Path to input image.', default='stuff.jpg')
args = parser.parse_args()

src = ncvslideio.imread(ncvslideio.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)

# Convert image to gray and blur it
src_gray = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)
src_gray = ncvslideio.blur(src_gray, (3,3))
## [setup]

## [createWindow]
# Create Window
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
## [createWindow]
## [trackbar]
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Canny thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
## [trackbar]

cv.waitKey()
