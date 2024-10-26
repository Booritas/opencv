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

    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = ncvslideio.minAreaRect(c)
        if c.shape[0] > 5:
            minEllipse[i] = ncvslideio.fitEllipse(c)

    # Draw contours + rotated rects + ellipses
    ## [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]
    ## [forContour]
    for i, c in enumerate(contours):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        # contour
        ncvslideio.drawContours(drawing, contours, i, color)
        # ellipse
        if c.shape[0] > 5:
            ncvslideio.ellipse(drawing, minEllipse[i], color, 2)
        # rotated rectangle
        box = ncvslideio.boxPoints(minRect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        ncvslideio.drawContours(drawing, [box], 0, color)
    ## [forContour]

    ## [showDrawings]
    # Show in a window
    ncvslideio.imshow('Contours', drawing)
    ## [showDrawings]

## [setup]
# Load source image
parser = argparse.ArgumentParser(description='Code for Creating Bounding rotated boxes and ellipses for contours tutorial.')
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
cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, thresh_callback)
thresh_callback(thresh)
## [trackbar]

cv.waitKey()
