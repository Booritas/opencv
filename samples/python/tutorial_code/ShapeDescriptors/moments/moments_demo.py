from __future__ import print_function
from __future__ import division
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

    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = ncvslideio.moments(contours[i])

    # Get the mass centers
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

    # Draw contours
    ## [zeroMat]
    drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    ## [zeroMat]
    ## [forContour]
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        ncvslideio.drawContours(drawing, contours, i, color, 2)
        ncvslideio.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
    ## [forContour]

    ## [showDrawings]
    # Show in a window
    ncvslideio.imshow('Contours', drawing)
    ## [showDrawings]

    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], ncvslideio.contourArea(contours[i]), ncvslideio.arcLength(contours[i], True)))

## [setup]
# Load source image
parser = argparse.ArgumentParser(description='Code for Image Moments tutorial.')
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
