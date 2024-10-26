from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import argparse

def Hist_and_Backproj(val):
    ## [initialize]
    bins = val
    histSize = max(bins, 2)
    ranges = [0, 180] # hue_range
    ## [initialize]

    ## [Get the Histogram and normalize it]
    hist = ncvslideio.calcHist([hue], [0], None, [histSize], ranges, accumulate=False)
    ncvslideio.normalize(hist, hist, alpha=0, beta=255, norm_type=ncvslideio.NORM_MINMAX)
    ## [Get the Histogram and normalize it]

    ## [Get Backprojection]
    backproj = ncvslideio.calcBackProject([hue], [0], hist, ranges, scale=1)
    ## [Get Backprojection]

    ## [Draw the backproj]
    ncvslideio.imshow('BackProj', backproj)
    ## [Draw the backproj]

    ## [Draw the histogram]
    w = 400
    h = 400
    bin_w = int(round(w / histSize))
    histImg = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(bins):
        ncvslideio.rectangle(histImg, (i*bin_w, h), ( (i+1)*bin_w, h - int(np.round( hist[i]*h/255.0 )) ), (0, 0, 255), ncvslideio.FILLED)

    ncvslideio.imshow('Histogram', histImg)
    ## [Draw the histogram]

## [Read the image]
parser = argparse.ArgumentParser(description='Code for Back Projection tutorial.')
parser.add_argument('--input', help='Path to input image.', default='home.jpg')
args = parser.parse_args()

src = ncvslideio.imread(ncvslideio.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
## [Read the image]

## [Transform it to HSV]
hsv = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2HSV)
## [Transform it to HSV]

## [Use only the Hue value]
ch = (0, 0)
hue = np.empty(hsv.shape, hsv.dtype)
cv.mixChannels([hsv], [hue], ch)
## [Use only the Hue value]

## [Create Trackbar to enter the number of bins]
window_image = 'Source image'
cv.namedWindow(window_image)
bins = 25
cv.createTrackbar('* Hue  bins: ', window_image, bins, 180, Hist_and_Backproj )
Hist_and_Backproj(bins)
## [Create Trackbar to enter the number of bins]

## [Show the image]
cv.imshow(window_image, src)
cv.waitKey()
## [Show the image]
