#!/usr/bin/env python

'''
    This program demonstrates Laplace point/edge detection using
    OpenCV function Laplacian()
    It captures from the camera of your choice: 0, 1, ... default 0
    Usage:
        python laplace.py <ddepth> <smoothType> <sigma>
        If no arguments given default arguments will be used.

    Keyboard Shortcuts:
    Press space bar to exit the program.
    '''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import sys

def main():
    # Declare the variables we are going to use
    ddepth = ncvslideio.CV_16S
    smoothType = "MedianBlur"
    sigma = 3
    if len(sys.argv)==4:
        ddepth = sys.argv[1]
        smoothType = sys.argv[2]
        sigma = sys.argv[3]
    # Taking input from the camera
    cap=ncvslideio.VideoCapture(0)
    # Create Window and Trackbar
    ncvslideio.namedWindow("Laplace of Image", ncvslideio.WINDOW_AUTOSIZE)
    ncvslideio.createTrackbar("Kernel Size Bar", "Laplace of Image", sigma, 15, lambda x:x)
    # Printing frame width, height and FPS
    print("=="*40)
    print("Frame Width: ", cap.get(ncvslideio.CAP_PROP_FRAME_WIDTH), "Frame Height: ", cap.get(ncvslideio.CAP_PROP_FRAME_HEIGHT), "FPS: ", cap.get(ncvslideio.CAP_PROP_FPS))
    while True:
        # Reading input from the camera
        ret, frame = cap.read()
        if ret == False:
            print("Can't open camera/video stream")
            break
        # Taking input/position from the trackbar
        sigma = ncvslideio.getTrackbarPos("Kernel Size Bar", "Laplace of Image")
        # Setting kernel size
        ksize = (sigma*5)|1
        # Removing noise by blurring with a filter
        if smoothType == "GAUSSIAN":
            smoothed = ncvslideio.GaussianBlur(frame, (ksize, ksize), sigma, sigma)
        if smoothType == "BLUR":
            smoothed = ncvslideio.blur(frame, (ksize, ksize))
        if smoothType == "MedianBlur":
            smoothed = ncvslideio.medianBlur(frame, ksize)

        # Apply Laplace function
        laplace = ncvslideio.Laplacian(smoothed, ddepth, 5)
        # Converting back to uint8
        result = ncvslideio.convertScaleAbs(laplace, (sigma+1)*0.25)
        # Display Output
        ncvslideio.imshow("Laplace of Image", result)
        k = ncvslideio.waitKey(30)
        if k == 27:
            return
if __name__ == "__main__":
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
