#!/usr/bin/python

'''
This example illustrates how to use ncvslideio.HoughCircles() function.

Usage:
    houghcircles.py [<image_name>]
    image argument defaults to board.jpg
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import sys

def main():
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'board.jpg'

    src = ncvslideio.imread(ncvslideio.samples.findFile(fn))
    img = ncvslideio.cvtColor(src, ncvslideio.COLOR_BGR2GRAY)
    img = ncvslideio.medianBlur(img, 5)
    cimg = src.copy() # numpy function

    circles = ncvslideio.HoughCircles(img, ncvslideio.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)

    if circles is not None: # Check if circles have been found and only then iterate over these and add them to the image
        circles = np.uint16(np.around(circles))
        _a, b, _c = circles.shape
        for i in range(b):
            ncvslideio.circle(cimg, (circles[0][i][0], circles[0][i][1]), circles[0][i][2], (0, 0, 255), 3, ncvslideio.LINE_AA)
            ncvslideio.circle(cimg, (circles[0][i][0], circles[0][i][1]), 2, (0, 255, 0), 3, ncvslideio.LINE_AA)  # draw center of circle

        ncvslideio.imshow("detected circles", cimg)

    ncvslideio.imshow("source", src)
    ncvslideio.waitKey(0)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
