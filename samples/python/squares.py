#!/usr/bin/env python

'''
Simple "Square Detector" program.

Loads several images sequentially and tries to find squares in each image.
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def find_squares(img):
    img = ncvslideio.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in ncvslideio.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = ncvslideio.Canny(gray, 0, 50, apertureSize=5)
                bin = ncvslideio.dilate(bin, None)
            else:
                _retval, bin = ncvslideio.threshold(gray, thrs, 255, ncvslideio.THRESH_BINARY)
            contours, _hierarchy = ncvslideio.findContours(bin, ncvslideio.RETR_LIST, ncvslideio.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cnt_len = ncvslideio.arcLength(cnt, True)
                cnt = ncvslideio.approxPolyDP(cnt, 0.02*cnt_len, True)
                if len(cnt) == 4 and ncvslideio.contourArea(cnt) > 1000 and ncvslideio.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                    if max_cos < 0.1:
                        squares.append(cnt)
    return squares

def main():
    from glob import glob
    for fn in glob('../data/pic*.png'):
        img = ncvslideio.imread(fn)
        squares = find_squares(img)
        ncvslideio.drawContours( img, squares, -1, (0, 255, 0), 3 )
        ncvslideio.imshow('squares', img)
        ch = ncvslideio.waitKey()
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
