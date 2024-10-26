#!/usr/bin/env python

'''
MSER detector demo
==================

Usage:
------
    mser.py [<video source>]

Keys:
-----
    ESC   - exit

'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

import video
import sys

def main():
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    cam = video.create_capture(video_src)
    mser = ncvslideio.MSER_create()

    while True:
        ret, img = cam.read()
        if ret == 0:
            break
        gray = ncvslideio.cvtColor(img, ncvslideio.COLOR_BGR2GRAY)
        vis = img.copy()

        regions, _ = mser.detectRegions(gray)
        hulls = [ncvslideio.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        ncvslideio.polylines(vis, hulls, 1, (0, 255, 0))

        ncvslideio.imshow('img', vis)
        if ncvslideio.waitKey(5) == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
