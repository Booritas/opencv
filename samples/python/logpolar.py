#!/usr/bin/env python

'''
plots image as logPolar and linearPolar

Usage:
    logpolar.py

Keys:
    ESC    - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

def main():
    import sys
    try:
        fn = sys.argv[1]
    except IndexError:
        fn = 'fruits.jpg'

    img = ncvslideio.imread(ncvslideio.samples.findFile(fn))
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)

    img2 = ncvslideio.logPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, ncvslideio.WARP_FILL_OUTLIERS)
    img3 = ncvslideio.linearPolar(img, (img.shape[0]/2, img.shape[1]/2), 40, ncvslideio.WARP_FILL_OUTLIERS)

    ncvslideio.imshow('before', img)
    ncvslideio.imshow('logpolar', img2)
    ncvslideio.imshow('linearpolar', img3)

    ncvslideio.waitKey(0)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
