#!/usr/bin/env python

'''
Coherence-enhancing filtering example
=====================================

inspired by
  Joachim Weickert "Coherence-Enhancing Shock Filters"
  http://www.mia.uni-saarland.de/Publications/weickert-dagm03.pdf
'''

# Python 2/3 compatibility
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

import numpy as np
import cv2 as cv

def coherence_filter(img, sigma = 11, str_sigma = 11, blend = 0.5, iter_n = 4):
    h, w = img.shape[:2]

    for i in xrange(iter_n):
        print(i)

        gray = ncvslideio.cvtColor(img, ncvslideio.COLOR_BGR2GRAY)
        eigen = ncvslideio.cornerEigenValsAndVecs(gray, str_sigma, 3)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]

        gxx = ncvslideio.Sobel(gray, ncvslideio.CV_32F, 2, 0, ksize=sigma)
        gxy = ncvslideio.Sobel(gray, ncvslideio.CV_32F, 1, 1, ksize=sigma)
        gyy = ncvslideio.Sobel(gray, ncvslideio.CV_32F, 0, 2, ksize=sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        ero = ncvslideio.erode(img, None)
        dil = ncvslideio.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img*(1.0 - blend) + img1*blend)
    print('done')
    return img


def main():
    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'baboon.jpg'

    src = ncvslideio.imread(ncvslideio.samples.findFile(fn))

    def nothing(*argv):
        pass

    def update():
        sigma = ncvslideio.getTrackbarPos('sigma', 'control')*2+1
        str_sigma = ncvslideio.getTrackbarPos('str_sigma', 'control')*2+1
        blend = ncvslideio.getTrackbarPos('blend', 'control') / 10.0
        print('sigma: %d  str_sigma: %d  blend_coef: %f' % (sigma, str_sigma, blend))
        dst = coherence_filter(src, sigma=sigma, str_sigma = str_sigma, blend = blend)
        ncvslideio.imshow('dst', dst)

    ncvslideio.namedWindow('control', 0)
    ncvslideio.createTrackbar('sigma', 'control', 9, 15, nothing)
    ncvslideio.createTrackbar('blend', 'control', 7, 10, nothing)
    ncvslideio.createTrackbar('str_sigma', 'control', 9, 15, nothing)


    print('Press SPACE to update the image\n')

    ncvslideio.imshow('src', src)
    update()
    while True:
        ch = ncvslideio.waitKey()
        if ch == ord(' '):
            update()
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    ncvslideio.destroyAllWindows()
