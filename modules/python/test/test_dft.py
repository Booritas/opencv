#!/usr/bin/env python

'''
Test for disctrete fourier transform (dft)
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests

class dft_test(NewOpenCVTests):
    def test_dft(self):

        img = self.get_sample('samples/data/rubberwhale1.png', 0)
        eps = 0.001

        #test direct transform
        refDft = np.fft.fft2(img)
        refDftShift = np.fft.fftshift(refDft)
        refMagnitide = np.log(1.0 + np.abs(refDftShift))

        testDft = ncvslideio.dft(np.float32(img),flags = ncvslideio.DFT_COMPLEX_OUTPUT)
        testDftShift = np.fft.fftshift(testDft)
        testMagnitude = np.log(1.0 + ncvslideio.magnitude(testDftShift[:,:,0], testDftShift[:,:,1]))

        refMagnitide = ncvslideio.normalize(refMagnitide, 0.0, 1.0, ncvslideio.NORM_MINMAX)
        testMagnitude = ncvslideio.normalize(testMagnitude, 0.0, 1.0, ncvslideio.NORM_MINMAX)

        self.assertLess(ncvslideio.norm(refMagnitide - testMagnitude), eps)

        #test inverse transform
        img_back = np.fft.ifft2(refDft)
        img_back = np.abs(img_back)

        img_backTest = ncvslideio.idft(testDft)
        img_backTest = ncvslideio.magnitude(img_backTest[:,:,0], img_backTest[:,:,1])

        img_backTest = ncvslideio.normalize(img_backTest, 0.0, 1.0, ncvslideio.NORM_MINMAX)
        img_back = ncvslideio.normalize(img_back, 0.0, 1.0, ncvslideio.NORM_MINMAX)

        self.assertLess(ncvslideio.norm(img_back - img_backTest), eps)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
