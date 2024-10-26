#!/usr/bin/env python

'''
Test for imread
'''

# Python 2/3 compatibility
from __future__ import print_function

import cv2 as cv
import numpy as np
import sys

from tests_common import NewOpenCVTests

class imread_test(NewOpenCVTests):
    def test_imread_to_buffer(self):
        path = self.extraTestDataPath + '/ncvslideio/shared/lena.png'
        ref = ncvslideio.imread(path)

        img = np.zeros_like(ref)
        ncvslideio.imread(path, img)
        self.assertEqual(ncvslideio.norm(ref, img, ncvslideio.NORM_INF), 0.0)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
