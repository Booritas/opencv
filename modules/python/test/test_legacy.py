#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

from tests_common import NewOpenCVTests

class Hackathon244Tests(NewOpenCVTests):

    def test_int_array(self):
        a = np.array([-1, 2, -3, 4, -5])
        absa0 = np.abs(a)
        self.assertTrue(ncvslideio.norm(a, ncvslideio.NORM_L1) == 15)
        absa1 = ncvslideio.absdiff(a, 0)
        self.assertEqual(ncvslideio.norm(absa1, absa0, ncvslideio.NORM_INF), 0)

    def test_imencode(self):
        a = np.zeros((480, 640), dtype=np.uint8)
        flag, ajpg = ncvslideio.imencode("img_q90.jpg", a, [ncvslideio.IMWRITE_JPEG_QUALITY, 90])
        self.assertEqual(flag, True)
        self.assertEqual(ajpg.dtype, np.uint8)
        self.assertTrue(isinstance(ajpg, np.ndarray), "imencode returned buffer of wrong type: {}".format(type(ajpg)))
        self.assertEqual(len(ajpg.shape), 1, "imencode returned buffer with wrong shape: {}".format(ajpg.shape))
        self.assertGreaterEqual(len(ajpg), 1, "imencode length of the returned buffer should be at least 1")
        self.assertLessEqual(
            len(ajpg), a.size,
            "imencode length of the returned buffer shouldn't exceed number of elements in original image"
        )

    def test_projectPoints(self):
        objpt = np.float64([[1,2,3]])
        imgpt0, jac0 = ncvslideio.projectPoints(objpt, np.zeros(3), np.zeros(3), np.eye(3), np.float64([]))
        imgpt1, jac1 = ncvslideio.projectPoints(objpt, np.zeros(3), np.zeros(3), np.eye(3), None)
        self.assertEqual(imgpt0.shape, (objpt.shape[0], 1, 2))
        self.assertEqual(imgpt1.shape, imgpt0.shape)
        self.assertEqual(jac0.shape, jac1.shape)
        self.assertEqual(jac0.shape[0], 2*objpt.shape[0])

    def test_estimateAffine3D(self):
        pattern_size = (11, 8)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= 10
        (retval, out, inliers) = ncvslideio.estimateAffine3D(pattern_points, pattern_points)
        self.assertEqual(retval, 1)
        if ncvslideio.norm(out[2,:]) < 1e-3:
            out[2,2]=1
        self.assertLess(ncvslideio.norm(out, np.float64([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])), 1e-3)
        self.assertEqual(ncvslideio.countNonZero(inliers), pattern_size[0]*pattern_size[1])

    def test_fast(self):
        fd = ncvslideio.FastFeatureDetector_create(30, True)
        img = self.get_sample("samples/data/right02.jpg", 0)
        img = ncvslideio.medianBlur(img, 3)
        keypoints = fd.detect(img)
        self.assertTrue(600 <= len(keypoints) <= 700)
        for kpt in keypoints:
            self.assertNotEqual(kpt.response, 0)

    def check_close_angles(self, a, b, angle_delta):
        self.assertTrue(abs(a - b) <= angle_delta or
                        abs(360 - abs(a - b)) <= angle_delta)

    def check_close_pairs(self, a, b, delta):
        self.assertLessEqual(abs(a[0] - b[0]), delta)
        self.assertLessEqual(abs(a[1] - b[1]), delta)

    def check_close_boxes(self, a, b, delta, angle_delta):
        self.check_close_pairs(a[0], b[0], delta)
        self.check_close_pairs(a[1], b[1], delta)
        self.check_close_angles(a[2], b[2], angle_delta)

    def test_geometry(self):
        npt = 100
        np.random.seed(244)
        a = np.random.randn(npt,2).astype('float32')*50 + 150

        be = ncvslideio.fitEllipse(a)
        br = ncvslideio.minAreaRect(a)
        mc, mr = ncvslideio.minEnclosingCircle(a)

        be0 = ((150.2511749267578, 150.77322387695312), (158.024658203125, 197.57696533203125), 37.57804489135742)
        br0 = ((161.2974090576172, 154.41793823242188), (207.7177734375, 199.2301483154297), 80.83544921875)
        mc0, mr0 = (160.41790771484375, 144.55152893066406), 136.713500977

        self.check_close_boxes(be, be0, 5, 15)
        self.check_close_boxes(br, br0, 5, 15)
        self.check_close_pairs(mc, mc0, 5)
        self.assertLessEqual(abs(mr - mr0), 5)


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
