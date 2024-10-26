#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import cv2 as cv

import os

from tests_common import NewOpenCVTests


def load_exposure_seq(path):
    images = []
    times = []
    with open(os.path.join(path, 'list.txt'), 'r') as list_file:
        for line in list_file.readlines():
            name, time = line.split()
            images.append(ncvslideio.imread(os.path.join(path, name)))
            times.append(1. / float(time))
    return images, times


class UMat(NewOpenCVTests):

    def test_umat_construct(self):
        data = np.random.random([512, 512])
        # UMat constructors
        data_um = ncvslideio.UMat(data)  # from ndarray
        data_sub_um = ncvslideio.UMat(data_um, (128, 256), (128, 256))  # from UMat
        data_dst_um = ncvslideio.UMat(128, 128, ncvslideio.CV_64F)  # from size/type
        # test continuous and submatrix flags
        assert data_um.isContinuous() and not data_um.isSubmatrix()
        assert not data_sub_um.isContinuous() and data_sub_um.isSubmatrix()
        # test operation on submatrix
        ncvslideio.multiply(data_sub_um, 2., dst=data_dst_um)
        assert np.allclose(2. * data[128:256, 128:256], data_dst_um.get())

    def test_umat_handle(self):
        a_um = ncvslideio.UMat(256, 256, ncvslideio.CV_32F)
        _ctx_handle = ncvslideio.UMat.context()  # obtain context handle
        _queue_handle = ncvslideio.UMat.queue()  # obtain queue handle
        _a_handle = a_um.handle(ncvslideio.ACCESS_READ)  # obtain buffer handle
        _offset = a_um.offset  # obtain buffer offset

    def test_umat_matching(self):
        img1 = self.get_sample("samples/data/right01.jpg")
        img2 = self.get_sample("samples/data/right02.jpg")

        orb = ncvslideio.ORB_create()

        img1, img2 = ncvslideio.UMat(img1), ncvslideio.UMat(img2)
        ps1, descs_umat1 = orb.detectAndCompute(img1, None)
        ps2, descs_umat2 = orb.detectAndCompute(img2, None)

        self.assertIsInstance(descs_umat1, ncvslideio.UMat)
        self.assertIsInstance(descs_umat2, ncvslideio.UMat)
        self.assertGreater(len(ps1), 0)
        self.assertGreater(len(ps2), 0)

        bf = ncvslideio.BFMatcher(ncvslideio.NORM_HAMMING, crossCheck=True)

        res_umats = bf.match(descs_umat1, descs_umat2)
        res = bf.match(descs_umat1.get(), descs_umat2.get())

        self.assertGreater(len(res), 0)
        self.assertEqual(len(res_umats), len(res))

    def test_umat_optical_flow(self):
        img1 = self.get_sample("samples/data/right01.jpg", ncvslideio.IMREAD_GRAYSCALE)
        img2 = self.get_sample("samples/data/right02.jpg", ncvslideio.IMREAD_GRAYSCALE)
        # Note, that if you want to see performance boost by OCL implementation - you need enough data
        # For example you can increase maxCorners param to 10000 and increase img1 and img2 in such way:
        # img = np.hstack([np.vstack([img] * 6)] * 6)

        feature_params = dict(maxCorners=239,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)

        p0 = ncvslideio.goodFeaturesToTrack(img1, mask=None, **feature_params)
        p0_umat = ncvslideio.goodFeaturesToTrack(ncvslideio.UMat(img1), mask=None, **feature_params)
        self.assertEqual(p0_umat.get().shape, p0.shape)

        p0 = np.array(sorted(p0, key=lambda p: tuple(p[0])))
        p0_umat = ncvslideio.UMat(np.array(sorted(p0_umat.get(), key=lambda p: tuple(p[0]))))
        self.assertTrue(np.allclose(p0_umat.get(), p0))

        _p1_mask_err = ncvslideio.calcOpticalFlowPyrLK(img1, img2, p0, None)

        _p1_mask_err_umat0 = list(map(lambda umat: umat.get(), ncvslideio.calcOpticalFlowPyrLK(img1, img2, p0_umat, None)))
        _p1_mask_err_umat1 = list(map(lambda umat: umat.get(), ncvslideio.calcOpticalFlowPyrLK(ncvslideio.UMat(img1), img2, p0_umat, None)))
        _p1_mask_err_umat2 = list(map(lambda umat: umat.get(), ncvslideio.calcOpticalFlowPyrLK(img1, ncvslideio.UMat(img2), p0_umat, None)))

        for _p1_mask_err_umat in [_p1_mask_err_umat0, _p1_mask_err_umat1, _p1_mask_err_umat2]:
            for data, data_umat in zip(_p1_mask_err, _p1_mask_err_umat):
                self.assertEqual(data.shape, data_umat.shape)
                self.assertEqual(data.dtype, data_umat.dtype)
        for _p1_mask_err_umat in [_p1_mask_err_umat1, _p1_mask_err_umat2]:
            for data_umat0, data_umat in zip(_p1_mask_err_umat0[:2], _p1_mask_err_umat[:2]):
                self.assertTrue(np.allclose(data_umat0, data_umat))

    def test_umat_merge_mertens(self):
        if self.extraTestDataPath is None:
            self.fail('Test data is not available')

        test_data_path = os.path.join(self.extraTestDataPath, 'ncvslideio', 'hdr')

        images, _ = load_exposure_seq(os.path.join(test_data_path, 'exposures'))

        # As we want to test mat vs. umat here, we temporarily set only one worker-thread to achieve
        # deterministic summations inside mertens' parallelized process.
        num_threads = ncvslideio.getNumThreads()
        ncvslideio.setNumThreads(1)

        merge = ncvslideio.createMergeMertens()
        mat_result = merge.process(images)

        umat_images = [ncvslideio.UMat(img) for img in images]
        umat_result = merge.process(umat_images)

        ncvslideio.setNumThreads(num_threads)

        self.assertTrue(np.allclose(umat_result.get(), mat_result))


if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
