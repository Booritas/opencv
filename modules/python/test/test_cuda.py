#!/usr/bin/env python

'''
CUDA-accelerated Computer Vision functions
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os

from tests_common import NewOpenCVTests, unittest

class cuda_test(NewOpenCVTests):
    def setUp(self):
        super(cuda_test, self).setUp()
        if not ncvslideio.cuda.getCudaEnabledDeviceCount():
            self.skipTest("No CUDA-capable device is detected")

    def test_cuda_upload_download(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = ncvslideio.cuda_GpuMat()
        cuMat.upload(npMat)

        self.assertTrue(np.allclose(cuMat.download(), npMat))

    def test_cuda_upload_download_stream(self):
        stream = ncvslideio.cuda_Stream()
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = ncvslideio.cuda_GpuMat(128,128, ncvslideio.CV_8UC3)
        cuMat.upload(npMat, stream)
        npMat2 = cuMat.download(stream=stream)
        stream.waitForCompletion()
        self.assertTrue(np.allclose(npMat2, npMat))

    def test_cuda_interop(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = ncvslideio.cuda_GpuMat()
        cuMat.upload(npMat)
        self.assertTrue(cuMat.cudaPtr() != 0)
        cuMatFromPtrSz = ncvslideio.cuda.createGpuMatFromCudaMemory(cuMat.size(),cuMat.type(),cuMat.cudaPtr(), cuMat.step)
        self.assertTrue(cuMat.cudaPtr() == cuMatFromPtrSz.cudaPtr())
        cuMatFromPtrRc = ncvslideio.cuda.createGpuMatFromCudaMemory(cuMat.size()[1],cuMat.size()[0],cuMat.type(),cuMat.cudaPtr(), cuMat.step)
        self.assertTrue(cuMat.cudaPtr() == cuMatFromPtrRc.cudaPtr())
        stream = ncvslideio.cuda_Stream()
        self.assertTrue(stream.cudaPtr() != 0)
        streamFromPtr = ncvslideio.cuda.wrapStream(stream.cudaPtr())
        self.assertTrue(stream.cudaPtr() == streamFromPtr.cudaPtr())
        asyncstream = ncvslideio.cuda_Stream(1)  # cudaStreamNonBlocking
        self.assertTrue(asyncstream.cudaPtr() != 0)

    def test_cuda_buffer_pool(self):
        ncvslideio.cuda.setBufferPoolUsage(True)
        ncvslideio.cuda.setBufferPoolConfig(ncvslideio.cuda.getDevice(), 1024 * 1024 * 64, 2)
        stream_a = ncvslideio.cuda.Stream()
        pool_a = ncvslideio.cuda.BufferPool(stream_a)
        cuMat = pool_a.getBuffer(1024, 1024, ncvslideio.CV_8UC3)
        ncvslideio.cuda.setBufferPoolUsage(False)
        self.assertEqual(cuMat.size(), (1024, 1024))
        self.assertEqual(cuMat.type(), ncvslideio.CV_8UC3)

    def test_cuda_release(self):
        npMat = (np.random.random((128, 128, 3)) * 255).astype(np.uint8)
        cuMat = ncvslideio.cuda_GpuMat()
        cuMat.upload(npMat)
        cuMat.release()
        self.assertTrue(cuMat.cudaPtr() == 0)
        self.assertTrue(cuMat.step == 0)
        self.assertTrue(cuMat.size() == (0, 0))

    def test_cuda_convertTo(self):
        # setup
        npMat_8UC4 = (np.random.random((128, 128, 4)) * 255).astype(np.uint8)
        npMat_32FC4 = npMat_8UC4.astype(np.single)
        new_type = ncvslideio.CV_32FC4

        # sync
        # in/out
        cuMat_8UC4 = ncvslideio.cuda_GpuMat(npMat_8UC4)
        cuMat_32FC4 = ncvslideio.cuda_GpuMat(cuMat_8UC4.size(), new_type)
        cuMat_32FC4_out = cuMat_8UC4.convertTo(new_type, cuMat_32FC4)
        self.assertTrue(cuMat_32FC4.cudaPtr() == cuMat_32FC4_out.cudaPtr())
        npMat_32FC4_out = cuMat_32FC4.download()
        self.assertTrue(np.array_equal(npMat_32FC4, npMat_32FC4_out))
        # out
        cuMat_32FC4_out = cuMat_8UC4.convertTo(new_type)
        npMat_32FC4_out = cuMat_32FC4.download()
        self.assertTrue(np.array_equal(npMat_32FC4, npMat_32FC4_out))

        # async
        stream = ncvslideio.cuda.Stream()
        cuMat_32FC4 = ncvslideio.cuda_GpuMat(cuMat_8UC4.size(), new_type)
        cuMat_32FC4_out = cuMat_8UC4.convertTo(new_type, cuMat_32FC4)
        # in/out
        cuMat_32FC4_out = cuMat_8UC4.convertTo(new_type, 1, 0, stream, cuMat_32FC4)
        self.assertTrue(cuMat_32FC4.cudaPtr() == cuMat_32FC4_out.cudaPtr())
        npMat_32FC4_out = cuMat_32FC4.download(stream)
        stream.waitForCompletion()
        self.assertTrue(np.array_equal(npMat_32FC4, npMat_32FC4_out))
        # out
        cuMat_32FC4_out = cuMat_8UC4.convertTo(new_type, 1, 0, stream)
        npMat_32FC4_out = cuMat_32FC4.download(stream)
        stream.waitForCompletion()
        self.assertTrue(np.array_equal(npMat_32FC4, npMat_32FC4_out))

    def test_cuda_copyTo(self):
        # setup
        npMat_8UC4 = (np.random.random((128, 128, 4)) * 255).astype(np.uint8)

        # sync
        # in/out
        cuMat_8UC4 = ncvslideio.cuda_GpuMat(npMat_8UC4)
        cuMat_8UC4_dst = ncvslideio.cuda_GpuMat(cuMat_8UC4.size(), cuMat_8UC4.type())
        cuMat_8UC4_out = cuMat_8UC4.copyTo(cuMat_8UC4_dst)
        self.assertTrue(cuMat_8UC4_out.cudaPtr() == cuMat_8UC4_dst.cudaPtr())
        npMat_8UC4_out = cuMat_8UC4_out.download()
        self.assertTrue(np.array_equal(npMat_8UC4, npMat_8UC4_out))
        # out
        cuMat_8UC4_out =  cuMat_8UC4.copyTo()
        npMat_8UC4_out = cuMat_8UC4_out.download()
        self.assertTrue(np.array_equal(npMat_8UC4, npMat_8UC4_out))

        # async
        stream = ncvslideio.cuda.Stream()
        # in/out
        cuMat_8UC4 = ncvslideio.cuda_GpuMat(npMat_8UC4)
        cuMat_8UC4_dst = ncvslideio.cuda_GpuMat(cuMat_8UC4.size(), cuMat_8UC4.type())
        cuMat_8UC4_out = cuMat_8UC4.copyTo(cuMat_8UC4_dst, stream)
        self.assertTrue(cuMat_8UC4_out.cudaPtr() == cuMat_8UC4_out.cudaPtr())
        npMat_8UC4_out = cuMat_8UC4_dst.download(stream)
        stream.waitForCompletion()
        self.assertTrue(np.array_equal(npMat_8UC4, npMat_8UC4_out))
        # out
        cuMat_8UC4_out = cuMat_8UC4.copyTo(stream)
        npMat_8UC4_out = cuMat_8UC4_out.download(stream)
        stream.waitForCompletion()
        self.assertTrue(np.array_equal(npMat_8UC4, npMat_8UC4_out))

    def test_cuda_denoising(self):
        self.assertEqual(True, hasattr(ncvslideio.cuda, 'fastNlMeansDenoising'))
        self.assertEqual(True, hasattr(ncvslideio.cuda, 'fastNlMeansDenoisingColored'))
        self.assertEqual(True, hasattr(ncvslideio.cuda, 'nonLocalMeans'))

if __name__ == '__main__':
    NewOpenCVTests.bootstrap()
