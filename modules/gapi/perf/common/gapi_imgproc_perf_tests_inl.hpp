// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP


#include "gapi_imgproc_perf_tests.hpp"

#include "../../test/common/gapi_imgproc_tests_common.hpp"

namespace opencv_test
{

  using namespace perf;

  namespace
  {
      void rgb2yuyv(const uchar* rgb_line, uchar* yuv422_line, int width)
      {
          CV_Assert(width % 2 == 0);
          for (int i = 0; i < width; i += 2)
          {
              uchar r = rgb_line[i * 3    ];
              uchar g = rgb_line[i * 3 + 1];
              uchar b = rgb_line[i * 3 + 2];

              yuv422_line[i * 2    ] = ncvslideio::saturate_cast<uchar>(-0.14713 * r - 0.28886 * g + 0.436   * b + 128.f);  // U0
              yuv422_line[i * 2 + 1] = ncvslideio::saturate_cast<uchar>( 0.299   * r + 0.587   * g + 0.114   * b        );  // Y0
              yuv422_line[i * 2 + 2] = ncvslideio::saturate_cast<uchar>(0.615    * r - 0.51499 * g - 0.10001 * b + 128.f);  // V0

              r = rgb_line[i * 3 + 3];
              g = rgb_line[i * 3 + 4];
              b = rgb_line[i * 3 + 5];

              yuv422_line[i * 2 + 3] = ncvslideio::saturate_cast<uchar>(0.299 * r + 0.587   * g + 0.114   * b);   // Y1
          }
      }

      void convertRGB2YUV422Ref(const ncvslideio::Mat& in, ncvslideio::Mat &out)
      {
          out.create(in.size(), CV_8UC2);

          for (int i = 0; i < in.rows; ++i)
          {
              const uchar* in_line_p  = in.ptr<uchar>(i);
              uchar* out_line_p = out.ptr<uchar>(i);
              rgb2yuyv(in_line_p, out_line_p, in.cols);
          }
      }
  }
//------------------------------------------------------------------------------

PERF_TEST_P_(SepFilterPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, compile_args) = GetParam();

    ncvslideio::Mat kernelX(kernSize, 1, CV_32F);
    ncvslideio::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);
    initMatrixRandN(type, sz, dtype, false);

    ncvslideio::Point anchor = ncvslideio::Point(-1, -1);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::sepFilter2D(in_mat1, out_mat_ocv, dtype, kernelX, kernelY );
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::sepFilter(in, dtype, kernelX, kernelY, anchor, ncvslideio::Scalar() );
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
      c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Filter2DPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, borderType = 0, dtype = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, borderType, dtype, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    ncvslideio::Point anchor = {-1, -1};
    double delta = 0;

    ncvslideio::Mat kernel = ncvslideio::Mat(kernSize, kernSize, CV_32FC1 );
    ncvslideio::Scalar kernMean = ncvslideio::Scalar::all(1.0);
    ncvslideio::Scalar kernStddev = ncvslideio::Scalar::all(2.0/3);
    randn(kernel, kernMean, kernStddev);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::filter2D(in_mat1, out_mat_ocv, dtype, kernel, anchor, delta, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::filter2D(in, dtype, kernel, anchor, delta, borderType);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }


    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BoxFilterPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int filterSize = 0, borderType = 0, dtype = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, filterSize, sz, borderType, dtype, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    ncvslideio::Point anchor = {-1, -1};
    bool normalize = true;

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::boxFilter(in_mat1, out_mat_ocv, dtype, ncvslideio::Size(filterSize, filterSize), anchor, normalize, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::boxFilter(in, dtype, ncvslideio::Size(filterSize, filterSize), anchor, normalize, borderType);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BlurPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int filterSize = 0, borderType = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, filterSize, sz, borderType, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    ncvslideio::Point anchor = {-1, -1};

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::blur(in_mat1, out_mat_ocv, ncvslideio::Size(filterSize, filterSize), anchor, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::blur(in, ncvslideio::Size(filterSize, filterSize), anchor, borderType);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(GaussianBlurPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, compile_args) = GetParam();

    ncvslideio::Size kSize = ncvslideio::Size(kernSize, kernSize);
    auto& rng = ncvslideio::theRNG();
    double sigmaX = rng();
    initMatrixRandN(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::GaussianBlur(in_mat1, out_mat_ocv, kSize, sigmaX);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::gaussianBlur(in, kSize, sigmaX);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }


    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MedianBlurPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::medianBlur(in_mat1, out_mat_ocv, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::medianBlur(in, kernSize);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(ErodePerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, kernType,  compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(kernType, ncvslideio::Size(kernSize, kernSize));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::erode(in_mat1, out_mat_ocv, kernel);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::erode(in, kernel);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(Erode3x3PerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int numIters = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, numIters, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(ncvslideio::MorphShapes::MORPH_RECT, ncvslideio::Size(3, 3));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::erode(in_mat1, out_mat_ocv, kernel, ncvslideio::Point(-1, -1), numIters);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::erode3x3(in, numIters);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(DilatePerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, kernType = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, kernType, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(kernType, ncvslideio::Size(kernSize, kernSize));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::dilate(in_mat1, out_mat_ocv, kernel);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::dilate(in, kernel);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(Dilate3x3PerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int numIters = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, numIters, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(ncvslideio::MorphShapes::MORPH_RECT, ncvslideio::Size(3, 3));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::dilate(in_mat1, out_mat_ocv, kernel, ncvslideio::Point(-1,-1), numIters);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::dilate3x3(in, numIters);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(MorphologyExPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    ncvslideio::MorphTypes op = ncvslideio::MORPH_ERODE;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, op, compile_args) = GetParam();

    initMatrixRandN(type, sz, type, false);

    ncvslideio::MorphShapes defShape = ncvslideio::MORPH_RECT;
    int defKernSize = 3;
    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(defShape, ncvslideio::Size(defKernSize, defKernSize));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::morphologyEx(in_mat1, out_mat_ocv, op, kernel);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::morphologyEx(in, op, kernel);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SobelPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0, dx = 0, dy = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, dx, dy, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Sobel(in_mat1, out_mat_ocv, dtype, dx, dy, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::Sobel(in, dtype, dx, dy, kernSize);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SobelXYPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0, order = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, order, compile_args) = GetParam();

    ncvslideio::Mat out_mat_ocv2;
    ncvslideio::Mat out_mat_gapi2;

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Sobel(in_mat1, out_mat_ocv, dtype, order, 0, kernSize);
        ncvslideio::Sobel(in_mat1, out_mat_ocv2, dtype, 0, order, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::SobelXY(in, dtype, order, kernSize);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));

    // Warm-up graph engine:
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi, out_mat_gapi2), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi, out_mat_gapi2));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat_gapi2, out_mat_ocv2));
        EXPECT_EQ(out_mat_gapi.size(), sz);
        EXPECT_EQ(out_mat_gapi2.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(LaplacianPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int kernSize = 0, dtype = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, kernSize, sz, dtype, compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Laplacian(in_mat1, out_mat_ocv, dtype, kernSize);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::Laplacian(in, dtype, kernSize);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BilateralFilterPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = 0;
    int dtype = 0, d = 0, borderType = BORDER_DEFAULT;
    double sigmaColor = 0, sigmaSpace = 0;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, dtype, sz, d, sigmaColor, sigmaSpace,
             compile_args) = GetParam();

    initMatrixRandN(type, sz, dtype, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::bilateralFilter(in_mat1, out_mat_ocv, d, sigmaColor, sigmaSpace, borderType);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::bilateralFilter(in, d, sigmaColor, sigmaSpace, borderType);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------


PERF_TEST_P_(CannyPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type;
    int apSize = 0;
    double thrLow = 0.0, thrUp = 0.0;
    ncvslideio::Size sz;
    bool l2gr = false;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, thrLow, thrUp, apSize, l2gr, compile_args) = GetParam();

    initMatrixRandN(type, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Canny(in_mat1, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(GoodFeaturesPerfTest, TestPerformance)
{
    double k = 0.04;

    compare_vector_f<ncvslideio::Point2f> cmpF;
    std::string fileName = "";
    int type = -1, maxCorners = -1, blockSize = -1;
    double qualityLevel = 0.0, minDistance = 0.0;
    bool useHarrisDetector = false;
    ncvslideio::GCompileArgs compileArgs;
    std::tie(cmpF, fileName, type, maxCorners, qualityLevel,
             minDistance, blockSize, useHarrisDetector, compileArgs) = GetParam();

    initMatFromImage(type, fileName);
    std::vector<ncvslideio::Point2f> outVecOCV, outVecGAPI;

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::goodFeaturesToTrack(in_mat1, outVecOCV, maxCorners, qualityLevel, minDistance,
                                ncvslideio::noArray(), blockSize, useHarrisDetector, k);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::goodFeaturesToTrack(in, maxCorners, qualityLevel, minDistance, ncvslideio::Mat(),
                                             blockSize, useHarrisDetector, k);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    // Warm-up graph engine:
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(outVecGAPI), std::move(compileArgs));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(outVecGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(outVecGAPI, outVecOCV));
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(FindContoursPerfTest, TestPerformance)
{
    CompareMats cmpF;
    MatType type;
    ncvslideio::Size sz;
    ncvslideio::RetrievalModes mode;
    ncvslideio::ContourApproximationModes method;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, mode, method, compile_args) = GetParam();

    ncvslideio::Mat in;
    initMatForFindingContours(in, sz, type);
    ncvslideio::Point offset = ncvslideio::Point();
    std::vector<ncvslideio::Vec4i> out_hier_gapi = std::vector<ncvslideio::Vec4i>();

    std::vector<std::vector<ncvslideio::Point>> out_cnts_gapi;
    ncvslideio::GComputation c(findContoursTestGAPI(in, mode, method, std::move(compile_args),
                                            out_cnts_gapi, out_hier_gapi, offset));

    TEST_CYCLE()
    {
        c.apply(gin(in, offset), gout(out_cnts_gapi));
    }

    findContoursTestOpenCVCompare(in, mode, method, out_cnts_gapi, out_hier_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FindContoursHPerfTest, TestPerformance)
{
    CompareMats cmpF;
    MatType type;
    ncvslideio::Size sz;
    ncvslideio::RetrievalModes mode;
    ncvslideio::ContourApproximationModes method;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, mode, method, compile_args) = GetParam();

    ncvslideio::Mat in;
    initMatForFindingContours(in, sz, type);
    ncvslideio::Point offset = ncvslideio::Point();

    std::vector<std::vector<ncvslideio::Point>> out_cnts_gapi;
    std::vector<ncvslideio::Vec4i>              out_hier_gapi;
    ncvslideio::GComputation c(findContoursTestGAPI<HIERARCHY>(in, mode, method, std::move(compile_args),
                                                       out_cnts_gapi, out_hier_gapi, offset));

    TEST_CYCLE()
    {
        c.apply(gin(in, offset), gout(out_cnts_gapi, out_hier_gapi));
    }

    findContoursTestOpenCVCompare<HIERARCHY>(in, mode, method, out_cnts_gapi, out_hier_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BoundingRectMatPerfTest, TestPerformance)
{
    CompareRects cmpF;
    ncvslideio::Size sz;
    MatType type;
    bool initByVector = false;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, initByVector, compile_args) = GetParam();

    if (initByVector)
    {
        initMatByPointsVectorRandU<ncvslideio::Point_>(type, sz, -1);
    }
    else
    {
        initMatrixRandU(type, sz, -1, false);
    }

    ncvslideio::Rect out_rect_gapi;
    ncvslideio::GComputation c(boundingRectTestGAPI(in_mat1, std::move(compile_args), out_rect_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_rect_gapi));
    }

    boundingRectTestOpenCVCompare(in_mat1, out_rect_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BoundingRectVector32SPerfTest, TestPerformance)
{
    CompareRects cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    std::vector<ncvslideio::Point2i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Rect out_rect_gapi;
    ncvslideio::GComputation c(boundingRectTestGAPI(in_vector, std::move(compile_args), out_rect_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_rect_gapi));
    }

    boundingRectTestOpenCVCompare(in_vector, out_rect_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BoundingRectVector32FPerfTest, TestPerformance)
{
    CompareRects cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    std::vector<ncvslideio::Point2f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Rect out_rect_gapi;
    ncvslideio::GComputation c(boundingRectTestGAPI(in_vector, std::move(compile_args), out_rect_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_rect_gapi));
    }

    boundingRectTestOpenCVCompare(in_vector, out_rect_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(FitLine2DMatVectorPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    ncvslideio::Size sz;
    MatType type;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, distType, compile_args) = GetParam();

    initMatByPointsVectorRandU<ncvslideio::Point_>(type, sz, -1);

    ncvslideio::Vec4f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_mat1, distType, std::move(compile_args), out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_mat1, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine2DVector32SPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    ncvslideio::Size sz;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<ncvslideio::Point2i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Vec4f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine2DVector32FPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    ncvslideio::Size sz;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<ncvslideio::Point2f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Vec4f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine2DVector64FPerfTest, TestPerformance)
{
    CompareVecs<float, 4> cmpF;
    ncvslideio::Size sz;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<ncvslideio::Point2d> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Vec4f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DMatVectorPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    ncvslideio::Size sz;
    MatType type;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, distType, compile_args) = GetParam();

    initMatByPointsVectorRandU<ncvslideio::Point3_>(type, sz, -1);

    ncvslideio::Vec6f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_mat1, distType, std::move(compile_args), out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_mat1, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DVector32SPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    ncvslideio::Size sz;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<ncvslideio::Point3i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Vec6f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DVector32FPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    ncvslideio::Size sz;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<ncvslideio::Point3f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Vec6f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(FitLine3DVector64FPerfTest, TestPerformance)
{
    CompareVecs<float, 6> cmpF;
    ncvslideio::Size sz;
    ncvslideio::DistanceTypes distType;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, distType, compile_args) = GetParam();

    std::vector<ncvslideio::Point3d> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    ncvslideio::Vec6f out_vec_gapi;
    ncvslideio::GComputation c(fitLineTestGAPI(in_vector, distType, std::move(compile_args),
                                       out_vec_gapi));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(out_vec_gapi));
    }

    fitLineTestOpenCVCompare(in_vector, distType, out_vec_gapi, cmpF);
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(EqHistPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC1, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::equalizeHist(in_mat1, out_mat_ocv);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::equalizeHist(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2RGBPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2RGB);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2RGB(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2GrayPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2GRAY);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2Gray(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2GrayPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2GRAY);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2Gray(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2YUVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2YUV);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2YUV(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(YUV2RGBPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2RGB);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::YUV2RGB(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2I420PerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2YUV_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2I420(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 3 / 2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2I420PerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC3, sz, CV_8UC1, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2YUV_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2I420(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 3 / 2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(I4202BGRPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC1, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2BGR_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::I4202BGR(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 2 / 3));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(I4202RGBPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandN(CV_8UC1, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2RGB_I420);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::I4202RGB(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), Size(sz.width, sz.height * 2 / 3));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(RGB2LabPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2Lab);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2Lab(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2LUVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2Luv);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2LUV(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(LUV2BGRPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_Luv2BGR);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::LUV2BGR(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }

    SANITY_CHECK_NOTHING();

}

//------------------------------------------------------------------------------

PERF_TEST_P_(BGR2YUVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2YUV);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2YUV(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(YUV2BGRPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);

    ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2BGR);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::YUV2BGR(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BayerGR2RGBPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC1, sz, CV_8UC3, false);

    ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BayerGR2RGB);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BayerGR2RGB(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RGB2HSVPerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC3, false);
    ncvslideio::cvtColor(in_mat1, in_mat1, ncvslideio::COLOR_BGR2RGB);

    ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2HSV);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2HSV(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(RGB2YUV422PerfTest, TestPerformance)
{
    compare_f cmpF = get<0>(GetParam());
    Size sz = get<1>(GetParam());
    ncvslideio::GCompileArgs compile_args = get<2>(GetParam());

    initMatrixRandN(CV_8UC3, sz, CV_8UC2, false);
    ncvslideio::cvtColor(in_mat1, in_mat1, ncvslideio::COLOR_BGR2RGB);

    convertRGB2YUV422Ref(in_mat1, out_mat_ocv);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2YUV422(in);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    c.apply(in_mat1, out_mat_gapi, std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(in_mat1, out_mat_gapi);
    }

    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(out_mat_gapi.size(), sz);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ResizePerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = -1;
    int interp = 1;
    ncvslideio::Size sz;
    ncvslideio::Size sz_out;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, interp, sz, sz_out, compile_args) = GetParam();

    in_mat1 = ncvslideio::Mat(sz, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);
    ncvslideio::randn(in_mat1, mean, stddev);
    out_mat_gapi = ncvslideio::Mat(sz_out, type);
    out_mat_ocv = ncvslideio::Mat(sz_out, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::resize(in_mat1, out_mat_ocv, sz_out, 0.0, 0.0, interp);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::resize(in, sz_out, 0.0, 0.0, interp);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ResizeFxFyPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = -1;
    int interp = 1;
    ncvslideio::Size sz;
    double fx = 1.0;
    double fy = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, interp, sz, fx, fy, compile_args) = GetParam();

    in_mat1 = ncvslideio::Mat(sz, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);
    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::Size sz_out = ncvslideio:: Size(saturate_cast<int>(sz.width*fx), saturate_cast<int>(sz.height*fy));
    out_mat_gapi = ncvslideio::Mat(sz_out, type);
    out_mat_ocv = ncvslideio::Mat(sz_out, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::resize(in_mat1, out_mat_ocv, sz_out, fx, fy, interp);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::resize(in, sz_out, fx, fy, interp);
    ncvslideio::GComputation c(in, out);

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ResizeInSimpleGraphPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type = -1;
    ncvslideio::Size sz;
    double fx = 0.5;
    double fy = 0.5;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type, sz, fx, fy, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    ncvslideio::Mat add_res_ocv;

    ncvslideio::add(in_mat1, in_mat2, add_res_ocv);
    ncvslideio::resize(add_res_ocv, out_mat_ocv, ncvslideio::Size(), fx, fy);

    ncvslideio::GMat in1, in2;
    ncvslideio::GMat add_res_gapi = ncvslideio::gapi::add(in1, in2);
    ncvslideio::GMat out = ncvslideio::gapi::resize(add_res_gapi, ncvslideio::Size(), fx, fy, INTER_LINEAR);
    ncvslideio::GComputation ac(GIn(in1, in2), GOut(out));

    auto cc = ac.compile(descr_of(gin(in_mat1, in_mat2)),
                         std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

// This test cases were created to control performance result of test scenario mentioned here:
// https://stackoverflow.com/questions/60629331/opencv-gapi-performance-not-good-as-expected

PERF_TEST_P_(BottleneckKernelsConstInputPerfTest, TestPerformance)
{
    compare_f cmpF;
    std::string fileName = "";
    ncvslideio::GCompileArgs compile_args;
    double fx = 0.5;
    double fy = 0.5;
    std::tie(cmpF, fileName, compile_args) = GetParam();

    in_mat1 = ncvslideio::imread(findDataFile(fileName));

    ncvslideio::Mat cvvga;
    ncvslideio::Mat cvgray;
    ncvslideio::Mat cvblurred;

    ncvslideio::resize(in_mat1, cvvga, ncvslideio::Size(), fx, fy);
    ncvslideio::cvtColor(cvvga, cvgray, ncvslideio::COLOR_BGR2GRAY);
    ncvslideio::blur(cvgray, cvblurred, ncvslideio::Size(3, 3));
    ncvslideio::Canny(cvblurred, out_mat_ocv, 32, 128, 3);

    ncvslideio::GMat in;
    ncvslideio::GMat vga = ncvslideio::gapi::resize(in, ncvslideio::Size(), fx, fy, INTER_LINEAR);
    ncvslideio::GMat gray = ncvslideio::gapi::BGR2Gray(vga);
    ncvslideio::GMat blurred = ncvslideio::gapi::blur(gray, ncvslideio::Size(3, 3));
    ncvslideio::GMat out = ncvslideio::gapi::Canny(blurred, 32, 128, 3);
    ncvslideio::GComputation ac(in, out);

    auto cc = ac.compile(descr_of(gin(in_mat1)),
        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

}
#endif //OPENCV_GAPI_IMGPROC_PERF_TESTS_INL_HPP
