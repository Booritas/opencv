// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_INL_HPP

#include <opencv2/gapi/imgproc.hpp>
#include "gapi_imgproc_tests.hpp"

#include "gapi_imgproc_tests_common.hpp"

namespace opencv_test
{

// FIXME avoid this code duplicate in perf tests
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
            yuv422_line[i * 2 + 2] = ncvslideio::saturate_cast<uchar>( 0.615   * r - 0.51499 * g - 0.10001 * b + 128.f);  // V0

            r = rgb_line[i * 3 + 3];
            g = rgb_line[i * 3 + 4];
            b = rgb_line[i * 3 + 5];

            yuv422_line[i * 2 + 3] = ncvslideio::saturate_cast<uchar>(0.299 * r + 0.587 * g + 0.114 * b);   // Y1
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

TEST_P(Filter2DTest, AccuracyTest)
{
    ncvslideio::Point anchor = {-1, -1};
    double delta = 0;

    ncvslideio::Mat kernel = ncvslideio::Mat(filterSize, CV_32FC1);
    ncvslideio::Scalar kernMean, kernStddev;

    const auto kernSize = filterSize.width * filterSize.height;
    const auto bigKernSize = 49;

    if (kernSize < bigKernSize)
    {
        kernMean = ncvslideio::Scalar(0.3);
        kernStddev = ncvslideio::Scalar(0.5);
    }
    else
    {
        kernMean = ncvslideio::Scalar(0.008);
        kernStddev = ncvslideio::Scalar(0.008);
    }

    randn(kernel, kernMean, kernStddev);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::filter2D(in, dtype, kernel, anchor, delta, borderType);

    ncvslideio::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::filter2D(in_mat1, out_mat_ocv, dtype, kernel, anchor, delta, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BoxFilterTest, AccuracyTest)
{
    ncvslideio::Point anchor = {-1, -1};
    bool normalize = true;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::boxFilter(in, dtype, ncvslideio::Size(filterSize, filterSize), anchor, normalize,
        borderType);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::boxFilter(in_mat1, out_mat_ocv, dtype, ncvslideio::Size(filterSize, filterSize), anchor,
            normalize, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(SepFilterTest, AccuracyTest)
{
    ncvslideio::Mat kernelX(kernSize, 1, CV_32F);
    ncvslideio::Mat kernelY(kernSize, 1, CV_32F);
    randu(kernelX, -1, 1);
    randu(kernelY, -1, 1);

    ncvslideio::Point anchor = ncvslideio::Point(-1, -1);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::sepFilter(in, dtype, kernelX, kernelY, anchor, ncvslideio::Scalar() );

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::sepFilter2D(in_mat1, out_mat_ocv, dtype, kernelX, kernelY );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BlurTest, AccuracyTest)
{
    ncvslideio::Point anchor = {-1, -1};

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::blur(in, ncvslideio::Size(filterSize, filterSize), anchor, borderType);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::blur(in_mat1, out_mat_ocv, ncvslideio::Size(filterSize, filterSize), anchor, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(GaussianBlurTest, AccuracyTest)
{
    ncvslideio::Size kSize = ncvslideio::Size(kernSize, kernSize);
    double sigmaX = rand();

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::gaussianBlur(in, kSize, sigmaX);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::GaussianBlur(in_mat1, out_mat_ocv, kSize, sigmaX);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(MedianBlurTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::medianBlur(in, kernSize);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::medianBlur(in_mat1, out_mat_ocv, kernSize);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(ErodeTest, AccuracyTest)
{
    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(kernType, ncvslideio::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::erode(in, kernel);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::erode(in_mat1, out_mat_ocv, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(Erode3x3Test, AccuracyTest)
{
    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(ncvslideio::MorphShapes::MORPH_RECT, ncvslideio::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::erode3x3(in, numIters);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::erode(in_mat1, out_mat_ocv, kernel, ncvslideio::Point(-1, -1), numIters);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(DilateTest, AccuracyTest)
{
    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(kernType, ncvslideio::Size(kernSize, kernSize));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::dilate(in, kernel);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::dilate(in_mat1, out_mat_ocv, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(Dilate3x3Test, AccuracyTest)
{
    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(ncvslideio::MorphShapes::MORPH_RECT, ncvslideio::Size(3,3));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::dilate3x3(in, numIters);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::dilate(in_mat1, out_mat_ocv, kernel, ncvslideio::Point(-1,-1), numIters);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(MorphologyExTest, AccuracyTest)
{
    ncvslideio::MorphShapes defShape = ncvslideio::MORPH_RECT;
    int defKernSize = 3;
    ncvslideio::Mat kernel = ncvslideio::getStructuringElement(defShape, ncvslideio::Size(defKernSize, defKernSize));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::morphologyEx(in, op, kernel);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::morphologyEx(in_mat1, out_mat_ocv, op, kernel);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(SobelTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::Sobel(in, dtype, dx, dy, kernSize );

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Sobel(in_mat1, out_mat_ocv, dtype, dx, dy, kernSize);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(SobelXYTest, AccuracyTest)
{
    ncvslideio::Mat out_mat_ocv2;
    ncvslideio::Mat out_mat_gapi2;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::SobelXY(in, dtype, order, kernSize, 1, 0, border_type, border_val);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi, out_mat_gapi2), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        // workaround for ncvslideio::Sobel
        ncvslideio::Mat temp_in;
        if(border_type == ncvslideio::BORDER_CONSTANT)
        {
            int n_pixels = (kernSize - 1) / 2;
            ncvslideio::copyMakeBorder(in_mat1, temp_in, n_pixels, n_pixels, n_pixels, n_pixels, border_type, border_val);
            in_mat1 = temp_in(ncvslideio::Rect(n_pixels, n_pixels, in_mat1.cols, in_mat1.rows));
        }
        ncvslideio::Sobel(in_mat1, out_mat_ocv, dtype, order, 0, kernSize, 1, 0, border_type);
        ncvslideio::Sobel(in_mat1, out_mat_ocv2, dtype, 0, order, kernSize, 1, 0, border_type);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat_gapi2, out_mat_ocv2));
        EXPECT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(sz, out_mat_gapi2.size());
    }
}

TEST_P(LaplacianTest, AccuracyTest)
{
    double delta = 10;
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::Laplacian(in, dtype, kernSize, scale, delta, borderType);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Laplacian(in_mat1, out_mat_ocv, dtype, kernSize, scale, delta, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BilateralFilterTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::bilateralFilter(in, d, sigmaColor, sigmaSpace, borderType);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::bilateralFilter(in_mat1, out_mat_ocv, d, sigmaColor, sigmaSpace, borderType);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(EqHistTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::equalizeHist(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::equalizeHist(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(CannyTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Canny(in_mat1, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(GoodFeaturesTest, AccuracyTest)
{
    double k = 0.04;

    initMatFromImage(type, fileName);

    std::vector<ncvslideio::Point2f> outVecOCV, outVecGAPI;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::goodFeaturesToTrack(in, maxCorners, qualityLevel, minDistance, ncvslideio::Mat(),
                                             blockSize, useHarrisDetector, k);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(outVecGAPI), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::goodFeaturesToTrack(in_mat1, outVecOCV, maxCorners, qualityLevel, minDistance,
                                ncvslideio::noArray(), blockSize, useHarrisDetector, k);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(outVecGAPI, outVecOCV));
    }
}

TEST_P(FindContoursNoOffsetTest, AccuracyTest)
{
    findContoursTestBody(sz, type, mode, method, cmpF, getCompileArgs());
}

TEST_P(FindContoursOffsetTest, AccuracyTest)
{
    const ncvslideio::Size sz(1280, 720);
    const MatType2 type = CV_8UC1;
    const ncvslideio::RetrievalModes mode = ncvslideio::RETR_EXTERNAL;
    const ncvslideio::ContourApproximationModes method = ncvslideio::CHAIN_APPROX_NONE;
    const CompareMats cmpF = AbsExact().to_compare_obj();
    const ncvslideio::Point offset(15, 15);

    findContoursTestBody(sz, type, mode, method, cmpF, getCompileArgs(), offset);
}

TEST_P(FindContoursHNoOffsetTest, AccuracyTest)
{
    findContoursTestBody<HIERARCHY>(sz, type, mode, method, cmpF, getCompileArgs());
}

TEST_P(FindContoursHOffsetTest, AccuracyTest)
{
    const ncvslideio::Size sz(1280, 720);
    const MatType2 type = CV_8UC1;
    const ncvslideio::RetrievalModes mode = ncvslideio::RETR_EXTERNAL;
    const ncvslideio::ContourApproximationModes method = ncvslideio::CHAIN_APPROX_NONE;
    const CompareMats cmpF = AbsExact().to_compare_obj();
    const ncvslideio::Point offset(15, 15);
    std::vector<std::vector<ncvslideio::Point>> outCtsOCV,  outCtsGAPI;
    std::vector<ncvslideio::Vec4i>              outHierOCV, outHierGAPI;

    findContoursTestBody<HIERARCHY>(sz, type, mode, method, cmpF, getCompileArgs(), offset);
}

TEST_P(BoundingRectMatTest, AccuracyTest)
{
    if (initByVector)
    {
        initMatByPointsVectorRandU<ncvslideio::Point_>(type, sz, dtype);
    }
    else
    {
        initMatrixRandU(type, sz, dtype);
    }
    boundingRectTestBody(in_mat1, cmpF, getCompileArgs());
}

TEST_P(BoundingRectVector32STest, AccuracyTest)

{
    std::vector<ncvslideio::Point2i> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    boundingRectTestBody(in_vector, cmpF, getCompileArgs());
}

TEST_P(BoundingRectVector32FTest, AccuracyTest)
{
    std::vector<ncvslideio::Point2f> in_vector;
    initPointsVectorRandU(sz.width, in_vector);

    boundingRectTestBody(in_vector, cmpF, getCompileArgs());
}

TEST_P(FitLine2DMatVectorTest, AccuracyTest)
{
    fitLineTestBody(in_mat1, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine2DVector32STest, AccuracyTest)
{
    std::vector<ncvslideio::Point2i> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine2DVector32FTest, AccuracyTest)
{
    std::vector<ncvslideio::Point2f> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine2DVector64FTest, AccuracyTest)
{
    std::vector<ncvslideio::Point2d> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DMatVectorTest, AccuracyTest)
{
    fitLineTestBody(in_mat1, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DVector32STest, AccuracyTest)
{
    std::vector<ncvslideio::Point3i> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DVector32FTest, AccuracyTest)
{
    std::vector<ncvslideio::Point3f> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(FitLine3DVector64FTest, AccuracyTest)
{
    std::vector<ncvslideio::Point3d> in_vec;
    initPointsVectorRandU(sz.width, in_vec);

    fitLineTestBody(in_vec, distType, cmpF, getCompileArgs());
}

TEST_P(BGR2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2RGB(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(RGB2GrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2Gray(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BGR2GrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2Gray(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2GRAY);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(RGB2YUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2YUV(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(YUV2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::YUV2RGB(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BGR2I420Test, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2I420(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2YUV_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(Size(sz.width, sz.height * 3 / 2), out_mat_gapi.size());
    }
}

TEST_P(RGB2I420Test, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2I420(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2YUV_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(Size(sz.width, sz.height * 3 / 2), out_mat_gapi.size());
    }
}

TEST_P(I4202BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::I4202BGR(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2BGR_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(Size(sz.width, sz.height * 2 / 3), out_mat_gapi.size());
    }
}

TEST_P(I4202RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::I4202RGB(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2RGB_I420);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(Size(sz.width, sz.height * 2 / 3), out_mat_gapi.size());
    }
}

TEST_P(NV12toRGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in_y;
    ncvslideio::GMat in_uv;
    auto out = ncvslideio::gapi::NV12toRGB(in_y, in_uv);

    // Additional mat for uv
    ncvslideio::Mat in_mat_uv(ncvslideio::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    ncvslideio::randn(in_mat_uv, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));

    ncvslideio::GComputation c(ncvslideio::GIn(in_y, in_uv), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat_uv), ncvslideio::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, ncvslideio::COLOR_YUV2RGB_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(NV12toBGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in_y;
    ncvslideio::GMat in_uv;
    auto out = ncvslideio::gapi::NV12toBGR(in_y, in_uv);

    // Additional mat for uv
    ncvslideio::Mat in_mat_uv(ncvslideio::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    ncvslideio::randn(in_mat_uv, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));

    ncvslideio::GComputation c(ncvslideio::GIn(in_y, in_uv), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat_uv), ncvslideio::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, ncvslideio::COLOR_YUV2BGR_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(NV12toGrayTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in_y;
    ncvslideio::GMat in_uv;
    auto out = ncvslideio::gapi::NV12toGray(in_y, in_uv);

    // Additional mat for uv
    ncvslideio::Mat in_mat_uv(ncvslideio::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    ncvslideio::randn(in_mat_uv, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));

    ncvslideio::GComputation c(ncvslideio::GIn(in_y, in_uv), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat_uv), ncvslideio::gout(out_mat_gapi), getCompileArgs());

    ncvslideio::Mat out_mat_ocv_planar;
    ncvslideio::Mat uv_planar(in_mat1.rows / 2, in_mat1.cols, CV_8UC1, in_mat_uv.data);
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::vconcat(in_mat1, uv_planar, out_mat_ocv_planar);
        ncvslideio::cvtColor(out_mat_ocv_planar, out_mat_ocv, ncvslideio::COLOR_YUV2GRAY_NV12);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

static void toPlanar(const ncvslideio::Mat& in, ncvslideio::Mat& out)
{
    GAPI_Assert(out.depth() == in.depth());
    GAPI_Assert(out.channels() == 1);
    GAPI_Assert(in.channels() == 3);
    GAPI_Assert(out.cols == in.cols);
    GAPI_Assert(out.rows == 3*in.rows);

    std::vector<ncvslideio::Mat> outs(3);
    for (int i = 0; i < 3; i++) {
        outs[i] = out(ncvslideio::Rect(0, i*in.rows, in.cols, in.rows));
    }
    ncvslideio::split(in, outs);
}

TEST_P(NV12toRGBpTest, AccuracyTest)
{
    ncvslideio::Size sz_p = ncvslideio::Size(sz.width, sz.height * 3);
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in_y;
    ncvslideio::GMat in_uv;
    auto out = ncvslideio::gapi::NV12toRGBp(in_y, in_uv);

    // Additional mat for uv
    ncvslideio::Mat in_mat_uv(ncvslideio::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    ncvslideio::randn(in_mat_uv, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));

    ncvslideio::GComputation c(ncvslideio::GIn(in_y, in_uv), ncvslideio::GOut(out));
    ncvslideio::Mat out_mat_gapi_planar(ncvslideio::Size(sz.width, sz.height * 3), CV_8UC1);
    c.apply(ncvslideio::gin(in_mat1, in_mat_uv), ncvslideio::gout(out_mat_gapi_planar), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::Mat out_mat_ocv_planar(ncvslideio::Size(sz.width, sz.height * 3), CV_8UC1);
    {
        ncvslideio::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, ncvslideio::COLOR_YUV2RGB_NV12);
        toPlanar(out_mat_ocv, out_mat_ocv_planar);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi_planar, out_mat_ocv_planar));
        EXPECT_EQ(sz_p, out_mat_gapi_planar.size());
    }
}


TEST_P(NV12toBGRpTest, AccuracyTest)
{
    ncvslideio::Size sz_p = ncvslideio::Size(sz.width, sz.height * 3);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in_y;
    ncvslideio::GMat in_uv;
    auto out = ncvslideio::gapi::NV12toBGRp(in_y, in_uv);

    // Additional mat for uv
    ncvslideio::Mat in_mat_uv(ncvslideio::Size(sz.width / 2, sz.height / 2), CV_8UC2);
    ncvslideio::randn(in_mat_uv, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));

    ncvslideio::GComputation c(ncvslideio::GIn(in_y, in_uv), ncvslideio::GOut(out));
    ncvslideio::Mat out_mat_gapi_planar(ncvslideio::Size(sz.width, sz.height * 3), CV_8UC1);
    c.apply(ncvslideio::gin(in_mat1, in_mat_uv), ncvslideio::gout(out_mat_gapi_planar), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::Mat out_mat_ocv_planar(ncvslideio::Size(sz.width, sz.height * 3), CV_8UC1);
    {
        ncvslideio::cvtColorTwoPlane(in_mat1, in_mat_uv, out_mat_ocv, ncvslideio::COLOR_YUV2BGR_NV12);
        toPlanar(out_mat_ocv, out_mat_ocv_planar);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi_planar, out_mat_ocv_planar));
        EXPECT_EQ(sz_p, out_mat_gapi_planar.size());
    }
}

TEST_P(RGB2LabTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2Lab(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2Lab);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BGR2LUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2LUV(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2Luv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(LUV2BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::LUV2BGR(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_Luv2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BGR2YUVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BGR2YUV(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BGR2YUV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(YUV2BGRTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::YUV2BGR(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_YUV2BGR);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(RGB2HSVTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2HSV(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_RGB2HSV);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(BayerGR2RGBTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::BayerGR2RGB(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cvtColor(in_mat1, out_mat_ocv, ncvslideio::COLOR_BayerGR2RGB);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(RGB2YUV422Test, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::RGB2YUV422(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        convertRGB2YUV422Ref(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

static void ResizeAccuracyTest(const CompareMats& cmpF, int type, int interp, ncvslideio::Size sz_in,
    ncvslideio::Size sz_out, double fx, double fy, ncvslideio::GCompileArgs&& compile_args)
{
    ncvslideio::Mat in_mat1 (sz_in, type );
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);

    auto out_mat_sz = sz_out.area() == 0 ? ncvslideio::Size(saturate_cast<int>(sz_in.width *fx),
                                                    saturate_cast<int>(sz_in.height*fy))
                                         : sz_out;
    ncvslideio::Mat out_mat(out_mat_sz, type);
    ncvslideio::Mat out_mat_ocv(out_mat_sz, type);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::resize(in, sz_out, fx, fy, interp);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat, std::move(compile_args));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::resize(in_mat1, out_mat_ocv, sz_out, fx, fy, interp);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat, out_mat_ocv));
    }
}

TEST_P(ResizeTest, AccuracyTest)
{
    ResizeAccuracyTest(cmpF, type, interp, sz, sz_out, 0.0, 0.0, getCompileArgs());
}

TEST_P(ResizeTestFxFy, AccuracyTest)
{
    ResizeAccuracyTest(cmpF, type, interp, sz, ncvslideio::Size{0, 0}, fx, fy, getCompileArgs());
}

TEST_P(ResizePTest, AccuracyTest)
{
    constexpr int planeNum = 3;
    ncvslideio::Size sz_in_p {sz.width,  sz.height*planeNum};
    ncvslideio::Size sz_out_p{sz_out.width, sz_out.height*planeNum};

    ncvslideio::Mat in_mat(sz_in_p, CV_8UC1);
    ncvslideio::randn(in_mat, ncvslideio::Scalar::all(127.0f), ncvslideio::Scalar::all(40.f));

    ncvslideio::Mat out_mat    (sz_out_p, CV_8UC1);
    ncvslideio::Mat out_mat_ocv_p(sz_out_p, CV_8UC1);

    ncvslideio::GMatP in;
    auto out = ncvslideio::gapi::resizeP(in, sz_out, interp);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    c.compile(ncvslideio::descr_of(in_mat).asPlanar(planeNum), getCompileArgs())
             (ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    for (int i = 0; i < planeNum; i++) {
        const ncvslideio::Mat in_mat_roi = in_mat(ncvslideio::Rect(0, i*sz.height,  sz.width,  sz.height));
        ncvslideio::Mat out_mat_roi = out_mat_ocv_p(ncvslideio::Rect(0, i*sz_out.height, sz_out.width, sz_out.height));
        ncvslideio::resize(in_mat_roi, out_mat_roi, sz_out, 0, 0, interp);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat, out_mat_ocv_p));
    }
}

} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_TESTS_INL_HPP
