// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcompoundkernel.hpp>

#include "backends/fluid/gfluidimgproc_func.hpp"


namespace {
    ncvslideio::Mat add_border(const ncvslideio::Mat& in, const int ksize, const int borderType, const ncvslideio::Scalar& bordVal){
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::Mat temp_in;
            int add = (ksize - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, add, add, add, add, borderType, bordVal);
            return temp_in(ncvslideio::Rect(add, add, in.cols, in.rows));
        }
        return in;
    }
}

GAPI_OCV_KERNEL(GCPUResize, ncvslideio::gapi::imgproc::GResize)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Size sz, double fx, double fy, int interp, ncvslideio::Mat &out)
    {
        ncvslideio::resize(in, out, sz, fx, fy, interp);
    }
};

GAPI_OCV_KERNEL(GCPUResizeP, ncvslideio::gapi::imgproc::GResizeP)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Size out_sz, int interp, ncvslideio::Mat& out)
    {
        int inH = in.rows / 3;
        int inW = in.cols;
        int outH = out.rows / 3;
        int outW = out.cols;
        for (int i = 0; i < 3; i++) {
            auto in_plane = in(ncvslideio::Rect(0, i*inH, inW, inH));
            auto out_plane = out(ncvslideio::Rect(0, i*outH, outW, outH));
            ncvslideio::resize(in_plane, out_plane, out_sz, 0, 0, interp);
        }
    }
};

GAPI_OCV_KERNEL(GCPUSepFilter, ncvslideio::gapi::imgproc::GSepFilter)
{
    static void run(const ncvslideio::Mat& in, int ddepth, const ncvslideio::Mat& kernX, const ncvslideio::Mat& kernY, const ncvslideio::Point& anchor, const ncvslideio::Scalar& delta,
                    int border, const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out)
    {
        if( border == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::Mat temp_in;
            int width_add = (kernY.cols - 1) / 2;
            int height_add =  (kernX.rows - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal);
            ncvslideio::Rect rect = ncvslideio::Rect(height_add, width_add, in.cols, in.rows);
            ncvslideio::sepFilter2D(temp_in(rect), out, ddepth, kernX, kernY, anchor, delta.val[0], border);
        }
        else
            ncvslideio::sepFilter2D(in, out, ddepth, kernX, kernY, anchor, delta.val[0], border);
    }
};

GAPI_OCV_KERNEL(GCPUBoxFilter, ncvslideio::gapi::imgproc::GBoxFilter)
{
    static void run(const ncvslideio::Mat& in, int ddepth, const ncvslideio::Size& ksize, const ncvslideio::Point& anchor, bool normalize, int borderType, const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::Mat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
            ncvslideio::Rect rect = ncvslideio::Rect(height_add, width_add, in.cols, in.rows);
            ncvslideio::boxFilter(temp_in(rect), out, ddepth, ksize, anchor, normalize, borderType);
        }
        else
            ncvslideio::boxFilter(in, out, ddepth, ksize, anchor, normalize, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUBlur, ncvslideio::gapi::imgproc::GBlur)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Size& ksize, const ncvslideio::Point& anchor, int borderType, const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::Mat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
            ncvslideio::Rect rect = ncvslideio::Rect(height_add, width_add, in.cols, in.rows);
            ncvslideio::blur(temp_in(rect), out, ksize, anchor, borderType);
        }
        else
            ncvslideio::blur(in, out, ksize, anchor, borderType);
    }
};


GAPI_OCV_KERNEL(GCPUFilter2D, ncvslideio::gapi::imgproc::GFilter2D)
{
    static void run(const ncvslideio::Mat& in, int ddepth, const ncvslideio::Mat& k, const ncvslideio::Point& anchor, const ncvslideio::Scalar& delta, int border,
                    const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out)
    {
        if( border == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::Mat temp_in;
            int width_add = (k.cols - 1) / 2;
            int height_add =  (k.rows - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, border, bordVal );
            ncvslideio::Rect rect = ncvslideio::Rect(height_add, width_add, in.cols, in.rows);
            ncvslideio::filter2D(temp_in(rect), out, ddepth, k, anchor, delta.val[0], border);
        }
        else
            ncvslideio::filter2D(in, out, ddepth, k, anchor, delta.val[0], border);
    }
};

GAPI_OCV_KERNEL(GCPUGaussBlur, ncvslideio::gapi::imgproc::GGaussBlur)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Size& ksize, double sigmaX, double sigmaY, int borderType, const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::Mat temp_in;
            int width_add = (ksize.width - 1) / 2;
            int height_add =  (ksize.height - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal );
            ncvslideio::Rect rect = ncvslideio::Rect(height_add, width_add, in.cols, in.rows);
            ncvslideio::GaussianBlur(temp_in(rect), out, ksize, sigmaX, sigmaY, borderType);
        }
        else
            ncvslideio::GaussianBlur(in, out, ksize, sigmaX, sigmaY, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUMedianBlur, ncvslideio::gapi::imgproc::GMedianBlur)
{
    static void run(const ncvslideio::Mat& in, int ksize, ncvslideio::Mat &out)
    {
        ncvslideio::medianBlur(in, out, ksize);
    }
};

GAPI_OCV_KERNEL(GCPUErode, ncvslideio::gapi::imgproc::GErode)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Mat& kernel, const ncvslideio::Point& anchor, int iterations, int borderType, const ncvslideio::Scalar& borderValue, ncvslideio::Mat &out)
    {
        ncvslideio::erode(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUDilate, ncvslideio::gapi::imgproc::GDilate)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Mat& kernel, const ncvslideio::Point& anchor, int iterations, int borderType, const ncvslideio::Scalar& borderValue, ncvslideio::Mat &out)
    {
        ncvslideio::dilate(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUMorphologyEx, ncvslideio::gapi::imgproc::GMorphologyEx)
{
    static void run(const ncvslideio::Mat &in, const ncvslideio::MorphTypes op, const ncvslideio::Mat &kernel,
                    const ncvslideio::Point &anchor, const int iterations,
                    const ncvslideio::BorderTypes borderType, const ncvslideio::Scalar &borderValue, ncvslideio::Mat &out)
    {
        ncvslideio::morphologyEx(in, out, op, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUSobel, ncvslideio::gapi::imgproc::GSobel)
{
    static void run(const ncvslideio::Mat& in, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType,
                    const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out)
    {
        ncvslideio::Mat temp_in = add_border(in, ksize, borderType, bordVal);
        ncvslideio::Sobel(temp_in, out, ddepth, dx, dy, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUSobelXY, ncvslideio::gapi::imgproc::GSobelXY)
{
    static void run(const ncvslideio::Mat& in, int ddepth, int order, int ksize, double scale, double delta, int borderType,
                    const ncvslideio::Scalar& bordVal, ncvslideio::Mat &out_dx, ncvslideio::Mat &out_dy)
    {
        ncvslideio::Mat temp_in = add_border(in, ksize, borderType, bordVal);
        ncvslideio::Sobel(temp_in, out_dx, ddepth, order, 0, ksize, scale, delta, borderType);
        ncvslideio::Sobel(temp_in, out_dy, ddepth, 0, order, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPULaplacian, ncvslideio::gapi::imgproc::GLaplacian)
{
    static void run(const ncvslideio::Mat& in, int ddepth, int ksize, double scale,
                    double delta, int borderType, ncvslideio::Mat &out)
    {
        ncvslideio::Laplacian(in, out, ddepth, ksize, scale, delta, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUBilateralFilter, ncvslideio::gapi::imgproc::GBilateralFilter)
{
    static void run(const ncvslideio::Mat& in, int d, double sigmaColor,
                    double sigmaSpace, int borderType, ncvslideio::Mat &out)
    {
        ncvslideio::bilateralFilter(in, out, d, sigmaColor, sigmaSpace, borderType);
    }
};

GAPI_OCV_KERNEL(GCPUEqualizeHist, ncvslideio::gapi::imgproc::GEqHist)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::equalizeHist(in, out);
    }
};

GAPI_OCV_KERNEL(GCPUCanny, ncvslideio::gapi::imgproc::GCanny)
{
    static void run(const ncvslideio::Mat& in, double thr1, double thr2, int apSize, bool l2gradient, ncvslideio::Mat &out)
    {
        ncvslideio::Canny(in, out, thr1, thr2, apSize, l2gradient);
    }
};

GAPI_OCV_KERNEL(GCPUGoodFeatures, ncvslideio::gapi::imgproc::GGoodFeatures)
{
    static void run(const ncvslideio::Mat& image, int maxCorners, double qualityLevel, double minDistance,
                    const ncvslideio::Mat& mask, int blockSize, bool useHarrisDetector, double k,
                    std::vector<ncvslideio::Point2f> &out)
    {
        ncvslideio::goodFeaturesToTrack(image, out, maxCorners, qualityLevel, minDistance,
                                mask, blockSize, useHarrisDetector, k);
    }
};

GAPI_OCV_KERNEL(GCPUFindContours, ncvslideio::gapi::imgproc::GFindContours)
{
    static void run(const ncvslideio::Mat& image, const ncvslideio::RetrievalModes mode,
                    const ncvslideio::ContourApproximationModes method, const ncvslideio::Point& offset,
                    std::vector<std::vector<ncvslideio::Point>> &outConts)
    {
        ncvslideio::findContours(image, outConts, mode, method, offset);
    }
};

GAPI_OCV_KERNEL(GCPUFindContoursNoOffset, ncvslideio::gapi::imgproc::GFindContoursNoOffset)
{
    static void run(const ncvslideio::Mat& image, const ncvslideio::RetrievalModes mode,
                    const ncvslideio::ContourApproximationModes method,
                    std::vector<std::vector<ncvslideio::Point>> &outConts)
    {
        ncvslideio::findContours(image, outConts, mode, method);
    }
};

GAPI_OCV_KERNEL(GCPUFindContoursH, ncvslideio::gapi::imgproc::GFindContoursH)
{
    static void run(const ncvslideio::Mat& image, const ncvslideio::RetrievalModes mode,
                    const ncvslideio::ContourApproximationModes method, const ncvslideio::Point& offset,
                    std::vector<std::vector<ncvslideio::Point>> &outConts, std::vector<ncvslideio::Vec4i> &outHier)
    {
        ncvslideio::findContours(image, outConts, outHier, mode, method, offset);
    }
};

GAPI_OCV_KERNEL(GCPUFindContoursHNoOffset, ncvslideio::gapi::imgproc::GFindContoursHNoOffset)
{
    static void run(const ncvslideio::Mat& image, const ncvslideio::RetrievalModes mode,
                    const ncvslideio::ContourApproximationModes method,
                    std::vector<std::vector<ncvslideio::Point>> &outConts, std::vector<ncvslideio::Vec4i> &outHier)
    {
        ncvslideio::findContours(image, outConts, outHier, mode, method);
    }
};

GAPI_OCV_KERNEL(GCPUBoundingRectMat, ncvslideio::gapi::imgproc::GBoundingRectMat)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Rect& out)
    {
        out = ncvslideio::boundingRect(in);
    }
};

GAPI_OCV_KERNEL(GCPUBoundingRectVector32S, ncvslideio::gapi::imgproc::GBoundingRectVector32S)
{
    static void run(const std::vector<ncvslideio::Point2i>& in, ncvslideio::Rect& out)
    {
        out = ncvslideio::boundingRect(in);
    }
};

GAPI_OCV_KERNEL(GCPUBoundingRectVector32F, ncvslideio::gapi::imgproc::GBoundingRectVector32F)
{
    static void run(const std::vector<ncvslideio::Point2f>& in, ncvslideio::Rect& out)
    {
        out = ncvslideio::boundingRect(in);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DMat, ncvslideio::gapi::imgproc::GFitLine2DMat)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::DistanceTypes distType, const double param,
                    const double reps, const double aeps, ncvslideio::Vec4f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DVector32S, ncvslideio::gapi::imgproc::GFitLine2DVector32S)
{
    static void run(const std::vector<ncvslideio::Point2i>& in, const ncvslideio::DistanceTypes distType,
                    const double param, const double reps, const double aeps, ncvslideio::Vec4f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DVector32F, ncvslideio::gapi::imgproc::GFitLine2DVector32F)
{
    static void run(const std::vector<ncvslideio::Point2f>& in, const ncvslideio::DistanceTypes distType,
                    const double param, const double reps, const double aeps, ncvslideio::Vec4f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine2DVector64F, ncvslideio::gapi::imgproc::GFitLine2DVector64F)
{
    static void run(const std::vector<ncvslideio::Point2d>& in, const ncvslideio::DistanceTypes distType,
                    const double param, const double reps, const double aeps, ncvslideio::Vec4f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DMat, ncvslideio::gapi::imgproc::GFitLine3DMat)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::DistanceTypes distType, const double param,
                    const double reps, const double aeps, ncvslideio::Vec6f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DVector32S, ncvslideio::gapi::imgproc::GFitLine3DVector32S)
{
    static void run(const std::vector<ncvslideio::Point3i>& in, const ncvslideio::DistanceTypes distType,
                    const double param, const double reps, const double aeps, ncvslideio::Vec6f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DVector32F, ncvslideio::gapi::imgproc::GFitLine3DVector32F)
{
    static void run(const std::vector<ncvslideio::Point3f>& in, const ncvslideio::DistanceTypes distType,
                    const double param, const double reps, const double aeps, ncvslideio::Vec6f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUFitLine3DVector64F, ncvslideio::gapi::imgproc::GFitLine3DVector64F)
{
    static void run(const std::vector<ncvslideio::Point3d>& in, const ncvslideio::DistanceTypes distType,
                    const double param, const double reps, const double aeps, ncvslideio::Vec6f& out)
    {
        ncvslideio::fitLine(in, out, distType, param, reps, aeps);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2RGB, ncvslideio::gapi::imgproc::GBGR2RGB)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2RGB);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2I420, ncvslideio::gapi::imgproc::GBGR2I420)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2YUV_I420);
    }
};

GAPI_OCV_KERNEL(GCPURGB2I420, ncvslideio::gapi::imgproc::GRGB2I420)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2YUV_I420);
    }
};

GAPI_OCV_KERNEL(GCPUI4202BGR, ncvslideio::gapi::imgproc::GI4202BGR)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2BGR_I420);
    }
};

GAPI_OCV_KERNEL(GCPUI4202RGB, ncvslideio::gapi::imgproc::GI4202RGB)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2RGB_I420);
    }
};

GAPI_OCV_KERNEL(GCPURGB2YUV, ncvslideio::gapi::imgproc::GRGB2YUV)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2YUV);
    }
};

GAPI_OCV_KERNEL(GCPUYUV2RGB, ncvslideio::gapi::imgproc::GYUV2RGB)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2RGB);
    }
};

GAPI_OCV_KERNEL(GCPUNV12toRGB, ncvslideio::gapi::imgproc::GNV12toRGB)
{
    static void run(const ncvslideio::Mat& in_y, const ncvslideio::Mat& in_uv, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColorTwoPlane(in_y, in_uv, out, ncvslideio::COLOR_YUV2RGB_NV12);
    }
};

GAPI_OCV_KERNEL(GCPUNV12toBGR, ncvslideio::gapi::imgproc::GNV12toBGR)
{
    static void run(const ncvslideio::Mat& in_y, const ncvslideio::Mat& in_uv, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColorTwoPlane(in_y, in_uv, out, ncvslideio::COLOR_YUV2BGR_NV12);
    }
};

GAPI_OCV_KERNEL(GCPURGB2Lab, ncvslideio::gapi::imgproc::GRGB2Lab)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2Lab);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2LUV, ncvslideio::gapi::imgproc::GBGR2LUV)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2Luv);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2YUV, ncvslideio::gapi::imgproc::GBGR2YUV)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2YUV);
    }
};

GAPI_OCV_KERNEL(GCPULUV2BGR, ncvslideio::gapi::imgproc::GLUV2BGR)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_Luv2BGR);
    }
};

GAPI_OCV_KERNEL(GCPUYUV2BGR, ncvslideio::gapi::imgproc::GYUV2BGR)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2BGR);
    }
};

GAPI_OCV_KERNEL(GCPURGB2Gray, ncvslideio::gapi::imgproc::GRGB2Gray)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2GRAY);
    }
};

GAPI_OCV_KERNEL(GCPUBGR2Gray, ncvslideio::gapi::imgproc::GBGR2Gray)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2GRAY);
    }
};

GAPI_OCV_KERNEL(GCPURGB2GrayCustom, ncvslideio::gapi::imgproc::GRGB2GrayCustom)
{
    static void run(const ncvslideio::Mat& in, float rY, float bY, float gY, ncvslideio::Mat &out)
    {
        ncvslideio::Mat planes[3];
        ncvslideio::split(in, planes);
        out = planes[0]*rY + planes[1]*bY + planes[2]*gY;
    }
};

GAPI_OCV_KERNEL(GCPUBayerGR2RGB, ncvslideio::gapi::imgproc::GBayerGR2RGB)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BayerGR2RGB);
    }
};

GAPI_OCV_KERNEL(GCPURGB2HSV, ncvslideio::gapi::imgproc::GRGB2HSV)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2HSV);
    }
};

GAPI_OCV_KERNEL(GCPURGB2YUV422, ncvslideio::gapi::imgproc::GRGB2YUV422)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        out.create(in.size(), CV_8UC2);

        for (int i = 0; i < in.rows; ++i)
        {
            const uchar* in_line_p  = in.ptr<uchar>(i);
            uchar* out_line_p = out.ptr<uchar>(i);
            ncvslideio::gapi::fluid::run_rgb2yuv422_impl(out_line_p, in_line_p, in.cols);
        }
    }
};

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


GAPI_OCV_KERNEL(GCPUNV12toRGBp, ncvslideio::gapi::imgproc::GNV12toRGBp)
{
    static void run(const ncvslideio::Mat& inY, const ncvslideio::Mat& inUV, ncvslideio::Mat& out)
    {
        ncvslideio::Mat rgb;
        ncvslideio::cvtColorTwoPlane(inY, inUV, rgb, ncvslideio::COLOR_YUV2RGB_NV12);
        toPlanar(rgb, out);
    }
};

G_TYPED_KERNEL(GYUV2Gray, <ncvslideio::GMat(ncvslideio::GMat)>, "yuvtogray") {
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in) {
        GAPI_Assert(in.depth  == CV_8U);
        GAPI_Assert(in.planar == false);
        GAPI_Assert(in.size.width  % 2 == 0);
        GAPI_Assert(in.size.height % 3 == 0);

        /* YUV format for this kernel:
         * Y Y Y Y Y Y Y Y
         * Y Y Y Y Y Y Y Y
         * Y Y Y Y Y Y Y Y
         * Y Y Y Y Y Y Y Y
         * U V U V U V U V
         * U V U V U V U V
         */

        return {CV_8U, 1, ncvslideio::Size{in.size.width, in.size.height - (in.size.height / 3)}, false};
    }
};

GAPI_OCV_KERNEL(GCPUYUV2Gray, GYUV2Gray)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat& out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2GRAY_NV12);
    }
};

G_TYPED_KERNEL(GConcatYUVPlanes, <ncvslideio::GMat(ncvslideio::GMat, ncvslideio::GMat)>, "concatyuvplanes") {
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc y, ncvslideio::GMatDesc uv) {
        return {CV_8U, 1, ncvslideio::Size{y.size.width, y.size.height + uv.size.height}, false};
    }
};

GAPI_OCV_KERNEL(GCPUConcatYUVPlanes, GConcatYUVPlanes)
{
    static void run(const ncvslideio::Mat& in_y, const ncvslideio::Mat& in_uv, ncvslideio::Mat& out)
    {
        ncvslideio::Mat uv_planar(in_uv.rows, in_uv.cols * 2, CV_8UC1, in_uv.data);
        ncvslideio::vconcat(in_y, uv_planar, out);
    }
};

GAPI_COMPOUND_KERNEL(GCPUNV12toGray, ncvslideio::gapi::imgproc::GNV12toGray)
{
    static ncvslideio::GMat expand(ncvslideio::GMat y, ncvslideio::GMat uv)
    {
        return GYUV2Gray::on(GConcatYUVPlanes::on(y, uv));
    }
};

GAPI_OCV_KERNEL(GCPUNV12toBGRp, ncvslideio::gapi::imgproc::GNV12toBGRp)
{
    static void run(const ncvslideio::Mat& inY, const ncvslideio::Mat& inUV, ncvslideio::Mat& out)
    {
        ncvslideio::Mat rgb;
        ncvslideio::cvtColorTwoPlane(inY, inUV, rgb, ncvslideio::COLOR_YUV2BGR_NV12);
        toPlanar(rgb, out);
    }
};

ncvslideio::GKernelPackage ncvslideio::gapi::imgproc::cpu::kernels()
{
    static auto pkg = ncvslideio::gapi::kernels
        < GCPUFilter2D
        , GCPUResize
        , GCPUResizeP
        , GCPUSepFilter
        , GCPUBoxFilter
        , GCPUBlur
        , GCPUGaussBlur
        , GCPUMedianBlur
        , GCPUErode
        , GCPUDilate
        , GCPUMorphologyEx
        , GCPUSobel
        , GCPUSobelXY
        , GCPULaplacian
        , GCPUBilateralFilter
        , GCPUCanny
        , GCPUGoodFeatures
        , GCPUEqualizeHist
        , GCPUFindContours
        , GCPUFindContoursNoOffset
        , GCPUFindContoursH
        , GCPUFindContoursHNoOffset
        , GCPUBGR2RGB
        , GCPURGB2YUV
        , GCPUBoundingRectMat
        , GCPUBoundingRectVector32S
        , GCPUBoundingRectVector32F
        , GCPUFitLine2DMat
        , GCPUFitLine2DVector32S
        , GCPUFitLine2DVector32F
        , GCPUFitLine2DVector64F
        , GCPUFitLine3DMat
        , GCPUFitLine3DVector32S
        , GCPUFitLine3DVector32F
        , GCPUFitLine3DVector64F
        , GCPUYUV2RGB
        , GCPUBGR2I420
        , GCPURGB2I420
        , GCPUI4202BGR
        , GCPUI4202RGB
        , GCPUNV12toRGB
        , GCPUNV12toBGR
        , GCPURGB2Lab
        , GCPUBGR2LUV
        , GCPUBGR2YUV
        , GCPUYUV2BGR
        , GCPULUV2BGR
        , GCPUBGR2Gray
        , GCPURGB2Gray
        , GCPURGB2GrayCustom
        , GCPUBayerGR2RGB
        , GCPURGB2HSV
        , GCPURGB2YUV422
        , GCPUYUV2Gray
        , GCPUNV12toRGBp
        , GCPUNV12toBGRp
        , GCPUNV12toGray
        , GCPUConcatYUVPlanes
        >();
    return pkg;
}
