// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>
#include "backends/ocl/goclimgproc.hpp"

GAPI_OCL_KERNEL(GOCLResize, ncvslideio::gapi::imgproc::GResize)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Size sz, double fx, double fy, int interp, ncvslideio::UMat &out)
    {
        ncvslideio::resize(in, out, sz, fx, fy, interp);
    }
};

GAPI_OCL_KERNEL(GOCLSepFilter, ncvslideio::gapi::imgproc::GSepFilter)
{
    static void run(const ncvslideio::UMat& in, int ddepth, const ncvslideio::Mat& kernX, const ncvslideio::Mat& kernY, const ncvslideio::Point& anchor, const ncvslideio::Scalar& delta,
                    int border, const ncvslideio::Scalar& bordVal, ncvslideio::UMat &out)
    {
        if( border == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::UMat temp_in;
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

GAPI_OCL_KERNEL(GOCLBoxFilter, ncvslideio::gapi::imgproc::GBoxFilter)
{
    static void run(const ncvslideio::UMat& in, int ddepth, const ncvslideio::Size& ksize, const ncvslideio::Point& anchor, bool normalize, int borderType, const ncvslideio::Scalar& bordVal, ncvslideio::UMat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::UMat temp_in;
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

GAPI_OCL_KERNEL(GOCLBlur, ncvslideio::gapi::imgproc::GBlur)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Size& ksize, const ncvslideio::Point& anchor, int borderType, const ncvslideio::Scalar& bordVal, ncvslideio::UMat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::UMat temp_in;
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


GAPI_OCL_KERNEL(GOCLFilter2D, ncvslideio::gapi::imgproc::GFilter2D)
{
    static void run(const ncvslideio::UMat& in, int ddepth, const ncvslideio::Mat& k, const ncvslideio::Point& anchor, const ncvslideio::Scalar& delta, int border,
                    const ncvslideio::Scalar& bordVal, ncvslideio::UMat &out)
    {
        if( border == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::UMat temp_in;
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

GAPI_OCL_KERNEL(GOCLGaussBlur, ncvslideio::gapi::imgproc::GGaussBlur)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Size& ksize, double sigmaX, double sigmaY, int borderType, const ncvslideio::Scalar& bordVal, ncvslideio::UMat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::UMat temp_in;
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

GAPI_OCL_KERNEL(GOCLMedianBlur, ncvslideio::gapi::imgproc::GMedianBlur)
{
    static void run(const ncvslideio::UMat& in, int ksize, ncvslideio::UMat &out)
    {
        ncvslideio::medianBlur(in, out, ksize);
    }
};

GAPI_OCL_KERNEL(GOCLErode, ncvslideio::gapi::imgproc::GErode)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Mat& kernel, const ncvslideio::Point& anchor, int iterations, int borderType, const ncvslideio::Scalar& borderValue, ncvslideio::UMat &out)
    {
        ncvslideio::erode(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCL_KERNEL(GOCLDilate, ncvslideio::gapi::imgproc::GDilate)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Mat& kernel, const ncvslideio::Point& anchor, int iterations, int borderType, const ncvslideio::Scalar& borderValue, ncvslideio::UMat &out)
    {
        ncvslideio::dilate(in, out, kernel, anchor, iterations, borderType, borderValue);
    }
};

GAPI_OCL_KERNEL(GOCLSobel, ncvslideio::gapi::imgproc::GSobel)
{
    static void run(const ncvslideio::UMat& in, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType,
                    const ncvslideio::Scalar& bordVal, ncvslideio::UMat &out)
    {
        if( borderType == ncvslideio::BORDER_CONSTANT )
        {
            ncvslideio::UMat temp_in;
            int add = (ksize - 1) / 2;
            ncvslideio::copyMakeBorder(in, temp_in, add, add, add, add, borderType, bordVal );
            ncvslideio::Rect rect = ncvslideio::Rect(add, add, in.cols, in.rows);
            ncvslideio::Sobel(temp_in(rect), out, ddepth, dx, dy, ksize, scale, delta, borderType);
        }
        else
        ncvslideio::Sobel(in, out, ddepth, dx, dy, ksize, scale, delta, borderType);
    }
};

GAPI_OCL_KERNEL(GOCLLaplacian, ncvslideio::gapi::imgproc::GLaplacian)
{
    static void run(const ncvslideio::UMat& in, int ddepth, int ksize, double scale,
                    double delta, int borderType, ncvslideio::UMat &out)
    {
        ncvslideio::Laplacian(in, out, ddepth, ksize, scale, delta, borderType);
    }
};

GAPI_OCL_KERNEL(GOCLBilateralFilter, ncvslideio::gapi::imgproc::GBilateralFilter)
{
    static void run(const ncvslideio::UMat& in, int ddepth, double sigmaColor,
                    double sigmaSpace, int borderType, ncvslideio::UMat &out)
    {
        ncvslideio::bilateralFilter(in, out, ddepth, sigmaColor, sigmaSpace, borderType);
    }
};

GAPI_OCL_KERNEL(GOCLEqualizeHist, ncvslideio::gapi::imgproc::GEqHist)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::equalizeHist(in, out);
    }
};

GAPI_OCL_KERNEL(GOCLCanny, ncvslideio::gapi::imgproc::GCanny)
{
    static void run(const ncvslideio::UMat& in, double thr1, double thr2, int apSize, bool l2gradient, ncvslideio::UMat &out)
    {
        ncvslideio::Canny(in, out, thr1, thr2, apSize, l2gradient);
    }
};

GAPI_OCL_KERNEL(GOCLRGB2YUV, ncvslideio::gapi::imgproc::GRGB2YUV)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2YUV);
    }
};

GAPI_OCL_KERNEL(GOCLYUV2RGB, ncvslideio::gapi::imgproc::GYUV2RGB)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2RGB);
    }
};

GAPI_OCL_KERNEL(GOCLRGB2Lab, ncvslideio::gapi::imgproc::GRGB2Lab)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2Lab);
    }
};

GAPI_OCL_KERNEL(GOCLBGR2LUV, ncvslideio::gapi::imgproc::GBGR2LUV)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2Luv);
    }
};

GAPI_OCL_KERNEL(GOCLBGR2YUV, ncvslideio::gapi::imgproc::GBGR2YUV)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2YUV);
    }
};

GAPI_OCL_KERNEL(GOCLLUV2BGR, ncvslideio::gapi::imgproc::GLUV2BGR)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_Luv2BGR);
    }
};

GAPI_OCL_KERNEL(GOCLYUV2BGR, ncvslideio::gapi::imgproc::GYUV2BGR)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_YUV2BGR);
    }
};

GAPI_OCL_KERNEL(GOCLRGB2Gray, ncvslideio::gapi::imgproc::GRGB2Gray)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_RGB2GRAY);
    }
};

GAPI_OCL_KERNEL(GOCLBGR2Gray, ncvslideio::gapi::imgproc::GBGR2Gray)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &out)
    {
        ncvslideio::cvtColor(in, out, ncvslideio::COLOR_BGR2GRAY);
    }
};

GAPI_OCL_KERNEL(GOCLRGB2GrayCustom, ncvslideio::gapi::imgproc::GRGB2GrayCustom)
{
    //TODO: avoid copy
    static void run(const ncvslideio::UMat& in, float rY, float bY, float gY, ncvslideio::UMat &out)
    {
        ncvslideio::Mat planes[3];
        ncvslideio::split(in.getMat(ncvslideio::ACCESS_READ), planes);
        ncvslideio::Mat tmp_out = (planes[0]*rY + planes[1]*bY + planes[2]*gY);
        tmp_out.copyTo(out);
    }
};


ncvslideio::GKernelPackage ncvslideio::gapi::imgproc::ocl::kernels()
{
    static auto pkg = ncvslideio::gapi::kernels
        < GOCLFilter2D
        , GOCLResize
        , GOCLSepFilter
        , GOCLBoxFilter
        , GOCLBlur
        , GOCLGaussBlur
        , GOCLMedianBlur
        , GOCLErode
        , GOCLDilate
        , GOCLSobel
        , GOCLLaplacian
        , GOCLBilateralFilter
        , GOCLCanny
        , GOCLEqualizeHist
        , GOCLRGB2YUV
        , GOCLYUV2RGB
        , GOCLRGB2Lab
        , GOCLBGR2LUV
        , GOCLBGR2YUV
        , GOCLYUV2BGR
        , GOCLLUV2BGR
        , GOCLBGR2Gray
        , GOCLRGB2Gray
        , GOCLRGB2GrayCustom
        >();
    return pkg;
}
