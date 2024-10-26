// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"
#include "gnnparsers.hpp"

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

GAPI_OCV_KERNEL(GCPUAdd, ncvslideio::gapi::core::GAdd)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::add(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUAddC, ncvslideio::gapi::core::GAddC)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::add(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUSub, ncvslideio::gapi::core::GSub)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::subtract(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUSubC, ncvslideio::gapi::core::GSubC)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::subtract(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUSubRC, ncvslideio::gapi::core::GSubRC)
{
    static void run(const ncvslideio::Scalar& a, const ncvslideio::Mat& b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::subtract(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMul, ncvslideio::gapi::core::GMul)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, double scale, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::multiply(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMulCOld, ncvslideio::gapi::core::GMulCOld)
{
    static void run(const ncvslideio::Mat& a, double b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMulC, ncvslideio::gapi::core::GMulC)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUDiv, ncvslideio::gapi::core::GDiv)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, double scale, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUDivC, ncvslideio::gapi::core::GDivC)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, double scale, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUDivRC, ncvslideio::gapi::core::GDivRC)
{
    static void run(const ncvslideio::Scalar& a, const ncvslideio::Mat& b, double scale, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUMask, ncvslideio::gapi::core::GMask)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Mat& mask, ncvslideio::Mat& out)
    {
        out = ncvslideio::Mat::zeros(in.size(), in.type());
        in.copyTo(out, mask);
    }
};

GAPI_OCV_KERNEL(GCPUMean, ncvslideio::gapi::core::GMean)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::mean(in);
    }
};

GAPI_OCV_KERNEL(GCPUPolarToCart, ncvslideio::gapi::core::GPolarToCart)
{
    static void run(const ncvslideio::Mat& magn, const ncvslideio::Mat& angle, bool angleInDegrees, ncvslideio::Mat& outx, ncvslideio::Mat& outy)
    {
        ncvslideio::polarToCart(magn, angle, outx, outy, angleInDegrees);
    }
};

GAPI_OCV_KERNEL(GCPUCartToPolar, ncvslideio::gapi::core::GCartToPolar)
{
    static void run(const ncvslideio::Mat& x, const ncvslideio::Mat& y, bool angleInDegrees, ncvslideio::Mat& outmagn, ncvslideio::Mat& outangle)
    {
        ncvslideio::cartToPolar(x, y, outmagn, outangle, angleInDegrees);
    }
};

GAPI_OCV_KERNEL(GCPUPhase, ncvslideio::gapi::core::GPhase)
{
    static void run(const ncvslideio::Mat &x, const ncvslideio::Mat &y, bool angleInDegrees, ncvslideio::Mat &out)
    {
        ncvslideio::phase(x, y, out, angleInDegrees);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGT, ncvslideio::gapi::core::GCmpGT)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGE, ncvslideio::gapi::core::GCmpGE)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLE, ncvslideio::gapi::core::GCmpLE)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLT, ncvslideio::gapi::core::GCmpLT)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpEQ, ncvslideio::gapi::core::GCmpEQ)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_EQ);
    }
};

GAPI_OCV_KERNEL(GCPUCmpNE, ncvslideio::gapi::core::GCmpNE)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_NE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGTScalar, ncvslideio::gapi::core::GCmpGTScalar)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpGEScalar, ncvslideio::gapi::core::GCmpGEScalar)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLEScalar, ncvslideio::gapi::core::GCmpLEScalar)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LE);
    }
};

GAPI_OCV_KERNEL(GCPUCmpLTScalar, ncvslideio::gapi::core::GCmpLTScalar)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LT);
    }
};

GAPI_OCV_KERNEL(GCPUCmpEQScalar, ncvslideio::gapi::core::GCmpEQScalar)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_EQ);
    }
};

GAPI_OCV_KERNEL(GCPUCmpNEScalar, ncvslideio::gapi::core::GCmpNEScalar)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_NE);
    }
};

GAPI_OCV_KERNEL(GCPUAnd, ncvslideio::gapi::core::GAnd)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_and(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUAndS, ncvslideio::gapi::core::GAndS)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_and(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUOr, ncvslideio::gapi::core::GOr)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_or(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUOrS, ncvslideio::gapi::core::GOrS)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_or(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUXor, ncvslideio::gapi::core::GXor)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_xor(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUXorS, ncvslideio::gapi::core::GXorS)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Scalar& b, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_xor(a, b, out);
    }
};

GAPI_OCV_KERNEL(GCPUNot, ncvslideio::gapi::core::GNot)
{
    static void run(const ncvslideio::Mat& a, ncvslideio::Mat& out)
    {
        ncvslideio::bitwise_not(a, out);
    }
};

GAPI_OCV_KERNEL(GCPUSelect, ncvslideio::gapi::core::GSelect)
{
    static void run(const ncvslideio::Mat& src1, const ncvslideio::Mat& src2, const ncvslideio::Mat& mask, ncvslideio::Mat& out)
    {
        src2.copyTo(out);
        src1.copyTo(out, mask);
    }
};

GAPI_OCV_KERNEL(GCPUMin, ncvslideio::gapi::core::GMin)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out)
    {
        out = ncvslideio::min(in1, in2);
    }
};

GAPI_OCV_KERNEL(GCPUMax, ncvslideio::gapi::core::GMax)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out)
    {
        out = ncvslideio::max(in1, in2);
    }
};

GAPI_OCV_KERNEL(GCPUAbsDiff, ncvslideio::gapi::core::GAbsDiff)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out)
    {
        ncvslideio::absdiff(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPUAbsDiffC, ncvslideio::gapi::core::GAbsDiffC)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Scalar& in2, ncvslideio::Mat& out)
    {
        ncvslideio::absdiff(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPUSum, ncvslideio::gapi::core::GSum)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::sum(in);
    }
};

GAPI_OCV_KERNEL(GCPUCountNonZero, ncvslideio::gapi::core::GCountNonZero)
{
    static void run(const ncvslideio::Mat& in, int& out)
    {
        out = ncvslideio::countNonZero(in);
    }
};

GAPI_OCV_KERNEL(GCPUAddW, ncvslideio::gapi::core::GAddW)
{
    static void run(const ncvslideio::Mat& in1, double alpha, const ncvslideio::Mat& in2, double beta, double gamma, int dtype, ncvslideio::Mat& out)
    {
        ncvslideio::addWeighted(in1, alpha, in2, beta, gamma, out, dtype);
    }
};

GAPI_OCV_KERNEL(GCPUNormL1, ncvslideio::gapi::core::GNormL1)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::norm(in, ncvslideio::NORM_L1);
    }
};

GAPI_OCV_KERNEL(GCPUNormL2, ncvslideio::gapi::core::GNormL2)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::norm(in, ncvslideio::NORM_L2);
    }
};

GAPI_OCV_KERNEL(GCPUNormInf, ncvslideio::gapi::core::GNormInf)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::norm(in, ncvslideio::NORM_INF);
    }
};

GAPI_OCV_KERNEL(GCPUIntegral, ncvslideio::gapi::core::GIntegral)
{
    static void run(const ncvslideio::Mat& in, int sdepth, int sqdepth, ncvslideio::Mat& out, ncvslideio::Mat& outSq)
    {
        ncvslideio::integral(in, out, outSq, sdepth, sqdepth);
    }
};

GAPI_OCV_KERNEL(GCPUThreshold, ncvslideio::gapi::core::GThreshold)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Scalar& a, const ncvslideio::Scalar& b, int type, ncvslideio::Mat& out)
    {
        ncvslideio::threshold(in, out, a.val[0], b.val[0], type);
    }
};

GAPI_OCV_KERNEL(GCPUThresholdOT, ncvslideio::gapi::core::GThresholdOT)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Scalar& b, int type, ncvslideio::Mat& out, ncvslideio::Scalar& outScalar)
    {
        outScalar = ncvslideio::threshold(in, out, b.val[0], b.val[0], type);
    }
};


GAPI_OCV_KERNEL(GCPUInRange, ncvslideio::gapi::core::GInRange)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Scalar& low, const ncvslideio::Scalar& up, ncvslideio::Mat& out)
    {
        ncvslideio::inRange(in, low, up, out);
    }
};

GAPI_OCV_KERNEL(GCPUSplit3, ncvslideio::gapi::core::GSplit3)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &m1, ncvslideio::Mat &m2, ncvslideio::Mat &m3)
    {
        std::vector<ncvslideio::Mat> outMats = {m1, m2, m3};
        ncvslideio::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
    }
};

GAPI_OCV_KERNEL(GCPUSplit4, ncvslideio::gapi::core::GSplit4)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &m1, ncvslideio::Mat &m2, ncvslideio::Mat &m3, ncvslideio::Mat &m4)
    {
        std::vector<ncvslideio::Mat> outMats = {m1, m2, m3, m4};
        ncvslideio::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
        m4 = outMats[3];
    }
};

GAPI_OCV_KERNEL(GCPUMerge3, ncvslideio::gapi::core::GMerge3)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, const ncvslideio::Mat& in3, ncvslideio::Mat &out)
    {
        std::vector<ncvslideio::Mat> inMats = {in1, in2, in3};
        ncvslideio::merge(inMats, out);
    }
};

GAPI_OCV_KERNEL(GCPUMerge4, ncvslideio::gapi::core::GMerge4)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, const ncvslideio::Mat& in3, const ncvslideio::Mat& in4, ncvslideio::Mat &out)
    {
        std::vector<ncvslideio::Mat> inMats = {in1, in2, in3, in4};
        ncvslideio::merge(inMats, out);
    }
};

GAPI_OCV_KERNEL(GCPURemap, ncvslideio::gapi::core::GRemap)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Mat& x, const ncvslideio::Mat& y, int a, int b, ncvslideio::Scalar s, ncvslideio::Mat& out)
    {
        ncvslideio::remap(in, out, x, y, a, b, s);
    }
};

GAPI_OCV_KERNEL(GCPUFlip, ncvslideio::gapi::core::GFlip)
{
    static void run(const ncvslideio::Mat& in, int code, ncvslideio::Mat& out)
    {
        ncvslideio::flip(in, out, code);
    }
};

GAPI_OCV_KERNEL(GCPUCrop, ncvslideio::gapi::core::GCrop)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Rect rect, ncvslideio::Mat& out)
    {
        ncvslideio::Mat(in, rect).copyTo(out);
    }
};

GAPI_OCV_KERNEL(GCPUConcatHor, ncvslideio::gapi::core::GConcatHor)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out)
    {
        ncvslideio::hconcat(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPUConcatVert, ncvslideio::gapi::core::GConcatVert)
{
    static void run(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out)
    {
        ncvslideio::vconcat(in1, in2, out);
    }
};

GAPI_OCV_KERNEL(GCPULUT, ncvslideio::gapi::core::GLUT)
{
    static void run(const ncvslideio::Mat& in, const ncvslideio::Mat& lut, ncvslideio::Mat& out)
    {
        ncvslideio::LUT(in, lut, out);
    }
};

GAPI_OCV_KERNEL(GCPUConvertTo, ncvslideio::gapi::core::GConvertTo)
{
    static void run(const ncvslideio::Mat& in, int rtype, double alpha, double beta, ncvslideio::Mat& out)
    {
        in.convertTo(out, rtype, alpha, beta);
    }
};

GAPI_OCV_KERNEL(GCPUSqrt, ncvslideio::gapi::core::GSqrt)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
    {
        ncvslideio::sqrt(in, out);
    }
};

GAPI_OCV_KERNEL(GCPUNormalize, ncvslideio::gapi::core::GNormalize)
{
    static void run(const ncvslideio::Mat& src, double a, double b,
                    int norm_type, int ddepth, ncvslideio::Mat& out)
    {
        ncvslideio::normalize(src, out, a, b, norm_type, ddepth);
    }
};

GAPI_OCV_KERNEL(GCPUWarpPerspective, ncvslideio::gapi::core::GWarpPerspective)
{
    static void run(const ncvslideio::Mat& src, const ncvslideio::Mat& M,  const ncvslideio::Size& dsize,
                    int flags, int borderMode, const ncvslideio::Scalar& borderValue, ncvslideio::Mat& out)
    {
        ncvslideio::warpPerspective(src, out, M, dsize, flags, borderMode, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUWarpAffine, ncvslideio::gapi::core::GWarpAffine)
{
    static void run(const ncvslideio::Mat& src, const ncvslideio::Mat& M,  const ncvslideio::Size& dsize,
                    int flags, int borderMode, const ncvslideio::Scalar& borderValue, ncvslideio::Mat& out)
    {
        ncvslideio::warpAffine(src, out, M, dsize, flags, borderMode, borderValue);
    }
};

GAPI_OCV_KERNEL(GCPUKMeansND, ncvslideio::gapi::core::GKMeansND)
{
    static void run(const ncvslideio::Mat& data, const int K, const ncvslideio::Mat& inBestLabels,
                    const ncvslideio::TermCriteria& criteria, const int attempts,
                    const ncvslideio::KmeansFlags flags,
                    double& compactness, ncvslideio::Mat& outBestLabels, ncvslideio::Mat& centers)
    {
        if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
        {
            inBestLabels.copyTo(outBestLabels);
        }
        compactness = ncvslideio::kmeans(data, K, outBestLabels, criteria, attempts, flags, centers);
    }
};

GAPI_OCV_KERNEL(GCPUKMeansNDNoInit, ncvslideio::gapi::core::GKMeansNDNoInit)
{
    static void run(const ncvslideio::Mat& data, const int K, const ncvslideio::TermCriteria& criteria,
                    const int attempts, const ncvslideio::KmeansFlags flags,
                    double& compactness, ncvslideio::Mat& outBestLabels, ncvslideio::Mat& centers)
    {
        compactness = ncvslideio::kmeans(data, K, outBestLabels, criteria, attempts, flags, centers);
    }
};

GAPI_OCV_KERNEL(GCPUKMeans2D, ncvslideio::gapi::core::GKMeans2D)
{
    static void run(const std::vector<ncvslideio::Point2f>& data, const int K,
                    const std::vector<int>& inBestLabels, const ncvslideio::TermCriteria& criteria,
                    const int attempts, const ncvslideio::KmeansFlags flags,
                    double& compactness, std::vector<int>& outBestLabels,
                    std::vector<ncvslideio::Point2f>& centers)
    {
        if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
        {
            outBestLabels = inBestLabels;
        }
        compactness = ncvslideio::kmeans(data, K, outBestLabels, criteria, attempts, flags, centers);
    }
};

GAPI_OCV_KERNEL(GCPUKMeans3D, ncvslideio::gapi::core::GKMeans3D)
{
    static void run(const std::vector<ncvslideio::Point3f>& data, const int K,
                    const std::vector<int>& inBestLabels, const ncvslideio::TermCriteria& criteria,
                    const int attempts, const ncvslideio::KmeansFlags flags,
                    double& compactness, std::vector<int>& outBestLabels,
                    std::vector<ncvslideio::Point3f>& centers)
    {
        if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
        {
            outBestLabels = inBestLabels;
        }
        compactness = ncvslideio::kmeans(data, K, outBestLabels, criteria, attempts, flags, centers);
    }
};

GAPI_OCV_KERNEL(GCPUTranspose, ncvslideio::gapi::core::GTranspose)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Mat& out)
    {
        ncvslideio::transpose(in, out);
    }
};


GAPI_OCV_KERNEL(GCPUParseSSDBL, ncvslideio::gapi::nn::parsers::GParseSSDBL)
{
    static void run(const ncvslideio::Mat&  in_ssd_result,
                    const ncvslideio::Size& in_size,
                    const float     confidence_threshold,
                    const int       filter_label,
                    std::vector<ncvslideio::Rect>& out_boxes,
                    std::vector<int>&      out_labels)
    {
        ncvslideio::ParseSSD(in_ssd_result, in_size,
                     confidence_threshold,
                     filter_label,
                     false,
                     false,
                     out_boxes, out_labels);
    }
};

GAPI_OCV_KERNEL(GOCVParseSSD, ncvslideio::gapi::nn::parsers::GParseSSD)
{
    static void run(const ncvslideio::Mat&  in_ssd_result,
                    const ncvslideio::Size& in_size,
                    const float     confidence_threshold,
                    const bool      alignment_to_square,
                    const bool      filter_out_of_bounds,
                    std::vector<ncvslideio::Rect>& out_boxes)
    {
        std::vector<int> unused_labels;
        ncvslideio::ParseSSD(in_ssd_result, in_size,
                     confidence_threshold,
                     -1,
                     alignment_to_square,
                     filter_out_of_bounds,
                     out_boxes, unused_labels);
    }
};

GAPI_OCV_KERNEL(GCPUParseYolo, ncvslideio::gapi::nn::parsers::GParseYolo)
{
    static void run(const ncvslideio::Mat&  in_yolo_result,
                    const ncvslideio::Size& in_size,
                    const float     confidence_threshold,
                    const float     nms_threshold,
                    const std::vector<float>& anchors,
                    std::vector<ncvslideio::Rect>& out_boxes,
                    std::vector<int>&      out_labels)
    {
        ncvslideio::parseYolo(in_yolo_result, in_size, confidence_threshold, nms_threshold, anchors, out_boxes, out_labels);
    }
};

GAPI_OCV_KERNEL(GCPUSize, ncvslideio::gapi::streaming::GSize)
{
    static void run(const ncvslideio::Mat& in, ncvslideio::Size& out)
    {
        out.width  = in.cols;
        out.height = in.rows;
    }
};

GAPI_OCV_KERNEL(GCPUSizeR, ncvslideio::gapi::streaming::GSizeR)
{
    static void run(const ncvslideio::Rect& in, ncvslideio::Size& out)
    {
        out.width  = in.width;
        out.height = in.height;
    }
};

GAPI_OCV_KERNEL(GCPUSizeMF, ncvslideio::gapi::streaming::GSizeMF)
{
    static void run(const ncvslideio::MediaFrame& in, ncvslideio::Size& out)
    {
        out = in.desc().size;
    }
};

ncvslideio::GKernelPackage ncvslideio::gapi::core::cpu::kernels()
{
    static auto pkg = ncvslideio::gapi::kernels
        <  GCPUAdd
         , GCPUAddC
         , GCPUSub
         , GCPUSubC
         , GCPUSubRC
         , GCPUMul
         , GCPUMulC
         , GCPUMulCOld
         , GCPUDiv
         , GCPUDivC
         , GCPUDivRC
         , GCPUMean
         , GCPUMask
         , GCPUPolarToCart
         , GCPUCartToPolar
         , GCPUPhase
         , GCPUCmpGT
         , GCPUCmpGE
         , GCPUCmpLE
         , GCPUCmpLT
         , GCPUCmpEQ
         , GCPUCmpNE
         , GCPUCmpGTScalar
         , GCPUCmpGEScalar
         , GCPUCmpLEScalar
         , GCPUCmpLTScalar
         , GCPUCmpEQScalar
         , GCPUCmpNEScalar
         , GCPUAnd
         , GCPUAndS
         , GCPUOr
         , GCPUOrS
         , GCPUXor
         , GCPUXorS
         , GCPUNot
         , GCPUSelect
         , GCPUMin
         , GCPUMax
         , GCPUAbsDiff
         , GCPUAbsDiffC
         , GCPUSum
         , GCPUCountNonZero
         , GCPUAddW
         , GCPUNormL1
         , GCPUNormL2
         , GCPUNormInf
         , GCPUIntegral
         , GCPUThreshold
         , GCPUThresholdOT
         , GCPUInRange
         , GCPUSplit3
         , GCPUSplit4
         , GCPUMerge3
         , GCPUMerge4
         , GCPURemap
         , GCPUFlip
         , GCPUCrop
         , GCPUConcatHor
         , GCPUConcatVert
         , GCPULUT
         , GCPUConvertTo
         , GCPUSqrt
         , GCPUNormalize
         , GCPUWarpPerspective
         , GCPUWarpAffine
         , GCPUKMeansND
         , GCPUKMeansNDNoInit
         , GCPUKMeans2D
         , GCPUKMeans3D
         , GCPUTranspose
         , GCPUParseSSDBL
         , GOCVParseSSD
         , GCPUParseYolo
         , GCPUSize
         , GCPUSizeR
         , GCPUSizeMF
        >();
    return pkg;
}
