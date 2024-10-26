// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"
#include "logger.hpp"

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/util/throw.hpp>

#include "backends/ocl/goclcore.hpp"

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#pragma comment(lib,"d3d11.lib")

// get rid of generate macro max/min/etc from DX side
#define D3D11_NO_HELPERS
#define NOMINMAX
#include <d3d11.h>
#pragma comment(lib, "dxgi")
#undef NOMINMAX
#undef D3D11_NO_HELPERS
#include <opencv2/core/directx.hpp>
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX

#include <opencv2/core/ocl.hpp>
#include "streaming/onevpl/accelerators/surface/dx11_frame_adapter.hpp"

GAPI_OCL_KERNEL(GOCLAdd, ncvslideio::gapi::core::GAdd)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::add(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLAddC, ncvslideio::gapi::core::GAddC)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::add(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLSub, ncvslideio::gapi::core::GSub)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::subtract(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLSubC, ncvslideio::gapi::core::GSubC)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::subtract(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLSubRC, ncvslideio::gapi::core::GSubRC)
{
    static void run(const ncvslideio::Scalar& a, const ncvslideio::UMat& b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::subtract(a, b, out, ncvslideio::noArray(), dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMul, ncvslideio::gapi::core::GMul)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, double scale, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::multiply(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMulCOld, ncvslideio::gapi::core::GMulCOld)
{
    static void run(const ncvslideio::UMat& a, double b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMulC, ncvslideio::gapi::core::GMulC)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::multiply(a, b, out, 1, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLDiv, ncvslideio::gapi::core::GDiv)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, double scale, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLDivC, ncvslideio::gapi::core::GDivC)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, double scale, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLDivRC, ncvslideio::gapi::core::GDivRC)
{
    static void run(const ncvslideio::Scalar& a, const ncvslideio::UMat& b, double scale, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::divide(a, b, out, scale, dtype);
    }
};

GAPI_OCL_KERNEL(GOCLMask, ncvslideio::gapi::core::GMask)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::UMat& mask, ncvslideio::UMat& out)
    {
        out = ncvslideio::UMat::zeros(in.size(), in.type());
        in.copyTo(out, mask);
    }
};


GAPI_OCL_KERNEL(GOCLMean, ncvslideio::gapi::core::GMean)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::mean(in);
    }
};

GAPI_OCL_KERNEL(GOCLPolarToCart, ncvslideio::gapi::core::GPolarToCart)
{
    static void run(const ncvslideio::UMat& magn, const ncvslideio::UMat& angle, bool angleInDegrees, ncvslideio::UMat& outx, ncvslideio::UMat& outy)
    {
        ncvslideio::polarToCart(magn, angle, outx, outy, angleInDegrees);
    }
};

GAPI_OCL_KERNEL(GOCLCartToPolar, ncvslideio::gapi::core::GCartToPolar)
{
    static void run(const ncvslideio::UMat& x, const ncvslideio::UMat& y, bool angleInDegrees, ncvslideio::UMat& outmagn, ncvslideio::UMat& outangle)
    {
        ncvslideio::cartToPolar(x, y, outmagn, outangle, angleInDegrees);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGT, ncvslideio::gapi::core::GCmpGT)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGE, ncvslideio::gapi::core::GCmpGE)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLE, ncvslideio::gapi::core::GCmpLE)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLT, ncvslideio::gapi::core::GCmpLT)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpEQ, ncvslideio::gapi::core::GCmpEQ)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_EQ);
    }
};

GAPI_OCL_KERNEL(GOCLCmpNE, ncvslideio::gapi::core::GCmpNE)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_NE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGTScalar, ncvslideio::gapi::core::GCmpGTScalar)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpGEScalar, ncvslideio::gapi::core::GCmpGEScalar)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_GE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLEScalar, ncvslideio::gapi::core::GCmpLEScalar)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LE);
    }
};

GAPI_OCL_KERNEL(GOCLCmpLTScalar, ncvslideio::gapi::core::GCmpLTScalar)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_LT);
    }
};

GAPI_OCL_KERNEL(GOCLCmpEQScalar, ncvslideio::gapi::core::GCmpEQScalar)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_EQ);
    }
};

GAPI_OCL_KERNEL(GOCLCmpNEScalar, ncvslideio::gapi::core::GCmpNEScalar)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::compare(a, b, out, ncvslideio::CMP_NE);
    }
};

GAPI_OCL_KERNEL(GOCLAnd, ncvslideio::gapi::core::GAnd)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_and(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLAndS, ncvslideio::gapi::core::GAndS)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_and(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLOr, ncvslideio::gapi::core::GOr)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_or(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLOrS, ncvslideio::gapi::core::GOrS)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_or(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLXor, ncvslideio::gapi::core::GXor)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::UMat& b, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_xor(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLXorS, ncvslideio::gapi::core::GXorS)
{
    static void run(const ncvslideio::UMat& a, const ncvslideio::Scalar& b, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_xor(a, b, out);
    }
};

GAPI_OCL_KERNEL(GOCLNot, ncvslideio::gapi::core::GNot)
{
    static void run(const ncvslideio::UMat& a, ncvslideio::UMat& out)
    {
        ncvslideio::bitwise_not(a, out);
    }
};

GAPI_OCL_KERNEL(GOCLSelect, ncvslideio::gapi::core::GSelect)
{
    static void run(const ncvslideio::UMat& src1, const ncvslideio::UMat& src2, const ncvslideio::UMat& mask, ncvslideio::UMat& out)
    {
        src2.copyTo(out);
        src1.copyTo(out, mask);
    }
};

////TODO: doesn't compiled with UMat
//GAPI_OCL_KERNEL(GOCLMin, ncvslideio::gapi::core::GMin)
//{
//    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, ncvslideio::UMat& out)
//    {
//        out = ncvslideio::min(in1, in2);
//    }
//};
//
////TODO: doesn't compiled with UMat
//GAPI_OCL_KERNEL(GOCLMax, ncvslideio::gapi::core::GMax)
//{
//    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, ncvslideio::UMat& out)
//    {
//        out = ncvslideio::max(in1, in2);
//    }
//};


GAPI_OCL_KERNEL(GOCLAbsDiff, ncvslideio::gapi::core::GAbsDiff)
{
    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, ncvslideio::UMat& out)
    {
        ncvslideio::absdiff(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLAbsDiffC, ncvslideio::gapi::core::GAbsDiffC)
{
    static void run(const ncvslideio::UMat& in1, const ncvslideio::Scalar& in2, ncvslideio::UMat& out)
    {
        ncvslideio::absdiff(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLSum, ncvslideio::gapi::core::GSum)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::sum(in);
    }
};

GAPI_OCL_KERNEL(GOCLCountNonZero, ncvslideio::gapi::core::GCountNonZero)
{
    static void run(const ncvslideio::UMat& in, int& out)
    {
        out = ncvslideio::countNonZero(in);
    }
};

GAPI_OCL_KERNEL(GOCLAddW, ncvslideio::gapi::core::GAddW)
{
    static void run(const ncvslideio::UMat& in1, double alpha, const ncvslideio::UMat& in2, double beta, double gamma, int dtype, ncvslideio::UMat& out)
    {
        ncvslideio::addWeighted(in1, alpha, in2, beta, gamma, out, dtype);
    }
};


GAPI_OCL_KERNEL(GOCLNormL1, ncvslideio::gapi::core::GNormL1)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::norm(in, ncvslideio::NORM_L1);
    }
};

GAPI_OCL_KERNEL(GOCLNormL2, ncvslideio::gapi::core::GNormL2)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::norm(in, ncvslideio::NORM_L2);
    }
};

GAPI_OCL_KERNEL(GOCLNormInf, ncvslideio::gapi::core::GNormInf)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Scalar& out)
    {
        out = ncvslideio::norm(in, ncvslideio::NORM_INF);
    }
};

GAPI_OCL_KERNEL(GOCLIntegral, ncvslideio::gapi::core::GIntegral)
{
    static void run(const ncvslideio::UMat& in, int sdepth, int sqdepth, ncvslideio::UMat& out, ncvslideio::UMat& outSq)
    {
        ncvslideio::integral(in, out, outSq, sdepth, sqdepth);
    }
};

GAPI_OCL_KERNEL(GOCLThreshold, ncvslideio::gapi::core::GThreshold)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Scalar& a, const ncvslideio::Scalar& b, int type, ncvslideio::UMat& out)
    {
        ncvslideio::threshold(in, out, a.val[0], b.val[0], type);
    }
};

GAPI_OCL_KERNEL(GOCLThresholdOT, ncvslideio::gapi::core::GThresholdOT)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Scalar& b, int type, ncvslideio::UMat& out, ncvslideio::Scalar& outScalar)
    {
        outScalar = ncvslideio::threshold(in, out, b.val[0], b.val[0], type);
    }
};


GAPI_OCL_KERNEL(GOCLInRange, ncvslideio::gapi::core::GInRange)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Scalar& low, const ncvslideio::Scalar& up, ncvslideio::UMat& out)
    {
        ncvslideio::inRange(in, low, up, out);
    }
};

GAPI_OCL_KERNEL(GOCLSplit3, ncvslideio::gapi::core::GSplit3)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &m1, ncvslideio::UMat &m2, ncvslideio::UMat &m3)
    {
        std::vector<ncvslideio::UMat> outMats = {m1, m2, m3};
        ncvslideio::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
    }
};

GAPI_OCL_KERNEL(GOCLSplit4, ncvslideio::gapi::core::GSplit4)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::UMat &m1, ncvslideio::UMat &m2, ncvslideio::UMat &m3, ncvslideio::UMat &m4)
    {
        std::vector<ncvslideio::UMat> outMats = {m1, m2, m3, m4};
        ncvslideio::split(in, outMats);

        // Write back FIXME: Write a helper or avoid this nonsense completely!
        m1 = outMats[0];
        m2 = outMats[1];
        m3 = outMats[2];
        m4 = outMats[3];
    }
};

GAPI_OCL_KERNEL(GOCLMerge3, ncvslideio::gapi::core::GMerge3)
{
    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, const ncvslideio::UMat& in3, ncvslideio::UMat &out)
    {
        std::vector<ncvslideio::UMat> inMats = {in1, in2, in3};
        ncvslideio::merge(inMats, out);
    }
};

GAPI_OCL_KERNEL(GOCLMerge4, ncvslideio::gapi::core::GMerge4)
{
    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, const ncvslideio::UMat& in3, const ncvslideio::UMat& in4, ncvslideio::UMat &out)
    {
        std::vector<ncvslideio::UMat> inMats = {in1, in2, in3, in4};
        ncvslideio::merge(inMats, out);
    }
};

GAPI_OCL_KERNEL(GOCLRemap, ncvslideio::gapi::core::GRemap)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Mat& x, const ncvslideio::Mat& y, int a, int b, ncvslideio::Scalar s, ncvslideio::UMat& out)
    {
        ncvslideio::remap(in, out, x, y, a, b, s);
    }
};

GAPI_OCL_KERNEL(GOCLFlip, ncvslideio::gapi::core::GFlip)
{
    static void run(const ncvslideio::UMat& in, int code, ncvslideio::UMat& out)
    {
        ncvslideio::flip(in, out, code);
    }
};

GAPI_OCL_KERNEL(GOCLCrop, ncvslideio::gapi::core::GCrop)
{
    static void run(const ncvslideio::UMat& in, ncvslideio::Rect rect, ncvslideio::UMat& out)
    {
        ncvslideio::UMat(in, rect).copyTo(out);
    }
};

GAPI_OCL_KERNEL(GOCLConcatHor, ncvslideio::gapi::core::GConcatHor)
{
    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, ncvslideio::UMat& out)
    {
        ncvslideio::hconcat(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLConcatVert, ncvslideio::gapi::core::GConcatVert)
{
    static void run(const ncvslideio::UMat& in1, const ncvslideio::UMat& in2, ncvslideio::UMat& out)
    {
        ncvslideio::vconcat(in1, in2, out);
    }
};

GAPI_OCL_KERNEL(GOCLLUT, ncvslideio::gapi::core::GLUT)
{
    static void run(const ncvslideio::UMat& in, const ncvslideio::Mat& lut, ncvslideio::UMat& out)
    {
        ncvslideio::LUT(in, lut, out);
    }
};

GAPI_OCL_KERNEL(GOCLConvertTo, ncvslideio::gapi::core::GConvertTo)
{
    static void run(const ncvslideio::UMat& in, int rtype, double alpha, double beta, ncvslideio::UMat& out)
    {
        in.convertTo(out, rtype, alpha, beta);
    }
};


GAPI_OCL_KERNEL(GOCLTranspose, ncvslideio::gapi::core::GTranspose)
{
    static void run(const ncvslideio::UMat& in,  ncvslideio::UMat& out)
    {
        ncvslideio::transpose(in, out);
    }
};

GAPI_OCL_KERNEL(GOCLBGR, ncvslideio::gapi::streaming::GBGR)
{
    static void run(const ncvslideio::MediaFrame& in, ncvslideio::UMat& out)
    {
        ncvslideio::util::suppress_unused_warning(in);
        ncvslideio::util::suppress_unused_warning(out);
#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
#ifdef HAVE_ONEVPL
        auto d = in.desc();
        if (d.fmt != ncvslideio::MediaFormat::NV12)
        {
            GAPI_LOG_FATAL(nullptr, "Unsupported format provided: " << static_cast<int>(d.fmt) <<
                           ". Expected ncvslideio::MediaFormat::NV12.");
            ncvslideio::util::throw_error(std::logic_error("Unsupported MediaFrame format provided"));
        }

        // FIXME: consider a better solution.
        // Current approach cannot be easily extended for other adapters (getHandle).
        auto adapterPtr = in.get<ncvslideio::gapi::wip::onevpl::VPLMediaFrameDX11Adapter>();
        if (adapterPtr == nullptr)
        {
            GAPI_LOG_FATAL(nullptr, "Unsupported adapter type. Only VPLMediaFrameDX11Adapter is supported");
            ncvslideio::util::throw_error(std::logic_error("Unsupported adapter type. Only VPLMediaFrameDX11Adapter is supported"));
        }

        auto params = adapterPtr->getHandle();
        auto handle = ncvslideio::util::any_cast<mfxHDLPair>(params);
        ID3D11Texture2D* texture = reinterpret_cast<ID3D11Texture2D*>(handle.first);
        if (texture == nullptr)
        {
            GAPI_LOG_FATAL(nullptr, "mfxHDLPair contains ID3D11Texture2D that is nullptr. Handle address" <<
                           reinterpret_cast<uint64_t>(handle.first));
            ncvslideio::util::throw_error(std::logic_error("mfxHDLPair contains ID3D11Texture2D that is nullptr"));
        }

        // FIXME: Assuming here that we only have 1 device
        // TODO: Textures are reusable, so to improve the peroformance here
        //       consider creating a hash map texture <-> device/ctx
        static thread_local ID3D11Device* pD3D11Device = nullptr;
        if (pD3D11Device == nullptr)
        {
            texture->GetDevice(&pD3D11Device);
        }
        if (pD3D11Device == nullptr)
        {
            GAPI_LOG_FATAL(nullptr, "D3D11Texture2D::GetDevice returns pD3D11Device that is nullptr");
            ncvslideio::util::throw_error(std::logic_error("D3D11Texture2D::GetDevice returns pD3D11Device that is nullptr"));
        }

        // FIXME: assuming here that the context is always the same
        // TODO: Textures are reusable, so to improve the peroformance here
        //       consider creating a hash map texture <-> device/ctx
        static thread_local ncvslideio::ocl::Context ctx = ncvslideio::directx::ocl::initializeContextFromD3D11Device(pD3D11Device);
        if (ctx.ptr() == nullptr)
        {
            GAPI_LOG_FATAL(nullptr, "initializeContextFromD3D11Device returned null context");
            ncvslideio::util::throw_error(std::logic_error("initializeContextFromD3D11Device returned null context"));
        }

        ncvslideio::directx::convertFromD3D11Texture2D(texture, out);
#else
        GAPI_LOG_FATAL(nullptr, "HAVE_ONEVPL is not set. Please, check your cmake flags");
        ncvslideio::util::throw_error(std::logic_error("HAVE_ONEVPL is not set. Please, check your cmake flags"));
#endif // HAVE_ONEVPL
#else
        GAPI_LOG_FATAL(nullptr, "HAVE_D3D11 or HAVE_DIRECTX is not set. Please, check your cmake flags");
        ncvslideio::util::throw_error(std::logic_error("HAVE_D3D11 or HAVE_DIRECTX is not set. Please, check your cmake flags"));
#endif // HAVE_D3D11
#endif // HAVE_DIRECTX
    }
};

ncvslideio::GKernelPackage ncvslideio::gapi::core::ocl::kernels()
{
    static auto pkg = ncvslideio::gapi::kernels
        <  GOCLAdd
         , GOCLAddC
         , GOCLSub
         , GOCLSubC
         , GOCLSubRC
         , GOCLMul
         , GOCLMulC
         , GOCLMulCOld
         , GOCLDiv
         , GOCLDivC
         , GOCLDivRC
         , GOCLMean
         , GOCLMask
         , GOCLPolarToCart
         , GOCLCartToPolar
         , GOCLCmpGT
         , GOCLCmpGE
         , GOCLCmpLE
         , GOCLCmpLT
         , GOCLCmpEQ
         , GOCLCmpNE
         , GOCLCmpGTScalar
         , GOCLCmpGEScalar
         , GOCLCmpLEScalar
         , GOCLCmpLTScalar
         , GOCLCmpEQScalar
         , GOCLCmpNEScalar
         , GOCLAnd
         , GOCLAndS
         , GOCLOr
         , GOCLOrS
         , GOCLXor
         , GOCLXorS
         , GOCLNot
         , GOCLSelect
         //, GOCLMin
         //, GOCLMax
         , GOCLAbsDiff
         , GOCLAbsDiffC
         , GOCLSum
         , GOCLCountNonZero
         , GOCLAddW
         , GOCLNormL1
         , GOCLNormL2
         , GOCLNormInf
         , GOCLIntegral
         , GOCLThreshold
         , GOCLThresholdOT
         , GOCLInRange
         , GOCLSplit3
         , GOCLSplit4
         , GOCLMerge3
         , GOCLMerge4
         , GOCLRemap
         , GOCLFlip
         , GOCLCrop
         , GOCLConcatHor
         , GOCLConcatVert
         , GOCLLUT
         , GOCLConvertTo
         , GOCLTranspose
         , GOCLBGR
         >();
    return pkg;
}
