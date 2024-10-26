// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2024 Intel Corporation


#include "test_precomp.hpp"

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/fluid/core.hpp>

namespace opencv_test
{
namespace
{
enum class KernelPackage: int
{
    OCV,
    OCL,
    FLUID,
};
std::ostream& operator<< (std::ostream &os, const KernelPackage &e)
{
    switch (e)
    {
#define _C(X) case KernelPackage::X: os << #X; break
        _C(OCV);
        _C(OCL);
        _C(FLUID);
#undef _C
    default: GAPI_Error("Unknown package");
    }
    return os;
}
} // namespace

struct GMatWithValue : public TestWithParam <KernelPackage> {
    ncvslideio::GKernelPackage getKernelPackage() {
        switch (GetParam()) {
        case KernelPackage::OCV: return ncvslideio::gapi::core::cpu::kernels();
        case KernelPackage::OCL: return ncvslideio::gapi::core::ocl::kernels();
        case KernelPackage::FLUID: return ncvslideio::gapi::core::fluid::kernels();
        default: GAPI_Error("Unknown package");
        }
    }
};

TEST_P(GMatWithValue, SingleIsland)
{
    ncvslideio::Size sz(2, 2);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8U);

    ncvslideio::GComputationT<ncvslideio::GMat(ncvslideio::GMat)> addEye([&](ncvslideio::GMat in) {
        return in + ncvslideio::GMat(ncvslideio::Mat::eye(sz, CV_8U));
    });

    ncvslideio::Mat out_mat;
    addEye.apply(in_mat, out_mat, ncvslideio::compile_args(ncvslideio::gapi::use_only{getKernelPackage()}));

    ncvslideio::Mat out_mat_ref = in_mat*2;
    EXPECT_EQ(0, cvtest::norm(out_mat, out_mat_ref, NORM_INF));
}

TEST_P(GMatWithValue, GraphWithNoInput)
{
    ncvslideio::Mat cval = ncvslideio::Mat::eye(ncvslideio::Size(2, 2), CV_8U);
    ncvslideio::GMat gval = ncvslideio::GMat(cval);
    ncvslideio::GMat out = ncvslideio::gapi::bitwise_not(gval);

    ncvslideio::Mat out_mat;
    ncvslideio::GComputation f(ncvslideio::GIn(), ncvslideio::GOut(out));

    // Compiling this isn't supported for now
    EXPECT_ANY_THROW(f.compile(ncvslideio::descr_of(cval),
                               ncvslideio::compile_args(ncvslideio::gapi::use_only{getKernelPackage()})));
}

INSTANTIATE_TEST_CASE_P(GAPI_GMat, GMatWithValue,
                        Values(KernelPackage::OCV,
                               KernelPackage::OCL,
                               KernelPackage::FLUID));

TEST(GAPI_MatWithValue, MultipleIslands)
{
    // This test employs a non-trivial island fusion process
    // as there's multiple backends in the graph

    ncvslideio::Size sz(2, 2);
    ncvslideio::Mat cval2 = ncvslideio::Mat::eye(sz, CV_8U) * 2;
    ncvslideio::Mat cval1 = ncvslideio::Mat::eye(sz, CV_8U);

    ncvslideio::GMat in;
    ncvslideio::GMat tmp = in  + ncvslideio::GMat(cval2); // Will be a Fluid operation
    ncvslideio::GMat out = tmp - ncvslideio::GMat(cval1); // Will be an OCV operation

    ncvslideio::GKernelPackage fluid_kernels = ncvslideio::gapi::core::fluid::kernels();
    ncvslideio::GKernelPackage opencv_kernels = ncvslideio::gapi::core::cpu::kernels();
    fluid_kernels.remove<ncvslideio::gapi::core::GSub>();
    opencv_kernels.remove<ncvslideio::gapi::core::GAdd>();
    auto kernels = ncvslideio::gapi::combine(fluid_kernels, opencv_kernels);

    ncvslideio::Mat in_mat = ncvslideio::Mat::zeros(sz, CV_8U);
    ncvslideio::Mat out_mat;
    auto cc = ncvslideio::GComputation(in, out)
        .compile(ncvslideio::descr_of(in_mat),
                 ncvslideio::compile_args(ncvslideio::gapi::use_only{kernels}));
    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    EXPECT_EQ(0, cvtest::norm(out_mat, ncvslideio::Mat::eye(sz, CV_8U), NORM_INF));
}

} // namespace opencv_test
