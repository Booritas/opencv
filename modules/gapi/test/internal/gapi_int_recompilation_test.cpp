// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_tests_common.hpp"
#include "api/gcomputation_priv.hpp"

#include <opencv2/gapi/fluid/gfluidkernel.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>

namespace opencv_test
{

TEST(GComputationCompile, NoRecompileWithSameMeta)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, in+in);

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye  (32, 32, CV_8UC1);
    ncvslideio::Mat in_mat2 = ncvslideio::Mat::zeros(32, 32, CV_8UC1);
    ncvslideio::Mat out_mat;

    cc.apply(in_mat1, out_mat);
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, NoRecompileWithWrongMeta)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, in+in);

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye  (32, 32, CV_8UC1);
    ncvslideio::Mat in_mat2 = ncvslideio::Mat::zeros(32, 32, CV_8UC1);
    ncvslideio::Mat out_mat;

    cc.apply(in_mat1, out_mat);
    auto comp1 = cc.priv().m_lastCompiled;

    EXPECT_THROW(cc.apply(ncvslideio::gin(ncvslideio::Scalar(128)), ncvslideio::gout(out_mat)), std::logic_error);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, RecompileWithDifferentMeta)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, in+in);

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye  (32, 32, CV_8UC1);
    ncvslideio::Mat in_mat2 = ncvslideio::Mat::zeros(64, 64, CV_32F);
    ncvslideio::Mat out_mat;

    cc.apply(in_mat1, out_mat);
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are different
    EXPECT_NE(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, FluidReshapeWithDifferentDims)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, in+in);

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye  (32, 32, CV_8UC1);
    ncvslideio::Mat in_mat2 = ncvslideio::Mat::zeros(64, 64, CV_8UC1);
    ncvslideio::Mat out_mat;

    cc.apply(in_mat1, out_mat, ncvslideio::compile_args(ncvslideio::gapi::core::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());
}

TEST(GComputationCompile, FluidReshapeResizeDownScale)
{
    ncvslideio::Size szOut(4, 4);
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, ncvslideio::gapi::resize(in, szOut));

    ncvslideio::Mat in_mat1( 8,  8, CV_8UC3);
    ncvslideio::Mat in_mat2(16, 16, CV_8UC3);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Mat out_mat1, out_mat2;

    cc.apply(in_mat1, out_mat1, ncvslideio::compile_args(ncvslideio::gapi::imgproc::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat2);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());

    ncvslideio::Mat cv_out_mat1, cv_out_mat2;
    ncvslideio::resize(in_mat1, cv_out_mat1, szOut);
    ncvslideio::resize(in_mat2, cv_out_mat2, szOut);
    // Fluid's and OpenCV's resizes aren't bit exact.
    // So 1 is here because it is max difference between them.
    EXPECT_TRUE(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()(out_mat1, cv_out_mat1));
    EXPECT_TRUE(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()(out_mat2, cv_out_mat2));
}

TEST(GComputationCompile, FluidReshapeSwitchToUpscaleFromDownscale)
{
    ncvslideio::Size szOut(4, 4);
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, ncvslideio::gapi::resize(in, szOut));

    ncvslideio::Mat in_mat1( 8,  8, CV_8UC3);
    ncvslideio::Mat in_mat2( 2,  2, CV_8UC3);
    ncvslideio::Mat in_mat3(16, 16, CV_8UC3);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat3, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Mat out_mat1, out_mat2, out_mat3;

    cc.apply(in_mat1, out_mat1, ncvslideio::compile_args(ncvslideio::gapi::imgproc::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat2);
    auto comp2 = cc.priv().m_lastCompiled;

    cc.apply(in_mat3, out_mat3);
    auto comp3 = cc.priv().m_lastCompiled;

    EXPECT_EQ(&comp1.priv(), &comp2.priv());
    EXPECT_EQ(&comp1.priv(), &comp3.priv());

    ncvslideio::Mat cv_out_mat1, cv_out_mat2, cv_out_mat3;
    ncvslideio::resize(in_mat1, cv_out_mat1, szOut);
    ncvslideio::resize(in_mat2, cv_out_mat2, szOut);
    ncvslideio::resize(in_mat3, cv_out_mat3, szOut);
    // Fluid's and OpenCV's Resizes aren't bit exact.
    // So 1 is here because it is max difference between them.
    EXPECT_TRUE(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()(out_mat1, cv_out_mat1));
    EXPECT_TRUE(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()(out_mat2, cv_out_mat2));
    EXPECT_TRUE(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()(out_mat3, cv_out_mat3));
}

TEST(GComputationCompile, ReshapeBlur)
{
    ncvslideio::Size kernelSize{3, 3};
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, ncvslideio::gapi::blur(in, kernelSize));

    ncvslideio::Mat in_mat1( 8,  8, CV_8UC1);
    ncvslideio::Mat in_mat2(16, 16, CV_8UC1);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Mat out_mat1, out_mat2;

    cc.apply(in_mat1, out_mat1, ncvslideio::compile_args(ncvslideio::gapi::imgproc::fluid::kernels()));
    auto comp1 = cc.priv().m_lastCompiled;

    cc.apply(in_mat2, out_mat2);
    auto comp2 = cc.priv().m_lastCompiled;

    // Both compiled objects are actually the same unique executable
    EXPECT_EQ(&comp1.priv(), &comp2.priv());

    ncvslideio::Mat cv_out_mat1, cv_out_mat2;
    ncvslideio::blur(in_mat1, cv_out_mat1, kernelSize);
    ncvslideio::blur(in_mat2, cv_out_mat2, kernelSize);

    EXPECT_EQ(0, cvtest::norm(out_mat1, cv_out_mat1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat2, cv_out_mat2, NORM_INF));
}

TEST(GComputationCompile, ReshapeRois)
{
    ncvslideio::Size kernelSize{3, 3};
    ncvslideio::Size szOut(8, 8);
    ncvslideio::GMat in;
    auto blurred = ncvslideio::gapi::blur(in, kernelSize);
    ncvslideio::GComputation cc(in, ncvslideio::gapi::resize(blurred, szOut));

    ncvslideio::Mat first_in_mat(8, 8, CV_8UC3);
    ncvslideio::randn(first_in_mat, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));
    ncvslideio::Mat first_out_mat;
    auto fluidKernels = ncvslideio::gapi::combine(gapi::imgproc::fluid::kernels(),
                                          gapi::core::fluid::kernels());
    cc.apply(first_in_mat, first_out_mat, ncvslideio::compile_args(fluidKernels));
    auto first_comp = cc.priv().m_lastCompiled;

    constexpr int niter = 4;
    for (int i = 0; i < niter; i++)
    {
        int width  = 4 + 2*i;
        int height = width;
        ncvslideio::Mat in_mat(width, height, CV_8UC3);
        ncvslideio::randn(in_mat, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));
        ncvslideio::Mat out_mat = ncvslideio::Mat::zeros(szOut, CV_8UC3);

        int x = 0;
        int y = szOut.height * i / niter;
        int roiW = szOut.width;
        int roiH = szOut.height / niter;
        ncvslideio::Rect roi{x, y, roiW, roiH};

        cc.apply(in_mat, out_mat, ncvslideio::compile_args(ncvslideio::GFluidOutputRois{{roi}}));
        auto comp = cc.priv().m_lastCompiled;

        EXPECT_EQ(&first_comp.priv(), &comp.priv());

        ncvslideio::Mat blur_mat, cv_out_mat;
        ncvslideio::blur(in_mat, blur_mat, kernelSize);
        ncvslideio::resize(blur_mat, cv_out_mat, szOut);
        // Fluid's and OpenCV's resizes aren't bit exact.
        // So 1 is here because it is max difference between them.
        EXPECT_TRUE(Tolerance_FloatRel_IntAbs(1e-5, 1).to_compare_f()(out_mat(roi), cv_out_mat(roi)));
    }
}

} // opencv_test
