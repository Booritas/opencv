// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP

#include <iostream>

#include "gapi_core_perf_tests.hpp"

#include "../../test/common/gapi_core_tests_common.hpp"

namespace opencv_test
{
using namespace perf;

//------------------------------------------------------------------------------

PERF_TEST_P_(PhasePerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////

    ncvslideio::phase(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::phase(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(SqrtPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::sqrt(in_mat1, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::sqrt(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

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

PERF_TEST_P_(AddPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::add(in_mat1, in_mat2, out_mat_ocv, ncvslideio::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::add(in1, in2, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // There is no need to qualify gin, gout, descr_of with namespace (ncvslideio::)
    // as they are in the same namespace as their actual argument (i.e. ncvslideio::Mat)
    // and thus are found via ADL, as in the examples below.
    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(AddCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    ncvslideio::GCompileArgs compile_args;

    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::add(in_mat1, sc, out_mat_ocv, ncvslideio::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::addC(in1, sc1, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SubPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::subtract(in_mat1, in_mat2, out_mat_ocv, ncvslideio::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::sub(in1, in2, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(SubCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    ncvslideio::GCompileArgs compile_args;

    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::subtract(in_mat1, sc, out_mat_ocv, ncvslideio::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::subC(in1, sc1, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SubRCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::subtract(sc, in_mat1, out_mat_ocv, ncvslideio::noArray(), dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::subRC(sc1, in1, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MulPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, scale, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2, scale, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(MulDoublePerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    auto& rng = ncvslideio::theRNG();
    double d = rng.uniform(0.0, 10.0);
    initMatrixRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, d, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    out = ncvslideio::gapi::mulC(in1, d, dtype);
    ncvslideio::GComputation c(in1, out);

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

PERF_TEST_P_(MulCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, sc, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::mulC(in1, sc1, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(DivPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, scale, compile_args) = GetParam();

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);

    //This condition need to workaround the #21044 issue in the OpenCV.
    //It reinitializes divider matrix without zero values for CV_16S DST type.
    if (dtype != type)
        ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(1), ncvslideio::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::divide(in_mat1, in_mat2, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::div(in1, in2, scale, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(DivCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, scale, compile_args) = GetParam();

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);

    //This condition need to workaround the #21044 issue in the OpenCV.
    //It reinitializes divider scalar without zero values for CV_16S DST type.
    if (dtype == CV_16S || (type == CV_16S && dtype == -1))
        ncvslideio::randu(sc, ncvslideio::Scalar::all(1), ncvslideio::Scalar::all(SHRT_MAX));

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::divide(in_mat1, sc, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::divC(in1, sc1, scale, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(DivRCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    double scale = 1.0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, scale, compile_args) = GetParam();

    // FIXIT Unstable input data for divide
    initMatsRandU(type, sz, dtype, false);
    //This condition need to workaround the #21044 issue in the OpenCV.
    //It reinitializes divider matrix without zero values for CV_16S DST type.
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(1), ncvslideio::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::divide(sc, in_mat1, out_mat_ocv, scale, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::divRC(sc1, in1, scale, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MaskPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);
    in_mat2 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    in_mat2 = in_mat2 > 128;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_mat_ocv = ncvslideio::Mat::zeros(in_mat1.size(), in_mat1.type());
    in_mat1.copyTo(out_mat_ocv, in_mat2);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in, m;
    auto out = ncvslideio::gapi::mask(in, m);
    ncvslideio::GComputation c(ncvslideio::GIn(in, m), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(MeanPerfTest, TestPerformance)
{
    compare_scalar_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, false);
    ncvslideio::Scalar out_norm;
    ncvslideio::Scalar out_norm_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_norm_ocv = ncvslideio::mean(in_mat1);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::mean(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_norm));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_norm));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_norm, out_norm_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Polar2CartPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatsRandU(CV_32FC1, sz, CV_32FC1, false);
    ncvslideio::Mat out_mat2;
    ncvslideio::Mat out_mat_ocv2;

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::polarToCart(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = ncvslideio::gapi::polarToCart(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out1, out2));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));
    }
    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat2, out_mat_ocv2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Cart2PolarPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatsRandU(CV_32FC1, sz, CV_32FC1, false);
    ncvslideio::Mat out_mat2(sz, CV_32FC1);
    ncvslideio::Mat out_mat_ocv2(sz, CV_32FC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::cartToPolar(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = ncvslideio::gapi::cartToPolar(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out1, out2));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2), gout(out_mat_gapi, out_mat2));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat2, out_mat_ocv2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(CmpPerfTest, TestPerformance)
{
    compare_f cmpF;
    CmpTypes opType = CMP_EQ;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, opType, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, CV_8U, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::compare(in_mat1, in_mat2, out_mat_ocv, opType);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    switch (opType)
    {
    case CMP_EQ: out = ncvslideio::gapi::cmpEQ(in1, in2); break;
    case CMP_GT: out = ncvslideio::gapi::cmpGT(in1, in2); break;
    case CMP_GE: out = ncvslideio::gapi::cmpGE(in1, in2); break;
    case CMP_LT: out = ncvslideio::gapi::cmpLT(in1, in2); break;
    case CMP_LE: out = ncvslideio::gapi::cmpLE(in1, in2); break;
    case CMP_NE: out = ncvslideio::gapi::cmpNE(in1, in2); break;
    default: FAIL() << "no such compare operation type for two matrices!";
    }
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(CmpWithScalarPerfTest, TestPerformance)
{
    compare_f cmpF;
    CmpTypes opType = CMP_EQ;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, opType, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, CV_8U, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::compare(in_mat1, sc, out_mat_ocv, opType);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar in2;
    switch (opType)
    {
    case CMP_EQ: out = ncvslideio::gapi::cmpEQ(in1, in2); break;
    case CMP_GT: out = ncvslideio::gapi::cmpGT(in1, in2); break;
    case CMP_GE: out = ncvslideio::gapi::cmpGE(in1, in2); break;
    case CMP_LT: out = ncvslideio::gapi::cmpLT(in1, in2); break;
    case CMP_LE: out = ncvslideio::gapi::cmpLE(in1, in2); break;
    case CMP_NE: out = ncvslideio::gapi::cmpNE(in1, in2); break;
    default: FAIL() << "no such compare operation type for matrix and scalar!";
    }
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BitwisePerfTest, TestPerformance)
{
    compare_f cmpF;
    bitwiseOp opType = AND;
    bool testWithScalar = false;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, opType, testWithScalar, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // G-API code & corresponding OpenCV code ////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    if( testWithScalar )
    {
        ncvslideio::GScalar sc1;
        switch (opType)
        {
        case AND:
            out = ncvslideio::gapi::bitwise_and(in1, sc1);
            ncvslideio::bitwise_and(in_mat1, sc, out_mat_ocv);
            break;
        case OR:
            out = ncvslideio::gapi::bitwise_or(in1, sc1);
            ncvslideio::bitwise_or(in_mat1, sc, out_mat_ocv);
            break;
        case XOR:
            out = ncvslideio::gapi::bitwise_xor(in1, sc1);
            ncvslideio::bitwise_xor(in_mat1, sc, out_mat_ocv);
            break;
        default:
            FAIL() << "no such bitwise operation type!";
        }
        ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

        // Warm-up graph engine:
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), std::move(compile_args));

        TEST_CYCLE()
        {
            c.apply(gin(in_mat1, sc), gout(out_mat_gapi));
        }
    }
    else
    {
        switch (opType)
        {
        case AND:
            out = ncvslideio::gapi::bitwise_and(in1, in2);
            ncvslideio::bitwise_and(in_mat1, in_mat2, out_mat_ocv);
            break;
        case OR:
            out = ncvslideio::gapi::bitwise_or(in1, in2);
            ncvslideio::bitwise_or(in_mat1, in_mat2, out_mat_ocv);
            break;
        case XOR:
            out = ncvslideio::gapi::bitwise_xor(in1, in2);
            ncvslideio::bitwise_xor(in_mat1, in_mat2, out_mat_ocv);
            break;
        default:
            FAIL() << "no such bitwise operation type!";
        }
        ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

        // Warm-up graph engine:
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), std::move(compile_args));

        TEST_CYCLE()
        {
            c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi));
        }
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(BitwiseNotPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::bitwise_not(in_mat1, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in, out;
    out = ncvslideio::gapi::bitwise_not(in);
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

PERF_TEST_P_(SelectPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);
    ncvslideio::Mat in_mask(sz, CV_8UC1);
    ncvslideio::randu(in_mask, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    in_mat2.copyTo(out_mat_ocv);
    in_mat1.copyTo(out_mat_ocv, in_mask);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, in3, out;
    out = ncvslideio::gapi::select(in1, in2, in3);
    ncvslideio::GComputation c(GIn(in1, in2, in3), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mask)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(MinPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::min(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::min(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(MaxPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::max(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::max(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(AbsDiffPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::absdiff(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::absDiff(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(AbsDiffCPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::absdiff(in_mat1, sc, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar sc1;
    out = ncvslideio::gapi::absDiffC(in1, sc1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, sc1), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, sc)),
                        std::move(compile_args));
    cc(gin(in_mat1, sc), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, sc), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SumPerfTest, TestPerformance)
{
    compare_scalar_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);
    ncvslideio::Scalar out_sum;
    ncvslideio::Scalar out_sum_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_sum_ocv = ncvslideio::sum(in_mat1);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::sum(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_sum));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_sum));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_sum, out_sum_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------
#pragma push_macro("countNonZero")
#undef countNonZero
PERF_TEST_P_(CountNonZeroPerfTest, TestPerformance)
{
    compare_scalar_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);
    int out_cnz_gapi, out_cnz_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_cnz_ocv = ncvslideio::countNonZero(in_mat1);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::countNonZero(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    // Warm-up graph engine:
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_cnz_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_cnz_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_cnz_gapi, out_cnz_ocv));
    }

    SANITY_CHECK_NOTHING();
}
#pragma pop_macro("countNonZero")
//------------------------------------------------------------------------------

PERF_TEST_P_(AddWeightedPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int dtype = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, dtype, compile_args) = GetParam();

    auto& rng = ncvslideio::theRNG();
    double alpha = rng.uniform(0.0, 1.0);
    double beta = rng.uniform(0.0, 1.0);
    double gamma = rng.uniform(0.0, 1.0);
    initMatsRandU(type, sz, dtype, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::addWeighted(in_mat1, alpha, in_mat2, beta, gamma, out_mat_ocv, dtype);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::addWeighted(in1, alpha, in2, beta, gamma, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(NormPerfTest, TestPerformance)
{
    compare_scalar_f cmpF;
    NormTypes opType = NORM_INF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, opType, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);
    ncvslideio::Scalar out_norm;
    ncvslideio::Scalar out_norm_ocv;

    // OpenCV code ///////////////////////////////////////////////////////////
    out_norm_ocv = ncvslideio::norm(in_mat1, opType);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    ncvslideio::GScalar out;
    switch (opType)
    {
    case NORM_L1: out = ncvslideio::gapi::normL1(in1); break;
    case NORM_L2: out = ncvslideio::gapi::normL2(in1); break;
    case NORM_INF: out = ncvslideio::gapi::normInf(in1); break;
    default: FAIL() << "no such norm operation type!";
    }
    ncvslideio::GComputation c(GIn(in1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_norm));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_norm));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_norm, out_norm_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(IntegralPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    MatType type_out = (type == CV_8U) ? CV_32SC1 : CV_64FC1;

    in_mat1 = ncvslideio::Mat(sz, type);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    ncvslideio::Size sz_out = ncvslideio::Size(sz.width + 1, sz.height + 1);
    ncvslideio::Mat out_mat1(sz_out, type_out);
    ncvslideio::Mat out_mat_ocv1(sz_out, type_out);

    ncvslideio::Mat out_mat2(sz_out, CV_64FC1);
    ncvslideio::Mat out_mat_ocv2(sz_out, CV_64FC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::integral(in_mat1, out_mat_ocv1, out_mat_ocv2);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out1, out2;
    std::tie(out1, out2) = ncvslideio::gapi::integral(in1, type_out, CV_64FC1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1), ncvslideio::GOut(out1, out2));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat1, out_mat2));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat1, out_mat2));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat1, out_mat_ocv1));
        EXPECT_TRUE(cmpF(out_mat2, out_mat_ocv2));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ThresholdPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int tt = 0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, tt, compile_args) = GetParam();

    ncvslideio::Scalar thr = initScalarRandU(50);
    ncvslideio::Scalar maxval = initScalarRandU(50) + ncvslideio::Scalar(50, 50, 50, 50);
    initMatrixRandU(type, sz, type, false);
    ncvslideio::Scalar out_scalar;

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::threshold(in_mat1, out_mat_ocv, thr.val[0], maxval.val[0], tt);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar th1, mv1;
    out = ncvslideio::gapi::threshold(in1, th1, mv1, tt);
    ncvslideio::GComputation c(GIn(in1, th1, mv1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, thr, maxval)),
                        std::move(compile_args));
    cc(gin(in_mat1, thr, maxval), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, thr, maxval), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ThresholdOTPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int tt = 0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, tt, compile_args) = GetParam();

    ncvslideio::Scalar maxval = initScalarRandU(50) + ncvslideio::Scalar(50, 50, 50, 50);
    initMatrixRandU(type, sz, type, false);
    ncvslideio::Scalar out_gapi_scalar;
    double ocv_res;

    // OpenCV code ///////////////////////////////////////////////////////////
    ocv_res = ncvslideio::threshold(in_mat1, out_mat_ocv, maxval.val[0], maxval.val[0], tt);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar mv1, scout;
    std::tie<ncvslideio::GMat, ncvslideio::GScalar>(out, scout) = ncvslideio::gapi::threshold(in1, mv1, tt);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, mv1), ncvslideio::GOut(out, scout));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, maxval)),
                        std::move(compile_args));
    cc(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(ocv_res, out_gapi_scalar.val[0]);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(InRangePerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    ncvslideio::Scalar thrLow = initScalarRandU(100);
    ncvslideio::Scalar thrUp = initScalarRandU(100) + ncvslideio::Scalar(100, 100, 100, 100);
    initMatrixRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::inRange(in_mat1, thrLow, thrUp, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    ncvslideio::GScalar th1, mv1;
    auto out = ncvslideio::gapi::inRange(in1, th1, mv1);
    ncvslideio::GComputation c(GIn(in1, th1, mv1), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, thrLow, thrUp)),
                        std::move(compile_args));
    cc(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Split3PerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandU(CV_8UC3, sz, CV_8UC1);
    ncvslideio::Mat out_mat2 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat3 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv2 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv3 = ncvslideio::Mat(sz, CV_8UC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<ncvslideio::Mat> out_mats_ocv = { out_mat_ocv, out_mat_ocv2, out_mat_ocv3 };
    ncvslideio::split(in_mat1, out_mats_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out1, out2, out3;
    std::tie(out1, out2, out3) = ncvslideio::gapi::split3(in1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1), ncvslideio::GOut(out1, out2, out3));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat2, out_mat_ocv2));
        EXPECT_TRUE(cmpF(out_mat3, out_mat_ocv3));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Split4PerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatrixRandU(CV_8UC4, sz, CV_8UC1);
    ncvslideio::Mat out_mat2 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat3 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat4 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv2 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv3 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv4 = ncvslideio::Mat(sz, CV_8UC1);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<ncvslideio::Mat> out_mats_ocv = { out_mat_ocv, out_mat_ocv2, out_mat_ocv3, out_mat_ocv4 };
    ncvslideio::split(in_mat1, out_mats_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4) = ncvslideio::gapi::split4(in1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1), ncvslideio::GOut(out1, out2, out3, out4));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1)),
                        std::move(compile_args));
    cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3, out_mat4));

    TEST_CYCLE()
    {
        cc(gin(in_mat1), gout(out_mat_gapi, out_mat2, out_mat3, out_mat4));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_TRUE(cmpF(out_mat2, out_mat_ocv2));
        EXPECT_TRUE(cmpF(out_mat3, out_mat_ocv3));
        EXPECT_TRUE(cmpF(out_mat4, out_mat_ocv4));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Merge3PerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatsRandU(type, sz, CV_MAKETYPE(type, 3));
    ncvslideio::Mat in_mat3(sz, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);
    ncvslideio::randn(in_mat3, mean, stddev);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<ncvslideio::Mat> in_mats_ocv = { in_mat1, in_mat2, in_mat3 };
    ncvslideio::merge(in_mats_ocv, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, in3;
    auto out = ncvslideio::gapi::merge3(in1, in2, in3);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(Merge4PerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, compile_args) = GetParam();

    initMatsRandU(CV_8UC1, sz, CV_8UC3);
    ncvslideio::Mat in_mat3(sz, CV_8UC1);
    ncvslideio::Mat in_mat4(sz, CV_8UC1);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);
    ncvslideio::randn(in_mat3, mean, stddev);
    ncvslideio::randn(in_mat4, mean, stddev);

    // OpenCV code ///////////////////////////////////////////////////////////
    std::vector<ncvslideio::Mat> in_mats_ocv = { in_mat1, in_mat2, in_mat3, in_mat4 };
    ncvslideio::merge(in_mats_ocv, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, in3, in4;
    auto out = ncvslideio::gapi::merge4(in1, in2, in3, in4);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3, in4), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3, in_mat4)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3, in_mat4), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3, in_mat4), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(RemapPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);
    ncvslideio::Mat in_map1(sz, CV_16SC2);
    ncvslideio::Mat in_map2 = ncvslideio::Mat();
    ncvslideio::randu(in_map1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Scalar bv = ncvslideio::Scalar();

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::remap(in_mat1, out_mat_ocv, in_map1, in_map2, ncvslideio::INTER_NEAREST, ncvslideio::BORDER_REPLICATE, bv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    auto out = ncvslideio::gapi::remap(in1, in_map1, in_map2, ncvslideio::INTER_NEAREST, ncvslideio::BORDER_REPLICATE, bv);
    ncvslideio::GComputation c(in1, out);

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

PERF_TEST_P_(FlipPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    int flipCode = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, flipCode, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::flip(in_mat1, out_mat_ocv, flipCode);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::flip(in, flipCode);
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

PERF_TEST_P_(CropPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::Rect rect_to;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, rect_to, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::Mat(in_mat1, rect_to).copyTo(out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::crop(in, rect_to);
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

PERF_TEST_P_(CopyPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::Mat(in_mat1).copyTo(out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::copy(in);
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

PERF_TEST_P_(ConcatHorPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    int wpart = sz.width / 4;

    ncvslideio::Size sz_in1 = ncvslideio::Size(wpart, sz.height);
    ncvslideio::Size sz_in2 = ncvslideio::Size(sz.width - wpart, sz.height);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);

    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);

    out_mat_gapi = ncvslideio::Mat(sz, type);
    out_mat_ocv = ncvslideio::Mat(sz, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::hconcat(in_mat1, in_mat2, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::concatHor(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(ConcatHorVecPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    int wpart1 = sz.width / 3;
    int wpart2 = sz.width / 2;

    ncvslideio::Size sz_in1 = ncvslideio::Size(wpart1, sz.height);
    ncvslideio::Size sz_in2 = ncvslideio::Size(wpart2, sz.height);
    ncvslideio::Size sz_in3 = ncvslideio::Size(sz.width - wpart1 - wpart2, sz.height);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);
    ncvslideio::Mat in_mat3(sz_in3, type);

    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);
    ncvslideio::randn(in_mat3, mean, stddev);

    out_mat_gapi = ncvslideio::Mat(sz, type);
    out_mat_ocv = ncvslideio::Mat(sz, type);

    std::vector <ncvslideio::Mat> cvmats = { in_mat1, in_mat2, in_mat3 };

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::hconcat(cvmats, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <ncvslideio::GMat> mats(3);
    auto out = ncvslideio::gapi::concatHor(mats);
    ncvslideio::GComputation c({ mats[0], mats[1], mats[2] }, { out });

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ConcatVertPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    int hpart = sz.height * 2 / 3;

    ncvslideio::Size sz_in1 = ncvslideio::Size(sz.width, hpart);
    ncvslideio::Size sz_in2 = ncvslideio::Size(sz.width, sz.height - hpart);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);

    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);

    out_mat_gapi = ncvslideio::Mat(sz, type);
    out_mat_ocv = ncvslideio::Mat(sz, type);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::vconcat(in_mat1, in_mat2, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::concatVert(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2)),
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

PERF_TEST_P_(ConcatVertVecPerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    int hpart1 = sz.height * 2 / 5;
    int hpart2 = sz.height / 5;

    ncvslideio::Size sz_in1 = ncvslideio::Size(sz.width, hpart1);
    ncvslideio::Size sz_in2 = ncvslideio::Size(sz.width, hpart2);
    ncvslideio::Size sz_in3 = ncvslideio::Size(sz.width, sz.height - hpart1 - hpart2);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);
    ncvslideio::Mat in_mat3(sz_in3, type);

    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);
    ncvslideio::randn(in_mat3, mean, stddev);

    out_mat_gapi = ncvslideio::Mat(sz, type);
    out_mat_ocv = ncvslideio::Mat(sz, type);

    std::vector <ncvslideio::Mat> cvmats = { in_mat1, in_mat2, in_mat3 };

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::vconcat(cvmats, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <ncvslideio::GMat> mats(3);
    auto out = ncvslideio::gapi::concatVert(mats);
    ncvslideio::GComputation c({ mats[0], mats[1], mats[2] }, { out });

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(gin(in_mat1, in_mat2, in_mat3)),
                        std::move(compile_args));
    cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));

    TEST_CYCLE()
    {
        cc(gin(in_mat1, in_mat2, in_mat3), gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(LUTPerfTest, TestPerformance)
{
    compare_f cmpF;
    MatType type_mat = -1;
    MatType type_lut = -1;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type_mat, type_lut, sz, compile_args) = GetParam();

    MatType type_out = CV_MAKETYPE(CV_MAT_DEPTH(type_lut), CV_MAT_CN(type_mat));

    initMatrixRandU(type_mat, sz, type_out);
    ncvslideio::Size sz_lut = ncvslideio::Size(1, 256);
    ncvslideio::Mat in_lut(sz_lut, type_lut);
    ncvslideio::randu(in_lut, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::LUT(in_mat1, in_lut, out_mat_ocv);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::LUT(in, in_lut);
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

PERF_TEST_P_(ConvertToPerfTest, TestPerformance)
{
    int depth_to     = -1;
    MatType type_mat = -1;
    double alpha = 0., beta = 0.;
    ncvslideio::Size sz;
    compare_f cmpF;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, type_mat, depth_to, sz, alpha, beta, compile_args) = GetParam();
    MatType type_out = CV_MAKETYPE(depth_to, CV_MAT_CN(type_mat));

    initMatrixRandU(type_mat, sz, type_out);

    // OpenCV code ///////////////////////////////////////////////////////////
    in_mat1.convertTo(out_mat_ocv, depth_to, alpha, beta);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::convertTo(in, depth_to, alpha, beta);
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

PERF_TEST_P_(KMeansNDPerfTest, TestPerformance)
{
    ncvslideio::Size sz;
    CompareMats cmpF;
    int K = -1;
    ncvslideio::KmeansFlags flags = ncvslideio::KMEANS_RANDOM_CENTERS;
    ncvslideio::GCompileArgs compile_args;
    std::tie(sz, cmpF, K, flags, compile_args) = GetParam();

    MatType2 type = CV_32FC1;
    initMatrixRandU(type, sz, -1, false);

    double compact_gapi = -1.;
    ncvslideio::Mat labels_gapi, centers_gapi;
    if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
    {
        const int amount = sz.height;
        ncvslideio::Mat bestLabels(ncvslideio::Size{1, amount}, CV_32SC1);
        ncvslideio::randu(bestLabels, 0, K);

        ncvslideio::GComputation c(kmeansTestGAPI(in_mat1, bestLabels, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(ncvslideio::gin(in_mat1, bestLabels),
                    ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestOpenCVCompare(in_mat1, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi, cmpF);
    }
    else
    {
        ncvslideio::GComputation c(kmeansTestGAPI(in_mat1, K, flags, std::move(compile_args), compact_gapi,
                                          labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestValidate(sz, type, K, compact_gapi, labels_gapi, centers_gapi);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KMeans2DPerfTest, TestPerformance)
{
    int amount = -1;
    int K = -1;
    ncvslideio::KmeansFlags flags = ncvslideio::KMEANS_RANDOM_CENTERS;
    ncvslideio::GCompileArgs compile_args;
    std::tie(amount, K, flags, compile_args) = GetParam();

    std::vector<ncvslideio::Point2f> in_vector{};
    initPointsVectorRandU(amount, in_vector);

    double compact_gapi = -1.;
    std::vector<int> labels_gapi{};
    std::vector<ncvslideio::Point2f> centers_gapi{};
    if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
    {
        std::vector<int> bestLabels(amount);
        ncvslideio::randu(bestLabels, 0, K);

        ncvslideio::GComputation c(kmeansTestGAPI(in_vector, bestLabels, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(ncvslideio::gin(in_vector, bestLabels),
                    ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestOpenCVCompare(in_vector, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi);
    }
    else
    {
        ncvslideio::GComputation c(kmeansTestGAPI(in_vector, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestValidate({-1, amount}, -1, K, compact_gapi, labels_gapi, centers_gapi);
    }
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KMeans3DPerfTest, TestPerformance)
{
    int amount = -1;
    int K = -1;
    ncvslideio::KmeansFlags flags = ncvslideio::KMEANS_RANDOM_CENTERS;
    ncvslideio::GCompileArgs compile_args;
    std::tie(amount, K, flags, compile_args) = GetParam();

    std::vector<ncvslideio::Point3f> in_vector{};
    initPointsVectorRandU(amount, in_vector);

    double compact_gapi = -1.;
    std::vector<int> labels_gapi;
    std::vector<ncvslideio::Point3f> centers_gapi;
    if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
    {
        std::vector<int> bestLabels(amount);
        ncvslideio::randu(bestLabels, 0, K);

        ncvslideio::GComputation c(kmeansTestGAPI(in_vector, bestLabels, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(ncvslideio::gin(in_vector, bestLabels),
                    ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestOpenCVCompare(in_vector, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi);
    }
    else
    {
        ncvslideio::GComputation c(kmeansTestGAPI(in_vector, K, flags, std::move(compile_args),
                                          compact_gapi, labels_gapi, centers_gapi));
        TEST_CYCLE()
        {
            c.apply(ncvslideio::gin(in_vector), ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi));
        }
        kmeansTestValidate({-1, amount}, -1, K, compact_gapi, labels_gapi, centers_gapi);
    }
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(TransposePerfTest, TestPerformance)
{
    compare_f cmpF;
    ncvslideio::Size sz;
    MatType type = -1;
    ncvslideio::GCompileArgs compile_args;
    std::tie(cmpF, sz, type, compile_args) = GetParam();

    initMatrixRandU(type, sz, type, false);

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::transpose(in_mat1, out_mat_ocv);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::transpose(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    // Warm-up graph engine:
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi), std::move(compile_args));

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ParseSSDBLPerfTest, TestPerformance)
{
    ncvslideio::Size sz;
    float confidence_threshold = 0.0f;
    int filter_label = 0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(sz, confidence_threshold, filter_label, compile_args) = GetParam();

    ncvslideio::Mat in_mat = generateSSDoutput(sz);
    std::vector<ncvslideio::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // Reference code //////////////////////////////////////////////////////////
    parseSSDBLref(in_mat, sz, confidence_threshold, filter_label, boxes_ref, labels_ref);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> op_sz;
    auto out = ncvslideio::gapi::parseSSD(in, op_sz, confidence_threshold, filter_label);
    ncvslideio::GComputation c(ncvslideio::GIn(in, op_sz), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat), descr_of(sz), std::move(compile_args));
    cc(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi, labels_gapi));

    TEST_CYCLE()
    {
        cc(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi, labels_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(boxes_gapi == boxes_ref);
        EXPECT_TRUE(labels_gapi == labels_ref);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ParseSSDPerfTest, TestPerformance)
{
    ncvslideio::Size sz;
    float confidence_threshold = 0;
    bool alignment_to_square = false, filter_out_of_bounds = false;
    ncvslideio::GCompileArgs compile_args;
    std::tie(sz, confidence_threshold, alignment_to_square, filter_out_of_bounds, compile_args) = GetParam();

    ncvslideio::Mat in_mat = generateSSDoutput(sz);
    std::vector<ncvslideio::Rect> boxes_gapi, boxes_ref;

    // Reference code //////////////////////////////////////////////////////////
    parseSSDref(in_mat, sz, confidence_threshold, alignment_to_square, filter_out_of_bounds, boxes_ref);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> op_sz;
    auto out = ncvslideio::gapi::parseSSD(in, op_sz, confidence_threshold, alignment_to_square, filter_out_of_bounds);
    ncvslideio::GComputation c(ncvslideio::GIn(in, op_sz), ncvslideio::GOut(out));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat), descr_of(sz), std::move(compile_args));
    cc(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi));

    TEST_CYCLE()
    {
        cc(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(boxes_gapi == boxes_ref);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(ParseYoloPerfTest, TestPerformance)
{
    ncvslideio::Size sz;
    float confidence_threshold = 0.0f, nms_threshold = 0.0f;
    int num_classes = 0;
    ncvslideio::GCompileArgs compile_args;
    std::tie(sz, confidence_threshold, nms_threshold, num_classes, compile_args) = GetParam();

    ncvslideio::Mat in_mat = generateYoloOutput(num_classes);
    auto anchors = ncvslideio::gapi::nn::parsers::GParseYolo::defaultAnchors();
    std::vector<ncvslideio::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // Reference code //////////////////////////////////////////////////////////
    parseYoloRef(in_mat, sz, confidence_threshold, nms_threshold, num_classes, anchors, boxes_ref, labels_ref);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> op_sz;
    auto out = ncvslideio::gapi::parseYolo(in, op_sz, confidence_threshold, nms_threshold, anchors);
    ncvslideio::GComputation c(ncvslideio::GIn(in, op_sz), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat), descr_of(sz), std::move(compile_args));
    cc(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi, labels_gapi));

    TEST_CYCLE()
    {
        cc(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi, labels_gapi));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(boxes_gapi == boxes_ref);
        EXPECT_TRUE(labels_gapi == labels_ref);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SizePerfTest, TestPerformance)
{
    MatType type = -1;
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(type, sz, compile_args) = GetParam();
    in_mat1 = ncvslideio::Mat(sz, type);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::streaming::size(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    ncvslideio::Size sz_out;

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(in_mat1), std::move(compile_args));
    cc(ncvslideio::gin(in_mat1), ncvslideio::gout(sz_out));

    TEST_CYCLE()
    {
        cc(ncvslideio::gin(in_mat1), ncvslideio::gout(sz_out));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(sz_out, sz);
    }

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(SizeRPerfTest, TestPerformance)
{
    ncvslideio::Size sz;
    ncvslideio::GCompileArgs compile_args;
    std::tie(sz, compile_args) = GetParam();
    ncvslideio::Rect rect(ncvslideio::Point(0,0), sz);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GOpaque<ncvslideio::Rect> op_rect;
    auto out = ncvslideio::gapi::streaming::size(op_rect);
    ncvslideio::GComputation c(ncvslideio::GIn(op_rect), ncvslideio::GOut(out));
    ncvslideio::Size sz_out;

    // Warm-up graph engine:
    auto cc = c.compile(descr_of(rect), std::move(compile_args));
    cc(ncvslideio::gin(rect), ncvslideio::gout(sz_out));

    TEST_CYCLE()
    {
        cc(ncvslideio::gin(rect), ncvslideio::gout(sz_out));
    }

    // Comparison ////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(sz_out, sz);
    }

    SANITY_CHECK_NOTHING();
}

}
#endif // OPENCV_GAPI_CORE_PERF_TESTS_INL_HPP
