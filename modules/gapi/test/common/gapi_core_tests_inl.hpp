// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_TESTS_INL_HPP
#define OPENCV_GAPI_CORE_TESTS_INL_HPP

#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/infer/parsers.hpp>
#include "gapi_core_tests.hpp"

#include "gapi_core_tests_common.hpp"

namespace opencv_test
{
TEST_P(MathOpTest, MatricesAccuracyTest)
{
    // G-API code & corresponding OpenCV code ////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    if( testWithScalar )
    {
        ncvslideio::GScalar sc1;
        switch(opType)
        {
        case (ADD):
        {
            out = ncvslideio::gapi::addC(in1, sc1, dtype);
            ncvslideio::add(in_mat1, sc, out_mat_ocv, ncvslideio::noArray(), dtype);
            break;
        }
        case (SUB):
        {
            if( doReverseOp )
            {
                out = ncvslideio::gapi::subRC(sc1, in1, dtype);
                ncvslideio::subtract(sc, in_mat1, out_mat_ocv, ncvslideio::noArray(), dtype);
            }
            else
            {
                out = ncvslideio::gapi::subC(in1, sc1, dtype);
                ncvslideio::subtract(in_mat1, sc, out_mat_ocv, ncvslideio::noArray(), dtype);
            }
            break;
        }
        case (DIV):
        {
            if( doReverseOp )
            {
                in_mat1.setTo(1, in_mat1 == 0);  // avoiding zeros in divide input data
                out = ncvslideio::gapi::divRC(sc1, in1, scale, dtype);
                ncvslideio::divide(sc, in_mat1, out_mat_ocv, scale, dtype);
                break;
            }
            else
            {
                sc += Scalar(sc[0] == 0, sc[1] == 0, sc[2] == 0, sc[3] == 0);  // avoiding zeros in divide input data
                out = ncvslideio::gapi::divC(in1, sc1, scale, dtype);
                ncvslideio::divide(in_mat1, sc, out_mat_ocv, scale, dtype);
                break;
            }
        }
        case (MUL):
        {
            // FIXME: add `scale` parameter to mulC
            out = ncvslideio::gapi::mulC(in1, sc1, /* scale, */ dtype);
            ncvslideio::multiply(in_mat1, sc, out_mat_ocv, 1., dtype);
            break;
        }
        default:
        {
            FAIL() << "no such math operation type for scalar and matrix!";
        }
        }
        ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());
    }
    else
    {
        switch(opType)
        {
        case (ADD):
        {
            out = ncvslideio::gapi::add(in1, in2, dtype);
            ncvslideio::add(in_mat1, in_mat2, out_mat_ocv, ncvslideio::noArray(), dtype);
            break;
        }
        case (SUB):
        {
            out = ncvslideio::gapi::sub(in1, in2, dtype);
            ncvslideio::subtract(in_mat1, in_mat2, out_mat_ocv, ncvslideio::noArray(), dtype);
            break;
        }
        case (DIV):
        {
            in_mat2.setTo(1, in_mat2 == 0);  // avoiding zeros in divide input data
            out = ncvslideio::gapi::div(in1, in2, scale, dtype);
            ncvslideio::divide(in_mat1, in_mat2, out_mat_ocv, scale, dtype);
            break;
        }
        case (MUL):
        {
            out = ncvslideio::gapi::mul(in1, in2, scale, dtype);
            ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv, scale, dtype);
            break;
        }
        default:
        {
            FAIL() << "no such math operation type for matrix and matrix!";
        }}
        ncvslideio::GComputation c(GIn(in1, in2), GOut(out));
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
    // TODO: make threshold vs bit-exact criteria be driven by testing parameter
    #if 1
        if (CV_MAT_DEPTH(out_mat_ocv.type()) != CV_32F &&
            CV_MAT_DEPTH(out_mat_ocv.type()) != CV_64F)
        {
            // integral: allow 1% of differences, and no diffs by >1 unit
            EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF), 1);  // check: abs(a[i] - b[i]) <= 1
            float tolerance = 0.01f;
#if defined(__arm__) || defined(__aarch64__)
            if (opType == DIV)
                tolerance = 0.05f;
#endif
            EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_L1), tolerance*out_mat_ocv.total());
        }
        else
        {
            // floating-point: expect 6 decimal digits - best we expect of F32
            EXPECT_LE(cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF | NORM_RELATIVE), 1e-6);
        }
    #else
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    #endif
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(MulDoubleTest, AccuracyTest)
{
    auto& rng = ncvslideio::theRNG();
    double d = rng.uniform(0.0, 10.0);

    // G-API code ////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    out = ncvslideio::gapi::mulC(in1, d, dtype);
    ncvslideio::GComputation c(in1, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code ///////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, d, out_mat_ocv, 1, dtype);

    // Comparison ////////////////////////////////////////////////////////////
#if 1
    if (CV_MAT_DEPTH(out_mat_ocv.type()) != CV_32F &&
        CV_MAT_DEPTH(out_mat_ocv.type()) != CV_64F)
    {
        // integral: allow 1% of differences, and no diffs by >1 unit
        EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF), 1);
        EXPECT_LE(cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_L1 | NORM_RELATIVE), 0.01);
    }
    else
    {
        // floating-point: expect 6 decimal digits - best we expect of F32
        EXPECT_LE(cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF | NORM_RELATIVE), 1e-6);
    }
#else
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
#endif
    EXPECT_EQ(sz, out_mat_gapi.size());
}

TEST_P(DivTest, DISABLED_DivByZeroTest)  // https://github.com/opencv/opencv/pull/12826
{
    in_mat2 = ncvslideio::Mat(sz, type);
    in_mat2.setTo(ncvslideio::Scalar::all(0));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::div(in1, in2, 1.0, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::divide(in_mat1, in_mat2, out_mat_ocv, 1.0, dtype);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(DivCTest, DISABLED_DivByZeroTest)  // https://github.com/opencv/opencv/pull/12826
{
    sc = ncvslideio::Scalar::all(0);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    ncvslideio::GScalar sc1;
    auto out = ncvslideio::gapi::divC(in1, sc1, dtype);
    ncvslideio::GComputation c(GIn(in1, sc1), GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::divide(in_mat1, sc, out_mat_ocv, dtype);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        ncvslideio::Mat zeros = ncvslideio::Mat::zeros(sz, type);
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, zeros, NORM_INF));
    }
}

TEST_P(MeanTest, AccuracyTest)
{
    ncvslideio::Scalar out_norm;
    ncvslideio::Scalar out_norm_ocv;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::mean(in);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_norm), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_norm_ocv = ncvslideio::mean(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(out_norm[0], out_norm_ocv[0]);
    }
}

TEST_P(MaskTest, AccuracyTest)
{
    in_mat2 = ncvslideio::Mat(sz, CV_8UC1);
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    in_mat2 = in_mat2 > 128;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in, m;
    auto out = ncvslideio::gapi::mask(in, m);

    ncvslideio::GComputation c(ncvslideio::GIn(in, m), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = ncvslideio::Mat::zeros(in_mat1.size(), in_mat1.type());
        in_mat1.copyTo(out_mat_ocv, in_mat2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(Polar2CartTest, AccuracyTest)
{
    ncvslideio::Mat out_mat2;
    ncvslideio::Mat out_mat_ocv2;
    if (dtype != -1)
    {
        out_mat2 = ncvslideio::Mat(sz, dtype);
        out_mat_ocv2 = ncvslideio::Mat(sz, dtype);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = ncvslideio::gapi::polarToCart(in1, in2);

    ncvslideio::GComputation c(GIn(in1, in2), GOut(out1, out2));
    c.apply(gin(in_mat1,in_mat2), gout(out_mat_gapi, out_mat2), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::polarToCart(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        // Note that we cannot rely on bit-exact sin/cos functions used for this
        // transform, so we need a threshold for verifying results vs reference.
        //
        // Relative threshold like 1e-6 is very restrictive, nearly best we can
        // expect of single-precision elementary functions implementation.
        //
        // However, good idea is making such threshold configurable: parameter
        // of this test - which a specific test instantiation could setup.
        //
        // Note that test instantiation for the OpenCV back-end could even let
        // the threshold equal to zero, as ncvslideio back-end calls the same kernel.
        //
        // TODO: Make threshold a configurable parameter of this test (ADE-221)

        ASSERT_EQ(sz, out_mat_gapi.size());

        ncvslideio::Mat &outx = out_mat_gapi,
                &outy = out_mat2;
        ncvslideio::Mat &refx = out_mat_ocv,
                &refy = out_mat_ocv2;

        EXPECT_LE(cvtest::norm(refx, outx, NORM_L1 | NORM_RELATIVE), 1e-6);
        EXPECT_LE(cvtest::norm(refy, outy, NORM_L1 | NORM_RELATIVE), 1e-6);
    }
}

TEST_P(Cart2PolarTest, AccuracyTest)
{
    ncvslideio::Mat out_mat2(sz, dtype);
    ncvslideio::Mat out_mat_ocv2(sz, dtype);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out1, out2;
    std::tie(out1, out2) = ncvslideio::gapi::cartToPolar(in1, in2);

    ncvslideio::GComputation c(GIn(in1, in2), GOut(out1, out2));
    c.apply(gin(in_mat1,in_mat2), gout(out_mat_gapi, out_mat2));
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::cartToPolar(in_mat1, in_mat2, out_mat_ocv, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        // Note that we cannot rely on bit-exact sin/cos functions used for this
        // transform, so we need a threshold for verifying results vs reference.
        //
        // Relative threshold like 1e-6 is very restrictive, nearly best we can
        // expect of single-precision elementary functions implementation.
        //
        // However, good idea is making such threshold configurable: parameter
        // of this test - which a specific test instantiation could setup.
        //
        // Note that test instantiation for the OpenCV back-end could even let
        // the threshold equal to zero, as ncvslideio back-end calls the same kernel.
        //
        // TODO: Make threshold a configurable parameter of this test (ADE-221)

        ASSERT_EQ(sz, out_mat_gapi.size());

        ncvslideio::Mat &outm = out_mat_gapi,
                &outa = out_mat2;
        ncvslideio::Mat &refm = out_mat_ocv,
                &refa = out_mat_ocv2;

        // FIXME: Angle result looks inaccurate at OpenCV
        //        (expected relative accuracy like 1e-6)
        EXPECT_LE(cvtest::norm(refm, outm, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(refa, outa, NORM_INF), 1e-3);
    }
}

TEST_P(CmpTest, AccuracyTest)
{
    // G-API code & corresponding OpenCV code ////////////////////////////////
    ncvslideio::GMat in1, out;
    if( testWithScalar )
    {
        ncvslideio::GScalar in2;
        switch(opType)
        {
        case CMP_EQ: out = ncvslideio::gapi::cmpEQ(in1, in2); break;
        case CMP_GT: out = ncvslideio::gapi::cmpGT(in1, in2); break;
        case CMP_GE: out = ncvslideio::gapi::cmpGE(in1, in2); break;
        case CMP_LT: out = ncvslideio::gapi::cmpLT(in1, in2); break;
        case CMP_LE: out = ncvslideio::gapi::cmpLE(in1, in2); break;
        case CMP_NE: out = ncvslideio::gapi::cmpNE(in1, in2); break;
        default: FAIL() << "no such compare operation type for matrix and scalar!";
        }

        ncvslideio::compare(in_mat1, sc, out_mat_ocv, opType);

        ncvslideio::GComputation c(GIn(in1, in2), GOut(out));
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());
    }
    else
    {
        ncvslideio::GMat in2;
        switch(opType)
        {
        case CMP_EQ: out = ncvslideio::gapi::cmpEQ(in1, in2); break;
        case CMP_GT: out = ncvslideio::gapi::cmpGT(in1, in2); break;
        case CMP_GE: out = ncvslideio::gapi::cmpGE(in1, in2); break;
        case CMP_LT: out = ncvslideio::gapi::cmpLT(in1, in2); break;
        case CMP_LE: out = ncvslideio::gapi::cmpLE(in1, in2); break;
        case CMP_NE: out = ncvslideio::gapi::cmpNE(in1, in2); break;
        default: FAIL() << "no such compare operation type for two matrices!";
        }

        ncvslideio::compare(in_mat1, in_mat2, out_mat_ocv, opType);

        ncvslideio::GComputation c(GIn(in1, in2), GOut(out));
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(BitwiseTest, AccuracyTest)
{
    // G-API code & corresponding OpenCV code ////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    if( testWithScalar )
    {
        ncvslideio::GScalar sc1;
        switch(opType)
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
        c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());
    }
    else
    {
        switch(opType)
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
        c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());
    }


    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(NotTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::bitwise_not(in);
    ncvslideio::GComputation c(in, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::bitwise_not(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(SelectTest, AccuracyTest)
{
    ncvslideio::Mat in_mask(sz, CV_8UC1);
    ncvslideio::randu(in_mask, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, in3;
    auto out = ncvslideio::gapi::select(in1, in2, in3);
    ncvslideio::GComputation c(GIn(in1, in2, in3), GOut(out));

    c.apply(gin(in_mat1, in_mat2, in_mask), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat2.copyTo(out_mat_ocv);
        in_mat1.copyTo(out_mat_ocv, in_mask);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(MinTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::min(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::min(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(MaxTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::max(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::max(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(AbsDiffTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::absDiff(in1, in2);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::absdiff(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(AbsDiffCTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    ncvslideio::GScalar sc1;
    auto out = ncvslideio::gapi::absDiffC(in1, sc1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, sc1), ncvslideio::GOut(out));

    c.apply(gin(in_mat1, sc), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::absdiff(in_mat1, sc, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    }
}

TEST_P(SumTest, AccuracyTest)
{
    ncvslideio::Scalar out_sum;
    ncvslideio::Scalar out_sum_ocv;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::sum(in);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_sum), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_sum_ocv = ncvslideio::sum(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_sum, out_sum_ocv));
    }
}

#pragma push_macro("countNonZero")
#undef countNonZero
TEST_P(CountNonZeroTest, AccuracyTest)
{
    int out_cnz_gapi = -1;
    int out_cnz_ocv = -2;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::countNonZero(in);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_cnz_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_cnz_ocv = ncvslideio::countNonZero(in_mat1);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_cnz_gapi, out_cnz_ocv));
    }
}
#pragma pop_macro("countNonZero")

TEST_P(AddWeightedTest, AccuracyTest)
{
    auto& rng = ncvslideio::theRNG();
    double alpha = rng.uniform(0.0, 1.0);
    double beta = rng.uniform(0.0, 1.0);
    double gamma = rng.uniform(0.0, 1.0);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::addWeighted(in1, alpha, in2, beta, gamma, dtype);
    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));

    c.apply(gin(in_mat1, in_mat2), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::addWeighted(in_mat1, alpha, in_mat2, beta, gamma, out_mat_ocv, dtype);
    }
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    EXPECT_EQ(sz, out_mat_gapi.size());
}

TEST_P(NormTest, AccuracyTest)
{
    ncvslideio::Scalar out_norm;
    ncvslideio::Scalar out_norm_ocv;

    // G-API code & corresponding OpenCV code ////////////////////////////////
    ncvslideio::GMat in1;
    ncvslideio::GScalar out;
    switch(opType)
    {
        case NORM_L1: out = ncvslideio::gapi::normL1(in1); break;
        case NORM_L2: out = ncvslideio::gapi::normL2(in1); break;
        case NORM_INF: out = ncvslideio::gapi::normInf(in1); break;
        default: FAIL() << "no such norm operation type!";
    }
    out_norm_ocv = ncvslideio::norm(in_mat1, opType);
    ncvslideio::GComputation c(GIn(in1), GOut(out));
    c.apply(gin(in_mat1), gout(out_norm), getCompileArgs());

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_norm, out_norm_ocv));
    }
}

TEST_P(IntegralTest, AccuracyTest)
{
    int type_out = (type == CV_8U) ? CV_32SC1 : CV_64FC1;
    in_mat1 = ncvslideio::Mat(sz, type);

    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    ncvslideio::Size sz_out = ncvslideio::Size(sz.width + 1, sz.height + 1);
    ncvslideio::Mat out_mat1(sz_out, type_out);
    ncvslideio::Mat out_mat_ocv1(sz_out, type_out);

    ncvslideio::Mat out_mat2(sz_out, CV_64FC1);
    ncvslideio::Mat out_mat_ocv2(sz_out, CV_64FC1);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out1, out2;
    std::tie(out1, out2)  = ncvslideio::gapi::integral(in1, type_out, CV_64FC1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1), ncvslideio::GOut(out1, out2));

    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat1, out_mat2), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::integral(in_mat1, out_mat_ocv1, out_mat_ocv2);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv1, out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat2, NORM_INF));
    }
}

TEST_P(ThresholdTest, AccuracyTestBinary)
{
    ncvslideio::Scalar thr = initScalarRandU(50);
    ncvslideio::Scalar out_scalar;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar th1, mv1;
    out = ncvslideio::gapi::threshold(in1, th1, mv1, tt);
    ncvslideio::GComputation c(GIn(in1, th1, mv1), GOut(out));

    c.apply(gin(in_mat1, thr, maxval), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::threshold(in_mat1, out_mat_ocv, thr.val[0], maxval.val[0], tt);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, ncvslideio::norm(out_mat_ocv, out_mat_gapi, NORM_L1));
    }
}

TEST_P(ThresholdOTTest, AccuracyTestOtsu)
{
    ncvslideio::Scalar maxval = initScalarRandU(50) + ncvslideio::Scalar(50, 50, 50, 50);
    ncvslideio::Scalar out_gapi_scalar;
    double ocv_res;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out;
    ncvslideio::GScalar mv1, scout;
    std::tie<ncvslideio::GMat, ncvslideio::GScalar>(out, scout) = ncvslideio::gapi::threshold(in1, mv1, tt);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, mv1), ncvslideio::GOut(out, scout));

    c.apply(gin(in_mat1, maxval), gout(out_mat_gapi, out_gapi_scalar), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ocv_res = ncvslideio::threshold(in_mat1, out_mat_ocv, maxval.val[0], maxval.val[0], tt);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(ocv_res, out_gapi_scalar.val[0]);
    }
}

TEST_P(InRangeTest, AccuracyTest)
{
    ncvslideio::Scalar thrLow = initScalarRandU(100);
    ncvslideio::Scalar thrUp = initScalarRandU(100) + ncvslideio::Scalar(100, 100, 100, 100);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    ncvslideio::GScalar th1, mv1;
    auto out = ncvslideio::gapi::inRange(in1, th1, mv1);
    ncvslideio::GComputation c(GIn(in1, th1, mv1), GOut(out));

    c.apply(gin(in_mat1, thrLow, thrUp), gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::inRange(in_mat1, thrLow, thrUp, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        ASSERT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(Split3Test, AccuracyTest)
{
    ncvslideio::Mat out_mat2 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat3 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat_ocv2 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat_ocv3 = ncvslideio::Mat(sz, dtype);
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out1, out2, out3;
    std::tie(out1, out2, out3)  = ncvslideio::gapi::split3(in1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1), ncvslideio::GOut(out1, out2, out3));

    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi, out_mat2, out_mat3), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<ncvslideio::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3};
        ncvslideio::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv,  out_mat_gapi, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat2, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv3, out_mat3, NORM_INF));
    }
}

TEST_P(Split4Test, AccuracyTest)
{
    ncvslideio::Mat out_mat2 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat3 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat4 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat_ocv2 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat_ocv3 = ncvslideio::Mat(sz, dtype);
    ncvslideio::Mat out_mat_ocv4 = ncvslideio::Mat(sz, dtype);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, out1, out2, out3, out4;
    std::tie(out1, out2, out3, out4)  = ncvslideio::gapi::split4(in1);
    ncvslideio::GComputation c(ncvslideio::GIn(in1), ncvslideio::GOut(out1, out2, out3, out4));

    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi, out_mat2, out_mat3, out_mat4), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<ncvslideio::Mat> out_mats_ocv = {out_mat_ocv, out_mat_ocv2, out_mat_ocv3, out_mat_ocv4};
        ncvslideio::split(in_mat1, out_mats_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv,  out_mat_gapi, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat2, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv3, out_mat3, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv4, out_mat4, NORM_INF));
    }
}

TEST_P(Merge3Test, AccuracyTest)
{
    ncvslideio::Mat in_mat3(sz, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat3, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, in3;
    auto out = ncvslideio::gapi::merge3(in1, in2, in3);

    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2, in_mat3), ncvslideio::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<ncvslideio::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3};
        ncvslideio::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(Merge4Test, AccuracyTest)
{
    ncvslideio::Mat in_mat3(sz, type);
    ncvslideio::Mat in_mat4(sz, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat3, mean, stddev);
    ncvslideio::randn(in_mat4, mean, stddev);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, in3, in4;
    auto out = ncvslideio::gapi::merge4(in1, in2, in3, in4);

    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3, in4), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2, in_mat3, in_mat4), ncvslideio::gout(out_mat_gapi), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        std::vector<ncvslideio::Mat> in_mats_ocv = {in_mat1, in_mat2, in_mat3, in_mat4};
        ncvslideio::merge(in_mats_ocv, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(RemapTest, AccuracyTest)
{
    ncvslideio::Mat in_map1(sz, CV_16SC2);
    ncvslideio::Mat in_map2 = ncvslideio::Mat();
    ncvslideio::randu(in_map1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Scalar bv = ncvslideio::Scalar();

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1;
    auto out = ncvslideio::gapi::remap(in1, in_map1, in_map2, ncvslideio::INTER_NEAREST,  ncvslideio::BORDER_REPLICATE, bv);
    ncvslideio::GComputation c(in1, out);

    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::remap(in_mat1, out_mat_ocv, in_map1, in_map2, ncvslideio::INTER_NEAREST, ncvslideio::BORDER_REPLICATE, bv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(FlipTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::flip(in, flipCode);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::flip(in_mat1, out_mat_ocv, flipCode);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(CropTest, AccuracyTest)
{
    ncvslideio::Size sz_out = ncvslideio::Size(rect_to.width, rect_to.height);
    if (dtype != -1)
    {
        out_mat_gapi = ncvslideio::Mat(sz_out, dtype);
        out_mat_ocv = ncvslideio::Mat(sz_out, dtype);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::crop(in, rect_to);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Mat(in_mat1, rect_to).copyTo(out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(sz_out, out_mat_gapi.size());
    }
}

TEST_P(CopyTest, AccuracyTest)
{
    ncvslideio::Size sz_out = sz;
    if (dtype != -1)
    {
        out_mat_gapi = ncvslideio::Mat(sz_out, dtype);
        out_mat_ocv = ncvslideio::Mat(sz_out, dtype);
    }

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::copy(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Mat(in_mat1).copyTo(out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(sz_out, out_mat_gapi.size());
    }
}

TEST_P(ConcatHorTest, AccuracyTest)
{
    ncvslideio::Size sz_out = sz;

    int wpart = sz_out.width / 4;
    ncvslideio::Size sz_in1 = ncvslideio::Size(wpart, sz_out.height);
    ncvslideio::Size sz_in2 = ncvslideio::Size(sz_out.width - wpart, sz_out.height);

    in_mat1 = ncvslideio::Mat(sz_in1, type );
    in_mat2 = ncvslideio::Mat(sz_in2, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);

    ncvslideio::Mat out_mat(sz_out, type);
    out_mat_ocv = ncvslideio::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::concatHor(in1, in2);

    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::hconcat(in_mat1, in_mat2, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(ConcatVertTest, AccuracyTest)
{
    ncvslideio::Size sz_out = sz;

    int hpart = sz_out.height * 2/3;
    ncvslideio::Size sz_in1 = ncvslideio::Size(sz_out.width, hpart);
    ncvslideio::Size sz_in2 = ncvslideio::Size(sz_out.width, sz_out.height - hpart);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);

    ncvslideio::Mat out_mat(sz_out, type);
    out_mat_ocv = ncvslideio::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::concatVert(in1, in2);

    ncvslideio::GComputation c(GIn(in1, in2), GOut(out));
    c.apply(gin(in_mat1, in_mat2), gout(out_mat), getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::vconcat(in_mat1, in_mat2, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(ConcatVertVecTest, AccuracyTest)
{
    ncvslideio::Size sz_out = sz;

    int hpart1 = sz_out.height * 2/5;
    int hpart2 = sz_out.height / 5;
    ncvslideio::Size sz_in1 = ncvslideio::Size(sz_out.width, hpart1);
    ncvslideio::Size sz_in2 = ncvslideio::Size(sz_out.width, hpart2);
    ncvslideio::Size sz_in3 = ncvslideio::Size(sz_out.width, sz_out.height - hpart1 - hpart2);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);
    ncvslideio::Mat in_mat3(sz_in3, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);
    ncvslideio::randn(in_mat3, mean, stddev);

    ncvslideio::Mat out_mat(sz_out, type);
    out_mat_ocv = ncvslideio::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <ncvslideio::GMat> mats(3);
    auto out = ncvslideio::gapi::concatVert(mats);

    std::vector <ncvslideio::Mat> cvmats = {in_mat1, in_mat2, in_mat3};

    ncvslideio::GComputation c({mats[0], mats[1], mats[2]}, {out});
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::vconcat(cvmats, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(ConcatHorVecTest, AccuracyTest)
{
    ncvslideio::Size sz_out = sz;

    int wpart1 = sz_out.width / 3;
    int wpart2 = sz_out.width / 4;
    ncvslideio::Size sz_in1 = ncvslideio::Size(wpart1, sz_out.height);
    ncvslideio::Size sz_in2 = ncvslideio::Size(wpart2, sz_out.height);
    ncvslideio::Size sz_in3 = ncvslideio::Size(sz_out.width - wpart1 - wpart2, sz_out.height);

    in_mat1 = ncvslideio::Mat(sz_in1, type);
    in_mat2 = ncvslideio::Mat(sz_in2, type);
    ncvslideio::Mat in_mat3 (sz_in3, type);
    ncvslideio::Scalar mean = ncvslideio::Scalar::all(127);
    ncvslideio::Scalar stddev = ncvslideio::Scalar::all(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);
    ncvslideio::randn(in_mat3, mean, stddev);

    ncvslideio::Mat out_mat(sz_out, type);
    out_mat_ocv = ncvslideio::Mat(sz_out, type);

    // G-API code //////////////////////////////////////////////////////////////
    std::vector <ncvslideio::GMat> mats(3);
    auto out = ncvslideio::gapi::concatHor(mats);

    std::vector <ncvslideio::Mat> cvmats = {in_mat1, in_mat2, in_mat3};

    ncvslideio::GComputation c({mats[0], mats[1], mats[2]}, {out});
    c.apply(gin(in_mat1, in_mat2, in_mat3), gout(out_mat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::hconcat(cvmats, out_mat_ocv );
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat, NORM_INF));
    }
}

TEST_P(LUTTest, AccuracyTest)
{
    int type_mat = type;
    int type_lut = dtype;
    int type_out = CV_MAKETYPE(CV_MAT_DEPTH(type_lut), CV_MAT_CN(type_mat));

    initMatrixRandU(type_mat, sz, type_out);
    ncvslideio::Size sz_lut = ncvslideio::Size(1, 256);
    ncvslideio::Mat in_lut(sz_lut, type_lut);
    ncvslideio::randu(in_lut, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::LUT(in, in_lut);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::LUT(in_mat1, in_lut, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(ConvertToTest, AccuracyTest)
{
    int type_mat = type;
    int depth_to = dtype;
    int type_out = CV_MAKETYPE(depth_to, CV_MAT_CN(type_mat));
    initMatrixRandU(type_mat, sz, type_out);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::convertTo(in, depth_to, alpha, beta);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        in_mat1.convertTo(out_mat_ocv, depth_to, alpha, beta);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(PhaseTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in_x, in_y;
    auto out = ncvslideio::gapi::phase(in_x, in_y, angle_in_degrees);

    ncvslideio::GComputation c(in_x, in_y, out);
    c.apply(in_mat1, in_mat2, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::phase(in_mat1, in_mat2, out_mat_ocv, angle_in_degrees);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: use a comparison functor instead (after enabling OpenCL)
    {
#if defined(__aarch64__) || defined(__arm__)
        EXPECT_NEAR(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF), 4e-6);
#else
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
#endif
    }
}

TEST_P(SqrtTest, AccuracyTest)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::sqrt(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::sqrt(in_mat1, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: use a comparison functor instead (after enabling OpenCL)
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}

TEST_P(WarpPerspectiveTest, AccuracyTest)
{
    ncvslideio::Point center{in_mat1.size() / 2};
    ncvslideio::Mat xy = ncvslideio::getRotationMatrix2D(center, angle, scale);
    ncvslideio::Matx13d z (0, 0, 1);
    ncvslideio::Mat transform_mat;
    ncvslideio::vconcat(xy, z, transform_mat);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::warpPerspective(in, transform_mat, in_mat1.size(), flags, border_mode, border_value);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::warpPerspective(in_mat1, out_mat_ocv, ncvslideio::Mat(transform_mat), in_mat1.size(), flags, border_mode, border_value);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(WarpAffineTest, AccuracyTest)
{
    ncvslideio::Point center{in_mat1.size() / 2};
    ncvslideio::Mat warp_mat = ncvslideio::getRotationMatrix2D(center, angle, scale);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::warpAffine(in, warp_mat, in_mat1.size(), flags, border_mode, border_value);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::warpAffine(in_mat1, out_mat_ocv, warp_mat, in_mat1.size(), flags, border_mode, border_value);

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
    }
}

TEST_P(NormalizeTest, Test)
{
    initMatrixRandN(type, sz, CV_MAKETYPE(ddepth, CV_MAT_CN(type)));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::normalize(in, a, b, norm_type, ddepth);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::normalize(in_mat1, out_mat_ocv, a, b, norm_type, ddepth);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(sz, out_mat_gapi.size());
    }
}

TEST_P(KMeansNDTest, AccuracyTest)
{
    kmeansTestBody(in_mat1, sz, type, K, flags, getCompileArgs(), cmpF);
}

TEST_P(KMeans2DTest, AccuracyTest)
{
    const int amount = sz.height;
    std::vector<ncvslideio::Point2f> in_vector{};
    initPointsVectorRandU(amount, in_vector);
    kmeansTestBody(in_vector, sz, type, K, flags, getCompileArgs());
}

TEST_P(KMeans3DTest, AccuracyTest)
{
    const int amount = sz.height;
    std::vector<ncvslideio::Point3f> in_vector{};
    initPointsVectorRandU(amount, in_vector);
    kmeansTestBody(in_vector, sz, type, K, flags, getCompileArgs());
}

TEST_P(TransposeTest, Test)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::transpose(in);

    ncvslideio::GComputation c(in, out);
    c.apply(in_mat1, out_mat_gapi, getCompileArgs());
    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::transpose(in_mat1, out_mat_ocv);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_ocv, out_mat_gapi));
    }
}
// PLEASE DO NOT PUT NEW ACCURACY TESTS BELOW THIS POINT! //////////////////////

TEST_P(BackendOutputAllocationTest, EmptyOutput)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    EXPECT_TRUE(out_mat_gapi.empty());
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());
    EXPECT_FALSE(out_mat_gapi.empty());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: output is allocated to the needed size
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());
}

TEST_P(BackendOutputAllocationTest, CorrectlyPreallocatedOutput)
{
    out_mat_gapi = ncvslideio::Mat(sz, type);
    auto out_mat_gapi_ref = out_mat_gapi;  // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::add(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::add(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: output is not reallocated
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());

    EXPECT_EQ(out_mat_gapi_ref.data, out_mat_gapi.data);
}

TEST_P(BackendOutputAllocationTest, IncorrectOutputMeta)
{
    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::add(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    const auto run_and_compare = [&c, this] ()
    {
        auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

        // G-API code //////////////////////////////////////////////////////////////
        c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());

        // OpenCV code /////////////////////////////////////////////////////////////
        ncvslideio::add(in_mat1, in_mat2, out_mat_ocv, ncvslideio::noArray());

        // Comparison //////////////////////////////////////////////////////////////
        // Expected: size is changed, type is changed, output is reallocated
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
        EXPECT_EQ(type, out_mat_gapi.type());

        EXPECT_NE(out_mat_gapi_ref.data, out_mat_gapi.data);
    };

    const auto chan = CV_MAT_CN(type);

    out_mat_gapi = ncvslideio::Mat(sz, CV_MAKE_TYPE(CV_64F, chan));
    run_and_compare();

    out_mat_gapi = ncvslideio::Mat(sz, CV_MAKE_TYPE(CV_MAT_DEPTH(type), chan + 1));
    run_and_compare();
}

TEST_P(BackendOutputAllocationTest, SmallerPreallocatedSize)
{
    out_mat_gapi = ncvslideio::Mat(sz / 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: size is changed, output is reallocated due to original size < curr size
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());

    EXPECT_NE(out_mat_gapi_ref.data, out_mat_gapi.data);
}

TEST_P(BackendOutputAllocationTest, SmallerPreallocatedSizeWithSubmatrix)
{
    out_mat_gapi = ncvslideio::Mat(sz / 2, type);

    ncvslideio::Mat out_mat_gapi_submat = out_mat_gapi(ncvslideio::Rect({10, 0}, sz / 5));
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);

    auto out_mat_gapi_submat_ref = out_mat_gapi_submat; // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi_submat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: submatrix is reallocated and is "detached", original matrix is unchanged
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi_submat, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi_submat.size());
    EXPECT_EQ(sz / 2, out_mat_gapi.size());

    EXPECT_NE(out_mat_gapi_submat_ref.data, out_mat_gapi_submat.data);
    EXPECT_NE(out_mat_gapi.data, out_mat_gapi_submat.datastart);
}

TEST_P(BackendOutputAllocationTest, LargerPreallocatedSize)
{
    out_mat_gapi = ncvslideio::Mat(sz * 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: size is changed, output is reallocated
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi.size());

    EXPECT_NE(out_mat_gapi_ref.data, out_mat_gapi.data);
}

TEST_P(BackendOutputAllocationLargeSizeWithCorrectSubmatrixTest,
    LargerPreallocatedSizeWithCorrectSubmatrix)
{
    out_mat_gapi = ncvslideio::Mat(sz * 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    ncvslideio::Mat out_mat_gapi_submat = out_mat_gapi(ncvslideio::Rect({5, 8}, sz));
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);

    auto out_mat_gapi_submat_ref = out_mat_gapi_submat;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi_submat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: submatrix is not reallocated, original matrix is not reallocated
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi_submat, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi_submat.size());
    EXPECT_EQ(sz * 2, out_mat_gapi.size());

    EXPECT_EQ(out_mat_gapi_ref.data, out_mat_gapi.data);
    EXPECT_EQ(out_mat_gapi_submat_ref.data, out_mat_gapi_submat.data);
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);
}

TEST_P(BackendOutputAllocationTest, LargerPreallocatedSizeWithSmallSubmatrix)
{
    out_mat_gapi = ncvslideio::Mat(sz * 2, type);
    auto out_mat_gapi_ref = out_mat_gapi; // shallow copy to ensure previous data is not deleted

    ncvslideio::Mat out_mat_gapi_submat = out_mat_gapi(ncvslideio::Rect({5, 8}, sz / 2));
    EXPECT_EQ(out_mat_gapi.data, out_mat_gapi_submat.datastart);

    auto out_mat_gapi_submat_ref = out_mat_gapi_submat;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::mul(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi_submat), getCompileArgs());

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::multiply(in_mat1, in_mat2, out_mat_ocv);

    // Comparison //////////////////////////////////////////////////////////////
    // Expected: submatrix is reallocated and is "detached", original matrix is unchanged
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi_submat, out_mat_ocv, NORM_INF));
    EXPECT_EQ(sz, out_mat_gapi_submat.size());
    EXPECT_EQ(sz * 2, out_mat_gapi.size());

    EXPECT_EQ(out_mat_gapi_ref.data, out_mat_gapi.data);
    EXPECT_NE(out_mat_gapi_submat_ref.data, out_mat_gapi_submat.data);
    EXPECT_NE(out_mat_gapi.data, out_mat_gapi_submat.datastart);
}

TEST_P(ReInitOutTest, TestWithAdd)
{
    in_mat1 = ncvslideio::Mat(sz, type);
    in_mat2 = ncvslideio::Mat(sz, type);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(100));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(100));

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in1, in2, out;
    out = ncvslideio::gapi::add(in1, in2, dtype);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    const auto run_and_compare = [&c, this] ()
    {
        // G-API code //////////////////////////////////////////////////////////////
        c.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat_gapi), getCompileArgs());

        // OpenCV code /////////////////////////////////////////////////////////////
        ncvslideio::add(in_mat1, in_mat2, out_mat_ocv, ncvslideio::noArray());

        // Comparison //////////////////////////////////////////////////////////////
        EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
        EXPECT_EQ(sz, out_mat_gapi.size());
    };

    // run for uninitialized output
    run_and_compare();

    // run for initialized output (can be initialized with a different size)
    initOutMats(out_sz, type);
    run_and_compare();
}

TEST_P(ParseSSDBLTest, ParseTest)
{
    ncvslideio::Mat in_mat = generateSSDoutput(sz);
    std::vector<ncvslideio::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> op_sz;
    auto out = ncvslideio::gapi::parseSSD(in, op_sz, confidence_threshold, filter_label);
    ncvslideio::GComputation c(ncvslideio::GIn(in, op_sz), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi, labels_gapi), getCompileArgs());

    // Reference code //////////////////////////////////////////////////////////
    parseSSDBLref(in_mat, sz, confidence_threshold, filter_label, boxes_ref, labels_ref);

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(boxes_gapi == boxes_ref);
    EXPECT_TRUE(labels_gapi == labels_ref);
}

TEST_P(ParseSSDTest, ParseTest)
{
    ncvslideio::Mat in_mat = generateSSDoutput(sz);
    std::vector<ncvslideio::Rect> boxes_gapi, boxes_ref;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> op_sz;
    auto out = ncvslideio::gapi::parseSSD(in, op_sz, confidence_threshold,
                                  alignment_to_square, filter_out_of_bounds);
    ncvslideio::GComputation c(ncvslideio::GIn(in, op_sz), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi), getCompileArgs());

    // Reference code //////////////////////////////////////////////////////////
    parseSSDref(in_mat, sz, confidence_threshold, alignment_to_square,
                filter_out_of_bounds, boxes_ref);

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(boxes_gapi == boxes_ref);
}

TEST_P(ParseYoloTest, ParseTest)
{
    ncvslideio::Mat in_mat = generateYoloOutput(num_classes, dims_config);
    auto anchors = ncvslideio::gapi::nn::parsers::GParseYolo::defaultAnchors();
    std::vector<ncvslideio::Rect> boxes_gapi, boxes_ref;
    std::vector<int> labels_gapi, labels_ref;

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    ncvslideio::GOpaque<ncvslideio::Size> op_sz;
    auto out = ncvslideio::gapi::parseYolo(in, op_sz, confidence_threshold, nms_threshold, anchors);
    ncvslideio::GComputation c(ncvslideio::GIn(in, op_sz), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(ncvslideio::gin(in_mat, sz), ncvslideio::gout(boxes_gapi, labels_gapi), getCompileArgs());

    // Reference code //////////////////////////////////////////////////////////
    parseYoloRef(in_mat, sz, confidence_threshold, nms_threshold, num_classes, anchors, boxes_ref, labels_ref);

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(boxes_gapi == boxes_ref);
    EXPECT_TRUE(labels_gapi == labels_ref);
}

TEST_P(SizeTest, ParseTest)
{
    ncvslideio::GMat in;
    ncvslideio::Size out_sz;

    auto out = ncvslideio::gapi::streaming::size(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_sz), getCompileArgs());

    EXPECT_EQ(sz, out_sz);
}

TEST_P(SizeRTest, ParseTest)
{
    ncvslideio::Rect rect(ncvslideio::Point(0,0), sz);
    ncvslideio::Size out_sz;

    ncvslideio::GOpaque<ncvslideio::Rect> op_rect;
    auto out = ncvslideio::gapi::streaming::size(op_rect);
    ncvslideio::GComputation c(ncvslideio::GIn(op_rect), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(rect), ncvslideio::gout(out_sz), getCompileArgs());

    EXPECT_EQ(sz, out_sz);
}

namespace {
    class TestMediaBGR final : public ncvslideio::MediaFrame::IAdapter {
        ncvslideio::Mat m_mat;

    public:
        explicit TestMediaBGR(ncvslideio::Mat m)
            : m_mat(m) {
        }
        ncvslideio::GFrameDesc meta() const override {
            return ncvslideio::GFrameDesc{ ncvslideio::MediaFormat::BGR, ncvslideio::Size(m_mat.cols, m_mat.rows) };
        }
        ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
            ncvslideio::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
            ncvslideio::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
            return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss));
        }
    };
}

namespace {
    class TestMediaGray final : public ncvslideio::MediaFrame::IAdapter {
        ncvslideio::Mat m_mat;

    public:
        explicit TestMediaGray(ncvslideio::Mat m)
            : m_mat(m) {
        }
        ncvslideio::GFrameDesc meta() const override {
            return ncvslideio::GFrameDesc{ ncvslideio::MediaFormat::GRAY, ncvslideio::Size(m_mat.cols, m_mat.rows) };
        }
        ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
            ncvslideio::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
            ncvslideio::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
            return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss));
        }
    };
}

TEST_P(SizeMFTest, ParseTest)
{
    ncvslideio::Size out_sz;
    ncvslideio::Mat bgr = ncvslideio::Mat::eye(sz.height, sz.width, CV_8UC3);
    ncvslideio::MediaFrame frame = ncvslideio::MediaFrame::Create<TestMediaBGR>(bgr);

    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::streaming::size(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(frame), ncvslideio::gout(out_sz), getCompileArgs());

    EXPECT_EQ(sz, out_sz);
}

TEST_P(SizeMFTest, ParseGrayTest)
{
    ncvslideio::Size out_sz;
    ncvslideio::Mat gray = ncvslideio::Mat::eye(sz.height, sz.width, CV_8UC1);
    ncvslideio::MediaFrame frame = ncvslideio::MediaFrame::Create<TestMediaGray>(gray);

    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::streaming::size(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(frame), ncvslideio::gout(out_sz), getCompileArgs());

    EXPECT_EQ(sz, out_sz);
}

} // opencv_test

#endif //OPENCV_GAPI_CORE_TESTS_INL_HPP
