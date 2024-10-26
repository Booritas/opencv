// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <iostream>

namespace opencv_test
{

TEST(GAPI_Scalar, Argument)
{
    ncvslideio::Size sz(2, 2);
    ncvslideio::Mat in_mat(sz, CV_8U);
    ncvslideio::randn(in_mat, ncvslideio::Scalar::all(127), ncvslideio::Scalar::all(40.f));

    ncvslideio::GComputationT<ncvslideio::GMat (ncvslideio::GMat, ncvslideio::GScalar)> mulS([](ncvslideio::GMat in, ncvslideio::GScalar c)
    {
        return in*c;
    });

    ncvslideio::Mat out_mat(sz, CV_8U);
    mulS.apply(in_mat, ncvslideio::Scalar(2), out_mat);

    ncvslideio::Mat reference = in_mat*2;
    EXPECT_EQ(0, cvtest::norm(out_mat, reference, NORM_INF));
}

TEST(GAPI_Scalar, ReturnValue)
{
    const ncvslideio::Size sz(2, 2);
    ncvslideio::Mat in_mat(sz, CV_8U, ncvslideio::Scalar(1));

    ncvslideio::GComputationT<ncvslideio::GScalar (ncvslideio::GMat)> sum_of_sum([](ncvslideio::GMat in)
    {
        return ncvslideio::gapi::sum(in + in);
    });

    ncvslideio::Scalar out;
    sum_of_sum.apply(in_mat, out);

    EXPECT_EQ(8, out[0]);
}

TEST(GAPI_Scalar, TmpScalar)
{
    const ncvslideio::Size sz(2, 2);
    ncvslideio::Mat in_mat(sz, CV_8U, ncvslideio::Scalar(1));

    ncvslideio::GComputationT<ncvslideio::GMat (ncvslideio::GMat)> mul_by_sum([](ncvslideio::GMat in)
    {
        return in * ncvslideio::gapi::sum(in);
    });

    ncvslideio::Mat out_mat(sz, CV_8U);
    mul_by_sum.apply(in_mat, out_mat);

    ncvslideio::Mat reference = ncvslideio::Mat(sz, CV_8U, ncvslideio::Scalar(4));
    EXPECT_EQ(0, cvtest::norm(out_mat, reference, NORM_INF));
}

TEST(GAPI_ScalarWithValue, Simple_Arithmetic_Pipeline)
{
    GMat in;
    GMat out = (in + 1) * 2;
    ncvslideio::GComputation comp(in, out);

    ncvslideio::Mat in_mat  = ncvslideio::Mat::eye(3, 3, CV_8UC1);
    ncvslideio::Mat ref_mat, out_mat;

    ref_mat = (in_mat + 1) * 2;
    comp.apply(in_mat, out_mat);

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_ScalarWithValue, GScalar_Initilization)
{
    ncvslideio::Scalar sc(2);
    ncvslideio::GMat in;
    ncvslideio::GScalar s(sc);
    ncvslideio::GComputation comp(in, ncvslideio::gapi::mulC(in, s));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC1);
    ncvslideio::Mat ref_mat, out_mat;
    ncvslideio::multiply(in_mat, sc, ref_mat, 1, CV_8UC1);
    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_ScalarWithValue, Constant_GScalar_In_Middle_Graph)
{
    ncvslideio::Scalar  sc(5);
    ncvslideio::GMat    in1;
    ncvslideio::GScalar in2;
    ncvslideio::GScalar s(sc);

    auto add_out = ncvslideio::gapi::addC(in1, in2);
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(ncvslideio::gapi::mulC(add_out, s)));

    ncvslideio::Mat    in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC1);
    ncvslideio::Scalar in_scalar(3);

    ncvslideio::Mat ref_mat, out_mat, add_mat;
    ncvslideio::add(in_mat, in_scalar, add_mat);
    ncvslideio::multiply(add_mat, sc, ref_mat, 1, CV_8UC1);
    comp.apply(ncvslideio::gin(in_mat, in_scalar), ncvslideio::gout(out_mat));

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

} // namespace opencv_test
