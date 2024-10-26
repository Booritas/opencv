// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

namespace opencv_test
{

namespace
{
    static ncvslideio::GMat DemoCC(ncvslideio::GMat in, ncvslideio::GScalar scale)
    {
        return ncvslideio::gapi::medianBlur(in + in*scale, 3);
    }

    struct GCompiledValidateMetaTyped: public ::testing::Test
    {
        ncvslideio::GComputationT<ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GScalar)> m_cc;

        GCompiledValidateMetaTyped() : m_cc(DemoCC)
        {
        }
    };

    struct GCompiledValidateMetaUntyped: public ::testing::Test
    {
        ncvslideio::GMat in;
        ncvslideio::GScalar scale;
        ncvslideio::GComputation m_ucc;

        GCompiledValidateMetaUntyped() : m_ucc(ncvslideio::GIn(in, scale),
                                               ncvslideio::GOut(DemoCC(in, scale)))
        {
        }
    };

    struct GCompiledValidateMetaEmpty: public ::testing::Test
    {
        ncvslideio::GMat in;
        ncvslideio::GScalar scale;
        ncvslideio::GComputation m_ucc;

        G_API_OP(GReturn42, <ncvslideio::GOpaque<int>(ncvslideio::GMat)>, "org.opencv.test.return_42")
        {
            static GOpaqueDesc outMeta(ncvslideio::GMatDesc /* in */) { return ncvslideio::empty_gopaque_desc(); }
        };

        GAPI_OCV_KERNEL(GOCVReturn42, GReturn42)
        {
            static void run(const ncvslideio::Mat &/* in */, int &out)
            {
                out = 42;
            }
        };

        GCompiledValidateMetaEmpty() : m_ucc(ncvslideio::GIn(in),
                                             ncvslideio::GOut(GReturn42::on(in)))
        {
        }
    };
} // anonymous namespace

TEST_F(GCompiledValidateMetaTyped, ValidMeta)
{
    ncvslideio::Mat in = ncvslideio::Mat::eye(ncvslideio::Size(128, 32), CV_8UC1);
    ncvslideio::Scalar sc(127);

    auto f = m_cc.compile(ncvslideio::descr_of(in),
                          ncvslideio::descr_of(sc));

    // Correct operation when meta is exactly the same
    ncvslideio::Mat out;
    EXPECT_NO_THROW(f(in, sc, out));

    // Correct operation on next invocation with same meta
    // taken from different input objects
    ncvslideio::Mat in2 = ncvslideio::Mat::zeros(ncvslideio::Size(128, 32), CV_8UC1);
    ncvslideio::Scalar sc2(64);
    ncvslideio::Mat out2;
    EXPECT_NO_THROW(f(in2, sc2, out2));
}

TEST_F(GCompiledValidateMetaTyped, InvalidMeta)
{
    auto f = m_cc.compile(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(64,32)},
                          ncvslideio::empty_scalar_desc());

    ncvslideio::Scalar sc(33);
    ncvslideio::Mat out;

    // 3 channels instead 1
    ncvslideio::Mat in1 = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_8UC3);
    EXPECT_THROW(f(in1, sc, out), std::logic_error);

    // 32f instead 8u
    ncvslideio::Mat in2 = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_32F);
    EXPECT_THROW(f(in2, sc, out), std::logic_error);

    // 32x32 instead of 64x32
    ncvslideio::Mat in3 = ncvslideio::Mat::eye(ncvslideio::Size(32,32), CV_8UC1);
    EXPECT_THROW(f(in3, sc, out), std::logic_error);

    // All is wrong
    ncvslideio::Mat in4 = ncvslideio::Mat::eye(ncvslideio::Size(128,64), CV_32FC3);
    EXPECT_THROW(f(in4, sc, out), std::logic_error);
}

TEST_F(GCompiledValidateMetaUntyped, ValidMeta)
{
    ncvslideio::Mat in1 = ncvslideio::Mat::eye(ncvslideio::Size(128, 32), CV_8UC1);
    ncvslideio::Scalar sc(127);

    auto f = m_ucc.compile(ncvslideio::descr_of(in1),
                           ncvslideio::descr_of(sc));

    // Correct operation when meta is exactly the same
    ncvslideio::Mat out1;
    EXPECT_NO_THROW(f(ncvslideio::gin(in1, sc), ncvslideio::gout(out1)));

    // Correct operation on next invocation with same meta
    // taken from different input objects
    ncvslideio::Mat in2 = ncvslideio::Mat::zeros(ncvslideio::Size(128, 32), CV_8UC1);
    ncvslideio::Scalar sc2(64);
    ncvslideio::Mat out2;
    EXPECT_NO_THROW(f(ncvslideio::gin(in2, sc2), ncvslideio::gout(out2)));
}

TEST_F(GCompiledValidateMetaUntyped, InvalidMetaValues)
{
    auto f = m_ucc.compile(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(64,32)},
                           ncvslideio::empty_scalar_desc());

    ncvslideio::Scalar sc(33);
    ncvslideio::Mat out;

    // 3 channels instead 1
    ncvslideio::Mat in1 = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_8UC3);
    EXPECT_THROW(f(ncvslideio::gin(in1, sc), ncvslideio::gout(out)), std::logic_error);

    // 32f instead 8u
    ncvslideio::Mat in2 = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_32F);
    EXPECT_THROW(f(ncvslideio::gin(in2, sc), ncvslideio::gout(out)), std::logic_error);

    // 32x32 instead of 64x32
    ncvslideio::Mat in3 = ncvslideio::Mat::eye(ncvslideio::Size(32,32), CV_8UC1);
    EXPECT_THROW(f(ncvslideio::gin(in3, sc), ncvslideio::gout(out)), std::logic_error);

    // All is wrong
    ncvslideio::Mat in4 = ncvslideio::Mat::eye(ncvslideio::Size(128,64), CV_32FC3);
    EXPECT_THROW(f(ncvslideio::gin(in4, sc), ncvslideio::gout(out)), std::logic_error);
}

TEST_F(GCompiledValidateMetaUntyped, InvalidMetaShape)
{
    auto f = m_ucc.compile(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(64,32)},
                           ncvslideio::empty_scalar_desc());

    ncvslideio::Mat in1 = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_8UC1);
    ncvslideio::Scalar sc(33);
    ncvslideio::Mat out1;

    // call as f(Mat,Mat) while f(Mat,Scalar) is expected
    EXPECT_THROW(f(ncvslideio::gin(in1, in1), ncvslideio::gout(out1)), std::logic_error);

    // call as f(Scalar,Mat) while f(Mat,Scalar) is expected
    EXPECT_THROW(f(ncvslideio::gin(sc, in1), ncvslideio::gout(out1)), std::logic_error);

    // call as f(Scalar,Scalar) while f(Mat,Scalar) is expected
    EXPECT_THROW(f(ncvslideio::gin(sc, sc), ncvslideio::gout(out1)), std::logic_error);
}

TEST_F(GCompiledValidateMetaUntyped, InvalidMetaNumber)
{
    auto f = m_ucc.compile(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(64,32)},
                           ncvslideio::empty_scalar_desc());

    ncvslideio::Mat in1 = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_8UC1);
    ncvslideio::Scalar sc(33);
    ncvslideio::Mat out1, out2;

    // call as f(Mat,Scalar,Scalar) while f(Mat,Scalar) is expected
    EXPECT_THROW(f(ncvslideio::gin(in1, sc, sc), ncvslideio::gout(out1)), std::logic_error);

    // call as f(Scalar,Mat,Scalar) while f(Mat,Scalar) is expected
    EXPECT_THROW(f(ncvslideio::gin(sc, in1, sc), ncvslideio::gout(out1)), std::logic_error);

    // call as f(Scalar) while f(Mat,Scalar) is expected
    EXPECT_THROW(f(ncvslideio::gin(sc), ncvslideio::gout(out1)), std::logic_error);

    // call as f(Mat,Scalar,[out1],[out2]) while f(Mat,Scalar,[out]) is expected
    EXPECT_THROW(f(ncvslideio::gin(in1, sc), ncvslideio::gout(out1, out2)), std::logic_error);
}

TEST_F(GCompiledValidateMetaEmpty, InvalidMatMetaCompile)
{
    EXPECT_THROW(m_ucc.compile(ncvslideio::empty_gmat_desc(),
                               ncvslideio::empty_scalar_desc()),
                 std::logic_error);
}

TEST_F(GCompiledValidateMetaEmpty, InvalidMatMetaApply)
{
    ncvslideio::Mat emptyIn;
    int out {};
    const auto pkg = ncvslideio::gapi::kernels<GCompiledValidateMetaEmpty::GOCVReturn42>();

    EXPECT_THROW(m_ucc.apply(ncvslideio::gin(emptyIn), ncvslideio::gout(out), ncvslideio::compile_args(pkg)),
                 std::logic_error);
}

TEST_F(GCompiledValidateMetaEmpty, ValidInvalidMatMetasApply)
{
    int out {};
    const auto pkg = ncvslideio::gapi::kernels<GCompiledValidateMetaEmpty::GOCVReturn42>();

    ncvslideio::Mat nonEmptyMat = ncvslideio::Mat::eye(ncvslideio::Size(64,32), CV_8UC1);
    m_ucc.apply(ncvslideio::gin(nonEmptyMat), ncvslideio::gout(out), ncvslideio::compile_args(pkg));
    EXPECT_EQ(out, 42);

    ncvslideio::Mat emptyIn;
    EXPECT_THROW(m_ucc.apply(ncvslideio::gin(emptyIn), ncvslideio::gout(out), ncvslideio::compile_args(pkg)),
                 std::logic_error);

    out = 0;
    m_ucc.apply(ncvslideio::gin(nonEmptyMat), ncvslideio::gout(out), ncvslideio::compile_args(pkg));
    EXPECT_EQ(out, 42);
}
} // namespace opencv_test
