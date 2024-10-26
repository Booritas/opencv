// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

namespace opencv_test
{
    typedef ::testing::Types<ncvslideio::GMat, ncvslideio::GMatP, ncvslideio::GFrame,
                             ncvslideio::GScalar, ncvslideio::GOpaque<int>,
                             ncvslideio::GArray<int>> VectorProtoTypes;

    template<typename T> struct DynamicGraphProtoArgs: public ::testing::Test { using Type = T; };

    TYPED_TEST_CASE(DynamicGraphProtoArgs, VectorProtoTypes);

    TYPED_TEST(DynamicGraphProtoArgs, AddProtoInputArgsSmoke)
    {
        using T = typename TestFixture::Type;
        auto ins = GIn();
        T in;
        EXPECT_NO_THROW(ins += GIn(in));
    }

    TYPED_TEST(DynamicGraphProtoArgs, AddProtoInputArgs)
    {
        using T = typename TestFixture::Type;
        T in1, in2;

        auto ins1 = GIn();
        ins1 += GIn(in1);
        ins1 += GIn(in2);

        auto ins2 = GIn(in1, in2);

        EXPECT_EQ(ins1.m_args.size(), ins2.m_args.size());
    }

    TYPED_TEST(DynamicGraphProtoArgs, AddProtoOutputArgsSmoke)
    {
        using T = typename TestFixture::Type;
        auto outs = GOut();
        T out;
        EXPECT_NO_THROW(outs += GOut(out));
    }

    TYPED_TEST(DynamicGraphProtoArgs, AddProtoOutputArgs)
    {
        using T = typename TestFixture::Type;
        T out1, out2;

        auto outs1 = GOut();
        outs1 += GOut(out1);
        outs1 += GOut(out2);

        auto outs2 = GOut(out1, out2);

        EXPECT_EQ(outs1.m_args.size(), outs2.m_args.size());
    }

    typedef ::testing::Types<ncvslideio::Mat,
#if !defined(GAPI_STANDALONE)
                             ncvslideio::UMat,
#endif // !defined(GAPI_STANDALONE)
                             ncvslideio::Scalar,
                             ncvslideio::detail::VectorRef,
                             ncvslideio::detail::OpaqueRef> VectorRunTypes;

    template<typename T> struct DynamicGraphRunArgs: public ::testing::Test { using Type = T; };

    TYPED_TEST_CASE(DynamicGraphRunArgs, VectorRunTypes);

    TYPED_TEST(DynamicGraphRunArgs, AddRunArgsSmoke)
    {
        auto in_vector = ncvslideio::gin();

        using T = typename TestFixture::Type;
        T in;
        EXPECT_NO_THROW(in_vector += ncvslideio::gin(in));
    }

    TYPED_TEST(DynamicGraphRunArgs, AddRunArgs)
    {
        using T = typename TestFixture::Type;
        T in1, in2;

        auto in_vector1 = ncvslideio::gin();
        in_vector1 += ncvslideio::gin(in1);
        in_vector1 += ncvslideio::gin(in2);

        auto in_vector2 = ncvslideio::gin(in1, in2);

        EXPECT_EQ(in_vector1.size(), in_vector2.size());
    }

    TYPED_TEST(DynamicGraphRunArgs, AddRunArgsPSmoke)
    {
        auto out_vector = ncvslideio::gout();

        using T = typename TestFixture::Type;
        T out;
        EXPECT_NO_THROW(out_vector += ncvslideio::gout(out));
    }

    TYPED_TEST(DynamicGraphRunArgs, AddRunArgsP)
    {
        using T = typename TestFixture::Type;
        T out1, out2;

        auto out_vector1 = ncvslideio::gout();
        out_vector1 += ncvslideio::gout(out1);
        out_vector1 += ncvslideio::gout(out2);

        auto out_vector2 = ncvslideio::gout(out1, out2);

        EXPECT_EQ(out_vector1.size(), out_vector2.size());
    }

    TEST(DynamicGraph, ProtoInputArgsExecute)
    {
        ncvslideio::GComputation cc([]() {
            ncvslideio::GMat in1;
            auto ins = GIn(in1);

            ncvslideio::GMat in2;
            ins += GIn(in2);

            ncvslideio::GMat out = ncvslideio::gapi::copy(in1 + in2);

            return ncvslideio::GComputation(std::move(ins), GOut(out));
        });

        ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat in_mat2 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat out_mat;

        EXPECT_NO_THROW(cc.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat)));
    }

    TEST(DynamicGraph, ProtoOutputArgsExecute)
    {
        ncvslideio::GComputation cc([]() {
            ncvslideio::GMat in;
            ncvslideio::GMat out1 = ncvslideio::gapi::copy(in);
            auto outs = GOut(out1);

            ncvslideio::GMat out2 = ncvslideio::gapi::copy(in);
            outs += GOut(out2);

            return ncvslideio::GComputation(ncvslideio::GIn(in), std::move(outs));
        });

        ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat out_mat1;
        ncvslideio::Mat out_mat2;

        EXPECT_NO_THROW(cc.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(out_mat1, out_mat1)));
    }

    TEST(DynamicGraph, ProtoOutputInputArgsExecute)
    {
        ncvslideio::GComputation cc([]() {
            ncvslideio::GMat in1;
            auto ins = GIn(in1);

            ncvslideio::GMat in2;
            ins += GIn(in2);

            ncvslideio::GMat out1 = ncvslideio::gapi::copy(in1 + in2);
            auto outs = GOut(out1);

            ncvslideio::GMat out2 = ncvslideio::gapi::copy(in1 + in2);
            outs += GOut(out2);

            return ncvslideio::GComputation(std::move(ins), std::move(outs));
        });

        ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat in_mat2 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat out_mat1, out_mat2;

        EXPECT_NO_THROW(cc.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat1, out_mat2)));
    }

    TEST(DynamicGraph, ProtoArgsExecute)
    {
        ncvslideio::GComputation cc([]() {
            ncvslideio::GMat in1;
            auto ins = GIn(in1);

            ncvslideio::GMat in2;
            ins += GIn(in2);

            ncvslideio::GMat out1 = ncvslideio::gapi::copy(in1 + in2);
            auto outs = GOut(out1);

            ncvslideio::GMat out2 = ncvslideio::gapi::copy(in1 + in2);
            outs += GOut(out2);

            return ncvslideio::GComputation(std::move(ins), std::move(outs));
        });

        ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat in_mat2 = ncvslideio::Mat::eye(32, 32, CV_8UC1);
        ncvslideio::Mat out_mat1, out_mat2;

        EXPECT_NO_THROW(cc.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat1, out_mat2)));
    }

    TEST(DynamicGraph, ProtoOutputInputArgsAccuracy)
    {
        ncvslideio::Size szOut(4, 4);
        ncvslideio::GComputation cc([&](){
            ncvslideio::GMat in1;
            auto ins = GIn(in1);

            ncvslideio::GMat in2;
            ins += GIn(in2);

            ncvslideio::GMat out1 = ncvslideio::gapi::resize(in1, szOut);
            auto outs = GOut(out1);

            ncvslideio::GMat out2 = ncvslideio::gapi::resize(in2, szOut);
            outs += GOut(out2);

            return ncvslideio::GComputation(std::move(ins), std::move(outs));
        });

        // G-API test code
        ncvslideio::Mat in_mat1( 8,  8, CV_8UC3);
        ncvslideio::Mat in_mat2(16, 16, CV_8UC3);
        ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
        ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

        auto in_vector = ncvslideio::gin();
        in_vector += ncvslideio::gin(in_mat1);
        in_vector += ncvslideio::gin(in_mat2);

        ncvslideio::Mat out_mat1, out_mat2;
        auto out_vector = ncvslideio::gout();
        out_vector += ncvslideio::gout(out_mat1);
        out_vector += ncvslideio::gout(out_mat2);

        cc.apply(std::move(in_vector), std::move(out_vector));

        // OCV ref code
        ncvslideio::Mat cv_out_mat1, cv_out_mat2;
        ncvslideio::resize(in_mat1, cv_out_mat1, szOut);
        ncvslideio::resize(in_mat2, cv_out_mat2, szOut);

        EXPECT_EQ(0, cvtest::norm(out_mat1, cv_out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat2, cv_out_mat2, NORM_INF));
    }

    TEST(DynamicGraph, Streaming)
    {
        ncvslideio::GComputation cc([&](){
            ncvslideio::Size szOut(4, 4);

            ncvslideio::GMat in1;
            auto ins = GIn(in1);

            ncvslideio::GMat in2;
            ins += GIn(in2);

            ncvslideio::GMat out1 = ncvslideio::gapi::resize(in1, szOut);
            auto outs = GOut(out1);

            ncvslideio::GMat out2 = ncvslideio::gapi::resize(in2, szOut);
            outs += GOut(out2);

            return ncvslideio::GComputation(std::move(ins), std::move(outs));
        });

        EXPECT_NO_THROW(cc.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::core::cpu::kernels())));
    }

    TEST(DynamicGraph, StreamingAccuracy)
    {
        ncvslideio::Size szOut(4, 4);
        ncvslideio::GComputation cc([&](){
            ncvslideio::GMat in1;
            auto ins = GIn(in1);

            ncvslideio::GMat in2;
            ins += GIn(in2);

            ncvslideio::GMat out1 = ncvslideio::gapi::resize(in1, szOut);
            ncvslideio::GProtoOutputArgs outs = GOut(out1);

            ncvslideio::GMat out2 = ncvslideio::gapi::resize(in2, szOut);
            outs += GOut(out2);
            return ncvslideio::GComputation(std::move(ins), std::move(outs));
        });

        // G-API test code
        ncvslideio::Mat in_mat1( 8,  8, CV_8UC3);
        ncvslideio::Mat in_mat2(16, 16, CV_8UC3);
        ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
        ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

        auto in_vector = ncvslideio::gin();
        in_vector += ncvslideio::gin(in_mat1);
        in_vector += ncvslideio::gin(in_mat2);

        ncvslideio::Mat out_mat1, out_mat2;
        auto out_vector = ncvslideio::gout();
        out_vector += ncvslideio::gout(out_mat1);
        out_vector += ncvslideio::gout(out_mat2);

        auto stream = cc.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::core::cpu::kernels()));
        stream.setSource(std::move(in_vector));

        stream.start();
        stream.pull(std::move(out_vector));
        stream.stop();

        // OCV ref code
        ncvslideio::Mat cv_out_mat1, cv_out_mat2;
        ncvslideio::resize(in_mat1, cv_out_mat1, szOut);
        ncvslideio::resize(in_mat2, cv_out_mat2, szOut);

        EXPECT_EQ(0, cvtest::norm(out_mat1, cv_out_mat1, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(out_mat2, cv_out_mat2, NORM_INF));
    }
} // namespace opencv_test
