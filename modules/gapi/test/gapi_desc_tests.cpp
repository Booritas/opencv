// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>

namespace opencv_test
{

namespace
{
    G_TYPED_KERNEL(KTest, <ncvslideio::GScalar(ncvslideio::GScalar)>, "org.opencv.test.scalar_kernel") {
        static ncvslideio::GScalarDesc outMeta(ncvslideio::GScalarDesc in) { return in; }
    };
    GAPI_OCV_KERNEL(GOCVScalarTest, KTest)
    {
        static void run(const ncvslideio::Scalar &in, ncvslideio::Scalar &out) { out = in+ncvslideio::Scalar(1); }
    };
}

TEST(GAPI_MetaDesc, MatDescOneCh)
{
    ncvslideio::Mat mat(240, 320, CV_8U);

    const auto desc = ncvslideio::descr_of(mat);

    EXPECT_EQ(CV_8U, desc.depth);
    EXPECT_EQ(1,     desc.chan);
    EXPECT_EQ(320,   desc.size.width);
    EXPECT_EQ(240,   desc.size.height);
    EXPECT_FALSE(desc.isND());
}

TEST(GAPI_MetaDesc, MatDescThreeCh)
{
    ncvslideio::Mat mat(480, 640, CV_8UC3);

    const auto desc = ncvslideio::descr_of(mat);

    EXPECT_EQ(CV_8U,   desc.depth);
    EXPECT_EQ(3,       desc.chan);
    EXPECT_EQ(640,     desc.size.width);
    EXPECT_EQ(480,     desc.size.height);
    EXPECT_FALSE(desc.isND());
}

TEST(GAPI_MetaDesc, MatDescND)
{
    std::vector<int> dims = {1,3,299,299};
    ncvslideio::Mat m(dims, CV_32F);
    const auto desc = ncvslideio::descr_of(m);
    EXPECT_EQ(CV_32F, desc.depth);
    EXPECT_EQ(-1,     desc.chan);
    EXPECT_EQ(1,      desc.dims[0]);
    EXPECT_EQ(3,      desc.dims[1]);
    EXPECT_EQ(299,    desc.dims[2]);
    EXPECT_EQ(299,    desc.dims[3]);
    EXPECT_TRUE(desc.isND());
}

TEST(GAPI_MetaDesc, VecMatDesc)
{
    std::vector<ncvslideio::Mat> vec1 = {
    ncvslideio::Mat(240, 320, CV_8U)};

    const auto desc1 = ncvslideio::descrs_of(vec1);
    EXPECT_EQ((GMatDesc{CV_8U, 1, {320, 240}}), get<GMatDesc>(desc1[0]));

    std::vector<ncvslideio::UMat> vec2 = {
    ncvslideio::UMat(480, 640, CV_8UC3)};

    const auto desc2 = ncvslideio::descrs_of(vec2);
    EXPECT_EQ((GMatDesc{CV_8U, 3, {640, 480}}), get<GMatDesc>(desc2[0]));
}

TEST(GAPI_MetaDesc, CanDescribe)
{
    constexpr int w = 15;
    constexpr int h = 7;
    ncvslideio::Mat m0(h, w, CV_8UC3);
    ncvslideio::GMatDesc md0{CV_8U,3,{w,h},false};

    ncvslideio::Mat m1(h*3, w, CV_8UC1);
    ncvslideio::GMatDesc md10{CV_8U,3,{w,h},true};
    ncvslideio::GMatDesc md11{CV_8U,1,{w,h*3},false};

    EXPECT_TRUE (md0 .canDescribe(m0));
    EXPECT_FALSE(md0 .canDescribe(m1));
    EXPECT_TRUE (md10.canDescribe(m1));
    EXPECT_TRUE (md11.canDescribe(m1));
}

TEST(GAPI_MetaDesc, OwnMatDescOneCh)
{
    ncvslideio::gapi::own::Mat mat(240, 320, CV_8U, nullptr);

    const auto desc = ncvslideio::gapi::own::descr_of(mat);

    EXPECT_EQ(CV_8U, desc.depth);
    EXPECT_EQ(1,     desc.chan);
    EXPECT_EQ(320,   desc.size.width);
    EXPECT_EQ(240,   desc.size.height);
    EXPECT_FALSE(desc.isND());
}

TEST(GAPI_MetaDesc, OwnMatDescThreeCh)
{
    ncvslideio::gapi::own::Mat mat(480, 640, CV_8UC3, nullptr);

    const auto desc = ncvslideio::gapi::own::descr_of(mat);

    EXPECT_EQ(CV_8U,   desc.depth);
    EXPECT_EQ(3,       desc.chan);
    EXPECT_EQ(640,     desc.size.width);
    EXPECT_EQ(480,     desc.size.height);
    EXPECT_FALSE(desc.isND());
}

TEST(GAPI_MetaDesc, OwnMatDescND)
{
    std::vector<int> dims = {1,3,224,224};
    ncvslideio::gapi::own::Mat m(dims, CV_32F, nullptr);

    const auto desc = ncvslideio::gapi::own::descr_of(m);

    EXPECT_EQ(CV_32F, desc.depth);
    EXPECT_EQ(-1,     desc.chan);
    EXPECT_EQ(1,      desc.dims[0]);
    EXPECT_EQ(3,      desc.dims[1]);
    EXPECT_EQ(224,    desc.dims[2]);
    EXPECT_EQ(224,    desc.dims[3]);
    EXPECT_TRUE(desc.isND());
}

TEST(GAPI_MetaDesc, VecOwnMatDesc)
{
    std::vector<ncvslideio::gapi::own::Mat> vec = {
    ncvslideio::gapi::own::Mat(240, 320, CV_8U, nullptr),
    ncvslideio::gapi::own::Mat(480, 640, CV_8UC3, nullptr)};

    const auto desc = ncvslideio::gapi::own::descrs_of(vec);

    EXPECT_EQ((GMatDesc{CV_8U, 1, {320, 240}}), get<GMatDesc>(desc[0]));
    EXPECT_EQ((GMatDesc{CV_8U, 3, {640, 480}}), get<GMatDesc>(desc[1]));
}

TEST(GAPI_MetaDesc, AdlVecOwnMatDesc)
{
    std::vector<ncvslideio::gapi::own::Mat> vec = {
    ncvslideio::gapi::own::Mat(240, 320, CV_8U, nullptr),
    ncvslideio::gapi::own::Mat(480, 640, CV_8UC3, nullptr)};

    const auto desc = descrs_of(vec);

    EXPECT_EQ((GMatDesc{CV_8U, 1, {320, 240}}), get<GMatDesc>(desc[0]));
    EXPECT_EQ((GMatDesc{CV_8U, 3, {640, 480}}), get<GMatDesc>(desc[1]));
}

TEST(GAPI_MetaDesc, Compare_Equal_MatDesc)
{
    const auto desc1 = ncvslideio::GMatDesc{CV_8U, 1, {64, 64}};
    const auto desc2 = ncvslideio::GMatDesc{CV_8U, 1, {64, 64}};

    EXPECT_TRUE(desc1 == desc2);
}

TEST(GAPI_MetaDesc, Compare_Not_Equal_MatDesc)
{
    const auto desc1 = ncvslideio::GMatDesc{CV_8U,  1, {64, 64}};
    const auto desc2 = ncvslideio::GMatDesc{CV_32F, 1, {64, 64}};

    EXPECT_TRUE(desc1 != desc2);
}

TEST(GAPI_MetaDesc, Compare_Equal_MatDesc_ND)
{
    const auto desc1 = ncvslideio::GMatDesc{CV_8U, {1,3,224,224}};
    const auto desc2 = ncvslideio::GMatDesc{CV_8U, {1,3,224,224}};

    EXPECT_TRUE(desc1 == desc2);
}

TEST(GAPI_MetaDesc, Compare_Not_Equal_MatDesc_ND_1)
{
    const auto desc1 = ncvslideio::GMatDesc{CV_8U,  {1,1000}};
    const auto desc2 = ncvslideio::GMatDesc{CV_32F, {1,1000}};

    EXPECT_TRUE(desc1 != desc2);
}

TEST(GAPI_MetaDesc, Compare_Not_Equal_MatDesc_ND_2)
{
    const auto desc1 = ncvslideio::GMatDesc{CV_8U, {1,1000}};
    const auto desc2 = ncvslideio::GMatDesc{CV_8U, {1,1400}};

    EXPECT_TRUE(desc1 != desc2);
}

TEST(GAPI_MetaDesc, Compare_Not_Equal_MatDesc_ND_3)
{
    const auto desc1 = ncvslideio::GMatDesc{CV_8U, {1,1000}};
    const auto desc2 = ncvslideio::GMatDesc{CV_8U, 1, {32,32}};

    EXPECT_TRUE(desc1 != desc2);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaNumber_1)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, in+in);

    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = ncvslideio::GMatDesc{CV_32F,1,{128,128}};

    EXPECT_NO_THROW(cc.compile(desc1));
    EXPECT_NO_THROW(cc.compile(desc2));

    // FIXME: custom exception type?
    // It is worth checking if compilation fails with different number
    // of meta parameters
    EXPECT_THROW(cc.compile(desc1, desc1),        std::logic_error);
    EXPECT_THROW(cc.compile(desc1, desc2, desc2), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaNumber_2)
{
    ncvslideio::GMat a, b;
    ncvslideio::GComputation cc(ncvslideio::GIn(a, b), ncvslideio::GOut(a+b));

    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    EXPECT_NO_THROW(cc.compile(desc1, desc1));

    const auto desc2 = ncvslideio::GMatDesc{CV_32F,1,{128,128}};
    EXPECT_NO_THROW(cc.compile(desc2, desc2));

    // FIXME: custom exception type?
    EXPECT_THROW(cc.compile(desc1),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2, desc2, desc2), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaType_Mat)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation cc(in, in+in);

    EXPECT_NO_THROW(cc.compile(ncvslideio::GMatDesc{CV_8U,1,{64,64}}));

    // FIXME: custom exception type?
    EXPECT_THROW(cc.compile(ncvslideio::empty_scalar_desc()), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaType_Scalar)
{
    ncvslideio::GScalar in;
    ncvslideio::GComputation cc(ncvslideio::GIn(in), ncvslideio::GOut(KTest::on(in)));

    const auto desc1 = ncvslideio::descr_of(ncvslideio::Scalar(128));
    const auto desc2 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    const auto pkg   = ncvslideio::gapi::kernels<GOCVScalarTest>();
    EXPECT_NO_THROW(cc.compile(desc1, ncvslideio::compile_args(pkg)));

    // FIXME: custom exception type?
    EXPECT_THROW(cc.compile(desc2, ncvslideio::compile_args(pkg)), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaType_Mixed)
{
    ncvslideio::GMat a;
    ncvslideio::GScalar v;
    ncvslideio::GComputation cc(ncvslideio::GIn(a, v), ncvslideio::GOut(ncvslideio::gapi::addC(a, v)));

    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = ncvslideio::descr_of(ncvslideio::Scalar(4));

    EXPECT_NO_THROW(cc.compile(desc1, desc2));

    // FIXME: custom exception type(s)?
    EXPECT_THROW(cc.compile(desc1),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2, desc1),        std::logic_error);
    EXPECT_THROW(cc.compile(desc1, desc1, desc1), std::logic_error);
    EXPECT_THROW(cc.compile(desc1, desc2, desc1), std::logic_error);
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaNumber_1)
{
    ncvslideio::GComputationT<ncvslideio::GMat(ncvslideio::GMat)> cc([](ncvslideio::GMat in)
    {
        return in+in;
    });

    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = ncvslideio::GMatDesc{CV_32F,1,{128,128}};

    EXPECT_NO_THROW(cc.compile(desc1));
    EXPECT_NO_THROW(cc.compile(desc2));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaNumber_2)
{
    ncvslideio::GComputationT<ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GMat)> cc([](ncvslideio::GMat a, ncvslideio::GMat b)
    {
        return a + b;
    });

    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    EXPECT_NO_THROW(cc.compile(desc1, desc1));

    const auto desc2 = ncvslideio::GMatDesc{CV_32F,1,{128,128}};
    EXPECT_NO_THROW(cc.compile(desc2, desc2));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaType_Mat)
{
    ncvslideio::GComputationT<ncvslideio::GMat(ncvslideio::GMat)> cc([](ncvslideio::GMat in)
    {
        return in+in;
    });

    EXPECT_NO_THROW(cc.compile(ncvslideio::GMatDesc{CV_8U,1,{64,64}}));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaType_Scalar)
{
    ncvslideio::GComputationT<ncvslideio::GScalar(ncvslideio::GScalar)> cc([](ncvslideio::GScalar in)
    {
        return KTest::on(in);
    });

    const auto desc1 = ncvslideio::descr_of(ncvslideio::Scalar(128));
    const auto pkg = ncvslideio::gapi::kernels<GOCVScalarTest>();
    //     EXPECT_NO_THROW(cc.compile(desc1, ncvslideio::compile_args(pkg)));
    cc.compile(desc1, ncvslideio::compile_args(pkg));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaType_Mixed)
{
    ncvslideio::GComputationT<ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GScalar)> cc([](ncvslideio::GMat a, ncvslideio::GScalar v)
    {
        return ncvslideio::gapi::addC(a, v);
    });

    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = ncvslideio::descr_of(ncvslideio::Scalar(4));

    EXPECT_NO_THROW(cc.compile(desc1, desc2));
}

TEST(GAPI_MetaDesc, Compare_Planar)
{
    const auto desc0 = ncvslideio::GMatDesc{CV_8U,3,{32,32},false};
    const auto desc1 = ncvslideio::GMatDesc{CV_8U,3,{32,32},false};
    const auto desc2 = ncvslideio::GMatDesc{CV_8U,3,{32,32},true};
    const auto desc3 = ncvslideio::GMatDesc{CV_8U,3,{64,64},true};

    EXPECT_TRUE(desc0 == desc1);
    EXPECT_TRUE(desc1 != desc2);
    EXPECT_TRUE(desc1 != desc3);
    EXPECT_TRUE(desc2 != desc3);
}

TEST(GAPI_MetaDesc, Sanity_asPlanar)
{
    constexpr int w = 32;
    constexpr int h = 16;
    const auto desc1 = ncvslideio::GMatDesc{CV_8U,3,{w,h},false};
    const auto desc2 = ncvslideio::GMatDesc{CV_8U,3,{w,h},true};

    EXPECT_NO_THROW(desc1.asPlanar());
    EXPECT_NO_THROW(desc2.asInterleaved());
    EXPECT_ANY_THROW(desc1.asInterleaved());
    EXPECT_ANY_THROW(desc2.asPlanar());
}

TEST(GAPI_MetaDesc, Compare_asPlanar)
{
    constexpr int w = 32;
    constexpr int h = 64;
    const auto desc0 = ncvslideio::GMatDesc{CV_8U,3,{w,h},false};
    const auto desc1 = ncvslideio::GMatDesc{CV_8U,3,{w,h},true};

    EXPECT_TRUE(desc0.asPlanar()      == desc1);
    EXPECT_TRUE(desc1.asInterleaved() == desc0);
}

TEST(GAPI_MetaDesc, Compare_asPlanarTransform)
{
    constexpr int w = 64;
    constexpr int h = 32;
    const auto desc0 = ncvslideio::GMatDesc{CV_8U,3,{w,h},true};
    const auto desc1 = ncvslideio::GMatDesc{CV_8U,1,{w,h*3},false};

    EXPECT_ANY_THROW(desc0.asPlanar(3));
    EXPECT_NO_THROW(desc1.asPlanar(3));
    EXPECT_TRUE(desc1.asPlanar(3) == desc0);
}

} // namespace opencv_test
