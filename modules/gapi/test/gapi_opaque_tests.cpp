// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#include "test_precomp.hpp"
#include <string>
#include <utility>

namespace opencv_test
{

namespace ThisTest
{
using GPointOpaque = ncvslideio::GOpaque<ncvslideio::Point>;

G_TYPED_KERNEL(GeneratePoint, <GPointOpaque(GMat)>, "test.opaque.gen_point")
{
    static GOpaqueDesc outMeta(const GMatDesc&) { return empty_gopaque_desc(); }
};

G_TYPED_KERNEL(FillMat, <GMat(ncvslideio::GOpaque<int>, int, int, ncvslideio::Size)>, "test.opaque.fill_mat")
{
    static GMatDesc outMeta(const GOpaqueDesc&, int depth, int chan, ncvslideio::Size size)
    {
        return ncvslideio::GMatDesc{depth, chan, size};
    }
};

G_TYPED_KERNEL(PaintPoint, <GMat(GPointOpaque, int, int, ncvslideio::Size)>, "test.opaque.paint_point")
{
    static GMatDesc outMeta(const GOpaqueDesc&, int depth, int chan, ncvslideio::Size size)
    {
        return ncvslideio::GMatDesc{depth, chan, size};
    }
};

struct MyCustomType{
    int num = -1;
    std::string s;
};

using GOpaq2 = std::tuple<GOpaque<MyCustomType>,GOpaque<MyCustomType>>;

G_TYPED_KERNEL_M(GenerateOpaque, <GOpaq2(GMat, GMat, std::string)>, "test.opaque.gen_point_multy")
{
    static std::tuple<GOpaqueDesc, GOpaqueDesc> outMeta(const GMatDesc&, const GMatDesc&, std::string)
    {
        return std::make_tuple(empty_gopaque_desc(), empty_gopaque_desc());
    }
};

} // namespace ThisTest

namespace
{
GAPI_OCV_KERNEL(OCVGeneratePoint, ThisTest::GeneratePoint)
{
    static void run(const ncvslideio::Mat&, ncvslideio::Point& out)
    {
        out = ncvslideio::Point(42, 42);
    }
};

GAPI_OCL_KERNEL(OCLGeneratePoint, ThisTest::GeneratePoint)
{
    static void run(const ncvslideio::UMat&, ncvslideio::Point& out)
    {
        out = ncvslideio::Point(42, 42);
    }
};

GAPI_OCV_KERNEL(OCVFillMat, ThisTest::FillMat)
{
    static void run(int a, int, int, ncvslideio::Size, ncvslideio::Mat& out)
    {
        out = ncvslideio::Scalar(a);
    }
};

GAPI_OCV_KERNEL(OCVPaintPoint, ThisTest::PaintPoint)
{
    static void run(ncvslideio::Point a, int, int, ncvslideio::Size, ncvslideio::Mat& out)
    {
        out.at<uint8_t>(a) = 77;
    }
};

GAPI_OCL_KERNEL(OCLPaintPoint, ThisTest::PaintPoint)
{
    static void run(ncvslideio::Point a, int depth, int chan, ncvslideio::Size size, ncvslideio::UMat& out)
    {
        GAPI_Assert(chan == 1);
        out.create(size, CV_MAKETYPE(depth, chan));
        ncvslideio::drawMarker(out, a, ncvslideio::Scalar(77));
    }
};

GAPI_OCV_KERNEL(OCVGenerateOpaque, ThisTest::GenerateOpaque)
{
    static void run(const ncvslideio::Mat& a, const ncvslideio::Mat& b, const std::string& s,
                    ThisTest::MyCustomType &out1, ThisTest::MyCustomType &out2)
    {
        out1.num = a.size().width * a.size().height;
        out1.s = s;

        out2.num = b.size().width * b.size().height;
        auto s2 = s;
        std::reverse(s2.begin(), s2.end());
        out2.s = s2;
    }
};
} // (anonymous namespace)

TEST(GOpaque, TestOpaqueOut)
{
    ncvslideio::Mat input = ncvslideio::Mat(52, 52, CV_8U);
    ncvslideio::Point point;

    ncvslideio::GMat in;
    auto out = ThisTest::GeneratePoint::on(in);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(input), ncvslideio::gout(point), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVGeneratePoint>()));

    EXPECT_TRUE(point == ncvslideio::Point(42, 42));
}

TEST(GOpaque, TestOpaqueIn)
{
    ncvslideio::Size sz = {42, 42};
    int depth = CV_8U;
    int chan = 1;
    ncvslideio::Mat mat = ncvslideio::Mat(sz, CV_MAKETYPE(depth, chan));
    int fill = 0;

    ncvslideio::GOpaque<int> in;
    auto out = ThisTest::FillMat::on(in, depth, chan, sz);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(fill), ncvslideio::gout(mat), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVFillMat>()));

    auto diff = ncvslideio::Mat(sz, CV_MAKETYPE(depth, chan), ncvslideio::Scalar(fill)) - mat;
    EXPECT_EQ(0, cvtest::norm(diff, NORM_INF));
}

TEST(GOpaque, TestOpaqueBetween)
{
    ncvslideio::Size sz = {50, 50};
    int depth = CV_8U;
    int chan = 1;
    ncvslideio::Mat mat_in = ncvslideio::Mat::zeros(sz, CV_MAKETYPE(depth, chan));
    ncvslideio::Mat mat_out = ncvslideio::Mat::zeros(sz, CV_MAKETYPE(depth, chan));

    ncvslideio::GMat in, out;
    auto betw = ThisTest::GeneratePoint::on(in);
    out = ThisTest::PaintPoint::on(betw, depth, chan, sz);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(mat_in), ncvslideio::gout(mat_out), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVGeneratePoint, OCVPaintPoint>()));

    int painted = mat_out.at<uint8_t>(42, 42);
    EXPECT_EQ(77, painted);
}

TEST(GOpaque, TestOpaqueBetweenIslands)
{
    ncvslideio::Size sz = {50, 50};
    int depth = CV_8U;
    int chan = 1;
    ncvslideio::Mat mat_in = ncvslideio::Mat::zeros(sz, CV_MAKETYPE(depth, chan));
    ncvslideio::Mat mat_out = ncvslideio::Mat::zeros(sz, CV_MAKETYPE(depth, chan));

    ncvslideio::GMat in, out;
    auto betw = ThisTest::GeneratePoint::on(in);
    out = ThisTest::PaintPoint::on(betw, depth, chan, sz);

    ncvslideio::gapi::island("test", ncvslideio::GIn(in), ncvslideio::GOut(betw));
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(mat_in), ncvslideio::gout(mat_out), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVGeneratePoint, OCVPaintPoint>()));

    int painted = mat_out.at<uint8_t>(42, 42);
    EXPECT_EQ(77, painted);
}

TEST(GOpaque, TestOpaqueCustomOut2)
{
    ncvslideio::Mat input1 = ncvslideio::Mat(52, 52, CV_8U);
    ncvslideio::Mat input2 = ncvslideio::Mat(42, 42, CV_8U);
    std::string str = "opaque";
    std::string str2 = str;
    std::reverse(str2.begin(), str2.end());

    ThisTest::MyCustomType out1, out2;

    ncvslideio::GMat in1, in2;
    auto out = ThisTest::GenerateOpaque::on(in1, in2, str);

    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));
    c.apply(ncvslideio::gin(input1, input2), ncvslideio::gout(out1, out2), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVGenerateOpaque>()));

    EXPECT_EQ(input1.size().width * input1.size().height, out1.num);
    EXPECT_EQ(str, out1.s);

    EXPECT_EQ(input2.size().width * input2.size().height, out2.num);
    EXPECT_EQ(str2, out2.s);
}

TEST(GOpaque, TestOpaqueOCLBackendIn)
{
    ncvslideio::Point p_in = {42, 42};
    ncvslideio::Mat mat_out;

    ThisTest::GPointOpaque in;
    ncvslideio::GMat out = ThisTest::PaintPoint::on(in, CV_8U, 1, {50, 50});

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(p_in), ncvslideio::gout(mat_out),
            ncvslideio::compile_args(ncvslideio::gapi::kernels<OCLPaintPoint>()));

    int painted = mat_out.at<uint8_t>(42, 42);
    EXPECT_EQ(77, painted);
}

TEST(GOpaque, TestOpaqueOCLBackendBetween)
{
    ncvslideio::Size sz = {50, 50};
    int depth   = CV_8U;
    int chan    = 1;
    ncvslideio::Mat mat_in = ncvslideio::Mat::zeros(sz, CV_MAKETYPE(depth, chan));
    ncvslideio::Mat mat_out;

    ncvslideio::GMat in;
    auto     betw = ThisTest::GeneratePoint::on(in);
    ncvslideio::GMat out  = ThisTest::PaintPoint::on(betw, depth, chan, sz);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(mat_in), ncvslideio::gout(mat_out),
            ncvslideio::compile_args(ncvslideio::gapi::kernels<OCLGeneratePoint, OCLPaintPoint>()));

    int painted = mat_out.at<uint8_t>(42, 42);
    EXPECT_EQ(77, painted);
}

TEST(GOpaque, TestOpaqueOCLBackendOut)
{
    ncvslideio::Mat input = ncvslideio::Mat(52, 52, CV_8U);
    ncvslideio::Point p_out;

    ncvslideio::GMat in;
    ThisTest::GPointOpaque out = ThisTest::GeneratePoint::on(in);

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(input), ncvslideio::gout(p_out),
            ncvslideio::compile_args(ncvslideio::gapi::kernels<OCLGeneratePoint>()));

    EXPECT_TRUE(p_out == ncvslideio::Point(42, 42));
}

TEST(GOpaque_OpaqueRef, TestMov)
{
    // Warning: this test is testing some not-very-public APIs
    // Test how OpaqueRef's mov() (aka poor man's move()) is working.

    using I = std::string;

    std::string str = "this string must be long due to short string optimization";
    const I gold(str);

    I test = gold;
    const char* ptr = test.data();

    ncvslideio::detail::OpaqueRef ref(test);
    ncvslideio::detail::OpaqueRef mov;
    mov.reset<I>();

    EXPECT_EQ(gold, ref.rref<I>());         // ref = gold

    mov.mov(ref);
    EXPECT_EQ(gold, mov.rref<I>());         // mov obtained the data
    EXPECT_EQ(ptr,  mov.rref<I>().data());  // pointer is unchanged (same data)
    EXPECT_EQ(test, ref.rref<I>());         // ref = test
    EXPECT_NE(test, mov.rref<I>());         // ref lost the data
}

// types from anonymous namespace doesn't work well with templates
inline namespace gapi_opaque_tests {
    struct MyTestStruct {
        int i;
        float f;
        std::string name;
    };
}

TEST(GOpaque_OpaqueRef, Kind)
{
    ncvslideio::detail::OpaqueRef v1(ncvslideio::Rect{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_RECT, v1.getKind());

    ncvslideio::detail::OpaqueRef v3(int{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_INT, v3.getKind());

    ncvslideio::detail::OpaqueRef v4(double{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_DOUBLE, v4.getKind());

    ncvslideio::detail::OpaqueRef v6(ncvslideio::Point{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_POINT, v6.getKind());

    ncvslideio::detail::OpaqueRef v7(ncvslideio::Size{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_SIZE, v7.getKind());

    ncvslideio::detail::OpaqueRef v8(std::string{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_STRING, v8.getKind());

    ncvslideio::detail::OpaqueRef v9(MyTestStruct{});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_UNKNOWN, v9.getKind());
}

TEST(GOpaque_OpaqueRef, TestReset)
{
    // Warning: this test is testing some not-very-public APIs
    ncvslideio::detail::OpaqueRef opref(int{42});
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_INT, opref.getKind());
    opref.reset<int>();
    EXPECT_EQ(ncvslideio::detail::OpaqueKind::CV_INT, opref.getKind());
}
} // namespace opencv_test
