// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "../test_precomp.hpp"

#include <ade/util/iota_range.hpp>
#include <opencv2/gapi/s11n.hpp>
#include "api/render_priv.hpp"
#include "../common/gapi_render_tests.hpp"

namespace opencv_test
{

TEST(S11N, Pipeline_Crop_Rect)
{
    ncvslideio::Rect rect_to{ 4,10,37,50 };
    ncvslideio::Size sz_in = ncvslideio::Size(1920, 1080);
    ncvslideio::Size sz_out = ncvslideio::Size(37, 50);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz_in, CV_8UC1);
    ncvslideio::Mat out_mat_gapi(sz_out, CV_8UC1);
    ncvslideio::Mat out_mat_ocv(sz_out, CV_8UC1);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::crop(in, rect_to);
    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(in, out));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);
    c.apply(in_mat, out_mat_gapi);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        out_mat_ocv = in_mat(rect_to);
    }
    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
    }
}


TEST(S11N, Pipeline_Canny_Bool)
{
    const ncvslideio::Size sz_in(1280, 720);
    ncvslideio::GMat in;
    double thrLow = 120.0;
    double thrUp = 240.0;
    int apSize = 5;
    bool l2gr = true;
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(1280, 720, CV_8UC1);
    ncvslideio::Mat out_mat_gapi(sz_in, CV_8UC1);
    ncvslideio::Mat out_mat_ocv(sz_in, CV_8UC1);

    // G-API code //////////////////////////////////////////////////////////////
    auto out = ncvslideio::gapi::Canny(in, thrLow, thrUp, apSize, l2gr);
    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(in, out));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);
    c.apply(in_mat, out_mat_gapi);

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Canny(in_mat, out_mat_ocv, thrLow, thrUp, apSize, l2gr);
    }
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

TEST(S11N, Pipeline_Not)
{
    ncvslideio::GMat in;
    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(in, ncvslideio::gapi::bitwise_not(in)));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat ref_mat = ~in_mat;

    ncvslideio::Mat out_mat;
    c.apply(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = ncvslideio::Mat();
    auto cc = c.compile(ncvslideio::descr_of(in_mat));
    cc(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(S11N, Pipeline_Sum_Scalar)
{
    ncvslideio::GMat in;
    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(in, ncvslideio::gapi::sum(in)));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Scalar ref_scl = ncvslideio::sum(in_mat);

    ncvslideio::Scalar out_scl;
    c.apply(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = ncvslideio::Scalar();
    auto cc = c.compile(ncvslideio::descr_of(in_mat));
    cc(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(S11N, Pipeline_BinaryOp)
{
    ncvslideio::GMat a, b;
    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(a, b, ncvslideio::gapi::add(a, b)));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat ref_mat = (in_mat + in_mat);

    ncvslideio::Mat out_mat;
    c.apply(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = ncvslideio::Mat();
    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_mat));
    cc(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(S11N, Pipeline_Binary_Sum_Scalar)
{
    ncvslideio::GMat a, b;
    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(a, b, ncvslideio::gapi::sum(a + b)));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Scalar ref_scl = ncvslideio::sum(in_mat + in_mat);
    ncvslideio::Scalar out_scl;
    c.apply(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = ncvslideio::Scalar();
    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_mat));
    cc(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(S11N, Pipeline_Sharpen)
{
    const ncvslideio::Size sz_in (1280, 720);
    const ncvslideio::Size sz_out( 640, 480);
    ncvslideio::Mat in_mat (sz_in,  CV_8UC3);
    in_mat = ncvslideio::Scalar(128, 33, 53);

    ncvslideio::Mat out_mat(sz_out, CV_8UC3);
    ncvslideio::Mat out_mat_y;
    ncvslideio::Mat out_mat_ocv(sz_out, CV_8UC3);

    float sharpen_coeffs[] = {
         0.0f, -1.f,  0.0f,
        -1.0f,  5.f, -1.0f,
         0.0f, -1.f,  0.0f
    };
    ncvslideio::Mat sharpen_kernel(3, 3, CV_32F, sharpen_coeffs);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::GMat in;
    auto vga     = ncvslideio::gapi::resize(in, sz_out);
    auto yuv     = ncvslideio::gapi::RGB2YUV(vga);
    auto yuv_p   = ncvslideio::gapi::split3(yuv);
    auto y_sharp = ncvslideio::gapi::filter2D(std::get<0>(yuv_p), -1, sharpen_kernel);
    auto yuv_new = ncvslideio::gapi::merge3(y_sharp, std::get<1>(yuv_p), std::get<2>(yuv_p));
    auto out     = ncvslideio::gapi::YUV2RGB(yuv_new);

    auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(y_sharp, out)));
    auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);
    c.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat_y, out_mat));

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Mat smaller;
        ncvslideio::resize(in_mat, smaller, sz_out);

        ncvslideio::Mat yuv_mat;
        ncvslideio::cvtColor(smaller, yuv_mat, ncvslideio::COLOR_RGB2YUV);
        std::vector<ncvslideio::Mat> yuv_planar(3);
        ncvslideio::split(yuv_mat, yuv_planar);
        ncvslideio::filter2D(yuv_planar[0], yuv_planar[0], -1, sharpen_kernel);
        ncvslideio::merge(yuv_planar, yuv_mat);
        ncvslideio::cvtColor(yuv_mat, out_mat_ocv, ncvslideio::COLOR_YUV2RGB);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        ncvslideio::Mat diff = out_mat_ocv != out_mat;
        std::vector<ncvslideio::Mat> diffBGR(3);
        ncvslideio::split(diff, diffBGR);
        EXPECT_EQ(0, cvtest::norm(diffBGR[0], NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diffBGR[1], NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diffBGR[2], NORM_INF));
    }

    // Metadata check /////////////////////////////////////////////////////////
    {
        auto cc    = c.compile(ncvslideio::descr_of(in_mat));
        auto metas = cc.outMetas();
        ASSERT_EQ(2u, metas.size());

        auto out_y_meta = ncvslideio::util::get<ncvslideio::GMatDesc>(metas[0]);
        auto out_meta   = ncvslideio::util::get<ncvslideio::GMatDesc>(metas[1]);

        // Y-output
        EXPECT_EQ(CV_8U,   out_y_meta.depth);
        EXPECT_EQ(1,       out_y_meta.chan);
        EXPECT_EQ(640,     out_y_meta.size.width);
        EXPECT_EQ(480,     out_y_meta.size.height);

        // Final output
        EXPECT_EQ(CV_8U,   out_meta.depth);
        EXPECT_EQ(3,       out_meta.chan);
        EXPECT_EQ(640,     out_meta.size.width);
        EXPECT_EQ(480,     out_meta.size.height);
    }
}

TEST(S11N, Pipeline_CustomRGB2YUV)
{
    const ncvslideio::Size sz(1280, 720);
    const int INS = 3;
    std::vector<ncvslideio::Mat> in_mats(INS);
    for (auto i : ade::util::iota(INS))
    {
        in_mats[i].create(sz, CV_8U);
        ncvslideio::randu(in_mats[i], ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    }

    const int OUTS = 3;
    std::vector<ncvslideio::Mat> out_mats_cv(OUTS);
    std::vector<ncvslideio::Mat> out_mats_gapi(OUTS);
    for (auto i : ade::util::iota(OUTS))
    {
        out_mats_cv[i].create(sz, CV_8U);
        out_mats_gapi[i].create(sz, CV_8U);
    }

    // G-API code //////////////////////////////////////////////////////////////
    {
        ncvslideio::GMat r, g, b;
        ncvslideio::GMat y = 0.299f*r + 0.587f*g + 0.114f*b;
        ncvslideio::GMat u = 0.492f*(b - y);
        ncvslideio::GMat v = 0.877f*(r - y);

        auto p = ncvslideio::gapi::serialize(ncvslideio::GComputation({r, g, b}, {y, u, v}));
        auto c = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);
        c.apply(in_mats, out_mats_gapi);
    }

    // OpenCV code /////////////////////////////////////////////////////////////
    {
        ncvslideio::Mat r = in_mats[0], g = in_mats[1], b = in_mats[2];
        ncvslideio::Mat y = 0.299f*r + 0.587f*g + 0.114f*b;
        ncvslideio::Mat u = 0.492f*(b - y);
        ncvslideio::Mat v = 0.877f*(r - y);

        out_mats_cv[0] = y;
        out_mats_cv[1] = u;
        out_mats_cv[2] = v;
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        const auto diff = [](ncvslideio::Mat m1, ncvslideio::Mat m2, int t) {
            return ncvslideio::abs(m1 - m2) > t;
        };

        // FIXME: Not bit-accurate even now!
        ncvslideio::Mat
            diff_y = diff(out_mats_cv[0], out_mats_gapi[0], 2),
            diff_u = diff(out_mats_cv[1], out_mats_gapi[1], 2),
            diff_v = diff(out_mats_cv[2], out_mats_gapi[2], 2);

        EXPECT_EQ(0, cvtest::norm(diff_y, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diff_u, NORM_INF));
        EXPECT_EQ(0, cvtest::norm(diff_v, NORM_INF));
    }
}

namespace ThisTest
{
    using GOpBool = GOpaque<bool>;
    using GOpInt = GOpaque<int>;
    using GOpDouble = GOpaque<double>;
    using GOpPoint = GOpaque<ncvslideio::Point>;
    using GOpSize = GOpaque<ncvslideio::Size>;
    using GOpRect = GOpaque<ncvslideio::Rect>;

    using GOpOut = std::tuple<GOpPoint, GOpSize, GOpRect>;

    G_TYPED_KERNEL_M(OpGenerate, <GOpOut(GOpBool, GOpInt, GOpDouble)>, "test.s11n.gopaque")
    {
        static std::tuple<GOpaqueDesc, GOpaqueDesc, GOpaqueDesc> outMeta(const GOpaqueDesc&, const GOpaqueDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_gopaque_desc(), empty_gopaque_desc(), empty_gopaque_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpGenerate, OpGenerate)
    {
        static void run(const bool& b, const int& i, const double& d,
                        ncvslideio::Point& p, ncvslideio::Size& s, ncvslideio::Rect& r)
        {
            p = ncvslideio::Point(i, i*2);
            s = b ? ncvslideio::Size(42, 42) : ncvslideio::Size(7, 7);
            int ii = static_cast<int>(d);
            r = ncvslideio::Rect(ii, ii, ii, ii);
        }
    };

    using GArrInt = GArray<int>;
    using GArrDouble = GArray<double>;
    using GArrPoint = GArray<ncvslideio::Point>;
    using GArrSize = GArray<ncvslideio::Size>;
    using GArrRect = GArray<ncvslideio::Rect>;
    using GArrMat = GArray<ncvslideio::Mat>;
    using GArrScalar = GArray<ncvslideio::Scalar>;

    using GArrOut = std::tuple<GArrPoint, GArrSize, GArrRect, GArrMat>;

    G_TYPED_KERNEL_M(ArrGenerate, <GArrOut(GArrInt, GArrInt, GArrDouble, GArrScalar)>, "test.s11n.garray")
    {
        static std::tuple<GArrayDesc, GArrayDesc, GArrayDesc, GArrayDesc> outMeta(const GArrayDesc&, const GArrayDesc&,
                                                                                  const GArrayDesc&, const GArrayDesc&) {
            return std::make_tuple(empty_array_desc(), empty_array_desc(), empty_array_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVArrGenerate, ArrGenerate)
    {
        static void run(const std::vector<int>& b, const std::vector<int>& i,
                        const std::vector<double>& d, const std::vector<ncvslideio::Scalar>& sc,
                        std::vector<ncvslideio::Point>& p, std::vector<ncvslideio::Size>& s,
                        std::vector<ncvslideio::Rect>& r, std::vector<ncvslideio::Mat>& m)
        {
            p.clear(); p.resize(b.size());
            s.clear(); s.resize(b.size());
            r.clear(); r.resize(b.size());
            m.clear(); m.resize(b.size());

            for (std::size_t idx = 0; idx < b.size(); ++idx)
            {
                p[idx] = ncvslideio::Point(i[idx], i[idx]*2);
                s[idx] = b[idx] == 1 ? ncvslideio::Size(42, 42) : ncvslideio::Size(7, 7);
                int ii = static_cast<int>(d[idx]);
                r[idx] = ncvslideio::Rect(ii, ii, ii, ii);
                m[idx] = ncvslideio::Mat(3, 3, CV_8UC1, sc[idx]);
            }
        }
    };

    G_TYPED_KERNEL_M(OpArrK1, <std::tuple<GArrInt,GOpSize>(GOpInt, GArrSize)>, "test.s11n.oparrk1")
    {
        static std::tuple<GArrayDesc, GOpaqueDesc> outMeta(const GOpaqueDesc&, const GArrayDesc&) {
            return std::make_tuple(empty_array_desc(), empty_gopaque_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK1, OpArrK1)
    {
        static void run(const int& i, const std::vector<ncvslideio::Size>& vs,
                        std::vector<int>& vi, ncvslideio::Size& s)
        {
            vi.clear(); vi.resize(vs.size());
            s = ncvslideio::Size(i, i);
            for (std::size_t idx = 0; idx < vs.size(); ++ idx)
                vi[idx] = vs[idx].area();
        }
    };

    G_TYPED_KERNEL_M(OpArrK2, <std::tuple<GOpDouble,GArrPoint>(GArrInt, GOpSize)>, "test.s11n.oparrk2")
    {
        static std::tuple<GOpaqueDesc, GArrayDesc> outMeta(const GArrayDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_gopaque_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK2, OpArrK2)
    {
        static void run(const std::vector<int>& vi, const ncvslideio::Size& s,
                        double& d, std::vector<ncvslideio::Point>& vp)
        {
            vp.clear(); vp.resize(vi.size());
            d = s.area() * 1.5;
            for (std::size_t idx = 0; idx < vi.size(); ++ idx)
                vp[idx] = ncvslideio::Point(vi[idx], vi[idx]);
        }
    };

    using GK3Out = std::tuple<ncvslideio::GArray<uint64_t>, ncvslideio::GArray<int32_t>>;
    G_TYPED_KERNEL_M(OpArrK3, <GK3Out(ncvslideio::GArray<bool>, ncvslideio::GArray<int32_t>, ncvslideio::GOpaque<float>)>, "test.s11n.oparrk3")
    {
        static std::tuple<GArrayDesc, GArrayDesc> outMeta(const GArrayDesc&, const GArrayDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_array_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK3, OpArrK3)
    {
        static void run(const std::vector<bool>& vb, const std::vector<int32_t>& vi_in, const float& f,
                        std::vector<uint64_t>& vui, std::vector<int32_t>& vi)
        {
            vui.clear(); vui.resize(vi_in.size());
            vi.clear();  vi.resize(vi_in.size());

            for (std::size_t idx = 0; idx < vi_in.size(); ++ idx)
            {
                vi[idx] = vb[idx] ? vi_in[idx] : -vi_in[idx];
                vui[idx] = vb[idx] ? static_cast<uint64_t>(vi_in[idx] * f) :
                                     static_cast<uint64_t>(vi_in[idx] / f);
            }
        }
    };

    using GK4Out = std::tuple<ncvslideio::GOpaque<int>, ncvslideio::GArray<std::string>>;
    G_TYPED_KERNEL_M(OpArrK4, <GK4Out(ncvslideio::GOpaque<bool>, ncvslideio::GOpaque<std::string>)>, "test.s11n.oparrk4")
    {
        static std::tuple<GOpaqueDesc, GArrayDesc> outMeta(const GOpaqueDesc&, const GOpaqueDesc&) {
            return std::make_tuple(empty_gopaque_desc(), empty_array_desc());
        }
    };

    GAPI_OCV_KERNEL(OCVOpArrK4, OpArrK4)
    {
        static void run(const bool& b, const std::string& s,
                        int& i, std::vector<std::string>& vs)
        {
            vs.clear();
            vs.resize(2);
            i = b ? 42 : 24;
            auto s_copy = s + " world";
            vs = std::vector<std::string>{s_copy, s_copy};
        }
    };
} // namespace ThisTest

TEST(S11N, Pipeline_GOpaque)
{
    using namespace ThisTest;
    GOpBool in1;
    GOpInt in2;
    GOpDouble in3;

    auto out = OpGenerate::on(in1, in2, in3);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3), ncvslideio::GOut(std::get<0>(out), std::get<1>(out), std::get<2>(out)));

    auto p = ncvslideio::gapi::serialize(c);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    bool b = true;
    int i = 33;
    double d = 128.7;
    ncvslideio::Point pp;
    ncvslideio::Size s;
    ncvslideio::Rect r;
    dc.apply(ncvslideio::gin(b, i, d), ncvslideio::gout(pp, s, r), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVOpGenerate>()));

    EXPECT_EQ(pp, ncvslideio::Point(i, i*2));
    EXPECT_EQ(s, ncvslideio::Size(42, 42));
    int ii = static_cast<int>(d);
    EXPECT_EQ(r, ncvslideio::Rect(ii, ii, ii, ii));
}

TEST(S11N, Pipeline_GArray)
{
    using namespace ThisTest;
    GArrInt in1, in2;
    GArrDouble in3;
    GArrScalar in4;

    auto out = ArrGenerate::on(in1, in2, in3, in4);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3, in4),
                       ncvslideio::GOut(std::get<0>(out), std::get<1>(out),
                                std::get<2>(out), std::get<3>(out)));

    auto p = ncvslideio::gapi::serialize(c);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    std::vector<int> b {1, 0, -1};
    std::vector<int> i {3, 0 , 59};
    std::vector<double> d {0.7, 120.5, 44.14};
    std::vector<ncvslideio::Scalar> sc {ncvslideio::Scalar::all(10), ncvslideio::Scalar::all(15), ncvslideio::Scalar::all(99)};
    std::vector<ncvslideio::Point> pp;
    std::vector<ncvslideio::Size> s;
    std::vector<ncvslideio::Rect> r;
    std::vector<ncvslideio::Mat> m;
    dc.apply(ncvslideio::gin(b, i, d, sc), ncvslideio::gout(pp, s, r, m), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVArrGenerate>()));

    for (std::size_t idx = 0; idx < b.size(); ++idx)
    {
        EXPECT_EQ(pp[idx], ncvslideio::Point(i[idx], i[idx]*2));
        EXPECT_EQ(s[idx], b[idx] == 1 ? ncvslideio::Size(42, 42) : ncvslideio::Size(7, 7));
        int ii = static_cast<int>(d[idx]);
        EXPECT_EQ(r[idx], ncvslideio::Rect(ii, ii, ii, ii));
    }
}

TEST(S11N, Pipeline_GArray_GOpaque_Multinode)
{
    using namespace ThisTest;
    GOpInt in1;
    GArrSize in2;

    auto tmp = OpArrK1::on(in1, in2);
    auto out = OpArrK2::on(std::get<0>(tmp), std::get<1>(tmp));

    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2),
                       ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));

    auto p = ncvslideio::gapi::serialize(c);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    int i = 42;
    std::vector<ncvslideio::Size> s{ncvslideio::Size(11, 22), ncvslideio::Size(13, 18)};
    double d;
    std::vector<ncvslideio::Point> pp;

    dc.apply(ncvslideio::gin(i, s), ncvslideio::gout(d, pp), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVOpArrK1, OCVOpArrK2>()));

    auto st = ncvslideio::Size(i ,i);
    EXPECT_EQ(d, st.area() * 1.5);

    for (std::size_t idx = 0; idx < s.size(); ++idx)
    {
        EXPECT_EQ(pp[idx], ncvslideio::Point(s[idx].area(), s[idx].area()));
    }
}

TEST(S11N, Pipeline_GArray_GOpaque_2)
{
    using namespace ThisTest;

    ncvslideio::GArray<bool> in1;
    ncvslideio::GArray<int32_t> in2;
    ncvslideio::GOpaque<float> in3;
    auto out = OpArrK3::on(in1, in2, in3);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2, in3),
                       ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));

    auto p = ncvslideio::gapi::serialize(c);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    std::vector<bool> b {true, false, false};
    std::vector<int32_t> i {234324, -234252, 999};
    float f = 0.85f;
    std::vector<int32_t> out_i;
    std::vector<uint64_t> out_ui;
    dc.apply(ncvslideio::gin(b, i, f), ncvslideio::gout(out_ui, out_i), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVOpArrK3>()));

    for (std::size_t idx = 0; idx < b.size(); ++idx)
    {
        EXPECT_EQ(out_i[idx], b[idx] ? i[idx] : -i[idx]);
        EXPECT_EQ(out_ui[idx], b[idx] ? static_cast<uint64_t>(i[idx] * f) :
                                        static_cast<uint64_t>(i[idx] / f));
    }
}

TEST(S11N, Pipeline_GArray_GOpaque_3)
{
    using namespace ThisTest;

    ncvslideio::GOpaque<bool> in1;
    ncvslideio::GOpaque<std::string> in2;
    auto out = OpArrK4::on(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2),
                       ncvslideio::GOut(std::get<0>(out), std::get<1>(out)));

    auto p = ncvslideio::gapi::serialize(c);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    bool b = false;
    std::string s("hello");
    int i = 0;
    std::vector<std::string> vs{};
    dc.apply(ncvslideio::gin(b, s), ncvslideio::gout(i, vs), ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVOpArrK4>()));

    EXPECT_EQ(24, i);
    std::vector<std::string> vs_ref{"hello world", "hello world"};
    EXPECT_EQ(vs_ref, vs);
}

TEST(S11N, Pipeline_Render_NV12)
{
    ncvslideio::Size sz (100, 200);
    int rects_num = 10;
    int text_num  = 10;
    int image_num = 10;

    int thick = 2;
    int lt = LINE_8;
    ncvslideio::Scalar color(111, 222, 77);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;

    // Rects
    int shift = 0;
    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect rect(200 + i, 200 + i, 200, 200);
        prims.emplace_back(ncvslideio::gapi::wip::draw::Rect(rect, color, thick, lt, shift));
    }

    // Mosaic
    int cellsz = 50;
    int decim = 0;
    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect mos(200 + i, 200 + i, 200, 200);
        prims.emplace_back(ncvslideio::gapi::wip::draw::Mosaic(mos, cellsz, decim));
    }

    // Text
    std::string text = "Some text";
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;
    for (int i = 0; i < text_num; ++i) {
        ncvslideio::Point org(200 + i, 200 + i);
        prims.emplace_back(ncvslideio::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));
    }

    // Image
    double transparency = 1.0;
    ncvslideio::Rect rect_img(0 ,0 , 50, 50);
    ncvslideio::Mat img(rect_img.size(), CV_8UC3, color);
    ncvslideio::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    for (int i = 0; i < image_num; ++i) {
        ncvslideio::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};

        prims.emplace_back(ncvslideio::gapi::wip::draw::Image({org_img, img, alpha}));
    }

    // Circle
    ncvslideio::Point center(300, 400);
    int rad = 25;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Circle({center, rad, color, thick, lt, shift}));

    // Line
    ncvslideio::Point point_next(300, 425);
    prims.emplace_back(ncvslideio::gapi::wip::draw::Line({center, point_next, color, thick, lt, shift}));

    // Poly
    std::vector<ncvslideio::Point> points = {{300, 400}, {290, 450}, {348, 410}, {300, 400}};
    prims.emplace_back(ncvslideio::gapi::wip::draw::Poly({points, color, thick, lt, shift}));

    ncvslideio::GMat y_in, uv_in, y_out, uv_out;
    ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim> arr;
    std::tie(y_out, uv_out) = ncvslideio::gapi::wip::draw::renderNV12(y_in, uv_in, arr);
    ncvslideio::GComputation comp(ncvslideio::GIn(y_in, uv_in, arr), ncvslideio::GOut(y_out, uv_out));

    auto serialized = ncvslideio::gapi::serialize(comp);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(serialized);

    ncvslideio::Mat y(1920, 1080, CV_8UC1);
    ncvslideio::Mat uv(960, 540, CV_8UC2);
    ncvslideio::randu(y, ncvslideio::Scalar(0), ncvslideio::Scalar(255));
    ncvslideio::randu(uv, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Mat y_ref_mat = y.clone(), uv_ref_mat = uv.clone();
    dc.apply(ncvslideio::gin(y, uv, prims), ncvslideio::gout(y, uv));

    // OpenCV code //////////////////////////////////////////////////////////////
    ncvslideio::Mat yuv;
    ncvslideio::gapi::wip::draw::cvtNV12ToYUV(y_ref_mat, uv_ref_mat, yuv);

    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect rect(200 + i, 200 + i, 200, 200);
        ncvslideio::rectangle(yuv, rect, cvtBGRToYUVC(color), thick, lt, shift);
    }

    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect mos(200 + i, 200 + i, 200, 200);
         drawMosaicRef(yuv, mos, cellsz);
    }

    for (int i = 0; i < text_num; ++i) {
        ncvslideio::Point org(200 + i, 200 + i);
        ncvslideio::putText(yuv, text, org, ff, fs, cvtBGRToYUVC(color), thick, lt, blo);
    }

    for (int i = 0; i < image_num; ++i) {
        ncvslideio::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};
        ncvslideio::Mat yuv_img;
        ncvslideio::cvtColor(img, yuv_img, ncvslideio::COLOR_BGR2YUV);
        blendImageRef(yuv, org_img, yuv_img, alpha);
    }

    ncvslideio::circle(yuv, center, rad, cvtBGRToYUVC(color), thick, lt, shift);
    ncvslideio::line(yuv, center, point_next, cvtBGRToYUVC(color), thick, lt, shift);
    std::vector<std::vector<ncvslideio::Point>> pp{points};
    ncvslideio::fillPoly(yuv, pp, cvtBGRToYUVC(color), lt, shift);

    // YUV -> NV12
    ncvslideio::gapi::wip::draw::cvtYUVToNV12(yuv, y_ref_mat, uv_ref_mat);

    EXPECT_EQ(ncvslideio::norm( y,  y_ref_mat), 0);
    EXPECT_EQ(ncvslideio::norm(uv, uv_ref_mat), 0);
}

TEST(S11N, Pipeline_Render_RGB)
{
    ncvslideio::Size sz (100, 200);
    int rects_num = 10;
    int text_num  = 10;
    int image_num = 10;

    int thick = 2;
    int lt = LINE_8;
    ncvslideio::Scalar color(111, 222, 77);

    // G-API code //////////////////////////////////////////////////////////////
    ncvslideio::gapi::wip::draw::Prims prims;

    // Rects
    int shift = 0;
    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect rect(200 + i, 200 + i, 200, 200);
        prims.emplace_back(ncvslideio::gapi::wip::draw::Rect(rect, color, thick, lt, shift));
    }

    // Mosaic
    int cellsz = 50;
    int decim = 0;
    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect mos(200 + i, 200 + i, 200, 200);
        prims.emplace_back(ncvslideio::gapi::wip::draw::Mosaic(mos, cellsz, decim));
    }

    // Text
    std::string text = "Some text";
    int ff = FONT_HERSHEY_SIMPLEX;
    double fs = 2.0;
    bool blo = false;
    for (int i = 0; i < text_num; ++i) {
        ncvslideio::Point org(200 + i, 200 + i);
        prims.emplace_back(ncvslideio::gapi::wip::draw::Text(text, org, ff, fs, color, thick, lt, blo));
    }

    // Image
    double transparency = 1.0;
    ncvslideio::Rect rect_img(0 ,0 , 50, 50);
    ncvslideio::Mat img(rect_img.size(), CV_8UC3, color);
    ncvslideio::Mat alpha(rect_img.size(), CV_32FC1, transparency);
    auto tl = rect_img.tl();
    for (int i = 0; i < image_num; ++i) {
        ncvslideio::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};

        prims.emplace_back(ncvslideio::gapi::wip::draw::Image({org_img, img, alpha}));
    }

    // Circle
    ncvslideio::Point center(300, 400);
    int rad = 25;
    prims.emplace_back(ncvslideio::gapi::wip::draw::Circle({center, rad, color, thick, lt, shift}));

    // Line
    ncvslideio::Point point_next(300, 425);
    prims.emplace_back(ncvslideio::gapi::wip::draw::Line({center, point_next, color, thick, lt, shift}));

    // Poly
    std::vector<ncvslideio::Point> points = {{300, 400}, {290, 450}, {348, 410}, {300, 400}};
    prims.emplace_back(ncvslideio::gapi::wip::draw::Poly({points, color, thick, lt, shift}));

    ncvslideio::GMat in, out;
    ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim> arr;
    out = ncvslideio::gapi::wip::draw::render3ch(in, arr);
    ncvslideio::GComputation comp(ncvslideio::GIn(in, arr), ncvslideio::GOut(out));

    auto serialized = ncvslideio::gapi::serialize(comp);
    auto dc = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(serialized);

    ncvslideio::Mat input(1920, 1080, CV_8UC3);
    ncvslideio::randu(input, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::Mat ref_mat = input.clone();
    dc.apply(ncvslideio::gin(input, prims), ncvslideio::gout(input));

    // OpenCV code //////////////////////////////////////////////////////////////
    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect rect(200 + i, 200 + i, 200, 200);
        ncvslideio::rectangle(ref_mat, rect, color, thick, lt, shift);
    }

    for (int i = 0; i < rects_num; ++i) {
        ncvslideio::Rect mos(200 + i, 200 + i, 200, 200);
         drawMosaicRef(ref_mat, mos, cellsz);
    }

    for (int i = 0; i < text_num; ++i) {
        ncvslideio::Point org(200 + i, 200 + i);
        ncvslideio::putText(ref_mat, text, org, ff, fs, color, thick, lt, blo);
    }

    for (int i = 0; i < image_num; ++i) {
        ncvslideio::Point org_img = {tl.x + i, tl.y + rect_img.size().height + i};
        blendImageRef(ref_mat, org_img, img, alpha);
    }

    ncvslideio::circle(ref_mat, center, rad, color, thick, lt, shift);
    ncvslideio::line(ref_mat, center, point_next, color, thick, lt, shift);
    std::vector<std::vector<ncvslideio::Point>> pp{points};
    ncvslideio::fillPoly(ref_mat, pp, color, lt, shift);

    EXPECT_EQ(ncvslideio::norm(input,  ref_mat), 0);
}

TEST(S11N, Pipeline_Const_GScalar)
{
    static constexpr auto in_scalar = 10;

    ncvslideio::GMat a;
    ncvslideio::GScalar s;

    ncvslideio::GComputation computation(GIn(a), GOut(ncvslideio::gapi::addC(a, in_scalar)));
    auto p = ncvslideio::gapi::serialize(computation);
    auto deserialized_computation = ncvslideio::gapi::deserialize<ncvslideio::GComputation>(p);

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat ref_mat;
    ncvslideio::add(in_mat, in_scalar, ref_mat);

    ncvslideio::Mat out_mat;
    computation.apply(ncvslideio::gin(in_mat/*, in_scalar*/), ncvslideio::gout(out_mat));
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = ncvslideio::Mat();
    deserialized_computation.apply(ncvslideio::gin(in_mat/*, in_scalar*/), ncvslideio::gout(out_mat));
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = ncvslideio::Mat();
    auto cc = deserialized_computation.compile(ncvslideio::descr_of(in_mat));
    cc(ncvslideio::gin(in_mat/*, in_scalar*/), ncvslideio::gout(out_mat));
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}
} // namespace opencv_test
