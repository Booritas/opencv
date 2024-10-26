// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <stdexcept>
#include <ade/util/iota_range.hpp>
#include "logger.hpp"

#include <opencv2/gapi/core.hpp>

#include "executor/thread_pool.hpp"

namespace opencv_test
{

namespace
{
    G_TYPED_KERNEL(GInvalidResize, <GMat(GMat,Size,double,double,int)>, "org.opencv.test.invalid_resize")
    {
         static GMatDesc outMeta(GMatDesc in, Size, double, double, int) { return in; }
    };

    GAPI_OCV_KERNEL(GOCVInvalidResize, GInvalidResize)
    {
        static void run(const ncvslideio::Mat& in, ncvslideio::Size sz, double fx, double fy, int interp, ncvslideio::Mat &out)
        {
            ncvslideio::resize(in, out, sz, fx, fy, interp);
        }
    };

    G_TYPED_KERNEL(GReallocatingCopy, <GMat(GMat)>, "org.opencv.test.reallocating_copy")
    {
         static GMatDesc outMeta(GMatDesc in) { return in; }
    };

    GAPI_OCV_KERNEL(GOCVReallocatingCopy, GReallocatingCopy)
    {
        static void run(const ncvslideio::Mat& in, ncvslideio::Mat &out)
        {
            out = in.clone();
        }
    };

    G_TYPED_KERNEL(GCustom, <GMat(GMat)>, "org.opencv.test.custom")
    {
         static GMatDesc outMeta(GMatDesc in) { return in; }
    };

    G_TYPED_KERNEL(GZeros, <GMat(GMat, GMatDesc)>, "org.opencv.test.zeros")
    {
        static GMatDesc outMeta(GMatDesc /*in*/, GMatDesc user_desc)
        {
            return user_desc;
        }
    };

    GAPI_OCV_KERNEL(GOCVZeros, GZeros)
    {
        static void run(const ncvslideio::Mat&      /*in*/,
                        const ncvslideio::GMatDesc& /*desc*/,
                        ncvslideio::Mat&            out)
        {
            out.setTo(0);
        }
    };

    G_TYPED_KERNEL(GBusyWait, <GMat(GMat, uint32_t)>, "org.busy_wait") {
        static GMatDesc outMeta(GMatDesc in, uint32_t)
        {
            return in;
        }
    };

    GAPI_OCV_KERNEL(GOCVBusyWait, GBusyWait)
    {
        static void run(const ncvslideio::Mat& in,
                        const uint32_t time_in_ms,
                        ncvslideio::Mat&       out)
        {
            using namespace std::chrono;
            auto s = high_resolution_clock::now();
            in.copyTo(out);
            auto e = high_resolution_clock::now();

            const auto elapsed_in_ms =
                static_cast<int32_t>(duration_cast<milliseconds>(e-s).count());

            int32_t diff = time_in_ms - elapsed_in_ms;
            const auto need_to_wait_in_ms = static_cast<uint32_t>(std::max(0, diff));

            s = high_resolution_clock::now();
            e = s;
            while (duration_cast<milliseconds>(e-s).count() < need_to_wait_in_ms) {
                e = high_resolution_clock::now();
            }
        }
    };

    // These definitions test the correct macro work if the kernel has multiple output values
    G_TYPED_KERNEL(GRetGArrayTupleOfGMat2Kernel,  <GArray<std::tuple<GMat, GMat>>(GMat, Scalar)>,                                         "org.opencv.test.retarrayoftupleofgmat2kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat3Kernel,  <GArray<std::tuple<GMat, GMat, GMat>>(GMat)>,                                           "org.opencv.test.retarrayoftupleofgmat3kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat4Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat>>(GMat)>,                                     "org.opencv.test.retarrayoftupleofgmat4kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat5Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat>>(GMat)>,                               "org.opencv.test.retarrayoftupleofgmat5kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat6Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,                         "org.opencv.test.retarrayoftupleofgmat6kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat7Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,                   "org.opencv.test.retarrayoftupleofgmat7kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat8Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,             "org.opencv.test.retarrayoftupleofgmat8kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat9Kernel,  <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>,       "org.opencv.test.retarrayoftupleofgmat9kernel")  {};
    G_TYPED_KERNEL(GRetGArraTupleyOfGMat10Kernel, <GArray<std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>>(GMat)>, "org.opencv.test.retarrayoftupleofgmat10kernel") {};

    G_TYPED_KERNEL_M(GRetGMat2Kernel,     <std::tuple<GMat, GMat>(GMat, GMat, GMat)>,                                     "org.opencv.test.retgmat2kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat3Kernel,     <std::tuple<GMat, GMat, GMat>(GMat, GScalar)>,                                  "org.opencv.test.retgmat3kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat4Kernel,     <std::tuple<GMat, GMat, GMat, GMat>(GMat, GArray<int>, GScalar)>,               "org.opencv.test.retgmat4kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat5Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat>(GMat)>,                               "org.opencv.test.retgmat5kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat6Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,                         "org.opencv.test.retgmat6kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat7Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,                   "org.opencv.test.retgmat7kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat8Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,             "org.opencv.test.retgmat8kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat9Kernel,     <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>,       "org.opencv.test.retgmat9kernel")      {};
    G_TYPED_KERNEL_M(GRetGMat10Kernel,    <std::tuple<GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat, GMat>(GMat)>, "org.opencv.test.retgmat10kernel")     {};
}

TEST(GAPI_Pipeline, OverloadUnary_MatMat)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation comp(in, ncvslideio::gapi::bitwise_not(in));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat ref_mat = ~in_mat;

    ncvslideio::Mat out_mat;
    comp.apply(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = ncvslideio::Mat();
    auto cc = comp.compile(ncvslideio::descr_of(in_mat));
    cc(in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_Pipeline, OverloadUnary_MatScalar)
{
    ncvslideio::GMat in;
    ncvslideio::GComputation comp(in, ncvslideio::gapi::sum(in));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Scalar ref_scl = ncvslideio::sum(in_mat);

    ncvslideio::Scalar out_scl;
    comp.apply(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = ncvslideio::Scalar();
    auto cc = comp.compile(ncvslideio::descr_of(in_mat));
    cc(in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(GAPI_Pipeline, OverloadBinary_Mat)
{
    ncvslideio::GMat a, b;
    ncvslideio::GComputation comp(a, b, ncvslideio::gapi::add(a, b));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat ref_mat = (in_mat+in_mat);

    ncvslideio::Mat out_mat;
    comp.apply(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

    out_mat = ncvslideio::Mat();
    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_mat));
    cc(in_mat, in_mat, out_mat);
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GAPI_Pipeline, OverloadBinary_Scalar)
{
    ncvslideio::GMat a, b;
    ncvslideio::GComputation comp(a, b, ncvslideio::gapi::sum(a + b));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Scalar ref_scl = ncvslideio::sum(in_mat+in_mat);

    ncvslideio::Scalar out_scl;
    comp.apply(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);

    out_scl = ncvslideio::Scalar();
    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_mat));
    cc(in_mat, in_mat, out_scl);
    EXPECT_EQ(out_scl, ref_scl);
}

TEST(GAPI_Pipeline, Sharpen)
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

    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(y_sharp, out));
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

TEST(GAPI_Pipeline, CustomRGB2YUV)
{
    const ncvslideio::Size sz(1280, 720);

    // BEWARE:
    //
    //    std::vector<ncvslideio::Mat> out_mats_cv(3, ncvslideio::Mat(sz, CV_8U))
    //
    // creates a vector of 3 elements pointing to the same Mat!
    // FIXME: Make a G-API check for that
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
        out_mats_cv  [i].create(sz, CV_8U);
        out_mats_gapi[i].create(sz, CV_8U);
    }

    // G-API code //////////////////////////////////////////////////////////////
    {
        ncvslideio::GMat r, g, b;
        ncvslideio::GMat y = 0.299f*r + 0.587f*g + 0.114f*b;
        ncvslideio::GMat u = 0.492f*(b - y);
        ncvslideio::GMat v = 0.877f*(r - y);

        ncvslideio::GComputation customCvt({r, g, b}, {y, u, v});
        customCvt.apply(in_mats, out_mats_gapi);
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
            return ncvslideio::abs(m1-m2) > t;
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

TEST(GAPI_Pipeline, PipelineWithInvalidKernel)
{
    ncvslideio::GMat in, out;
    ncvslideio::Mat in_mat(500, 500, CV_8UC1), out_mat;
    out = GInvalidResize::on(in, ncvslideio::Size(300, 300), 0.0, 0.0, ncvslideio::INTER_LINEAR);

    const auto pkg = ncvslideio::gapi::kernels<GOCVInvalidResize>();
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    EXPECT_THROW(comp.apply(in_mat, out_mat, ncvslideio::compile_args(pkg)), std::logic_error);
}

TEST(GAPI_Pipeline, InvalidOutputComputation)
{
    ncvslideio::GMat in1, out1, out2, out3;

    std::tie(out1, out2, out2) = ncvslideio::gapi::split3(in1);
    ncvslideio::GComputation c({in1}, {out1, out2, out3});
    ncvslideio::Mat in_mat;
    ncvslideio::Mat out_mat1, out_mat2, out_mat3, out_mat4;
    std::vector<ncvslideio::Mat> u_outs = {out_mat1, out_mat2, out_mat3, out_mat4};
    std::vector<ncvslideio::Mat> u_ins = {in_mat};

    EXPECT_THROW(c.apply(u_ins, u_outs), std::logic_error);
}

TEST(GAPI_Pipeline, PipelineAllocatingKernel)
{
    ncvslideio::GMat in, out;
    ncvslideio::Mat in_mat(500, 500, CV_8UC1), out_mat;
    out = GReallocatingCopy::on(in);

    const auto pkg = ncvslideio::gapi::kernels<GOCVReallocatingCopy>();
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    EXPECT_THROW(comp.apply(in_mat, out_mat, ncvslideio::compile_args(pkg)), std::logic_error);
}

TEST(GAPI_Pipeline, CreateKernelImplFromLambda)
{
    ncvslideio::Size size(300, 300);
    int type = CV_8UC3;
    ncvslideio::Mat in_mat(size, type);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    int value = 5;

    ncvslideio::GMat in;
    ncvslideio::GMat out = GCustom::on(in);
    ncvslideio::GComputation comp(in, out);

    // OpenCV //////////////////////////////////////////////////////////////////////////
    auto ref_mat = in_mat + value;

    // G-API //////////////////////////////////////////////////////////////////////////
    auto impl = ncvslideio::gapi::cpu::ocv_kernel<GCustom>([&value](const ncvslideio::Mat& src, ncvslideio::Mat& dst)
                {
                    dst = src + value;
                });

    ncvslideio::Mat out_mat;
    auto pkg = ncvslideio::gapi::kernels(impl);
    comp.apply(in_mat, out_mat, ncvslideio::compile_args(pkg));

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
}

TEST(GAPI_Pipeline, ReplaceDefaultByLambda)
{
    ncvslideio::Size size(300, 300);
    int type = CV_8UC3;
    ncvslideio::Mat in_mat1(size, type);
    ncvslideio::Mat in_mat2(size, type);
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    ncvslideio::GMat in1, in2;
    ncvslideio::GMat out = ncvslideio::gapi::add(in1, in2);
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    // OpenCV //////////////////////////////////////////////////////////////////////////
    ncvslideio::Mat ref_mat = in_mat1 + in_mat2;


    // G-API //////////////////////////////////////////////////////////////////////////
    bool is_called = false;
    auto impl = ncvslideio::gapi::cpu::ocv_kernel<ncvslideio::gapi::core::GAdd>([&is_called]
                (const ncvslideio::Mat& src1, const ncvslideio::Mat& src2, int, ncvslideio::Mat& dst)
                {
                    is_called = true;
                    dst = src1 + src2;
                });

    ncvslideio::Mat out_mat;
    auto pkg = ncvslideio::gapi::kernels(impl);
    comp.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
    EXPECT_TRUE(is_called);
}

struct AddImpl
{
    void operator()(const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, int, ncvslideio::Mat& out)
    {
        out = in1 + in2;
        is_called = true;
    }

    bool is_called = false;
};

TEST(GAPI_Pipeline, ReplaceDefaultByFunctor)
{
    ncvslideio::Size size(300, 300);
    int type = CV_8UC3;
    ncvslideio::Mat in_mat1(size, type);
    ncvslideio::Mat in_mat2(size, type);
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    ncvslideio::GMat in1, in2;
    ncvslideio::GMat out = ncvslideio::gapi::add(in1, in2);
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    // OpenCV //////////////////////////////////////////////////////////////////////////
    ncvslideio::Mat ref_mat = in_mat1 + in_mat2;


    // G-API ///////////////////////////////////////////////////////////////////////////
    AddImpl f;
    EXPECT_FALSE(f.is_called);
    auto impl = ncvslideio::gapi::cpu::ocv_kernel<ncvslideio::gapi::core::GAdd>(f);

    ncvslideio::Mat out_mat;
    auto pkg = ncvslideio::gapi::kernels(impl);
    comp.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(pkg));

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
    EXPECT_TRUE(f.is_called);
}

TEST(GAPI_Pipeline, GraphOutputIs1DMat)
{
    int dim = 100;
    ncvslideio::Mat in_mat(1, 1, CV_8UC3);
    ncvslideio::Mat out_mat;

    ncvslideio::GMat in;
    auto cc = ncvslideio::GComputation(in, GZeros::on(in, ncvslideio::GMatDesc(CV_8U, {dim})))
        .compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(ncvslideio::gapi::kernels<GOCVZeros>()));

    // NB: Computation is able to write 1D output ncvslideio::Mat to empty out_mat.
    ASSERT_NO_THROW(cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat)));
    ASSERT_EQ(1, out_mat.size.dims());
    ASSERT_EQ(dim, out_mat.size[0]);

    // NB: Computation is able to write 1D output ncvslideio::Mat
    // to pre-allocated with the same meta out_mat.
    ASSERT_NO_THROW(cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat)));
    ASSERT_EQ(1, out_mat.size.dims());
    ASSERT_EQ(dim, out_mat.size[0]);
}

TEST(GAPI_Pipeline, 1DMatBetweenIslands)
{
    int dim = 100;
    ncvslideio::Mat in_mat(1, 1, CV_8UC3);
    ncvslideio::Mat out_mat;

    ncvslideio::Mat ref_mat({dim}, CV_8U);
    ref_mat.dims = 1;
    ref_mat.setTo(0);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::copy(GZeros::on(ncvslideio::gapi::copy(in), ncvslideio::GMatDesc(CV_8U, {dim})));
    auto cc = ncvslideio::GComputation(in, out)
        .compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(ncvslideio::gapi::kernels<GOCVZeros>()));

    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
}

TEST(GAPI_Pipeline, 1DMatWithinSingleIsland)
{
    int dim = 100;
    ncvslideio::Size blur_sz(3, 3);
    ncvslideio::Mat in_mat(10, 10, CV_8UC3);
    ncvslideio::randu(in_mat, 0, 255);
    ncvslideio::Mat out_mat;

    ncvslideio::Mat ref_mat({dim}, CV_8U);
    ref_mat.dims = 1;
    ref_mat.setTo(0);

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::blur(
            GZeros::on(ncvslideio::gapi::blur(in, blur_sz), ncvslideio::GMatDesc(CV_8U, {dim})), blur_sz);
    auto cc = ncvslideio::GComputation(in, out)
        .compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(ncvslideio::gapi::kernels<GOCVZeros>()));

    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
}

TEST(GAPI_Pipeline, BranchesExecutedInParallel)
{
    ncvslideio::GMat in;
    // NB: ncvslideio::gapi::copy used to prevent fusing OCV backend operations
    // into the single island where they will be executed in turn
    auto out0 = GBusyWait::on(ncvslideio::gapi::copy(in), 1000u /*1sec*/);
    auto out1 = GBusyWait::on(ncvslideio::gapi::copy(in), 1000u /*1sec*/);
    auto out2 = GBusyWait::on(ncvslideio::gapi::copy(in), 1000u /*1sec*/);
    auto out3 = GBusyWait::on(ncvslideio::gapi::copy(in), 1000u /*1sec*/);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out0,out1,out2,out3));
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(32, 32, CV_8UC1);
    ncvslideio::Mat out_mat0, out_mat1, out_mat2, out_mat3;

    using namespace std::chrono;
    auto s = high_resolution_clock::now();
    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat0, out_mat1, out_mat2, out_mat3),
               ncvslideio::compile_args(ncvslideio::use_threaded_executor(4u),
                                ncvslideio::gapi::kernels<GOCVBusyWait>()));
    auto e = high_resolution_clock::now();
    const auto elapsed_in_ms = duration_cast<milliseconds>(e-s).count();;

    EXPECT_GE(1200u, elapsed_in_ms);
}

} // namespace opencv_test
