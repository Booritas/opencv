// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "test_precomp.hpp"

#include <opencv2/gapi/media.hpp>

////////////////////////////////////////////////////////////////////////////////
// ncvslideio::GFrame tests

namespace opencv_test {

G_API_OP(GBlurFrame, <GMat(GFrame)>, "test.blur_frame") {
    static GMatDesc outMeta(GFrameDesc in) {
        return ncvslideio::GMatDesc(CV_8U,3,in.size);
    }
};

GAPI_OCV_KERNEL(OCVBlurFrame, GBlurFrame) {
    static void run(const ncvslideio::MediaFrame &in, ncvslideio::Mat& out) {
        GAPI_Assert(in.desc().fmt == ncvslideio::MediaFormat::BGR);
        ncvslideio::MediaFrame::View view = in.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::blur(ncvslideio::Mat(in.desc().size, CV_8UC3, view.ptr[0], view.stride[0]),
                 out,
                 ncvslideio::Size{3,3});
    }
};

G_API_OP(GBlurFrameGray, <GMat(GFrame)>, "test.blur_frame_gray") {
    static GMatDesc outMeta(GFrameDesc in) {
        return ncvslideio::GMatDesc(CV_8U, 1, in.size);
    }
};

GAPI_OCV_KERNEL(OCVBlurFrameGray, GBlurFrameGray) {
    static void run(const ncvslideio::MediaFrame & in, ncvslideio::Mat & out) {
        GAPI_Assert(in.desc().fmt == ncvslideio::MediaFormat::GRAY);
        ncvslideio::MediaFrame::View view = in.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::blur(ncvslideio::Mat(in.desc().size, CV_8UC1, view.ptr[0], view.stride[0]),
        out,
        ncvslideio::Size{ 3,3 });
    }
};


////////////////////////////////////////////////////////////////////////////////
// ncvslideio::MediaFrame tests
namespace {
class TestMediaBGR final: public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_mat;
    using Cb = ncvslideio::MediaFrame::View::Callback;
    Cb m_cb;

public:
    explicit TestMediaBGR(ncvslideio::Mat m, Cb cb = [](){})
        : m_mat(m), m_cb(cb) {
    }
    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ncvslideio::MediaFormat::BGR, ncvslideio::Size(m_mat.cols, m_mat.rows)};
    }
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        ncvslideio::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss), Cb{m_cb});
    }
};

class TestMediaNV12 final: public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_y;
    ncvslideio::Mat m_uv;
public:
    TestMediaNV12(ncvslideio::Mat y, ncvslideio::Mat uv) : m_y(y), m_uv(uv) {
    }
    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ncvslideio::MediaFormat::NV12, ncvslideio::Size(m_y.cols, m_y.rows)};
    }
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = {
            m_y.ptr(), m_uv.ptr(), nullptr, nullptr
        };
        ncvslideio::MediaFrame::View::Strides ss = {
            m_y.step, m_uv.step, 0u, 0u
        };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss));
    }
};

class TestMediaGray final : public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_mat;
    using Cb = ncvslideio::MediaFrame::View::Callback;
    Cb m_cb;

public:
    explicit TestMediaGray(ncvslideio::Mat m, Cb cb = []() {})
        : m_mat(m), m_cb(cb) {
    }
    ncvslideio::GFrameDesc meta() const override {
        return ncvslideio::GFrameDesc{ ncvslideio::MediaFormat::GRAY, ncvslideio::Size(m_mat.cols, m_mat.rows) };
    }
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override {
        ncvslideio::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        ncvslideio::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return ncvslideio::MediaFrame::View(std::move(pp), std::move(ss), Cb{ m_cb });
    }
};

} // anonymous namespace

struct MediaFrame_Test: public ::testing::Test {
    using M = ncvslideio::Mat;
    using MF = ncvslideio::MediaFrame;
    MF frame;
};

struct MediaFrame_BGR: public MediaFrame_Test {
    M bgr;
    MediaFrame_BGR()
        : bgr(M::eye(240, 320, CV_8UC3)) {
        ncvslideio::randn(bgr, ncvslideio::Scalar::all(127.0f), ncvslideio::Scalar::all(40.f));
        frame = MF::Create<TestMediaBGR>(bgr);
    }
};

TEST_F(MediaFrame_BGR, Meta) {
    auto meta = frame.desc();
    EXPECT_EQ(ncvslideio::MediaFormat::BGR, meta.fmt);
    EXPECT_EQ(ncvslideio::Size(320,240),    meta.size);
}

TEST_F(MediaFrame_BGR, Access) {
    ncvslideio::MediaFrame::View view1 = frame.access(ncvslideio::MediaFrame::Access::R);
    EXPECT_EQ(bgr.ptr(), view1.ptr[0]);
    EXPECT_EQ(bgr.step,  view1.stride[0]);

    ncvslideio::MediaFrame::View view2 = frame.access(ncvslideio::MediaFrame::Access::R);
    EXPECT_EQ(bgr.ptr(), view2.ptr[0]);
    EXPECT_EQ(bgr.step,  view2.stride[0]);
}

TEST_F(MediaFrame_BGR, Input) {
    // Run the OpenCV code
    ncvslideio::Mat out_mat_ocv, out_mat_gapi;
    ncvslideio::blur(bgr, out_mat_ocv, ncvslideio::Size{3,3});

    // Run the G-API code
    ncvslideio::GFrame in;
    ncvslideio::GMat out = GBlurFrame::on(in);
    ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
        .apply(ncvslideio::gin(frame),
               ncvslideio::gout(out_mat_gapi),
               ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVBlurFrame>()));

    // Compare
    EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
}

struct MediaFrame_Gray : public MediaFrame_Test {
    M gray;
    MediaFrame_Gray()
        : gray(M::eye(240, 320, CV_8UC1)) {
        ncvslideio::randn(gray, ncvslideio::Scalar::all(127.0f), ncvslideio::Scalar::all(40.f));
        frame = MF::Create<TestMediaGray>(gray);
    }
};

TEST_F(MediaFrame_Gray, Meta) {
    auto meta = frame.desc();
    EXPECT_EQ(ncvslideio::MediaFormat::GRAY, meta.fmt);
    EXPECT_EQ(ncvslideio::Size(320, 240), meta.size);
}

TEST_F(MediaFrame_Gray, Access) {
    ncvslideio::MediaFrame::View view1 = frame.access(ncvslideio::MediaFrame::Access::R);
    EXPECT_EQ(gray.ptr(), view1.ptr[0]);
    EXPECT_EQ(gray.step, view1.stride[0]);

    ncvslideio::MediaFrame::View view2 = frame.access(ncvslideio::MediaFrame::Access::R);
    EXPECT_EQ(gray.ptr(), view2.ptr[0]);
    EXPECT_EQ(gray.step, view2.stride[0]);
}

TEST_F(MediaFrame_Gray, Input) {
    // Run the OpenCV code
    ncvslideio::Mat out_mat_ocv, out_mat_gapi;
    ncvslideio::blur(gray, out_mat_ocv, ncvslideio::Size{ 3,3 });

    // Run the G-API code
    ncvslideio::GFrame in;
    ncvslideio::GMat out = GBlurFrameGray::on(in);
    ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
        .apply(ncvslideio::gin(frame),
            ncvslideio::gout(out_mat_gapi),
            ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVBlurFrameGray>()));

    // Compare
    EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
}


struct MediaFrame_NV12: public MediaFrame_Test {
    ncvslideio::Size sz;
    ncvslideio::Mat buf, y, uv;
    MediaFrame_NV12()
        : sz {320, 240}
        , buf(M::eye(sz.height*3/2, sz.width, CV_8UC1))
        , y  (buf.rowRange(0, sz.height))
        , uv (buf.rowRange(sz.height, sz.height*3/2)) {
        frame = MF::Create<TestMediaNV12>(y, uv);
    }
};

TEST_F(MediaFrame_NV12, Meta) {
    auto meta = frame.desc();
    EXPECT_EQ(ncvslideio::MediaFormat::NV12, meta.fmt);
    EXPECT_EQ(ncvslideio::Size(320,240),     meta.size);
}

TEST_F(MediaFrame_NV12, Access) {
    ncvslideio::MediaFrame::View view1 = frame.access(ncvslideio::MediaFrame::Access::R);
    EXPECT_EQ(y. ptr(), view1.ptr   [0]);
    EXPECT_EQ(y. step,  view1.stride[0]);
    EXPECT_EQ(uv.ptr(), view1.ptr   [1]);
    EXPECT_EQ(uv.step,  view1.stride[1]);

    ncvslideio::MediaFrame::View view2 = frame.access(ncvslideio::MediaFrame::Access::R);
    EXPECT_EQ(y. ptr(), view2.ptr   [0]);
    EXPECT_EQ(y. step,  view2.stride[0]);
    EXPECT_EQ(uv.ptr(), view2.ptr   [1]);
    EXPECT_EQ(uv.step,  view2.stride[1]);
}

TEST(MediaFrame, Callback) {
    int counter = 0;
    ncvslideio::Mat bgr = ncvslideio::Mat::eye(240, 320, CV_8UC3);
    ncvslideio::MediaFrame frame = ncvslideio::MediaFrame::Create<TestMediaBGR>(bgr, [&counter](){counter++;});

    // Test that the callback (in this case, incrementing the counter)
    // is called only on View destruction.
    EXPECT_EQ(0, counter);
    {
        ncvslideio::MediaFrame::View v1 = frame.access(ncvslideio::MediaFrame::Access::R);
        EXPECT_EQ(0, counter);
    }
    EXPECT_EQ(1, counter);
    {
        ncvslideio::MediaFrame::View v1 = frame.access(ncvslideio::MediaFrame::Access::R);
        EXPECT_EQ(1, counter);
        ncvslideio::MediaFrame::View v2 = frame.access(ncvslideio::MediaFrame::Access::W);
        EXPECT_EQ(1, counter);
    }
    EXPECT_EQ(3, counter);
}

TEST(MediaFrame, blobParams) {
    ncvslideio::Mat bgr = ncvslideio::Mat::eye(240, 320, CV_8UC3);
    ncvslideio::MediaFrame frame = ncvslideio::MediaFrame::Create<TestMediaBGR>(bgr);

    EXPECT_NO_THROW(frame.blobParams());
}

} // namespace opencv_test
