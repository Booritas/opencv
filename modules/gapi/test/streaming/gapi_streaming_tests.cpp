// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_streaming_tests_common.hpp"

#include <thread> // sleep_for (Delay)

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>
#include <opencv2/gapi/gstreaming.hpp>


namespace opencv_test
{
namespace
{

enum class KernelPackage: int
{
    OCV,
    OCV_FLUID,
    OCL,
    OCL_FLUID,
};
std::ostream& operator<< (std::ostream &os, const KernelPackage &e)
{
    switch (e)
    {
#define _C(X) case KernelPackage::X: os << #X; break
        _C(OCV);
        _C(OCV_FLUID);
        _C(OCL);
        _C(OCL_FLUID);
#undef _C
    default: GAPI_Error("InternalError");
    }
    return os;
}

struct GAPI_Streaming: public ::testing::TestWithParam<std::tuple<KernelPackage,
                                                                  ncvslideio::optional<size_t>>> {
    GAPI_Streaming() {
        KernelPackage pkg_kind;
        std::tie(pkg_kind, cap) = GetParam();
        pkg = getKernelPackage(pkg_kind);
    }

    const ncvslideio::optional<size_t>& getQueueCapacity()
    {
        return cap;
    }

    ncvslideio::GKernelPackage getKernelPackage(KernelPackage pkg_kind)
    {
        using namespace ncvslideio::gapi;
        switch (pkg_kind)
        {
        case KernelPackage::OCV:
            return ncvslideio::gapi::combine(core::cpu::kernels(),
                                     imgproc::cpu::kernels());
            break;

        case KernelPackage::OCV_FLUID:
            return ncvslideio::gapi::combine(core::cpu::kernels(),
                                     imgproc::cpu::kernels(),
                                     core::fluid::kernels());
            break;

        // FIXME: OpenCL backend seem to work fine with Streaming
        // however the results are not very bit exact with CPU
        // It may be a problem but may be just implementation innacuracy.
        // Need to customize the comparison function in tests where OpenCL
        // is involved.
        case KernelPackage::OCL:
            return ncvslideio::gapi::combine(core::ocl::kernels(),
                                     imgproc::ocl::kernels());
            break;

        case KernelPackage::OCL_FLUID:
            return ncvslideio::gapi::combine(core::ocl::kernels(),
                                     imgproc::ocl::kernels(),
                                     core::fluid::kernels());
            break;
        }
        throw std::logic_error("Unknown package");
    }

    ncvslideio::GCompileArgs getCompileArgs() {
        using namespace ncvslideio::gapi;
        auto args = ncvslideio::compile_args(use_only{pkg});
        if (cap) {
            args += ncvslideio::compile_args(ncvslideio::gapi::streaming::queue_capacity{cap.value()});
        }
        return args;
    }

    ncvslideio::GKernelPackage       pkg;
    ncvslideio::optional<size_t>     cap;
};

G_API_OP(Delay, <ncvslideio::GMat(ncvslideio::GMat, int)>, "org.opencv.test.delay") {
    static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in, int) { return in; }
};
GAPI_OCV_KERNEL(OCVDelay, Delay) {
    static void run(const ncvslideio::Mat &in, int ms, ncvslideio::Mat &out) {
        std::this_thread::sleep_for(std::chrono::milliseconds{ms});
        in.copyTo(out);
    }
};

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
        return ncvslideio::GFrameDesc{ncvslideio::MediaFormat::NV12, m_y.size()};
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

class TestMediaGRAY final : public ncvslideio::MediaFrame::IAdapter {
    ncvslideio::Mat m_mat;
    using Cb = ncvslideio::MediaFrame::View::Callback;
    Cb m_cb;

public:
    explicit TestMediaGRAY(ncvslideio::Mat m, Cb cb = []() {})
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


class BGRSource : public ncvslideio::gapi::wip::GCaptureSource {
public:
    explicit BGRSource(const std::string& pipeline)
        : ncvslideio::gapi::wip::GCaptureSource(pipeline) {
    }

    bool pull(ncvslideio::gapi::wip::Data& data) override {
        if (ncvslideio::gapi::wip::GCaptureSource::pull(data)) {
            data = ncvslideio::MediaFrame::Create<TestMediaBGR>(ncvslideio::util::get<ncvslideio::Mat>(data));
            return true;
        }
        return false;
    }

    GMetaArg descr_of() const override {
        return ncvslideio::GMetaArg{ncvslideio::GFrameDesc{ncvslideio::MediaFormat::BGR,
                                           ncvslideio::util::get<ncvslideio::GMatDesc>(
                                                   ncvslideio::gapi::wip::GCaptureSource::descr_of()).size}};
    }
};

void cvtBGR2NV12(const ncvslideio::Mat& bgr, ncvslideio::Mat& y, ncvslideio::Mat& uv) {
    ncvslideio::Size frame_sz = bgr.size();
    ncvslideio::Size half_sz  = frame_sz / 2;

    ncvslideio::Mat yuv;
    ncvslideio::cvtColor(bgr, yuv, ncvslideio::COLOR_BGR2YUV_I420);

    // Copy Y plane
    yuv.rowRange(0, frame_sz.height).copyTo(y);

    // Merge sampled U and V planes
    std::vector<int> dims = {half_sz.height, half_sz.width};
    auto start = frame_sz.height;
    auto range_h = half_sz.height/2;
    std::vector<ncvslideio::Mat> uv_planes = {
        yuv.rowRange(start,           start + range_h)  .reshape(0, dims),
        yuv.rowRange(start + range_h, start + range_h*2).reshape(0, dims)
    };
    ncvslideio::merge(uv_planes, uv);
}

class NV12Source : public ncvslideio::gapi::wip::GCaptureSource {
public:
    explicit NV12Source(const std::string& pipeline)
        : ncvslideio::gapi::wip::GCaptureSource(pipeline) {
    }

    bool pull(ncvslideio::gapi::wip::Data& data) override {
        if (ncvslideio::gapi::wip::GCaptureSource::pull(data)) {
            ncvslideio::Mat bgr = ncvslideio::util::get<ncvslideio::Mat>(data);
            ncvslideio::Mat y, uv;
            cvtBGR2NV12(bgr, y, uv);
            data = ncvslideio::MediaFrame::Create<TestMediaNV12>(y, uv);
            return true;
        }
        return false;
    }

    GMetaArg descr_of() const override {
        return ncvslideio::GMetaArg{ncvslideio::GFrameDesc{ncvslideio::MediaFormat::NV12,
            ncvslideio::util::get<ncvslideio::GMatDesc>(
                    ncvslideio::gapi::wip::GCaptureSource::descr_of()).size}};
    }
};

class GRAYSource : public ncvslideio::gapi::wip::GCaptureSource {
public:
    explicit GRAYSource(const std::string& pipeline)
        : ncvslideio::gapi::wip::GCaptureSource(pipeline) {
    }

    bool pull(ncvslideio::gapi::wip::Data& data) override {
        if (ncvslideio::gapi::wip::GCaptureSource::pull(data)) {
            ncvslideio::Mat bgr = ncvslideio::util::get<ncvslideio::Mat>(data);
            ncvslideio::Mat gray;
            cvtColor(bgr, gray, ncvslideio::COLOR_BGR2GRAY);
            data = ncvslideio::MediaFrame::Create<TestMediaGRAY>(gray);
            return true;
        }
        return false;
    }

    GMetaArg descr_of() const override {
        return ncvslideio::GMetaArg{ ncvslideio::GFrameDesc{ncvslideio::MediaFormat::GRAY,
                                            ncvslideio::util::get<ncvslideio::GMatDesc>(
                                            ncvslideio::gapi::wip::GCaptureSource::descr_of()).size} };
    }
};


void checkPullOverload(const ncvslideio::Mat& ref,
                       const bool has_output,
                       ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>& args) {
    EXPECT_TRUE(has_output);
    using runArgs = ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>;
    ncvslideio::Mat out_mat;
    switch (args.index()) {
        case runArgs::index_of<ncvslideio::GRunArgs>():
        {
            auto outputs = util::get<ncvslideio::GRunArgs>(args);
            EXPECT_EQ(1u, outputs.size());
            out_mat = ncvslideio::util::get<ncvslideio::Mat>(outputs[0]);
            break;
        }
        case runArgs::index_of<ncvslideio::GOptRunArgs>():
        {
            auto outputs = util::get<ncvslideio::GOptRunArgs>(args);
            EXPECT_EQ(1u, outputs.size());
            auto opt_mat = ncvslideio::util::get<ncvslideio::optional<ncvslideio::Mat>>(outputs[0]);
            ASSERT_TRUE(opt_mat.has_value());
            out_mat = *opt_mat;
            break;
        }
        default: GAPI_Error("Incorrect type of Args");
    }

    EXPECT_EQ(0., ncvslideio::norm(ref, out_mat, ncvslideio::NORM_INF));
}

class InvalidSource : public ncvslideio::gapi::wip::IStreamSource {
public:
    InvalidSource(const size_t throw_every_nth_frame,
                  const size_t num_frames)
        : m_throw_every_nth_frame(throw_every_nth_frame),
          m_curr_frame_id(0u),
          m_num_frames(num_frames),
          m_mat(1, 1, CV_8U) {
    }

    static std::string exception_msg()
    {
        return "InvalidSource successfully failed!";
    }

    bool pull(ncvslideio::gapi::wip::Data& d) override {
        ++m_curr_frame_id;
        if (m_curr_frame_id > m_num_frames) {
            return false;
        }

        if (m_curr_frame_id % m_throw_every_nth_frame == 0) {
            throw std::logic_error(InvalidSource::exception_msg());
            return true;
        } else {
            d = ncvslideio::Mat(m_mat);
        }

        return true;
    }

    ncvslideio::GMetaArg descr_of() const override {
        return ncvslideio::GMetaArg{ncvslideio::descr_of(m_mat)};
    }

private:
    size_t m_throw_every_nth_frame;
    size_t m_curr_frame_id;
    size_t m_num_frames;
    ncvslideio::Mat m_mat;
};

G_TYPED_KERNEL(GThrowExceptionOp, <GMat(GMat)>, "org.opencv.test.throw_error_op")
{
     static GMatDesc outMeta(GMatDesc in) { return in; }
};

GAPI_OCV_KERNEL(GThrowExceptionKernel, GThrowExceptionOp)
{
    static std::string exception_msg()
    {
        return "GThrowExceptionKernel successfully failed";
    }

    static void run(const ncvslideio::Mat&, ncvslideio::Mat&)
    {
        throw std::logic_error(GThrowExceptionKernel::exception_msg());
    }
};

} // anonymous namespace

TEST_P(GAPI_Streaming, SmokeTest_ConstInput_GMat)
{
    // This graph models the following use-case:
    // Canny here is used as some "feature detector"
    //
    // Island/device layout may be different given the contents
    // of the passed kernel package.
    //
    // The expectation is that we get as much islands in the
    // graph as backends the GKernelPackage contains.
    //
    // [Capture] --> Crop --> Resize --> Canny --> [out]

    const auto crop_rc = ncvslideio::Rect(13, 75, 377, 269);
    const auto resample_sz = ncvslideio::Size(224, 224);
    const auto thr_lo = 64.;
    const auto thr_hi = 192.;

    ncvslideio::GMat in;
    auto roi = ncvslideio::gapi::crop(in, crop_rc);
    auto res = ncvslideio::gapi::resize(roi, resample_sz);
    auto out = ncvslideio::gapi::Canny(res, thr_lo, thr_hi);
    ncvslideio::GComputation c(in, out);

    // Input data
    ncvslideio::Mat in_mat = ncvslideio::imread(findDataFile("ncvslideio/edgefilter/kodim23.png"));
    ncvslideio::Mat out_mat_gapi;

    // OpenCV reference image
    ncvslideio::Mat out_mat_ocv;
    {
        ncvslideio::Mat tmp;
        ncvslideio::resize(in_mat(crop_rc), tmp, resample_sz);
        ncvslideio::Canny(tmp, out_mat_ocv, thr_lo, thr_hi);
    }

    // Compilation & testing
    auto ccomp = c.compileStreaming(ncvslideio::descr_of(in_mat), getCompileArgs());
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(ncvslideio::gin(in_mat));

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Fetch the result 15 times
    for (int i = 0; i < 15; i++) {
        // With constant inputs, the stream is endless so
        // the blocking pull() should never return `false`.
        EXPECT_TRUE(ccomp.pull(ncvslideio::gout(out_mat_gapi)));
        // Fluid's and OpenCV's Resizes aren't bit exact.
        // So 1% is here because it is max difference between them.
        EXPECT_TRUE(AbsSimilarPoints(0, 1).to_compare_f()(out_mat_gapi, out_mat_ocv));
    }

    EXPECT_TRUE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());
}

TEST_P(GAPI_Streaming, SmokeTest_VideoInput_GMat)
{
    const auto crop_rc = ncvslideio::Rect(13, 75, 377, 269);
    const auto resample_sz = ncvslideio::Size(224, 224);
    const auto thr_lo = 64.;
    const auto thr_hi = 192.;

    ncvslideio::GMat in;
    auto roi = ncvslideio::gapi::crop(in, crop_rc);
    auto res = ncvslideio::gapi::resize(roi, resample_sz);
    auto out = ncvslideio::gapi::Canny(res, thr_lo, thr_hi);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(ncvslideio::gapi::copy(in), out));

    // OpenCV reference image code
    auto opencv_ref = [&](const ncvslideio::Mat &in_mat, ncvslideio::Mat &out_mat) {
        ncvslideio::Mat tmp;
        ncvslideio::resize(in_mat(crop_rc), tmp, resample_sz);
        ncvslideio::Canny(tmp, out_mat, thr_lo, thr_hi);
    };

    // Compilation & testing
    auto ccomp = c.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                                    getCompileArgs());
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        ccomp.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Process the full video
    ncvslideio::Mat in_mat_gapi, out_mat_gapi;

    std::size_t frames = 0u;
    while (ccomp.pull(ncvslideio::gout(in_mat_gapi, out_mat_gapi))) {
        frames++;
        ncvslideio::Mat out_mat_ocv;
        opencv_ref(in_mat_gapi, out_mat_ocv);
        // Fluid's and OpenCV's Resizes aren't bit exact.
        // So 1% is here because it is max difference between them.
        EXPECT_TRUE(AbsSimilarPoints(0, 1).to_compare_f()(out_mat_gapi, out_mat_ocv));
    }
    EXPECT_LT(0u, frames);
    EXPECT_FALSE(ccomp.running());

    // Stop can be called at any time (even if the pipeline is not running)
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());
}

TEST_P(GAPI_Streaming, Regression_CompileTimeScalar)
{
    // There was a bug with compile-time GScalars.  Compile-time
    // GScalars generate their own DATA nodes at GModel/GIslandModel
    // level, resulting in an extra link at the GIslandModel level, so
    // GStreamingExecutor automatically assigned an input queue to
    // such edges. Since there were no in-graph producer for that
    // data, no data were pushed to such queue what lead to a
    // deadlock.

    ncvslideio::GMat in;
    ncvslideio::GMat tmp = ncvslideio::gapi::copy(in);
    for (int i = 0; i < 3; i++) {
        tmp = tmp & ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));
    }
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(tmp, tmp + 1));

    auto ccomp = c.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,512}},
                                    getCompileArgs());

    ncvslideio::Mat in_mat = ncvslideio::imread(findDataFile("ncvslideio/edgefilter/kodim23.png"));
    ncvslideio::Mat out_mat1, out_mat2;

    // Fetch the result 15 times
    ccomp.setSource(ncvslideio::gin(in_mat));
    ccomp.start();
    for (int i = 0; i < 15; i++) {
        EXPECT_TRUE(ccomp.pull(ncvslideio::gout(out_mat1, out_mat2)));
    }

    ccomp.stop();
}

TEST_P(GAPI_Streaming, SmokeTest_StartRestart)
{
    ncvslideio::GMat in;
    auto res = ncvslideio::gapi::resize(in, ncvslideio::Size{300,200});
    auto out = ncvslideio::gapi::Canny(res, 95, 220);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(ncvslideio::gapi::copy(in), out));

    auto ccomp = c.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                                    getCompileArgs());
    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // Run 1
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    std::size_t num_frames1 = 0u;
    try {
        ccomp.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    ncvslideio::Mat out1, out2;
    while (ccomp.pull(ncvslideio::gout(out1, out2))) num_frames1++;

    EXPECT_FALSE(ccomp.running());

    // Run 2
    std::size_t num_frames2 = 0u;
    try {
        ccomp.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();
    EXPECT_TRUE(ccomp.running());
    while (ccomp.pull(ncvslideio::gout(out1, out2))) num_frames2++;

    EXPECT_FALSE(ccomp.running());

    EXPECT_LT(0u, num_frames1);
    EXPECT_LT(0u, num_frames2);
    EXPECT_EQ(num_frames1, num_frames2);
}

TEST_P(GAPI_Streaming, SmokeTest_VideoConstSource_NoHang)
{
    // A video source is a finite one, while const source is not.
    // Check that pipeline completes when a video source completes.
    auto refc = ncvslideio::GComputation([](){
        ncvslideio::GMat in;
        return ncvslideio::GComputation(in, ncvslideio::gapi::copy(in));
    }).compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}}, getCompileArgs());

    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        refc.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    refc.start();
    std::size_t ref_frames = 0u;
    ncvslideio::Mat tmp;
    while (refc.pull(ncvslideio::gout(tmp))) ref_frames++;
    EXPECT_EQ(100u, ref_frames);

    ncvslideio::GMat in;
    ncvslideio::GMat in2;
    ncvslideio::GMat roi = ncvslideio::gapi::crop(in2, ncvslideio::Rect{1,1,256,256});
    ncvslideio::GMat blr = ncvslideio::gapi::blur(roi, ncvslideio::Size(3,3));
    ncvslideio::GMat out = blr - in;
    auto testc = ncvslideio::GComputation(ncvslideio::GIn(in, in2), ncvslideio::GOut(out))
        .compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{256,256}},
                          ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                          getCompileArgs());

    ncvslideio::Mat in_const = ncvslideio::Mat::eye(ncvslideio::Size(256,256), CV_8UC3);
    testc.setSource(ncvslideio::gin(in_const,
                            gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    testc.start();
    std::size_t test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;

    EXPECT_EQ(ref_frames, test_frames);
}

TEST_P(GAPI_Streaming, SmokeTest_AutoMeta)
{
    ncvslideio::GMat in;
    ncvslideio::GMat in2;
    ncvslideio::GMat roi = ncvslideio::gapi::crop(in2, ncvslideio::Rect{1,1,256,256});
    ncvslideio::GMat blr = ncvslideio::gapi::blur(roi, ncvslideio::Size(3,3));
    ncvslideio::GMat out = blr - in;

    auto testc = ncvslideio::GComputation(ncvslideio::GIn(in, in2), ncvslideio::GOut(out))
        .compileStreaming(getCompileArgs());

    ncvslideio::Mat in_const = ncvslideio::Mat::eye(ncvslideio::Size(256,256), CV_8UC3);
    ncvslideio::Mat tmp;

    // Test with one video source
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        testc.setSource(ncvslideio::gin(in_const, gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    testc.start();

    std::size_t test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;
    EXPECT_EQ(100u, test_frames);

    // Now test with another one
    path = findDataFile("ncvslideio/video/1920x1080.avi");
    try {
        testc.setSource(ncvslideio::gin(in_const, gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    testc.start();

    test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;
    EXPECT_EQ(165u, test_frames);
}


TEST_P(GAPI_Streaming, SmokeTest_AutoMeta_2xConstMat)
{
    ncvslideio::GMat in;
    ncvslideio::GMat in2;
    ncvslideio::GMat roi = ncvslideio::gapi::crop(in2, ncvslideio::Rect{1,1,256,256});
    ncvslideio::GMat blr = ncvslideio::gapi::blur(roi, ncvslideio::Size(3,3));
    ncvslideio::GMat out = blr - in;

    auto testc = ncvslideio::GComputation(ncvslideio::GIn(in, in2), ncvslideio::GOut(out))
        .compileStreaming(getCompileArgs());

    ncvslideio::Mat in_const = ncvslideio::Mat::eye(ncvslideio::Size(256,256), CV_8UC3);
    ncvslideio::Mat tmp;

    // Test with first image
    auto in_src = ncvslideio::imread(findDataFile("ncvslideio/edgefilter/statue.png"));
    testc.setSource(ncvslideio::gin(in_const, in_src));
    testc.start();

    ASSERT_TRUE(testc.pull(ncvslideio::gout(tmp)));

    testc.stop();

    // Now test with second image
    in_src = ncvslideio::imread(findDataFile("ncvslideio/edgefilter/kodim23.png"));
    testc.setSource(ncvslideio::gin(in_const, in_src));
    testc.start();

    ASSERT_TRUE(testc.pull(ncvslideio::gout(tmp)));

    testc.stop();
}

TEST_P(GAPI_Streaming, SmokeTest_AutoMeta_VideoScalar)
{
    ncvslideio::GMat in_m;
    ncvslideio::GScalar in_s;
    ncvslideio::GMat out_m = in_m * in_s;

    auto testc = ncvslideio::GComputation(ncvslideio::GIn(in_m, in_s), ncvslideio::GOut(out_m))
        .compileStreaming(getCompileArgs());

    ncvslideio::Mat tmp;
    // Test with one video source and scalar
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        testc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path), ncvslideio::Scalar{1.25}));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    testc.start();

    std::size_t test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;
    EXPECT_EQ(100u, test_frames);

    // Now test with another one video source and scalar
    path = findDataFile("ncvslideio/video/1920x1080.avi");
    try {
        testc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path), ncvslideio::Scalar{0.75}));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    testc.start();

    test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;
    EXPECT_EQ(165u, test_frames);
}

// Instantiate tests with different backends, but default queue capacity
INSTANTIATE_TEST_CASE_P(TestStreaming, GAPI_Streaming,
                        Combine(Values( KernelPackage::OCV
                                      , KernelPackage::OCV_FLUID),
                                Values(ncvslideio::optional<size_t>{})));

// Instantiate tests with the same backend but various queue capacity
INSTANTIATE_TEST_CASE_P(TestStreaming_QC, GAPI_Streaming,
                        Combine(Values(KernelPackage::OCV_FLUID),
                                Values(1u, 4u)));

namespace TypesTest
{
    G_API_OP(SumV, <ncvslideio::GArray<int>(ncvslideio::GMat)>, "test.gapi.sumv") {
        static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc &) {
            return ncvslideio::empty_array_desc();
        }
    };
    G_API_OP(AddV, <ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GArray<int>)>, "test.gapi.addv") {
        static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in, const ncvslideio::GArrayDesc &) {
            return in;
        }
    };

    GAPI_OCV_KERNEL(OCVSumV, SumV) {
        static void run(const ncvslideio::Mat &in, std::vector<int> &out) {
            CV_Assert(in.depth() == CV_8U);
            const auto length = in.cols * in.channels();
            out.resize(length);

            const uchar *ptr = in.ptr(0);
            for (int c = 0; c < length; c++) {
                out[c] = ptr[c];
            }
            for (int r = 1; r < in.rows; r++) {
                ptr = in.ptr(r);
                for (int c = 0; c < length; c++) {
                    out[c] += ptr[c];
                }
            }
        }
    };

    GAPI_OCV_KERNEL(OCVAddV, AddV) {
        static void run(const ncvslideio::Mat &in, const std::vector<int> &inv, ncvslideio::Mat &out) {
            CV_Assert(in.depth() == CV_8U);
            const auto length = in.cols * in.channels();
            CV_Assert(length == static_cast<int>(inv.size()));

            for (int r = 0; r < in.rows; r++) {
                const uchar *in_ptr = in.ptr(r);
                uchar *out_ptr = out.ptr(r);

                for (int c = 0; c < length; c++) {
                    out_ptr[c] = ncvslideio::saturate_cast<uchar>(in_ptr[c] + inv[c]);
                }
            }
        }
    };

    GAPI_FLUID_KERNEL(FluidAddV, AddV, false) {
        static const int Window = 1;

        static void run(const ncvslideio::gapi::fluid::View &in,
                        const std::vector<int> &inv,
                        ncvslideio::gapi::fluid::Buffer &out) {
            const uchar *in_ptr = in.InLineB(0);
            uchar *out_ptr = out.OutLineB(0);

            const auto length = in.meta().size.width * in.meta().chan;
            CV_Assert(length == static_cast<int>(inv.size()));

            for (int c = 0; c < length; c++) {
                out_ptr[c] = ncvslideio::saturate_cast<uchar>(in_ptr[c] + inv[c]);
            }
        }
    };
} // namespace TypesTest

TEST_P(GAPI_Streaming, SmokeTest_AutoMeta_VideoArray)
{
    ncvslideio::GMat in_m;
    ncvslideio::GArray<int> in_v;
    ncvslideio::GMat out_m = TypesTest::AddV::on(in_m, in_v) - in_m;

    // Run pipeline
    auto args = ncvslideio::compile_args(ncvslideio::gapi::kernels<TypesTest::OCVAddV>());
    auto capacity = getQueueCapacity();
    if (capacity)
    {
        args += ncvslideio::compile_args(
                    ncvslideio::gapi::streaming::queue_capacity{capacity.value()});
    }
    auto testc = ncvslideio::GComputation(ncvslideio::GIn(in_m, in_v), ncvslideio::GOut(out_m))
                    .compileStreaming(std::move(args));

    ncvslideio::Mat tmp;
    // Test with one video source and vector
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    std::vector<int> first_in_vec(768*3, 1);
    try {
        testc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path), first_in_vec));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    testc.start();

    std::size_t test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;
    EXPECT_EQ(100u, test_frames);

    // Now test with another one
    path = findDataFile("ncvslideio/video/1920x1080.avi");
    std::vector<int> second_in_vec(1920*3, 1);
    try {
        testc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path), second_in_vec));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    testc.start();

    test_frames = 0u;
    while (testc.pull(ncvslideio::gout(tmp))) test_frames++;
    EXPECT_EQ(165u, test_frames);
}

TEST(GAPI_Streaming_Types, InputScalar)
{
    // This test verifies if Streaming works with Scalar data @ input.

    ncvslideio::GMat in_m;
    ncvslideio::GScalar in_s;
    ncvslideio::GMat out_m = in_m * in_s;
    ncvslideio::GComputation c(ncvslideio::GIn(in_m, in_s), ncvslideio::GOut(out_m));

    // Input data
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(256, 256, CV_8UC1);
    ncvslideio::Scalar in_scl = 32;

    // Run pipeline
    auto sc = c.compileStreaming(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_scl));
    sc.setSource(ncvslideio::gin(in_mat, in_scl));
    sc.start();

    for (int i = 0; i < 10; i++)
    {
        ncvslideio::Mat out;
        EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));
        EXPECT_EQ(0., ncvslideio::norm(out, in_mat.mul(in_scl), ncvslideio::NORM_INF));
    }
}

TEST(GAPI_Streaming_Types, InputVector)
{
    // This test verifies if Streaming works with Vector data @ input.

    ncvslideio::GMat in_m;
    ncvslideio::GArray<int> in_v;
    ncvslideio::GMat out_m = TypesTest::AddV::on(in_m, in_v) - in_m;
    ncvslideio::GComputation c(ncvslideio::GIn(in_m, in_v), ncvslideio::GOut(out_m));

    // Input data
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(256, 256, CV_8UC1);
    std::vector<int> in_vec;
    TypesTest::OCVSumV::run(in_mat, in_vec);
    EXPECT_EQ(std::vector<int>(256,1), in_vec); // self-sanity-check

    auto opencv_ref = [&](const ncvslideio::Mat &in, const std::vector<int> &inv, ncvslideio::Mat &out) {
        ncvslideio::Mat tmp = in_mat.clone(); // allocate the same amount of memory as graph does
        TypesTest::OCVAddV::run(in, inv, tmp);
        out = tmp - in;
    };

    // Run pipeline
    auto sc = c.compileStreaming(ncvslideio::descr_of(in_mat),
                                 ncvslideio::descr_of(in_vec),
                                 ncvslideio::compile_args(ncvslideio::gapi::kernels<TypesTest::OCVAddV>()));
    sc.setSource(ncvslideio::gin(in_mat, in_vec));
    sc.start();

    for (int i = 0; i < 10; i++)
    {
        ncvslideio::Mat out_mat;
        EXPECT_TRUE(sc.pull(ncvslideio::gout(out_mat)));

        ncvslideio::Mat ref_mat;
        opencv_ref(in_mat, in_vec, ref_mat);
        EXPECT_EQ(0., ncvslideio::norm(ref_mat, out_mat, ncvslideio::NORM_INF));
    }
}

TEST(GAPI_Streaming_Types, XChangeScalar)
{
    // This test verifies if Streaming works when pipeline steps
    // (islands) exchange Scalar data.

    ncvslideio::GMat in;
    ncvslideio::GScalar m = ncvslideio::gapi::mean(in);
    ncvslideio::GMat tmp = ncvslideio::gapi::convertTo(in, CV_32F) - m;
    ncvslideio::GMat out = ncvslideio::gapi::blur(tmp, ncvslideio::Size(3,3));
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(ncvslideio::gapi::copy(in),
                                             ncvslideio::gapi::convertTo(out, CV_8U)));

    auto ocv_ref = [](const ncvslideio::Mat &in_mat, ncvslideio::Mat &out_mat) {
        ncvslideio::Scalar ocv_m = ncvslideio::mean(in_mat);
        ncvslideio::Mat ocv_tmp;
        in_mat.convertTo(ocv_tmp, CV_32F);
        ocv_tmp -= ocv_m;
        ncvslideio::blur(ocv_tmp, ocv_tmp, ncvslideio::Size(3,3));
        ocv_tmp.convertTo(out_mat, CV_8U);
    };

    // Here we want mean & convertTo run on OCV
    // and subC & blur3x3 on Fluid.
    // FIXME: With the current API it looks quite awful:
    auto ocv_kernels = ncvslideio::gapi::core::cpu::kernels(); // convertTo
    ocv_kernels.remove<ncvslideio::gapi::core::GSubC>();

    auto fluid_kernels = ncvslideio::gapi::combine(ncvslideio::gapi::core::fluid::kernels(),     // subC
                                           ncvslideio::gapi::imgproc::fluid::kernels()); // box3x3
    fluid_kernels.remove<ncvslideio::gapi::core::GConvertTo>();
    fluid_kernels.remove<ncvslideio::gapi::core::GMean>();

    // FIXME: Now
    // - fluid kernels take over ocv kernels (including Copy, SubC, & Box3x3)
    // - selected kernels (which were removed from the fluid package) remain in OCV
    //   (ConvertTo + some others)
    // FIXME: This is completely awful. User should easily pick up specific kernels
    // to an empty kernel package to craft his own but not do it via exclusion.
    // Need to expose kernel declarations to public headers to enable kernels<..>()
    // on user side.
    auto kernels = ncvslideio::gapi::combine(ocv_kernels, fluid_kernels);

    // Compile streaming pipeline
    auto sc = c.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                                ncvslideio::compile_args(ncvslideio::gapi::use_only{kernels}));
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    ncvslideio::Mat in_frame;
    ncvslideio::Mat out_mat_gapi;
    ncvslideio::Mat out_mat_ref;

    std::size_t num_frames = 0u;
    while (sc.pull(ncvslideio::gout(in_frame, out_mat_gapi))) {
        num_frames++;
        ocv_ref(in_frame, out_mat_ref);
        EXPECT_EQ(0., ncvslideio::norm(out_mat_gapi, out_mat_ref, ncvslideio::NORM_INF));
    }
    EXPECT_LT(0u, num_frames);
}

TEST(GAPI_Streaming_Types, XChangeVector)
{
    // This test verifies if Streaming works when pipeline steps
    // (islands) exchange Vector data.

    ncvslideio::GMat in1, in2;
    ncvslideio::GMat in = ncvslideio::gapi::crop(in1, ncvslideio::Rect{0,0,576,576});
    ncvslideio::GScalar m = ncvslideio::gapi::mean(in);
    ncvslideio::GArray<int> s = TypesTest::SumV::on(in2); // (in2 = eye, so s = [1,0,0,1,..])
    ncvslideio::GMat out = TypesTest::AddV::on(in - m, s);

    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(ncvslideio::gapi::copy(in), out));

    auto ocv_ref = [](const ncvslideio::Mat &in_mat1, const ncvslideio::Mat &in_mat2, ncvslideio::Mat &out_mat) {
        ncvslideio::Mat in_roi = in_mat1(ncvslideio::Rect{0,0,576,576});
        ncvslideio::Scalar ocv_m = ncvslideio::mean(in_roi);
        std::vector<int> ocv_v;
        TypesTest::OCVSumV::run(in_mat2, ocv_v);

        out_mat.create(ncvslideio::Size(576,576), CV_8UC3);
        ncvslideio::Mat in_tmp = in_roi - ocv_m;
        TypesTest::OCVAddV::run(in_tmp, ocv_v, out_mat);
    };

    // Let crop/mean/sumV be calculated via OCV,
    // and AddV/subC be calculated via Fluid
    auto ocv_kernels = ncvslideio::gapi::core::cpu::kernels();
    ocv_kernels.remove<ncvslideio::gapi::core::GSubC>();
    ocv_kernels.include<TypesTest::OCVSumV>();

    auto fluid_kernels = ncvslideio::gapi::core::fluid::kernels();
    fluid_kernels.include<TypesTest::FluidAddV>();

    // Here OCV takes precedense over Fluid, with SubC & SumV remaining
    // in Fluid.
    auto kernels = ncvslideio::gapi::combine(fluid_kernels, ocv_kernels);

    // Compile streaming pipeline
    ncvslideio::Mat in_eye = ncvslideio::Mat::eye(ncvslideio::Size(576, 576), CV_8UC3);
    auto sc = c.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                                 ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{576,576}},
                                 ncvslideio::compile_args(ncvslideio::gapi::use_only{kernels}));
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path),
                             in_eye));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    ncvslideio::Mat in_frame;
    ncvslideio::Mat out_mat_gapi;
    ncvslideio::Mat out_mat_ref;

    std::size_t num_frames = 0u;
    while (sc.pull(ncvslideio::gout(in_frame, out_mat_gapi))) {
        num_frames++;
        ocv_ref(in_frame, in_eye, out_mat_ref);
        EXPECT_EQ(0., ncvslideio::norm(out_mat_gapi, out_mat_ref, ncvslideio::NORM_INF));
    }
    EXPECT_LT(0u, num_frames);
}

TEST(GAPI_Streaming_Types, OutputScalar)
{
    // This test verifies if Streaming works when pipeline
    // produces scalar data only

    ncvslideio::GMat in;
    ncvslideio::GScalar out = ncvslideio::gapi::mean(in);
    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
        .compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}});

    std::string video_path;
    video_path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(video_path));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    ncvslideio::VideoCapture cap;
    cap.open(video_path);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::Mat tmp;
    ncvslideio::Scalar out_scl;
    std::size_t num_frames = 0u;
    while (sc.pull(ncvslideio::gout(out_scl)))
    {
        num_frames++;
        cap >> tmp;
        ncvslideio::Scalar out_ref = ncvslideio::mean(tmp);
        EXPECT_EQ(out_ref, out_scl);
    }
    EXPECT_LT(0u, num_frames);
}

TEST(GAPI_Streaming_Types, OutputVector)
{
    // This test verifies if Streaming works when pipeline
    // produces vector data only

    auto pkg = ncvslideio::gapi::kernels<TypesTest::OCVSumV>();

    ncvslideio::GMat in1, in2;
    ncvslideio::GMat roi = ncvslideio::gapi::crop(in2, ncvslideio::Rect(3,3,256,256));
    ncvslideio::GArray<int> out = TypesTest::SumV::on(ncvslideio::gapi::mul(roi, in1));
    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out))
        .compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{256,256}},
                          ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                          ncvslideio::compile_args(pkg));

    auto ocv_ref = [](const ncvslideio::Mat &ocv_in1,
                      const ncvslideio::Mat &ocv_in2,
                      std::vector<int> &ocv_out) {
        auto ocv_roi = ocv_in2(ncvslideio::Rect{3,3,256,256});
        TypesTest::OCVSumV::run(ocv_roi.mul(ocv_in1), ocv_out);
    };

    ncvslideio::Mat in_eye = ncvslideio::Mat::eye(ncvslideio::Size(256, 256), CV_8UC3);
    std::string video_path;
    video_path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(ncvslideio::gin(in_eye, gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(video_path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    ncvslideio::VideoCapture cap;
    cap.open(video_path);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::Mat tmp;
    std::vector<int> ref_vec;
    std::vector<int> out_vec;
    std::size_t num_frames = 0u;
    while (sc.pull(ncvslideio::gout(out_vec)))
    {
        num_frames++;
        cap >> tmp;
        ref_vec.clear();
        ocv_ref(in_eye, tmp, ref_vec);
        EXPECT_EQ(ref_vec, out_vec);
    }
    EXPECT_LT(0u, num_frames);
}

G_API_OP(DimsChans,
         <std::tuple<ncvslideio::GArray<int>, ncvslideio::GOpaque<int>>(ncvslideio::GMat)>,
         "test.streaming.dims_chans") {
    static std::tuple<ncvslideio::GArrayDesc, ncvslideio::GOpaqueDesc> outMeta(const ncvslideio::GMatDesc &) {
        return std::make_tuple(ncvslideio::empty_array_desc(),
                               ncvslideio::empty_gopaque_desc());
    }
};

GAPI_OCV_KERNEL(OCVDimsChans, DimsChans) {
    static void run(const ncvslideio::Mat &in, std::vector<int> &ov, int &oi) {
        ov = {in.cols, in.rows};
        oi = in.channels();
    }
};

struct GAPI_Streaming_TemplateTypes: ::testing::Test {
    // There was a problem in GStreamingExecutor
    // when outputs were formally not used by the graph
    // but still should be in place as operation need
    // to produce them, and host data type constructors
    // were missing for GArray and GOpaque in this case.
    // This test tests exactly this.

    GAPI_Streaming_TemplateTypes() {
        // Prepare everything for the test:
        // Graph itself
        blur = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

        ncvslideio::GMat blur_d = ncvslideio::gapi::streaming::desync(blur);
        std::tie(vec, opq) = DimsChans::on(blur_d);

        // Kernel package
        pkg = ncvslideio::gapi::kernels<OCVDimsChans>();

        // Input mat
        in_mat = ncvslideio::Mat::eye(ncvslideio::Size(320,240), CV_8UC3);
    }

    ncvslideio::GMat in;
    ncvslideio::GMat blur;
    ncvslideio::GArray<int> vec;
    ncvslideio::GOpaque<int> opq;
    ncvslideio::GKernelPackage pkg;
    ncvslideio::Mat in_mat;
};

TEST_F(GAPI_Streaming_TemplateTypes, UnusedVectorIsOK)
{
    // Declare graph without listing vec as output
    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(blur, opq))
        .compileStreaming(ncvslideio::compile_args(pkg));
    sc.setSource(ncvslideio::gin(in_mat));
    sc.start();

    ncvslideio::optional<ncvslideio::Mat> out_mat;
    ncvslideio::optional<int> out_int;

    int counter = 0;
    while (sc.pull(ncvslideio::gout(out_mat, out_int))) {
        if (counter++ == 10) {
            // Stop the test after 10 iterations
            sc.stop();
            break;
        }
        GAPI_Assert(out_mat || out_int);
        if (out_int) {
            EXPECT_EQ(3, out_int.value());
        }
    }
}

TEST_F(GAPI_Streaming_TemplateTypes, UnusedOpaqueIsOK)
{
    // Declare graph without listing opq as output
    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(blur, vec))
        .compileStreaming(ncvslideio::compile_args(pkg));
    sc.setSource(ncvslideio::gin(in_mat));
    sc.start();

    ncvslideio::optional<ncvslideio::Mat> out_mat;
    ncvslideio::optional<std::vector<int> > out_vec;

    int counter = 0;
    while (sc.pull(ncvslideio::gout(out_mat, out_vec))) {
        if (counter++ == 10) {
            // Stop the test after 10 iterations
            sc.stop();
            break;
        }
        GAPI_Assert(out_mat || out_vec);
        if (out_vec) {
            EXPECT_EQ(320, out_vec.value()[0]);
            EXPECT_EQ(240, out_vec.value()[1]);
        }
    }
}

struct GAPI_Streaming_Unit: public ::testing::Test {
    ncvslideio::Mat m;

    ncvslideio::GComputation cc;
    ncvslideio::GStreamingCompiled sc;

    ncvslideio::GCompiled ref;

    GAPI_Streaming_Unit()
        : m(ncvslideio::Mat::ones(224,224,CV_8UC3))
        , cc([]{
                ncvslideio::GMat a, b;
                ncvslideio::GMat c = a + b*2;
                return ncvslideio::GComputation(ncvslideio::GIn(a, b), ncvslideio::GOut(c));
            })
    {

        const auto a_desc = ncvslideio::descr_of(m);
        const auto b_desc = ncvslideio::descr_of(m);
        sc  = cc.compileStreaming(a_desc, b_desc);
        ref = cc.compile(a_desc, b_desc);
    }
};

// FIXME: (GAPI_Streaming_Types,   InputOpaque) test is missing here!
// FIXME: (GAPI_Streaming_Types, XChangeOpaque) test is missing here!
// FIXME: (GAPI_Streaming_Types,  OutputOpaque) test is missing here!

TEST(GAPI_Streaming, TestTwoVideosDifferentLength)
{
    auto desc = ncvslideio::GMatDesc{CV_8U,3,{768,576}};
    auto path1 = findDataFile("ncvslideio/video/768x576.avi");
    auto path2 = findDataFile("highgui/video/big_buck_bunny.avi");

    ncvslideio::GMat in1, in2;
    auto out = in1 + ncvslideio::gapi::resize(in2, desc.size);

    ncvslideio::GComputation cc(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    auto sc = cc.compileStreaming();
    try {
        sc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path1),
                             gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path2)));
    } catch(...) {
        throw SkipTestException("Video file can not be found");
    }
    sc.start();

    ncvslideio::Mat out_mat;
    std::size_t frames = 0u;
    while(sc.pull(ncvslideio::gout(out_mat))) {
        frames++;
    }

    // big_buck_bunny.avi has 125 frames, 768x576.avi - 100 frames,
    // expect framework to stop after 100 frames
    EXPECT_EQ(100u, frames);
}

TEST_F(GAPI_Streaming_Unit, TestStartWithoutnSetSource)
{
    EXPECT_ANY_THROW(sc.start());
}

TEST_F(GAPI_Streaming_Unit, TestStopWithoutStart1)
{
    // It is ok!
    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, TestStopWithoutStart2)
{
    // It should be ok as well
    sc.setSource(ncvslideio::gin(m, m));
    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, StopStartStop)
{
    ncvslideio::Mat out;
    EXPECT_NO_THROW(sc.stop());
    EXPECT_NO_THROW(sc.setSource(ncvslideio::gin(m, m)));
    EXPECT_NO_THROW(sc.start());

    std::size_t i = 0u;
    while (i++ < 10u) {EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));};

    EXPECT_NO_THROW(sc.stop());
}

TEST_F(GAPI_Streaming_Unit, ImplicitStop)
{
    EXPECT_NO_THROW(sc.setSource(ncvslideio::gin(m, m)));
    EXPECT_NO_THROW(sc.start());
    // No explicit stop here - pipeline stops successfully at the test exit
}

TEST_F(GAPI_Streaming_Unit, StartStopStart_NoSetSource)
{
    EXPECT_NO_THROW(sc.setSource(ncvslideio::gin(m, m)));
    EXPECT_NO_THROW(sc.start());
    EXPECT_NO_THROW(sc.stop());
    EXPECT_ANY_THROW(sc.start()); // Should fail since setSource was not called
}

TEST_F(GAPI_Streaming_Unit, StartStopStress_Const)
{
    // Runs 100 times with no deadlock - assumed stable (robust) enough
    for (int i = 0; i < 100; i++)
    {
        sc.stop();
        sc.setSource(ncvslideio::gin(m, m));
        sc.start();
        ncvslideio::Mat out;
        for (int j = 0; j < 5; j++) EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));
    }
}

TEST_F(GAPI_Streaming_Unit, StartStopStress_Video)
{
    // Runs 100 times with no deadlock - assumed stable (robust) enough
    sc = cc.compileStreaming(ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}},
                             ncvslideio::GMatDesc{CV_8U,3,ncvslideio::Size{768,576}});
    m = ncvslideio::Mat::eye(ncvslideio::Size{768,576}, CV_8UC3);
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    for (int i = 0; i < 100; i++)
    {
        sc.stop();
        try {
            sc.setSource(ncvslideio::gin(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path), m));
        } catch(...) {
            throw SkipTestException("Video file can not be opened");
        }
        sc.start();
        ncvslideio::Mat out;
        for (int j = 0; j < 5; j++) EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));
    }
}

TEST_F(GAPI_Streaming_Unit, PullNoStart)
{
    sc.setSource(ncvslideio::gin(m, m));

    ncvslideio::Mat out;
    EXPECT_ANY_THROW(sc.pull(ncvslideio::gout(out)));
}


TEST_F(GAPI_Streaming_Unit, SetSource_Multi_BeforeStart)
{
    ncvslideio::Mat eye = ncvslideio::Mat::eye  (224, 224, CV_8UC3);
    ncvslideio::Mat zrs = ncvslideio::Mat::zeros(224, 224, CV_8UC3);

    // Call setSource two times, data specified last time
    // should be actually processed.
    sc.setSource(ncvslideio::gin(zrs, zrs));
    sc.setSource(ncvslideio::gin(eye, eye));

    // Run the pipeline, acquire result once
    sc.start();
    ncvslideio::Mat out, out_ref;
    EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));
    sc.stop();

    // Pipeline should process `eye` mat, not `zrs`
    ref(ncvslideio::gin(eye, eye), ncvslideio::gout(out_ref));
    EXPECT_EQ(0., ncvslideio::norm(out, out_ref, ncvslideio::NORM_INF));
}

TEST_F(GAPI_Streaming_Unit, SetSource_During_Execution)
{
    ncvslideio::Mat zrs = ncvslideio::Mat::zeros(224, 224, CV_8UC3);

    sc.setSource(ncvslideio::gin(m, m));
    sc.start();
    EXPECT_ANY_THROW(sc.setSource(ncvslideio::gin(zrs, zrs)));
    EXPECT_ANY_THROW(sc.setSource(ncvslideio::gin(zrs, zrs)));
    EXPECT_ANY_THROW(sc.setSource(ncvslideio::gin(zrs, zrs)));
    sc.stop();
}

TEST_F(GAPI_Streaming_Unit, SetSource_After_Completion)
{
    sc.setSource(ncvslideio::gin(m, m));

    // Test pipeline with `m` input
    sc.start();
    ncvslideio::Mat out, out_ref;
    EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));
    sc.stop();

    // Test against ref
    ref(ncvslideio::gin(m, m), ncvslideio::gout(out_ref));
    EXPECT_EQ(0., ncvslideio::norm(out, out_ref, ncvslideio::NORM_INF));

    // Now set another source
    ncvslideio::Mat eye = ncvslideio::Mat::eye(224, 224, CV_8UC3);
    sc.setSource(ncvslideio::gin(eye, m));
    sc.start();
    EXPECT_TRUE(sc.pull(ncvslideio::gout(out)));
    sc.stop();

    // Test against new ref
    ref(ncvslideio::gin(eye, m), ncvslideio::gout(out_ref));
    EXPECT_EQ(0., ncvslideio::norm(out, out_ref, ncvslideio::NORM_INF));
}

// NB: Check pull overload for python
TEST(Streaming, Python_Pull_Overload)
{
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::copy(in);
    ncvslideio::GComputation c(in, out);

    ncvslideio::Size sz(3,3);
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar(255));

    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(ncvslideio::gin(in_mat));

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    bool has_output;
    ncvslideio::GRunArgs outputs;
    using RunArgs = ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>;
    RunArgs args;

    std::tie(has_output, args) = ccomp.pull();

    checkPullOverload(in_mat, has_output, args);

    ccomp.stop();
    EXPECT_FALSE(ccomp.running());
}

TEST(GAPI_Streaming_Desync, Python_Pull_Overload)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out = ncvslideio::gapi::streaming::desync(in);
    ncvslideio::GComputation c(in, out);

    ncvslideio::Size sz(3,3);
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar(255));

    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ccomp.setSource(ncvslideio::gin(in_mat));

    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    bool has_output;
    ncvslideio::GRunArgs outputs;
    using RunArgs = ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>;
    RunArgs args;

    std::tie(has_output, args) = ccomp.pull();

    checkPullOverload(in_mat, has_output, args);

    ccomp.stop();
    EXPECT_FALSE(ccomp.running());
}

TEST(GAPI_Streaming_Desync, SmokeTest_Regular)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));
    ncvslideio::GMat out1 = ncvslideio::gapi::Canny(tmp1, 32, 128, 3);

    // FIXME: Unary desync should not require tie!
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out2 = tmp2 / ncvslideio::gapi::Sobel(tmp2, CV_8U, 1, 1);;

    ncvslideio::Mat test_in = ncvslideio::Mat::eye(ncvslideio::Size(32,32), CV_8UC3);
    ncvslideio::Mat test_out1, test_out2;
    ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
        .apply(ncvslideio::gin(test_in), ncvslideio::gout(test_out1, test_out2));
}

TEST(GAPI_Streaming_Desync, SmokeTest_Streaming)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));
    ncvslideio::GMat out1 = ncvslideio::gapi::Canny(tmp1, 32, 128, 3);

    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out2 = Delay::on(tmp2,10) / ncvslideio::gapi::Sobel(tmp2, CV_8U, 1, 1);

    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<OCVDelay>()));
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    std::size_t out1_hits = 0u;
    std::size_t out2_hits = 0u;
    ncvslideio::optional<ncvslideio::Mat> test_out1, test_out2;
    while (sc.pull(ncvslideio::gout(test_out1, test_out2))) {
        GAPI_Assert(test_out1 || test_out2);
        if (test_out1) out1_hits++;
        if (test_out2) out2_hits++;
    }
    EXPECT_EQ(100u, out1_hits);      // out1 must be available for all frames
    EXPECT_LE(out2_hits, out1_hits); // out2 must appear less times than out1
}

TEST(GAPI_Streaming_Desync, SmokeTest_Streaming_TwoParts)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));
    ncvslideio::GMat out1 = ncvslideio::gapi::Canny(tmp1, 32, 128, 3);

    // Desynchronized path 1
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out2 = tmp2 / ncvslideio::gapi::Sobel(tmp2, CV_8U, 1, 1);

    // Desynchronized path 2
    ncvslideio::GMat tmp3 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out3 = 0.5*tmp3 +  0.5*ncvslideio::gapi::medianBlur(tmp3, 7);

    // The code should compile and execute well (desynchronized parts don't cross)
    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2, out3))
        .compileStreaming();
    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    std::size_t test_frames = 0u;
    ncvslideio::optional<ncvslideio::Mat> test_out1, test_out2, test_out3;
    while (sc.pull(ncvslideio::gout(test_out1, test_out2, test_out3))) {
        GAPI_Assert(test_out1 || test_out2 || test_out3);
        if (test_out1) {
            // count frames only for synchronized output
            test_frames++;
        }
    }
    EXPECT_EQ(100u, test_frames);
}

TEST(GAPI_Streaming_Desync, Negative_NestedDesync_Tier0)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    // Desynchronized path 1
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out1 = ncvslideio::gapi::medianBlur(tmp2, 3);

    // Desynchronized path 2, nested from 1 (directly from desync)
    ncvslideio::GMat tmp3 = ncvslideio::gapi::streaming::desync(tmp2);
    ncvslideio::GMat out2 = 0.5*tmp3;

    // This shouldn't compile
    EXPECT_ANY_THROW(ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
                     .compileStreaming());
}

TEST(GAPI_Streaming_Desync, Negative_NestedDesync_Tier1)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    // Desynchronized path 1
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out1 = ncvslideio::gapi::medianBlur(tmp2, 3);

    // Desynchronized path 2, nested from 1 (indirectly from desync)
    ncvslideio::GMat tmp3 = ncvslideio::gapi::streaming::desync(out1);
    ncvslideio::GMat out2 = 0.5*tmp3;

    // This shouldn't compile
    EXPECT_ANY_THROW(ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
                     .compileStreaming());
}

TEST(GAPI_Streaming_Desync, Negative_CrossMainPart_Tier0)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    // Desynchronized path: depends on both tmp1 and tmp2
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out1 = 0.5*tmp1 + 0.5*tmp2;

    // This shouldn't compile
    EXPECT_ANY_THROW(ncvslideio::GComputation(in, out1).compileStreaming());
}

TEST(GAPI_Streaming_Desync, Negative_CrossMainPart_Tier1)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    // Desynchronized path: depends on both tmp1 and tmp2
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out1 = 0.5*tmp1 + 0.5*ncvslideio::gapi::medianBlur(tmp2, 3);

    // This shouldn't compile
    EXPECT_ANY_THROW(ncvslideio::GComputation(in, out1).compileStreaming());
}

TEST(GAPI_Streaming_Desync, Negative_CrossOtherDesync_Tier0)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    // Desynchronized path 1
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out1 = 0.5*tmp2;

    // Desynchronized path 2 (depends on 1)
    ncvslideio::GMat tmp3 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out2 = 0.5*tmp3 + tmp2;

    // This shouldn't compile
    EXPECT_ANY_THROW(ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
                     .compileStreaming());
}

TEST(GAPI_Streaming_Desync, Negative_CrossOtherDesync_Tier1)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    // Desynchronized path 1
    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out1 = 0.5*tmp2;

    // Desynchronized path 2 (depends on 1)
    ncvslideio::GMat tmp3 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat out2 = 0.5*ncvslideio::gapi::medianBlur(tmp3,3) + 1.0*tmp2;

    // This shouldn't compile
    EXPECT_ANY_THROW(ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
                     .compileStreaming());
}

TEST(GAPI_Streaming_Desync, Negative_SynchronizedPull)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    ncvslideio::GMat tmp1 = ncvslideio::gapi::streaming::desync(out1);
    ncvslideio::GMat out2 = 0.5*tmp1;

    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
        .compileStreaming();

    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    ncvslideio::Mat o1, o2;
    EXPECT_ANY_THROW(sc.pull(ncvslideio::gout(o1, o2)));
}

TEST(GAPI_Streaming_Desync, UseSpecialPull)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out1 = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    ncvslideio::GMat tmp1 = ncvslideio::gapi::streaming::desync(out1);
    ncvslideio::GMat out2 = 0.5*tmp1;

    auto sc = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2))
        .compileStreaming();

    auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        sc.setSource(ncvslideio::gin(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path)));
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    sc.start();

    ncvslideio::optional<ncvslideio::Mat> o1, o2;
    std::size_t num_frames = 0u;

    while (sc.pull(ncvslideio::gout(o1, o2))) {
        if (o1) num_frames++;
    }
    EXPECT_EQ(100u, num_frames);
}

G_API_OP(ProduceVector, <ncvslideio::GArray<int>(ncvslideio::GMat)>, "test.desync.vector") {
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc &) {
        return ncvslideio::empty_array_desc();
    }
};

G_API_OP(ProduceOpaque, <ncvslideio::GOpaque<int>(ncvslideio::GMat)>, "test.desync.opaque") {
    static ncvslideio::GOpaqueDesc outMeta(const ncvslideio::GMatDesc &) {
        return ncvslideio::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL(OCVVector, ProduceVector) {
    static void run(const ncvslideio::Mat& in, std::vector<int> &out) {
        out = {in.cols, in.rows};
    }
};

GAPI_OCV_KERNEL(OCVOpaque, ProduceOpaque) {
    static void run(const ncvslideio::Mat &in, int &v) {
        v = in.channels();
    }
};

namespace {
ncvslideio::GStreamingCompiled desyncTestObject() {
    ncvslideio::GMat in;
    ncvslideio::GMat blur = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    ncvslideio::GMat blur_d = ncvslideio::gapi::copy(ncvslideio::gapi::streaming::desync(blur));
    ncvslideio::GMat d1 = Delay::on(blur_d, 10);
    ncvslideio::GMat d2 = Delay::on(blur_d, 30);

    ncvslideio::GArray<int>  vec = ProduceVector::on(d1);
    ncvslideio::GOpaque<int> opq = ProduceOpaque::on(d2);

    auto pkg = ncvslideio::gapi::kernels<OCVDelay, OCVVector, OCVOpaque>();
    return ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(blur, vec, opq))
        .compileStreaming(ncvslideio::compile_args(pkg));
}
} // anonymous namespace

TEST(GAPI_Streaming_Desync, MultipleDesyncOutputs_1) {
    auto sc = desyncTestObject();
    const ncvslideio::Mat in_mat = ncvslideio::Mat::eye(ncvslideio::Size(320,240), CV_8UC3);

    sc.setSource(ncvslideio::gin(in_mat));
    sc.start();

    ncvslideio::optional<ncvslideio::Mat> out_mat;
    ncvslideio::optional<std::vector<int> > out_vec;
    ncvslideio::optional<int> out_int;

    int counter = 0;
    while (sc.pull(ncvslideio::gout(out_mat, out_vec, out_int))) {
        if (counter++ == 1000) {
            // Stop the test after 1000 iterations
            sc.stop();
            break;
        }
        GAPI_Assert(out_mat || out_vec || out_int);

        // out_vec and out_int are on the same desynchronized path
        // they MUST arrive together. If one is available, the other
        // also must be available.
        if (out_vec) { ASSERT_TRUE(out_int.has_value()); }
        if (out_int) { ASSERT_TRUE(out_vec.has_value()); }

        if (out_vec || out_int) {
            EXPECT_EQ(320, out_vec.value()[0]);
            EXPECT_EQ(240, out_vec.value()[1]);
            EXPECT_EQ(3, out_int.value());
        }
    }
}

TEST(GAPI_Streaming_Desync, StartStop_Stress) {
    auto sc = desyncTestObject();
    const ncvslideio::Mat in_mat = ncvslideio::Mat::eye(ncvslideio::Size(320,240), CV_8UC3);

    ncvslideio::optional<ncvslideio::Mat> out_mat;
    ncvslideio::optional<std::vector<int> > out_vec;
    ncvslideio::optional<int> out_int;

    for (int i = 0; i < 10; i++) {
        sc.setSource(ncvslideio::gin(in_mat));
        sc.start();
        int counter = 0;
        while (counter++ < 100) {
            sc.pull(ncvslideio::gout(out_mat, out_vec, out_int));
            GAPI_Assert(out_mat || out_vec || out_int);
            if (out_vec) { ASSERT_TRUE(out_int.has_value()); }
            if (out_int) { ASSERT_TRUE(out_vec.has_value()); }
        }
        sc.stop();
    }
}

TEST(GAPI_Streaming_Desync, DesyncObjectConsumedByTwoIslandsViaSeparateDesync) {
    // See comment in the implementation of ncvslideio::gapi::streaming::desync (.cpp)
    ncvslideio::GMat in;
    ncvslideio::GMat tmp = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    ncvslideio::GMat tmp1 = ncvslideio::gapi::streaming::desync(tmp);
    ncvslideio::GMat out1 = ncvslideio::gapi::copy(tmp1); // ran via Streaming backend

    ncvslideio::GMat tmp2 = ncvslideio::gapi::streaming::desync(tmp);
    ncvslideio::GMat out2 = tmp2 * 0.5;           // ran via OCV backend

    auto c = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    EXPECT_NO_THROW(c.compileStreaming());
}

TEST(GAPI_Streaming_Desync, DesyncObjectConsumedByTwoIslandsViaSameDesync) {
    // See comment in the implementation of ncvslideio::gapi::streaming::desync (.cpp)
    ncvslideio::GMat in;
    ncvslideio::GMat tmp = ncvslideio::gapi::boxFilter(in, -1, ncvslideio::Size(3,3));

    ncvslideio::GMat tmp1 = ncvslideio::gapi::streaming::desync(tmp);
    ncvslideio::GMat out1 = ncvslideio::gapi::copy(tmp1); // ran via Streaming backend
    ncvslideio::GMat out2 = out1 - 0.5*tmp1;      // ran via OCV backend

    auto c = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    EXPECT_NO_THROW(c.compileStreaming());
}

TEST(GAPI_Streaming, CopyFrame)
{
    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::copy(in);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compileStreaming();
    try {
        cc.setSource<BGRSource>(filepath);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::MediaFrame frame;
    ncvslideio::Mat ocv_mat;
    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (cc.pull(ncvslideio::gout(frame)) && num_frames < max_frames)
    {
        auto view = frame.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat gapi_mat(frame.desc().size, CV_8UC3, view.ptr[0]);
        num_frames++;
        cap >> ocv_mat;

        EXPECT_EQ(0, cvtest::norm(ocv_mat, gapi_mat, NORM_INF));
    }
}

TEST(GAPI_Streaming, CopyFrameGray)
{
    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::copy(in);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compileStreaming();
    try {
        cc.setSource<GRAYSource>(filepath);
    }
    catch (...) {
        throw SkipTestException("Video file can not be opened");
    }

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::MediaFrame frame;
    ncvslideio::Mat ocv_mat;
    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (cc.pull(ncvslideio::gout(frame)) && num_frames < max_frames)
    {
        auto view = frame.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat gapi_mat(frame.desc().size, CV_8UC1, view.ptr[0]);
        num_frames++;
        cap >> ocv_mat;
        ncvslideio::Mat gray;
        cvtColor(ocv_mat, gray, ncvslideio::COLOR_BGR2GRAY);
        EXPECT_EQ(0, cvtest::norm(gray, gapi_mat, NORM_INF));
    }
}

TEST(GAPI_Streaming, CopyMat)
{
    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::copy(in);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compileStreaming();
    try {
        cc.setSource<ncvslideio::gapi::wip::GCaptureSource>(filepath);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::Mat out_mat;
    ncvslideio::Mat ocv_mat;
    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (cc.pull(ncvslideio::gout(out_mat)) && num_frames < max_frames)
    {
        num_frames++;
        cap >> ocv_mat;

        EXPECT_EQ(0, cvtest::norm(ocv_mat, out_mat, NORM_INF));
    }
}

TEST(GAPI_Streaming, Reshape)
{
    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::copy(in);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compileStreaming();
    try {
        cc.setSource<BGRSource>(filepath);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::MediaFrame frame;
    ncvslideio::Mat ocv_mat;
    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (cc.pull(ncvslideio::gout(frame)) && num_frames < max_frames)
    {
        auto view = frame.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat gapi_mat(frame.desc().size, CV_8UC3, view.ptr[0]);
        num_frames++;
        cap >> ocv_mat;

        EXPECT_EQ(0, cvtest::norm(ocv_mat, gapi_mat, NORM_INF));
    }

    // Reshape the graph meta
    filepath = findDataFile("ncvslideio/video/1920x1080.avi");
    cc.stop();
    try {
        cc.setSource<BGRSource>(filepath);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::MediaFrame frame2;
    ncvslideio::Mat ocv_mat2;

    num_frames = 0u;

    cc.start();
    while (cc.pull(ncvslideio::gout(frame2)) && num_frames < max_frames)
    {
        auto view = frame2.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat gapi_mat(frame2.desc().size, CV_8UC3, view.ptr[0]);
        num_frames++;
        cap >> ocv_mat2;

        EXPECT_EQ(0, cvtest::norm(ocv_mat2, gapi_mat, NORM_INF));
    }
}

TEST(GAPI_Streaming, ReshapeGray)
{
    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");

    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::copy(in);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compileStreaming();
    try {
        cc.setSource<GRAYSource>(filepath);
    }
    catch (...) {
        throw SkipTestException("Video file can not be opened");
    }

    ncvslideio::VideoCapture cap;
    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::MediaFrame frame;
    ncvslideio::Mat ocv_mat;
    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (cc.pull(ncvslideio::gout(frame)) && num_frames < max_frames)
    {
        auto view = frame.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat gapi_mat(frame.desc().size, CV_8UC1, view.ptr[0]);
        num_frames++;
        cap >> ocv_mat;
        ncvslideio::Mat gray;
        cvtColor(ocv_mat, gray, ncvslideio::COLOR_BGR2GRAY);
        EXPECT_EQ(0, cvtest::norm(gray, gapi_mat, NORM_INF));
    }

    // Reshape the graph meta
    filepath = findDataFile("ncvslideio/video/1920x1080.avi");
    cc.stop();
    try {
        cc.setSource<GRAYSource>(filepath);
    }
    catch (...) {
        throw SkipTestException("Video file can not be opened");
    }

    cap.open(filepath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::MediaFrame frame2;
    ncvslideio::Mat ocv_mat2;

    num_frames = 0u;

    cc.start();
    while (cc.pull(ncvslideio::gout(frame2)) && num_frames < max_frames)
    {
        auto view = frame2.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat gapi_mat(frame2.desc().size, CV_8UC1, view.ptr[0]);
        num_frames++;
        cap >> ocv_mat2;
        ncvslideio::Mat gray;
        cvtColor(ocv_mat2, gray, ncvslideio::COLOR_BGR2GRAY);
        EXPECT_EQ(0, cvtest::norm(gray, gapi_mat, NORM_INF));
    }
}


namespace {
    enum class TestSourceType {
        BGR,
        NV12,
        GRAY
    };
    std::ostream& operator<<(std::ostream& os, TestSourceType a) {
        os << "Source:";
        switch (a) {
            case TestSourceType::BGR:  return os << "BGR";
            case TestSourceType::NV12: return os << "NV12";
            case TestSourceType::GRAY: return os << "GRAY";
            default: CV_Assert(false && "unknown TestSourceType");
        }
    }

    ncvslideio::gapi::wip::IStreamSource::Ptr createTestSource(TestSourceType sourceType,
                                                       const std::string& pipeline) {
        assert(sourceType == TestSourceType::BGR || sourceType == TestSourceType::NV12 || sourceType == TestSourceType::GRAY);

        ncvslideio::gapi::wip::IStreamSource::Ptr ptr { };

        switch (sourceType) {
            case TestSourceType::BGR: {
                try {
                    ptr = ncvslideio::gapi::wip::make_src<BGRSource>(pipeline);
                }
                catch(...) {
                    throw SkipTestException(std::string("BGRSource for '") + pipeline +
                                            "' couldn't be created!");
                }
                break;
            }
            case TestSourceType::NV12: {
                try {
                    ptr = ncvslideio::gapi::wip::make_src<NV12Source>(pipeline);
                }
                catch(...) {
                    throw SkipTestException(std::string("NV12Source for '") + pipeline +
                                            "' couldn't be created!");
                }
                break;
            }
            case TestSourceType::GRAY: {
                try {
                    ptr = ncvslideio::gapi::wip::make_src<GRAYSource>(pipeline);
                }
                catch (...) {
                    throw SkipTestException(std::string("GRAYSource for '") + pipeline +
                        "' couldn't be created!");
                }
                break;
            }
            default: {
                throw SkipTestException("Incorrect type of source! "
                                        "Something went wrong in the test!");
            }
        }

        return ptr;
    }

    enum class TestAccessType {
        BGR,
        Y,
        UV
    };
    std::ostream& operator<<(std::ostream& os, TestAccessType a) {
        os << "Accessor:";
        switch (a) {
            case TestAccessType::BGR: return os << "BGR";
            case TestAccessType::Y:   return os << "Y";
            case TestAccessType::UV:  return os << "UV";
            default: CV_Assert(false && "unknown TestAccessType");
        }
    }

    using GapiFunction = std::function<ncvslideio::GMat(const ncvslideio::GFrame&)>;
    static std::map<TestAccessType, GapiFunction> gapi_functions = {
        { TestAccessType::BGR, ncvslideio::gapi::streaming::BGR },
        { TestAccessType::Y,   ncvslideio::gapi::streaming::Y   },
        { TestAccessType::UV,  ncvslideio::gapi::streaming::UV  }
    };

    using RefFunction = std::function<ncvslideio::Mat(const ncvslideio::Mat&)>;
    static std::map<std::pair<TestSourceType,TestAccessType>, RefFunction> ref_functions = {
        { std::make_pair(TestSourceType::BGR, TestAccessType::BGR),
          [](const ncvslideio::Mat& bgr) { return bgr; } },
        { std::make_pair(TestSourceType::BGR, TestAccessType::Y),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat y, uv;
              cvtBGR2NV12(bgr, y, uv);
              return y;
          } },
        { std::make_pair(TestSourceType::BGR, TestAccessType::UV),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat y, uv;
              cvtBGR2NV12(bgr, y, uv);
              return uv;
          } },
        { std::make_pair(TestSourceType::NV12, TestAccessType::BGR),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat y, uv, out_bgr;
              cvtBGR2NV12(bgr, y, uv);
              ncvslideio::cvtColorTwoPlane(y, uv, out_bgr,
                                   ncvslideio::COLOR_YUV2BGR_NV12);
              return out_bgr;
          } },
        { std::make_pair(TestSourceType::NV12, TestAccessType::Y),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat y, uv;
              cvtBGR2NV12(bgr, y, uv);
              return y;
          } },
        { std::make_pair(TestSourceType::NV12, TestAccessType::UV),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat y, uv;
              cvtBGR2NV12(bgr, y, uv);
              return uv;
          } },
        { std::make_pair(TestSourceType::GRAY, TestAccessType::BGR),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat gray;
              ncvslideio::cvtColor(bgr, gray, ncvslideio::COLOR_BGR2GRAY);
              ncvslideio::Mat out_bgr;
              ncvslideio::cvtColor(gray, out_bgr, ncvslideio::COLOR_GRAY2BGR);
              return out_bgr;
          } },
        { std::make_pair(TestSourceType::GRAY, TestAccessType::Y),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat gray;
              ncvslideio::cvtColor(bgr, gray, ncvslideio::COLOR_BGR2GRAY);
              return gray;
          } },
        { std::make_pair(TestSourceType::GRAY, TestAccessType::UV),
          [](const ncvslideio::Mat& bgr) {
              ncvslideio::Mat uv(bgr.size() / 2, CV_8UC2, ncvslideio::Scalar::all(127));
              return uv;
          } },
    };
} // anonymous namespace

struct GAPI_Accessors_In_Streaming : public TestWithParam<
    std::tuple<std::string,TestSourceType,TestAccessType>>
{ };


TEST_P(GAPI_Accessors_In_Streaming, AccuracyTest)
{
    std::string filepath{};
    TestSourceType sourceType = TestSourceType::BGR;
    TestAccessType accessType = TestAccessType::BGR;
    std::tie(filepath, sourceType, accessType) = GetParam();
    auto accessor = gapi_functions[accessType];
    auto fromBGR = ref_functions[std::make_pair(sourceType, accessType)];

    const std::string& absFilePath = findDataFile(filepath);

    ncvslideio::GFrame in;
    ncvslideio::GMat out = accessor(in);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compileStreaming();
    auto src = createTestSource(sourceType, absFilePath);
    cc.setSource(src);

    ncvslideio::VideoCapture cap;
    cap.open(absFilePath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::Mat cap_mat, ocv_mat, gapi_mat;
    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (num_frames < max_frames && cc.pull(ncvslideio::gout(gapi_mat)))
    {
        num_frames++;
        cap >> cap_mat;
        ocv_mat = fromBGR(cap_mat);

        EXPECT_EQ(0, cvtest::norm(ocv_mat, gapi_mat, NORM_INF));
    }

    cc.stop();
}

INSTANTIATE_TEST_CASE_P(TestAccessor, GAPI_Accessors_In_Streaming,
                        Combine(Values("ncvslideio/video/768x576.avi"),
                                Values(TestSourceType::BGR, TestSourceType::NV12, TestSourceType::GRAY),
                                Values(TestAccessType::BGR, TestAccessType::Y, TestAccessType::UV)
                        ));


struct GAPI_Accessors_Meta_In_Streaming : public TestWithParam<
    std::tuple<std::string,TestSourceType,TestAccessType>>
{ };

TEST_P(GAPI_Accessors_Meta_In_Streaming, AccuracyTest)
{
    std::string filepath{};
    TestSourceType sourceType = TestSourceType::BGR;
    TestAccessType accessType = TestAccessType::BGR;
    std::tie(filepath, sourceType, accessType) = GetParam();
    auto accessor = gapi_functions[accessType];
    auto fromBGR = ref_functions[std::make_pair(sourceType, accessType)];

    const std::string& absFilePath = findDataFile(filepath);

    ncvslideio::GFrame in;
    ncvslideio::GMat gmat = accessor(in);
    ncvslideio::GMat resized = ncvslideio::gapi::resize(gmat, ncvslideio::Size(1920, 1080));
    ncvslideio::GOpaque<int64_t> outId = ncvslideio::gapi::streaming::seq_id(resized);
    ncvslideio::GOpaque<int64_t> outTs = ncvslideio::gapi::streaming::timestamp(resized);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(resized, outId, outTs));

    auto cc = comp.compileStreaming();
    auto src = createTestSource(sourceType, absFilePath);
    cc.setSource(src);

    ncvslideio::VideoCapture cap;
    cap.open(absFilePath);
    if (!cap.isOpened())
        throw SkipTestException("Video file can not be opened");

    ncvslideio::Mat cap_mat, req_mat, ocv_mat, gapi_mat;
    int64_t seq_id = 0, timestamp = 0;
    std::set<int64_t> all_seq_ids;
    std::vector<int64_t> all_timestamps;

    std::size_t num_frames = 0u;
    std::size_t max_frames = 10u;

    cc.start();
    while (num_frames < max_frames && cc.pull(ncvslideio::gout(gapi_mat, seq_id, timestamp)))
    {
        num_frames++;

        cap >> cap_mat;
        req_mat = fromBGR(cap_mat);
        ncvslideio::resize(req_mat, ocv_mat, ncvslideio::Size(1920, 1080));
        EXPECT_EQ(0, cvtest::norm(ocv_mat, gapi_mat, NORM_INF));

        all_seq_ids.insert(seq_id);
        all_timestamps.push_back(timestamp);
    }

    cc.stop();

    EXPECT_EQ(all_seq_ids.begin(), all_seq_ids.find(0L));
    auto last_elem_it = --all_seq_ids.end();
    EXPECT_EQ(last_elem_it, all_seq_ids.find(int64_t(max_frames - 1L)));
    EXPECT_EQ(max_frames, all_seq_ids.size());

    EXPECT_EQ(max_frames, all_timestamps.size());
    EXPECT_TRUE(std::is_sorted(all_timestamps.begin(), all_timestamps.end()));
}

INSTANTIATE_TEST_CASE_P(AccessorMeta, GAPI_Accessors_Meta_In_Streaming,
                        Combine(Values("ncvslideio/video/768x576.avi"),
                                Values(TestSourceType::BGR, TestSourceType::NV12, TestSourceType::GRAY),
                                Values(TestAccessType::BGR, TestAccessType::Y, TestAccessType::UV)
                        ));

TEST(GAPI_Streaming, TestPythonAPI)
{
    ncvslideio::Size sz(200, 200);
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar(255));
    const auto crop_rc = ncvslideio::Rect(13, 75, 100, 100);

    // OpenCV reference image
    ncvslideio::Mat ocv_mat;
    {
        ocv_mat = in_mat(crop_rc);
    }

    ncvslideio::GMat in;
    auto roi = ncvslideio::gapi::crop(in, crop_rc);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(roi));

    // NB: Used by python bridge
    auto cc = comp.compileStreaming(ncvslideio::detail::ExtractMetaCallback{[&](const ncvslideio::GTypesInfo& info)
            {
                GAPI_Assert(info.size() == 1u);
                GAPI_Assert(info[0].shape == ncvslideio::GShape::GMAT);
                return ncvslideio::GMetaArgs{ncvslideio::GMetaArg{ncvslideio::descr_of(in_mat)}};
            }});

    // NB: Used by python bridge
    cc.setSource(ncvslideio::detail::ExtractArgsCallback{[&](const ncvslideio::GTypesInfo& info)
            {
                GAPI_Assert(info.size() == 1u);
                GAPI_Assert(info[0].shape == ncvslideio::GShape::GMAT);
                return ncvslideio::GRunArgs{in_mat};
            }});

    cc.start();

    bool is_over = false;
    ncvslideio::GRunArgs out_args;
    using RunArgs = ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>;
    RunArgs args;

    // NB: Used by python bridge
    std::tie(is_over, args) = cc.pull();

    switch (args.index()) {
        case RunArgs::index_of<ncvslideio::GRunArgs>():
            out_args = util::get<ncvslideio::GRunArgs>(args); break;
        default: GAPI_Error("Incorrect type of return value");
    }

    ASSERT_EQ(1u, out_args.size());
    ASSERT_TRUE(ncvslideio::util::holds_alternative<ncvslideio::Mat>(out_args[0]));

    EXPECT_EQ(0, cvtest::norm(ocv_mat, ncvslideio::util::get<ncvslideio::Mat>(out_args[0]), NORM_INF));
    EXPECT_TRUE(is_over);

    cc.stop();
}

#ifdef HAVE_ONEVPL

TEST(OneVPL_Source, Init)
{
    using CfgParam = ncvslideio::gapi::wip::onevpl::CfgParam;

    std::vector<CfgParam> src_params;
    src_params.push_back(CfgParam::create_implementation(MFX_IMPL_TYPE_HARDWARE));
#ifdef _WIN32
    src_params.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_D3D11));
#elif defined(__linux__)
    src_params.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_VAAPI));
#endif
    src_params.push_back(CfgParam::create_decoder_id(MFX_CODEC_HEVC));
    std::stringstream stream(std::ios_base::in | std::ios_base::out | std::ios_base::binary);

    EXPECT_TRUE(stream.write(reinterpret_cast<char*>(const_cast<unsigned char *>(streaming::onevpl::hevc_header)),
                             sizeof(streaming::onevpl::hevc_header)));
    std::shared_ptr<ncvslideio::gapi::wip::onevpl::IDataProvider> stream_data_provider =
                std::make_shared<streaming::onevpl::StreamDataProvider>(stream);

    ncvslideio::Ptr<ncvslideio::gapi::wip::IStreamSource> cap;
    bool cap_created = false;
    try {
        cap = ncvslideio::gapi::wip::make_onevpl_src(stream_data_provider, src_params);
        cap_created = true;
    } catch (const std::exception&) {
    }
    ASSERT_TRUE(cap_created);

    ncvslideio::gapi::wip::Data out;
    while (cap->pull(out)) {
        (void)out;
    }
    EXPECT_TRUE(stream_data_provider->empty());
}
#endif // HAVE_ONEVPL

TEST(GAPI_Streaming, TestDesyncRMat) {
    ncvslideio::GMat in;
    auto blurred = ncvslideio::gapi::blur(in, ncvslideio::Size{3,3});
    auto desynced = ncvslideio::gapi::streaming::desync(blurred);
    auto out = in - blurred;
    auto pipe = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(desynced, out)).compileStreaming();

    ncvslideio::Size sz(32,32);
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar(255));
    pipe.setSource(ncvslideio::gin(in_mat));
    pipe.start();

    ncvslideio::optional<ncvslideio::RMat> out_desync;
    ncvslideio::optional<ncvslideio::RMat> out_rmat;
    while (true) {
        // Initially it threw "bad variant access" since there was
        // no RMat handling in wrap_opt_arg
        EXPECT_NO_THROW(pipe.pull(ncvslideio::gout(out_desync, out_rmat)));
        if (out_rmat) break;
    }
}

G_API_OP(GTestBlur, <GFrame(GFrame)>, "test.blur") {
    static GFrameDesc outMeta(GFrameDesc d) { return d; }
};
GAPI_OCV_KERNEL(GOcvTestBlur, GTestBlur) {
    static void run(const ncvslideio::MediaFrame& in, ncvslideio::MediaFrame& out) {
        auto d = in.desc();
        GAPI_Assert(d.fmt == ncvslideio::MediaFormat::BGR);
        auto view = in.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat mat(d.size, CV_8UC3, view.ptr[0]);
        ncvslideio::Mat blurred;
        ncvslideio::blur(mat, blurred, ncvslideio::Size{3,3});
        out = ncvslideio::MediaFrame::Create<TestMediaBGR>(blurred);
    }
};

TEST(GAPI_Streaming, TestDesyncMediaFrame) {
    ncvslideio::GFrame in;
    auto blurred = GTestBlur::on(in);
    auto desynced = ncvslideio::gapi::streaming::desync(blurred);
    auto out = GTestBlur::on(blurred);
    auto pipe = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(desynced, out))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GOcvTestBlur>()));

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");
    try {
        pipe.setSource<BGRSource>(filepath);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    pipe.start();

    ncvslideio::optional<ncvslideio::MediaFrame> out_desync;
    ncvslideio::optional<ncvslideio::MediaFrame> out_frame;
    while (true) {
        // Initially it threw "bad variant access" since there was
        // no MediaFrame handling in wrap_opt_arg
        EXPECT_NO_THROW(pipe.pull(ncvslideio::gout(out_desync, out_frame)));
        if (out_frame) break;
    }
}

G_API_OP(GTestBlurGray, <GFrame(GFrame)>, "test.blur_gray") {
    static GFrameDesc outMeta(GFrameDesc d) { return d; }
};
GAPI_OCV_KERNEL(GOcvTestBlurGray, GTestBlurGray) {
    static void run(const ncvslideio::MediaFrame & in, ncvslideio::MediaFrame & out) {
        auto d = in.desc();
        GAPI_Assert(d.fmt == ncvslideio::MediaFormat::GRAY);
        auto view = in.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat mat(d.size, CV_8UC1, view.ptr[0]);
        ncvslideio::Mat blurred;
        ncvslideio::blur(mat, blurred, ncvslideio::Size{ 3,3 });
        out = ncvslideio::MediaFrame::Create<TestMediaGRAY>(blurred);
    }
};

TEST(GAPI_Streaming, TestDesyncMediaFrameGray) {
    ncvslideio::GFrame in;
    auto blurred = GTestBlurGray::on(in);
    auto desynced = ncvslideio::gapi::streaming::desync(blurred);
    auto out = GTestBlurGray::on(blurred);
    auto pipe = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(desynced, out))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GOcvTestBlurGray>()));

    std::string filepath = findDataFile("ncvslideio/video/768x576.avi");
    try {
        pipe.setSource<GRAYSource>(filepath);
    }
    catch (...) {
        throw SkipTestException("Video file can not be opened");
    }
    pipe.start();

    ncvslideio::optional<ncvslideio::MediaFrame> out_desync;
    ncvslideio::optional<ncvslideio::MediaFrame> out_frame;
    while (true) {
        // Initially it threw "bad variant access" since there was
        // no MediaFrame handling in wrap_opt_arg
        EXPECT_NO_THROW(pipe.pull(ncvslideio::gout(out_desync, out_frame)));
        if (out_frame) break;
    }
}

TEST(GAPI_Streaming_Exception, SingleKernelThrow) {
    ncvslideio::GMat in;
    auto pipeline = ncvslideio::GComputation(in, GThrowExceptionOp::on(in))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GThrowExceptionKernel>()));

    ncvslideio::Mat in_mat(ncvslideio::Size(300, 300), CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    pipeline.setSource(ncvslideio::gin(in_mat));
    pipeline.start();

    EXPECT_THROW(
            try {
                ncvslideio::Mat out_mat;
                pipeline.pull(ncvslideio::gout(out_mat));
            } catch (const std::logic_error& e) {
                EXPECT_EQ(GThrowExceptionKernel::exception_msg(), e.what());
                throw;
            }, std::logic_error);
}

TEST(GAPI_Streaming_Exception, StreamingBackendExceptionAsInput) {
    ncvslideio::GMat in;
    auto pipeline = ncvslideio::GComputation(in,
            ncvslideio::gapi::copy(GThrowExceptionOp::on(in)))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GThrowExceptionKernel>()));

    ncvslideio::Mat in_mat(ncvslideio::Size(300, 300), CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    pipeline.setSource(ncvslideio::gin(in_mat));
    pipeline.start();

    EXPECT_THROW(
            try {
                ncvslideio::Mat out_mat;
                pipeline.pull(ncvslideio::gout(out_mat));
            } catch (const std::logic_error& e) {
                EXPECT_EQ(GThrowExceptionKernel::exception_msg(), e.what());
                throw;
            }, std::logic_error);
}

TEST(GAPI_Streaming_Exception, RegularBacckendsExceptionAsInput) {
    ncvslideio::GMat in;
    auto pipeline = ncvslideio::GComputation(in,
            ncvslideio::gapi::add(GThrowExceptionOp::on(in), GThrowExceptionOp::on(in)))
        .compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GThrowExceptionKernel>()));

    ncvslideio::Mat in_mat(ncvslideio::Size(300, 300), CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    pipeline.setSource(ncvslideio::gin(in_mat));
    pipeline.start();

    EXPECT_THROW(
            try {
                ncvslideio::Mat out_mat;
                pipeline.pull(ncvslideio::gout(out_mat));
            } catch (const std::logic_error& e) {
                EXPECT_EQ(GThrowExceptionKernel::exception_msg(), e.what());
                throw;
            }, std::logic_error);
}

TEST(GAPI_Streaming_Exception, SourceThrow) {
    ncvslideio::GMat in;
    auto pipeline = ncvslideio::GComputation(in, ncvslideio::gapi::copy(in)).compileStreaming();

    pipeline.setSource(std::make_shared<InvalidSource>(1u, 1u));
    pipeline.start();

    EXPECT_THROW(
            try {
                ncvslideio::Mat out_mat;
                pipeline.pull(ncvslideio::gout(out_mat));
            } catch (const std::logic_error& e) {
                EXPECT_EQ(InvalidSource::exception_msg(), e.what());
                throw;
            }, std::logic_error);
}

TEST(GAPI_Streaming_Exception, SourceThrowEverySecondFrame) {
    constexpr size_t throw_every_nth_frame = 2u;
    constexpr size_t num_frames = 10u;
    size_t curr_frame = 0;
    bool has_frame = true;
    ncvslideio::Mat out_mat;

    ncvslideio::GMat in;
    auto pipeline = ncvslideio::GComputation(in, ncvslideio::gapi::copy(in)).compileStreaming();

    pipeline.setSource(std::make_shared<InvalidSource>(throw_every_nth_frame, num_frames));
    pipeline.start();
    while (has_frame) {
        ++curr_frame;
        try {
            has_frame = pipeline.pull(ncvslideio::gout(out_mat));
        } catch (const std::exception& e) {
            EXPECT_TRUE(curr_frame % throw_every_nth_frame == 0);
            EXPECT_EQ(InvalidSource::exception_msg(), e.what());
        }
    }

    // NB: Pull was called num_frames + 1(stop).
    EXPECT_EQ(num_frames, curr_frame - 1);
}

} // namespace opencv_test
