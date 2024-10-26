// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test/common/gapi_tests_common.hpp"

#include <opencv2/gapi/streaming/gstreamer/gstreamerpipeline.hpp>
#include <opencv2/gapi/streaming/gstreamer/gstreamersource.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcomputation.hpp>

#include <opencv2/ts.hpp>

#include <regex>

#ifdef HAVE_GSTREAMER

namespace opencv_test
{

struct GStreamerSourceTest : public TestWithParam<std::tuple<std::string, ncvslideio::Size, std::size_t>>
{ };


TEST_P(GStreamerSourceTest, AccuracyTest)
{
    std::string pipeline;
    ncvslideio::Size expectedFrameSize;
    std::size_t streamLength { };
    std::tie(pipeline, expectedFrameSize, streamLength) = GetParam();

    // Graph declaration:
    ncvslideio::GMat in;
    auto out = ncvslideio::gapi::copy(in);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));

    // Graph compilation for streaming mode:
    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // GStreamer streaming source configuration:
    ccomp.setSource<ncvslideio::gapi::wip::GStreamerSource>(pipeline);

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    ncvslideio::Mat in_mat_gapi;

    EXPECT_TRUE(ccomp.pull(ncvslideio::gout(in_mat_gapi)));
    EXPECT_TRUE(!in_mat_gapi.empty());
    EXPECT_EQ(expectedFrameSize, in_mat_gapi.size());
    EXPECT_EQ(CV_8UC3, in_mat_gapi.type());

    std::size_t framesCount = 1UL;
    while (ccomp.pull(ncvslideio::gout(in_mat_gapi))) {
        EXPECT_TRUE(!in_mat_gapi.empty());
        EXPECT_EQ(expectedFrameSize, in_mat_gapi.size());
        EXPECT_EQ(CV_8UC3, in_mat_gapi.type());

        framesCount++;
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());

    EXPECT_EQ(streamLength, framesCount);
}

TEST_P(GStreamerSourceTest, TimestampsTest)
{
    std::string pipeline;
    std::size_t streamLength { };
    std::tie(pipeline, std::ignore, streamLength) = GetParam();

    // Graph declaration:
    ncvslideio::GMat in;
    ncvslideio::GMat copied = ncvslideio::gapi::copy(in);
    ncvslideio::GOpaque<int64_t> outId = ncvslideio::gapi::streaming::seq_id(copied);
    ncvslideio::GOpaque<int64_t> outTs = ncvslideio::gapi::streaming::timestamp(copied);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(outId, outTs));

    // Graph compilation for streaming mode:
    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // GStreamer streaming source configuration:
    ccomp.setSource<ncvslideio::gapi::wip::GStreamerSource>(pipeline);

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    int64_t seqId;
    int64_t timestamp;

    std::vector<int64_t> allSeqIds;
    std::vector<int64_t> allTimestamps;

    while (ccomp.pull(ncvslideio::gout(seqId, timestamp))) {
        allSeqIds.push_back(seqId);
        allTimestamps.push_back(timestamp);
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());

    EXPECT_EQ(0L, allSeqIds.front());
    EXPECT_EQ(int64_t(streamLength) - 1, allSeqIds.back());
    EXPECT_EQ(streamLength, allSeqIds.size());
    EXPECT_TRUE(std::is_sorted(allSeqIds.begin(), allSeqIds.end()));
    EXPECT_EQ(allSeqIds.size(), std::set<int64_t>(allSeqIds.begin(), allSeqIds.end()).size());

    EXPECT_EQ(streamLength, allTimestamps.size());
    EXPECT_TRUE(std::is_sorted(allTimestamps.begin(), allTimestamps.end()));
}

G_TYPED_KERNEL(GGstFrameCopyToNV12, <std::tuple<ncvslideio::GMat,ncvslideio::GMat>(GFrame)>,
    "org.opencv.test.gstframe_copy_to_nv12")
{
    static std::tuple<GMatDesc, GMatDesc> outMeta(GFrameDesc desc) {
        GMatDesc y  { CV_8U, 1, desc.size, false };
        GMatDesc uv { CV_8U, 2, desc.size / 2, false };

        return std::make_tuple(y, uv);
    }
};

G_TYPED_KERNEL(GGstFrameCopyToGRAY8, <ncvslideio::GMat(GFrame)>,
    "org.opencv.test.gstframe_copy_to_gray8")
{
    static GMatDesc outMeta(GFrameDesc desc) {
        GMatDesc y{ CV_8U, 1, desc.size, false };
        return y;
    }
};


GAPI_OCV_KERNEL(GOCVGstFrameCopyToNV12, GGstFrameCopyToNV12)
{
    static void run(const ncvslideio::MediaFrame& in, ncvslideio::Mat& y, ncvslideio::Mat& uv)
    {
        auto view = in.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat ly(y.size(), y.type(), view.ptr[0], view.stride[0]);
        ncvslideio::Mat luv(uv.size(), uv.type(), view.ptr[1], view.stride[1]);

        ly.copyTo(y);
        luv.copyTo(uv);
    }
};

GAPI_OCV_KERNEL(GOCVGstFrameCopyToGRAY8, GGstFrameCopyToGRAY8)
{
    static void run(const ncvslideio::MediaFrame & in, ncvslideio::Mat & y)
    {
        auto view = in.access(ncvslideio::MediaFrame::Access::R);
        ncvslideio::Mat ly(y.size(), y.type(), view.ptr[0], view.stride[0]);
        ly.copyTo(y);
    }
};


TEST_P(GStreamerSourceTest, GFrameTest)
{
    std::string pipeline;
    ncvslideio::Size expectedFrameSize;
    std::size_t streamLength { };
    bool isNV12 = false;
    std::tie(pipeline, expectedFrameSize, streamLength) = GetParam();

    //Check if pipline string contains NV12 sub-string
    if (pipeline.find("NV12") != std::string::npos) {
        isNV12 = true;
    }

    // Graph declaration:
    ncvslideio::GFrame in;
    ncvslideio::GMat copiedY, copiedUV;
    if (isNV12) {
        std::tie(copiedY, copiedUV) = GGstFrameCopyToNV12::on(in);
    }
    else {
        copiedY = GGstFrameCopyToGRAY8::on(in);
    }

    ncvslideio::GComputation c(ncvslideio::GIn(in), isNV12 ? ncvslideio::GOut(copiedY, copiedUV) : ncvslideio::GOut(copiedY));

    // Graph compilation for streaming mode:
    ncvslideio::GStreamingCompiled ccomp;
    if (isNV12) {
        ccomp = c.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GOCVGstFrameCopyToNV12>()));
    } else {
        ccomp = c.compileStreaming(ncvslideio::compile_args(ncvslideio::gapi::kernels<GOCVGstFrameCopyToGRAY8>()));
    }


    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // GStreamer streaming source configuration:
    ccomp.setSource<ncvslideio::gapi::wip::GStreamerSource>
        (pipeline, ncvslideio::gapi::wip::GStreamerSource::OutputType::FRAME);

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    ncvslideio::Mat y_mat, uv_mat;

    EXPECT_TRUE(isNV12 ? ccomp.pull(ncvslideio::gout(y_mat, uv_mat)) : ccomp.pull(ncvslideio::gout(y_mat)));
    EXPECT_TRUE(!y_mat.empty());
    if (isNV12) {
        EXPECT_TRUE(!uv_mat.empty());
    }

    ncvslideio::Size expectedYSize = expectedFrameSize;
    ncvslideio::Size expectedUVSize = expectedFrameSize / 2;

    EXPECT_EQ(expectedYSize, y_mat.size());
    if (isNV12) {
        EXPECT_EQ(expectedUVSize, uv_mat.size());
    }

    EXPECT_EQ(CV_8UC1, y_mat.type());
    if (isNV12) {
        EXPECT_EQ(CV_8UC2, uv_mat.type());
    }

    std::size_t framesCount = 1UL;
    while (isNV12 ? ccomp.pull(ncvslideio::gout(y_mat, uv_mat)) : ccomp.pull(ncvslideio::gout(y_mat))) {
        EXPECT_TRUE(!y_mat.empty());
        if (isNV12) {
            EXPECT_TRUE(!uv_mat.empty());
        }

        EXPECT_EQ(expectedYSize, y_mat.size());
        if (isNV12) {
            EXPECT_EQ(expectedUVSize, uv_mat.size());
        }

        EXPECT_EQ(CV_8UC1, y_mat.type());
        if (isNV12) {
            EXPECT_EQ(CV_8UC2, uv_mat.type());
        }

        framesCount++;
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());

    EXPECT_EQ(streamLength, framesCount);
}


// FIXME: Need to launch with sudo. May be infrastructure problems.
// TODO: It is needed to add tests for streaming from native KMB camera: kmbcamsrc
//       GStreamer element.
INSTANTIATE_TEST_CASE_P(CameraEmulatingPipeline, GStreamerSourceTest,
                        Combine(Values("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=NV12,width=1920,height=1080,framerate=3/1 ! "
                                       "appsink",
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=GRAY8,width=1920,height=1080,framerate=3/1 ! "
                                       "appsink"),
                                Values(ncvslideio::Size(1920, 1080)),
                                Values(10UL)));


INSTANTIATE_TEST_CASE_P(FileEmulatingPipeline, GStreamerSourceTest,
                        Combine(Values("videotestsrc pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=NV12,width=640,height=420,framerate=3/1 ! "
                                       "appsink",
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=GRAY8,width=640,height=420,framerate=3/1 ! "
                                       "appsink"),
                                Values(ncvslideio::Size(640, 420)),
                                Values(10UL)));


INSTANTIATE_TEST_CASE_P(MultipleLiveSources, GStreamerSourceTest,
                        Combine(Values("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=NV12,width=1280,height=720 ! appsink "
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "fakesink",
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=GRAY8,width=1280,height=720 ! appsink "
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "fakesink"),
                                Values(ncvslideio::Size(1280, 720)),
                                Values(10UL)));


INSTANTIATE_TEST_CASE_P(MultipleNotLiveSources, GStreamerSourceTest,
                        Combine(Values("videotestsrc pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=NV12,width=1280,height=720 ! appsink "
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "fakesink",
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=GRAY8,width=1280,height=720 ! appsink "
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "fakesink"),
                                Values(ncvslideio::Size(1280, 720)),
                                Values(10UL)));


TEST(GStreamerMultiSourceSmokeTest, Test)
{
    // Graph declaration:
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::add(in1, in2);
    ncvslideio::GComputation c(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));

    // Graph compilation for streaming mode:
    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    ncvslideio::gapi::wip::GStreamerPipeline
        pipeline("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                 "videorate ! videoscale ! "
                 "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                 "appsink name=sink1 "
                 "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                 "videorate ! videoscale ! "
                 "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                 "appsink name=sink2");

    // GStreamer streaming sources configuration:
    auto src1 = pipeline.getStreamingSource("sink1");
    auto src2 = pipeline.getStreamingSource("sink2");

    ccomp.setSource(ncvslideio::gin(src1, src2));

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    ncvslideio::Mat in_mat_gapi;

    EXPECT_TRUE(ccomp.pull(ncvslideio::gout(in_mat_gapi)));
    EXPECT_TRUE(!in_mat_gapi.empty());
    EXPECT_EQ(CV_8UC3, in_mat_gapi.type());

    while (ccomp.pull(ncvslideio::gout(in_mat_gapi))) {
        EXPECT_TRUE(!in_mat_gapi.empty());
        EXPECT_EQ(CV_8UC3, in_mat_gapi.type());
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());
}

struct GStreamerMultiSourceTestNV12 :
    public TestWithParam<std::tuple<ncvslideio::GComputation, ncvslideio::gapi::wip::GStreamerSource::OutputType>>
{ };

TEST_P(GStreamerMultiSourceTestNV12, ImageDataTest)
{
    std::string pathToLeftIm = findDataFile("ncvslideio/stereomatching/datasets/tsukuba/im6.png");
    std::string pathToRightIm = findDataFile("ncvslideio/stereomatching/datasets/tsukuba/im2.png");

    std::string pipelineToReadImage("filesrc location=LOC ! pngdec ! videoconvert ! "
        "videoscale ! video/x-raw,format=NV12 ! appsink");

    ncvslideio::gapi::wip::GStreamerSource leftImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToLeftIm));
    ncvslideio::gapi::wip::GStreamerSource rightImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToRightIm));

    ncvslideio::gapi::wip::Data leftImData, rightImData;
    leftImageProvider.pull(leftImData);
    rightImageProvider.pull(rightImData);

    ncvslideio::Mat leftRefMat =  ncvslideio::util::get<ncvslideio::Mat>(leftImData);
    ncvslideio::Mat rightRefMat = ncvslideio::util::get<ncvslideio::Mat>(rightImData);

    // Retrieve test parameters:
    std::tuple<ncvslideio::GComputation, ncvslideio::gapi::wip::GStreamerSource::OutputType> params = GetParam();
    ncvslideio::GComputation extractImage = std::move(std::get<0>(params));
    ncvslideio::gapi::wip::GStreamerSource::OutputType outputType = std::get<1>(params);

    // Graph compilation for streaming mode:
    auto compiled =
        extractImage.compileStreaming();

    EXPECT_TRUE(compiled);
    EXPECT_FALSE(compiled.running());

    ncvslideio::gapi::wip::GStreamerPipeline
        pipeline(std::string("multifilesrc location=" + pathToLeftIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=NV12 ! "
                 "appsink name=sink1 ") +
                 std::string("multifilesrc location=" + pathToRightIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=NV12 ! "
                 "appsink name=sink2"));

    // GStreamer streaming sources configuration:
    auto src1 = pipeline.getStreamingSource("sink1", outputType);
    auto src2 = pipeline.getStreamingSource("sink2", outputType);

    compiled.setSource(ncvslideio::gin(src1, src2));

    // Start of streaming:
    compiled.start();
    EXPECT_TRUE(compiled.running());

    // Streaming - pulling of frames:
    ncvslideio::Mat in_mat1, in_mat2;

    std::size_t counter { }, limit { 10 };
    while(compiled.pull(ncvslideio::gout(in_mat1, in_mat2)) && (counter < limit)) {
        EXPECT_EQ(0, ncvslideio::norm(in_mat1, leftRefMat, ncvslideio::NORM_INF));
        EXPECT_EQ(0, ncvslideio::norm(in_mat2, rightRefMat, ncvslideio::NORM_INF));
        ++counter;
    }

    compiled.stop();

    EXPECT_FALSE(compiled.running());
}

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGMatsTest, GStreamerMultiSourceTestNV12,
                        Combine(Values(ncvslideio::GComputation([]()
                                       {
                                           ncvslideio::GMat in1, in2;
                                           return ncvslideio::GComputation(ncvslideio::GIn(in1, in2),
                                                                   ncvslideio::GOut(ncvslideio::gapi::copy(in1),
                                                                            ncvslideio::gapi::copy(in2)));
                                       })),
                               Values(ncvslideio::gapi::wip::GStreamerSource::OutputType::MAT)));

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGFramesTest, GStreamerMultiSourceTestNV12,
                        Combine(Values(ncvslideio::GComputation([]()
                                       {
                                           ncvslideio::GFrame in1, in2;
                                           return ncvslideio::GComputation(ncvslideio::GIn(in1, in2),
                                                ncvslideio::GOut(ncvslideio::gapi::streaming::BGR(in1),
                                                         ncvslideio::gapi::streaming::BGR(in2)));
                                       })),
                               Values(ncvslideio::gapi::wip::GStreamerSource::OutputType::FRAME)));

struct GStreamerMultiSourceTestGRAY8 :
    public TestWithParam<std::tuple<ncvslideio::GComputation, ncvslideio::gapi::wip::GStreamerSource::OutputType>>
{ };

TEST_P(GStreamerMultiSourceTestGRAY8, ImageDataTest)
{
    std::string pathToLeftIm = findDataFile("ncvslideio/stereomatching/datasets/tsukuba/im6.png");
    std::string pathToRightIm = findDataFile("ncvslideio/stereomatching/datasets/tsukuba/im2.png");

    std::string pipelineToReadImage("filesrc location=LOC ! pngdec ! videoconvert ! "
        "videoscale ! video/x-raw,format=GRAY8 ! appsink");

    ncvslideio::gapi::wip::GStreamerSource leftImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToLeftIm));
    ncvslideio::gapi::wip::GStreamerSource rightImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToRightIm));

    ncvslideio::gapi::wip::Data leftImData, rightImData;
    leftImageProvider.pull(leftImData);
    rightImageProvider.pull(rightImData);

    ncvslideio::Mat leftRefMat =  ncvslideio::util::get<ncvslideio::Mat>(leftImData);
    ncvslideio::Mat rightRefMat = ncvslideio::util::get<ncvslideio::Mat>(rightImData);

    // Retrieve test parameters:
    std::tuple<ncvslideio::GComputation, ncvslideio::gapi::wip::GStreamerSource::OutputType> params = GetParam();
    ncvslideio::GComputation extractImage = std::move(std::get<0>(params));
    ncvslideio::gapi::wip::GStreamerSource::OutputType outputType = std::get<1>(params);

    // Graph compilation for streaming mode:
    auto compiled =
        extractImage.compileStreaming();

    EXPECT_TRUE(compiled);
    EXPECT_FALSE(compiled.running());

    ncvslideio::gapi::wip::GStreamerPipeline
        pipeline(std::string("multifilesrc location=" + pathToLeftIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! "
                 "appsink name=sink1 ") +
                 std::string("multifilesrc location=" + pathToRightIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! "
                 "appsink name=sink2"));

    // GStreamer streaming sources configuration:
    auto src1 = pipeline.getStreamingSource("sink1", outputType);
    auto src2 = pipeline.getStreamingSource("sink2", outputType);

    compiled.setSource(ncvslideio::gin(src1, src2));

    // Start of streaming:
    compiled.start();
    EXPECT_TRUE(compiled.running());

    // Streaming - pulling of frames:
    ncvslideio::Mat in_mat1, in_mat2;

    std::size_t counter { }, limit { 10 };
    while(compiled.pull(ncvslideio::gout(in_mat1, in_mat2)) && (counter < limit)) {
        EXPECT_EQ(0, ncvslideio::norm(in_mat1, leftRefMat, ncvslideio::NORM_INF));
        EXPECT_EQ(0, ncvslideio::norm(in_mat2, rightRefMat, ncvslideio::NORM_INF));
        ++counter;
    }

    compiled.stop();

    EXPECT_FALSE(compiled.running());
}

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGMatsTest, GStreamerMultiSourceTestGRAY8,
                        Combine(Values(ncvslideio::GComputation([]()
                                       {
                                           ncvslideio::GMat in1, in2;
                                           return ncvslideio::GComputation(ncvslideio::GIn(in1, in2),
                                                                   ncvslideio::GOut(ncvslideio::gapi::copy(in1),
                                                                            ncvslideio::gapi::copy(in2)));
                                       })),
                               Values(ncvslideio::gapi::wip::GStreamerSource::OutputType::MAT)));

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGFramesTest, GStreamerMultiSourceTestGRAY8,
                        Combine(Values(ncvslideio::GComputation([]()
                                       {
                                           ncvslideio::GFrame in1, in2;
                                           return ncvslideio::GComputation(ncvslideio::GIn(in1, in2),
                                                ncvslideio::GOut(ncvslideio::gapi::streaming::BGR(in1),
                                                         ncvslideio::gapi::streaming::BGR(in2)));
                                       })),
                               Values(ncvslideio::gapi::wip::GStreamerSource::OutputType::FRAME)));

} // namespace opencv_test

#endif // HAVE_GSTREAMER
