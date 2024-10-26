// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <tuple>
#include <unordered_set>

#include "test_precomp.hpp"
#include "opencv2/gapi/streaming/meta.hpp"
#include "opencv2/gapi/streaming/cap.hpp"

namespace opencv_test {

TEST(GraphMeta, Trad_AccessInput) {
    ncvslideio::GMat in;
    ncvslideio::GMat out1 = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));
    ncvslideio::GOpaque<int> out2 = ncvslideio::gapi::streaming::meta<int>(in, "foo");
    ncvslideio::GComputation graph(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(ncvslideio::Size(64, 64), CV_8UC1);
    ncvslideio::Mat out_mat;
    int out_meta = 0;

    // manually set metadata in the input fields
    auto inputs = ncvslideio::gin(in_mat);
    inputs[0].meta["foo"] = 42;

    graph.apply(std::move(inputs), ncvslideio::gout(out_mat, out_meta));
    EXPECT_EQ(42, out_meta);
}

TEST(GraphMeta, Trad_AccessTmp) {
    ncvslideio::GMat in;
    ncvslideio::GMat tmp = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));
    ncvslideio::GMat out1 = tmp+1;
    ncvslideio::GOpaque<float> out2 = ncvslideio::gapi::streaming::meta<float>(tmp, "bar");
    ncvslideio::GComputation graph(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(ncvslideio::Size(64, 64), CV_8UC1);
    ncvslideio::Mat out_mat;
    float out_meta = 0.f;

    // manually set metadata in the input fields
    auto inputs = ncvslideio::gin(in_mat);
    inputs[0].meta["bar"] = 1.f;

    graph.apply(std::move(inputs), ncvslideio::gout(out_mat, out_meta));
    EXPECT_EQ(1.f, out_meta);
}

TEST(GraphMeta, Trad_AccessOutput) {
    ncvslideio::GMat in;
    ncvslideio::GMat out1 = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));
    ncvslideio::GOpaque<std::string> out2 = ncvslideio::gapi::streaming::meta<std::string>(out1, "baz");
    ncvslideio::GComputation graph(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(ncvslideio::Size(64, 64), CV_8UC1);
    ncvslideio::Mat out_mat;
    std::string out_meta;

    // manually set metadata in the input fields
    auto inputs = ncvslideio::gin(in_mat);

    // NOTE: Assigning explicitly an std::string is important,
    // otherwise a "const char*" will be stored and won't be
    // translated properly by util::any since std::string is
    // used within the graph.
    inputs[0].meta["baz"] = std::string("opencv");

    graph.apply(std::move(inputs), ncvslideio::gout(out_mat, out_meta));
    EXPECT_EQ("opencv", out_meta);
}

TEST(GraphMeta, Streaming_AccessInput) {
    ncvslideio::GMat in;
    ncvslideio::GMat out1 = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));
    ncvslideio::GOpaque<int64_t> out2 = ncvslideio::gapi::streaming::seq_id(in);
    ncvslideio::GComputation graph(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    auto ccomp = graph.compileStreaming();
    const auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        ccomp.setSource<ncvslideio::gapi::wip::GCaptureSource>(path);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();

    ncvslideio::Mat out_mat;
    int64_t out_meta = 0;
    int64_t expected_counter = 0;

    while (ccomp.pull(ncvslideio::gout(out_mat, out_meta))) {
        EXPECT_EQ(expected_counter, out_meta);
        ++expected_counter;
    }
}

TEST(GraphMeta, Streaming_AccessOutput) {
    ncvslideio::GMat in;
    ncvslideio::GMat out1 = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));
    ncvslideio::GOpaque<int64_t> out2 = ncvslideio::gapi::streaming::seq_id(out1);
    ncvslideio::GOpaque<int64_t> out3 = ncvslideio::gapi::streaming::timestamp(out1);
    ncvslideio::GComputation graph(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2, out3));

    auto ccomp = graph.compileStreaming();
    const auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        ccomp.setSource<ncvslideio::gapi::wip::GCaptureSource>(path);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();

    ncvslideio::Mat out_mat;
    int64_t out_meta = 0;
    int64_t out_timestamp = 0;
    int64_t expected_counter = 0;
    int64_t prev_timestamp = -1;

    while (ccomp.pull(ncvslideio::gout(out_mat, out_meta, out_timestamp))) {
        EXPECT_EQ(expected_counter, out_meta);
        ++expected_counter;

        EXPECT_NE(prev_timestamp, out_timestamp);
        prev_timestamp = out_timestamp;
    }
}

TEST(GraphMeta, Streaming_AccessDesync) {
    ncvslideio::GMat in;
    ncvslideio::GOpaque<int64_t> out1 = ncvslideio::gapi::streaming::seq_id(in);
    ncvslideio::GOpaque<int64_t> out2 = ncvslideio::gapi::streaming::timestamp(in);
    ncvslideio::GMat             out3 = ncvslideio::gapi::blur(in, ncvslideio::Size(3,3));

    ncvslideio::GMat tmp = ncvslideio::gapi::streaming::desync(in);
    ncvslideio::GScalar mean = ncvslideio::gapi::mean(tmp);
    ncvslideio::GOpaque<int64_t> out4 = ncvslideio::gapi::streaming::seq_id(mean);
    ncvslideio::GOpaque<int64_t> out5 = ncvslideio::gapi::streaming::timestamp(mean);
    ncvslideio::GComputation graph(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2, out3, out4, out5));

    auto ccomp = graph.compileStreaming();
    const auto path = findDataFile("ncvslideio/video/768x576.avi");
    try {
        ccomp.setSource<ncvslideio::gapi::wip::GCaptureSource>(path);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    ccomp.start();

    ncvslideio::optional<int64_t> out_sync_id;
    ncvslideio::optional<int64_t> out_sync_ts;
    ncvslideio::optional<ncvslideio::Mat> out_sync_mat;

    ncvslideio::optional<int64_t> out_desync_id;
    ncvslideio::optional<int64_t> out_desync_ts;

    std::unordered_set<int64_t> sync_ids;
    std::unordered_set<int64_t> desync_ids;

    while (ccomp.pull(ncvslideio::gout(out_sync_id, out_sync_ts, out_sync_mat,
                               out_desync_id, out_desync_ts))) {
        if (out_sync_id.has_value()) {
            CV_Assert(out_sync_ts.has_value());
            CV_Assert(out_sync_mat.has_value());
            sync_ids.insert(out_sync_id.value());
        }
        if (out_desync_id.has_value()) {
            CV_Assert(out_desync_ts.has_value());
            desync_ids.insert(out_desync_id.value());
        }
    }
    // Visually report that everything is really ok
    std::cout << sync_ids.size() << " vs " << desync_ids.size() << std::endl;

    // Desync path should generate less objects than the synchronized one
    EXPECT_GE(sync_ids.size(), desync_ids.size());

    // ..but all desynchronized IDs must be present in the synchronized set
    for (auto &&d_id : desync_ids) {
        EXPECT_TRUE(sync_ids.count(d_id) > 0);
    }
}

} // namespace opencv_test
