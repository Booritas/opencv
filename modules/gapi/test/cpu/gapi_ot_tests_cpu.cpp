// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation


#include "../test_precomp.hpp"

#include <opencv2/gapi/ot.hpp>
#include <opencv2/gapi/cpu/ot.hpp>

#include "opencv2/gapi/streaming/meta.hpp"
#include "opencv2/gapi/streaming/cap.hpp"

namespace {
ncvslideio::gapi::ot::TrackingStatus from_string(const std::string& status) {
    if (status == "NEW") {
        return ncvslideio::gapi::ot::TrackingStatus::NEW;
    }
    else if (status == "TRACKED") {
        return ncvslideio::gapi::ot::TrackingStatus::TRACKED;
    }
    else if (status == "LOST") {
        return ncvslideio::gapi::ot::TrackingStatus::LOST;
    }

    throw std::runtime_error("String representation for ncvslideio::gapi::ot::TrackingStatus: \""
        + status + "\" contains incorrect value!");
}
} // anonymous namespace

namespace opencv_test {
struct FrameDetections {
    std::size_t frame_no{};
    std::vector<std::vector<ncvslideio::Rect>> boxes;
    std::vector<std::vector<int32_t>> box_ids;
    FrameDetections() {}
    FrameDetections(std::size_t in_frame_no, const std::vector<std::vector<ncvslideio::Rect>>& in_boxes,
                    const std::vector<std::vector<int32_t>>& in_box_ids) :
        frame_no(in_frame_no),
        boxes(in_boxes),
        box_ids(in_box_ids) {}
};

struct TrackerReference {
    std::size_t frame_no{};
    std::vector<std::vector<ncvslideio::Rect>> tracked_boxes;
    std::vector<std::vector<int32_t>> tracked_box_ids;
    std::vector<std::vector<uint64_t>> tracking_ids;
    std::vector<std::vector<ncvslideio::gapi::ot::TrackingStatus>> tracking_statuses;
    TrackerReference() {}
    TrackerReference(std::size_t in_frame_no,
                     const std::vector<std::vector<ncvslideio::Rect>>& in_tracked_boxes,
                     const std::vector<std::vector<int32_t>>& in_tracked_box_ids,
                     const std::vector<std::vector<uint64_t>>& in_tracking_ids,
                     const std::vector<std::vector<ncvslideio::gapi::ot::TrackingStatus>>&
                         in_tracking_statuses) :
        frame_no(in_frame_no),
        tracked_boxes(in_tracked_boxes),
        tracked_box_ids(in_tracked_box_ids),
        tracking_ids(in_tracking_ids),
        tracking_statuses(in_tracking_statuses) {}
};

struct FrameDetectionsParams {
    FrameDetections value;
};

struct TrackerReferenceParams {
    TrackerReference value;
};
} // namespace opencv_test

namespace ncvslideio {
    namespace detail {
        template<> struct CompileArgTag<opencv_test::FrameDetectionsParams> {
            static const char* tag() {
                return "org.opencv.test.frame_detections_params";
            }
        };
    } // namespace detail

    namespace detail {
        template<> struct CompileArgTag<opencv_test::TrackerReferenceParams> {
            static const char* tag() {
                return "org.opencv.test.tracker_reference_params";
            }
        };
    } // namespace detail
} // namespace ncvslideio

namespace opencv_test {
G_API_OP(CvVideo768x576_Detect, <std::tuple<ncvslideio::GArray<ncvslideio::Rect>, ncvslideio::GArray<int32_t>>(ncvslideio::GMat)>,
    "test.custom.cv_video_768x576_detect") {
    static std::tuple<ncvslideio::GArrayDesc, ncvslideio::GArrayDesc> outMeta(ncvslideio::GMatDesc) {
        return std::make_tuple(ncvslideio::empty_array_desc(), ncvslideio::empty_array_desc());
    }
};

GAPI_OCV_KERNEL_ST(OCV_CvVideo768x576_Detect, CvVideo768x576_Detect, FrameDetections) {
    static void setup(ncvslideio::GMatDesc,
                      std::shared_ptr<FrameDetections> &state,
                      const ncvslideio::GCompileArgs &compileArgs) {
        auto params = ncvslideio::gapi::getCompileArg<opencv_test::FrameDetectionsParams>(compileArgs)
            .value_or(opencv_test::FrameDetectionsParams{ });
        state = std::make_shared<FrameDetections>(params.value);
    }

    static void run(const ncvslideio::Mat&,
                    std::vector<ncvslideio::Rect> &out_boxes,
                    std::vector<int32_t> &out_box_ids,
                    FrameDetections &state) {
        if (state.frame_no < state.boxes.size()) {
            out_boxes = state.boxes[state.frame_no];
            out_box_ids = state.box_ids[state.frame_no];
            ++state.frame_no;
        }
    }
};

G_API_OP(CheckTrackerResults, <ncvslideio::GOpaque<bool>(ncvslideio::GArray<ncvslideio::Rect>, ncvslideio::GArray<int32_t>,
                                                 ncvslideio::GArray<uint64_t>, ncvslideio::GArray<int>)>,
    "test.custom.check_tracker_results") {
    static ncvslideio::GOpaqueDesc outMeta(ncvslideio::GArrayDesc, ncvslideio::GArrayDesc, ncvslideio::GArrayDesc, ncvslideio::GArrayDesc) {
        return ncvslideio::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL_ST(OCVCheckTrackerResults, CheckTrackerResults, TrackerReference) {
    static void setup(ncvslideio::GArrayDesc, ncvslideio::GArrayDesc,
                      ncvslideio::GArrayDesc, ncvslideio::GArrayDesc,
                      std::shared_ptr<TrackerReference> &state,
                      const ncvslideio::GCompileArgs &compileArgs) {
        auto params = ncvslideio::gapi::getCompileArg<opencv_test::TrackerReferenceParams>(compileArgs)
            .value_or(opencv_test::TrackerReferenceParams{ });
        state = std::make_shared<TrackerReference>(params.value);
    }

    static void run(const std::vector<ncvslideio::Rect> &in_tr_rcts,
                    const std::vector<int32_t> &in_det_ids,
                    const std::vector<uint64_t> &in_tr_ids,
                    const std::vector<int> &in_tr_statuses,
                    bool& success,
                    TrackerReference& state) {

        if (state.frame_no < state.tracked_boxes.size()) {
            auto reference_boxes = state.tracked_boxes[state.frame_no];
            auto reference_box_ids = state.tracked_box_ids[state.frame_no];
            auto reference_tr_ids = state.tracking_ids[state.frame_no];
            auto reference_tr_statuses = state.tracking_statuses[state.frame_no];

            success = true;
            GAPI_Assert(in_tr_rcts.size() == reference_boxes.size());
            GAPI_Assert(in_det_ids.size() == reference_box_ids.size());
            GAPI_Assert(in_tr_ids.size() == reference_tr_ids.size());
            GAPI_Assert(in_tr_statuses.size() == reference_tr_statuses.size());
            for (uint32_t i = 0; (i < in_tr_rcts.size() && success); ++i) {
                const ncvslideio::Rect& reference_rc = reference_boxes[i];
                const ncvslideio::Rect& in_rc = in_tr_rcts[i];

                success &= (reference_rc == in_rc);
                success &= (reference_box_ids[i] == in_det_ids[i]);
                success &= (reference_tr_ids[i] == in_tr_ids[i]);
                success &= (reference_tr_statuses[i] == in_tr_statuses[i]);
            }

            ++state.frame_no;
        }
        else {
            success = true;
        }
    }
};

TEST(VASObjectTracker, PipelineTest)
{
    constexpr int32_t frames_to_handle = 30;
    std::string pathToVideo = opencv_test::findDataFile("ncvslideio/video/768x576.avi");

    std::vector<std::vector<ncvslideio::Rect>> input_boxes(frames_to_handle);
    std::vector<std::vector<int32_t>> input_det_ids(frames_to_handle);

    std::string path_to_boxes = opencv_test::findDataFile("ncvslideio/video/vas_object_tracking/detections_30_frames.yml");

    ncvslideio::FileStorage fs_input_boxes(path_to_boxes, ncvslideio::FileStorage::READ);
    ncvslideio::FileNode fn_input_boxes = fs_input_boxes.root();
    for (auto it = fn_input_boxes.begin(); it != fn_input_boxes.end(); ++it) {
        ncvslideio::FileNode fn_frame = *it;
        std::string frame_name = fn_frame.name();
        int frame_no = std::stoi(frame_name.substr(frame_name.find("_") + 1));

        for (auto fit = fn_frame.begin(); fit != fn_frame.end(); ++fit) {
            ncvslideio::FileNode fn_box = *fit;

            ncvslideio::Rect box((int)fn_box["x"], (int)fn_box["y"],
                (int)fn_box["width"], (int)fn_box["height"]);
            input_boxes[frame_no].push_back(box);
            input_det_ids[frame_no].push_back(fn_box["id"]);
        }
    }

    std::vector<std::vector<ncvslideio::Rect>> reference_trackings(frames_to_handle);
    std::vector<std::vector<int32_t>> reference_trackings_det_ids(frames_to_handle);
    std::vector<std::vector<uint64_t>> reference_trackings_tr_ids(frames_to_handle);
    std::vector<std::vector<ncvslideio::gapi::ot::TrackingStatus>> reference_trackings_tr_statuses(frames_to_handle);

    std::string path_to_trackings = opencv_test::findDataFile("ncvslideio/video/vas_object_tracking/trackings_30_frames.yml");

    ncvslideio::FileStorage fs_reference_trackings(path_to_trackings, ncvslideio::FileStorage::READ);
    ncvslideio::FileNode fn_reference_trackings = fs_reference_trackings.root();
    for (auto it =  fn_reference_trackings.begin(); it != fn_reference_trackings.end(); ++it) {
        ncvslideio::FileNode fn_frame = *it;
        std::string frame_name = fn_frame.name();
        int frame_no = std::stoi(frame_name.substr(frame_name.find("_") + 1));

        for (auto fit = fn_frame.begin(); fit != fn_frame.end(); ++fit) {
            ncvslideio::FileNode fn_tracked_box = *fit;

            ncvslideio::Rect tracked_box((int)fn_tracked_box["x"], (int)fn_tracked_box["y"],
                (int)fn_tracked_box["width"], (int)fn_tracked_box["height"]);
            reference_trackings[frame_no].push_back(tracked_box);
            reference_trackings_det_ids[frame_no].push_back(fn_tracked_box["id"]);
            reference_trackings_tr_ids[frame_no].push_back(int(fn_tracked_box["tracking_id"]));
            reference_trackings_tr_statuses[frame_no].push_back(
                from_string(fn_tracked_box["tracking_status"]));
        }
    }

    ncvslideio::GMat in;

    ncvslideio::GArray<ncvslideio::Rect> detections;
    ncvslideio::GArray<int> det_ids;
    std::tie(detections, det_ids) = CvVideo768x576_Detect::on(in);

    constexpr float delta_time = 0.055f;
    ncvslideio::GArray<ncvslideio::Rect> tracking_rects;
    ncvslideio::GArray<int32_t> tracking_det_ids;
    ncvslideio::GArray<uint64_t> tracking_ids;
    ncvslideio::GArray<int> tracking_statuses;
    std::tie(tracking_rects, tracking_det_ids, tracking_ids, tracking_statuses) =
        ncvslideio::gapi::ot::track(in, detections, det_ids, delta_time);

    ncvslideio::GOpaque<bool> check_result =
        CheckTrackerResults::on(tracking_rects, tracking_det_ids, tracking_ids, tracking_statuses);

    ncvslideio::GComputation ccomp(ncvslideio::GIn(in), ncvslideio::GOut(check_result));


    opencv_test::FrameDetections fds { 0, input_boxes, input_det_ids };
    opencv_test::TrackerReference tr { 0, reference_trackings,
                                       reference_trackings_det_ids,
                                       reference_trackings_tr_ids,
                                       reference_trackings_tr_statuses };

    // Graph compilation for streaming mode:
    auto compiled =
        ccomp.compileStreaming(ncvslideio::compile_args(
            ncvslideio::gapi::combine(ncvslideio::gapi::kernels<OCV_CvVideo768x576_Detect,
                                                OCVCheckTrackerResults>(),
                              ncvslideio::gapi::ot::cpu::kernels()),
            opencv_test::FrameDetectionsParams{ fds },
            opencv_test::TrackerReferenceParams{ tr }));

    EXPECT_TRUE(compiled);
    EXPECT_FALSE(compiled.running());

    compiled.setSource<ncvslideio::gapi::wip::GCaptureSource>(pathToVideo);

    // Start of streaming:
    compiled.start();
    EXPECT_TRUE(compiled.running());

    // Streaming:
    bool success;

    std::size_t counter { }, limit { 30 };
    while(compiled.pull(ncvslideio::gout(success)) && (counter < limit)) {
         ++counter;
     }

     compiled.stop();

     EXPECT_TRUE(success);
     EXPECT_FALSE(compiled.running());
}
} // namespace opencv_test
