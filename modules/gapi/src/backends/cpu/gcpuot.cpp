// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include <opencv2/gapi/ot.hpp>
#include <opencv2/gapi/cpu/ot.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <vas/ot.hpp>

namespace ncvslideio
{
namespace gapi
{
namespace ot
{

// Helper functions for OT kernels
namespace {
void GTrackImplSetup(ncvslideio::GArrayDesc, ncvslideio::GArrayDesc, float,
                     std::shared_ptr<vas::ot::ObjectTracker>& state,
                     const ObjectTrackerParams& params) {
    vas::ot::ObjectTracker::Builder ot_builder;
    ot_builder.max_num_objects = params.max_num_objects;
    ot_builder.input_image_format = vas::ColorFormat(params.input_image_format);
    ot_builder.tracking_per_class = params.tracking_per_class;

    state = ot_builder.Build(vas::ot::TrackingType::ZERO_TERM_IMAGELESS);
}

void GTrackImplPrepare(const std::vector<ncvslideio::Rect>& in_rects,
                       const std::vector<int32_t>& in_class_labels,
                       float delta,
                       std::vector<vas::ot::DetectedObject>& detected_objs,
                       vas::ot::ObjectTracker& state)
{
    if (in_rects.size() != in_class_labels.size())
    {
        ncvslideio::util::throw_error(std::invalid_argument("Track() implementation run() method: in_rects and in_class_labels "
                                                    "sizes are different."));
    }

    detected_objs.reserve(in_rects.size());

    for (std::size_t i = 0; i < in_rects.size(); ++i)
    {
        detected_objs.emplace_back(in_rects[i], in_class_labels[i]);
    }

    state.SetFrameDeltaTime(delta);
}
} // anonymous namespace

GAPI_OCV_KERNEL_ST(GTrackFromMatImpl, ncvslideio::gapi::ot::GTrackFromMat, vas::ot::ObjectTracker)
{
    static void setup(ncvslideio::GMatDesc, ncvslideio::GArrayDesc rects_desc,
                      ncvslideio::GArrayDesc labels_desc, float delta,
                      std::shared_ptr<vas::ot::ObjectTracker>& state,
                      const ncvslideio::GCompileArgs& compile_args)
    {
        auto params = ncvslideio::gapi::getCompileArg<ObjectTrackerParams>(compile_args)
            .value_or(ObjectTrackerParams{});

        GAPI_Assert(params.input_image_format == 0 && "Only BGR input as ncvslideio::GMat is supported for now");
        GTrackImplSetup(rects_desc, labels_desc, delta, state, params);
    }

    static void run(const ncvslideio::Mat& in_mat, const std::vector<ncvslideio::Rect>& in_rects,
                    const std::vector<int32_t>& in_class_labels, float delta,
                    std::vector<ncvslideio::Rect>& out_tr_rects,
                    std::vector<int32_t>& out_rects_classes,
                    std::vector<uint64_t>& out_tr_ids,
                    std::vector<int>& out_tr_statuses,
                    vas::ot::ObjectTracker& state)
    {
        std::vector<vas::ot::DetectedObject> detected_objs;
        GTrackImplPrepare(in_rects, in_class_labels, delta, detected_objs, state);

        GAPI_Assert(in_mat.type() == CV_8UC3 && "Input mat is not in BGR format");

        auto objects = state.Track(in_mat, detected_objs);

        for (auto&& object : objects)
        {
            out_tr_rects.push_back(object.rect);
            out_rects_classes.push_back(object.class_label);
            out_tr_ids.push_back(object.tracking_id);
            out_tr_statuses.push_back(static_cast<int>(object.status));
        }
    }
};

GAPI_OCV_KERNEL_ST(GTrackFromFrameImpl, ncvslideio::gapi::ot::GTrackFromFrame, vas::ot::ObjectTracker)
{
    static void setup(ncvslideio::GFrameDesc, ncvslideio::GArrayDesc rects_desc,
                      ncvslideio::GArrayDesc labels_desc, float delta,
                      std::shared_ptr<vas::ot::ObjectTracker>& state,
                      const ncvslideio::GCompileArgs& compile_args)
    {
        auto params = ncvslideio::gapi::getCompileArg<ObjectTrackerParams>(compile_args)
            .value_or(ObjectTrackerParams{});

        GAPI_Assert(params.input_image_format == 1 && "Only NV12 input as ncvslideio::GFrame is supported for now");
        GTrackImplSetup(rects_desc, labels_desc, delta, state, params);
    }

    static void run(const ncvslideio::MediaFrame& in_frame, const std::vector<ncvslideio::Rect>& in_rects,
                    const std::vector<int32_t>& in_class_labels, float delta,
                    std::vector<ncvslideio::Rect>& out_tr_rects,
                    std::vector<int32_t>& out_rects_classes,
                    std::vector<uint64_t>& out_tr_ids,
                    std::vector<int>& out_tr_statuses,
                    vas::ot::ObjectTracker& state)
    {
        std::vector<vas::ot::DetectedObject> detected_objs;
        GTrackImplPrepare(in_rects, in_class_labels, delta, detected_objs, state);

        // Extract metadata from MediaFrame and construct ncvslideio::Mat atop of it
        ncvslideio::MediaFrame::View view = in_frame.access(ncvslideio::MediaFrame::Access::R);
        auto ptrs = view.ptr;
        auto strides = view.stride;
        auto desc = in_frame.desc();

        GAPI_Assert((desc.fmt == ncvslideio::MediaFormat::NV12 || desc.fmt == ncvslideio::MediaFormat::BGR) \
                    && "Input frame is not in NV12 or BGR format");

        ncvslideio::Mat in;
        if (desc.fmt == ncvslideio::MediaFormat::NV12) {
            GAPI_Assert(ptrs[0] != nullptr && "Y plane pointer is empty");
            GAPI_Assert(ptrs[1] != nullptr && "UV plane pointer is empty");
            if (strides[0] > 0) {
                in = ncvslideio::Mat(desc.size, CV_8UC1, ptrs[0], strides[0]);
            } else {
                in = ncvslideio::Mat(desc.size, CV_8UC1, ptrs[0]);
            }
        }

        auto objects = state.Track(in, detected_objs);

        for (auto&& object : objects)
        {
            out_tr_rects.push_back(object.rect);
            out_rects_classes.push_back(object.class_label);
            out_tr_ids.push_back(object.tracking_id);
            out_tr_statuses.push_back(static_cast<int>(object.status));
        }
    }
};

ncvslideio::gapi::GKernelPackage cpu::kernels()
{
    return ncvslideio::gapi::kernels
        <
          GTrackFromFrameImpl,
          GTrackFromMatImpl
        >();
}

}   // namespace ot
}   // namespace gapi
}   // namespace ncvslideio
