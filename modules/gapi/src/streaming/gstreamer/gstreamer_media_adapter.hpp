// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_MEDIA_ADAPTER_HPP
#define OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_MEDIA_ADAPTER_HPP

// #include <opencv2/gapi/garray.hpp>
// #include <opencv2/gapi/streaming/meta.hpp>

#include "gstreamerptr.hpp"
#include <opencv2/gapi/streaming/gstreamer/gstreamersource.hpp>

#include <atomic>
#include <mutex>

#ifdef HAVE_GSTREAMER
#include <gst/gstbuffer.h>
#include <gst/video/video-frame.h>

namespace ncvslideio {
namespace gapi {
namespace wip {
namespace gst {

class GStreamerMediaAdapter : public ncvslideio::MediaFrame::IAdapter {
public:
    explicit GStreamerMediaAdapter(const ncvslideio::GFrameDesc& frameDesc,
                                   GstVideoInfo* videoInfo,
                                   GstBuffer* buffer);

    ~GStreamerMediaAdapter() override;

    virtual ncvslideio::GFrameDesc meta() const override;

    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access access) override;

    ncvslideio::util::any blobParams() const override;

protected:
    ncvslideio::GFrameDesc m_frameDesc;

    GStreamerPtr<GstVideoInfo> m_videoInfo;
    GStreamerPtr<GstBuffer> m_buffer;

    std::vector<gint> m_strides;
    std::vector<gsize> m_offsets;

    GstVideoFrame m_videoFrame;

    std::atomic<bool> m_isMapped;
    std::atomic<bool> m_mappedForWrite;
    std::mutex m_mutex;
};

} // namespace gst
} // namespace wip
} // namespace gapi
} // namespace ncvslideio
#endif // HAVE_GSTREAMER
#endif // OPENCV_GAPI_STREAMING_GSTREAMER_GSTREAMER_MEDIA_ADAPTER_HPP
