// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include <opencv2/gapi/ot.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <vas/ot.hpp>

namespace ncvslideio
{
namespace gapi
{
namespace ot
{
GAPI_EXPORTS_W std::tuple<ncvslideio::GArray<ncvslideio::Rect>,
                          ncvslideio::GArray<int32_t>,
                          ncvslideio::GArray<uint64_t>,
                          ncvslideio::GArray<int>>
    track(const ncvslideio::GMat& mat,
          const ncvslideio::GArray<ncvslideio::Rect>& detected_rects,
          const ncvslideio::GArray<int>& detected_class_labels,
          float delta)
{
    return GTrackFromMat::on(mat, detected_rects, detected_class_labels, delta);
}

GAPI_EXPORTS_W std::tuple<ncvslideio::GArray<ncvslideio::Rect>,
                          ncvslideio::GArray<int32_t>,
                          ncvslideio::GArray<uint64_t>,
                          ncvslideio::GArray<int>>
    track(const ncvslideio::GFrame& frame,
          const ncvslideio::GArray<ncvslideio::Rect>& detected_rects,
          const ncvslideio::GArray<int>& detected_class_labels,
          float delta)
{
    return GTrackFromFrame::on(frame, detected_rects, detected_class_labels, delta);
}
}   // namespace ot
}   // namespace gapi
}   // namespace ncvslideio
