// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gframe.hpp>
#include <opencv2/gapi/media.hpp>

#include "api/gorigin.hpp"

// ncvslideio::GFrame public implementation //////////////////////////////////////////////
ncvslideio::GFrame::GFrame()
    : m_priv(new GOrigin(GShape::GFRAME, GNode::Param())) {
}

ncvslideio::GFrame::GFrame(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GFRAME, n, out)) {
}

ncvslideio::GOrigin& ncvslideio::GFrame::priv() {
    return *m_priv;
}

const ncvslideio::GOrigin& ncvslideio::GFrame::priv() const {
    return *m_priv;
}

namespace ncvslideio {

bool GFrameDesc::operator== (const GFrameDesc &rhs) const {
    return fmt == rhs.fmt && size == rhs.size;
}

GFrameDesc descr_of(const ncvslideio::MediaFrame &frame) {
    return frame.desc();
}

std::ostream& operator<<(std::ostream& os, const ncvslideio::GFrameDesc &d) {
    os << '[';
    switch (d.fmt) {
    case MediaFormat::BGR:  os << "BGR"; break;
    case MediaFormat::NV12: os << "NV12"; break;
    case MediaFormat::GRAY: os << "GRAY"; break;
    default: GAPI_Error("Invalid media format");
    }
    os << ' ' << d.size << ']';
    return os;
}

} // namespace ncvslideio
