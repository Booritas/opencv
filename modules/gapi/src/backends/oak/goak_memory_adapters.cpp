// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "oak_memory_adapters.hpp"

namespace ncvslideio {
namespace gapi {
namespace oak {

OAKMediaAdapter::OAKMediaAdapter(ncvslideio::Size sz, ncvslideio::MediaFormat fmt, std::vector<uint8_t>&& buffer)
: m_sz(sz), m_fmt(fmt), m_buffer(buffer) {
    GAPI_Assert(fmt == ncvslideio::MediaFormat::NV12 && "OAKMediaAdapter only supports NV12 format for now");
}

MediaFrame::View OAKMediaAdapter::OAKMediaAdapter::access(MediaFrame::Access) {
    uint8_t* y_ptr = m_buffer.data();
    uint8_t* uv_ptr = m_buffer.data() + static_cast<long>(m_buffer.size() / 3 * 2);
    return MediaFrame::View{ncvslideio::MediaFrame::View::Ptrs{y_ptr, uv_ptr},
                            ncvslideio::MediaFrame::View::Strides{static_cast<long unsigned int>(m_sz.width),
                                                          static_cast<long unsigned int>(m_sz.width)}};
}

ncvslideio::GFrameDesc OAKMediaAdapter::OAKMediaAdapter::meta() const { return {m_fmt, m_sz}; }

OAKRMatAdapter::OAKRMatAdapter(const ncvslideio::Size& size,
                               int precision,
                               std::vector<float>&& buffer)
    : m_size(size), m_precision(precision), m_buffer(buffer) {
    GAPI_Assert(m_precision == CV_16F);

    std::vector<int> wrapped_dims{1, 1, m_size.width, m_size.height};

    // FIXME: check layout and add strides
    m_desc = ncvslideio::GMatDesc(m_precision, wrapped_dims);
    m_mat = ncvslideio::Mat(static_cast<int>(wrapped_dims.size()),
                    wrapped_dims.data(),
                    CV_16FC1, // FIXME: cover other precisions
                    m_buffer.data());
}

ncvslideio::GMatDesc OAKRMatAdapter::desc() const {
    return m_desc;
}

ncvslideio::RMat::View OAKRMatAdapter::access(ncvslideio::RMat::Access) {
    return ncvslideio::RMat::View{m_desc, m_mat.data};
}

} // namespace oak
} // namespace gapi
} // namespace ncvslideio
