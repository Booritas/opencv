// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
#define OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP

#include <memory>

#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/rmat.hpp>

namespace ncvslideio {
namespace gapi {
namespace oak {

// Used for OAK backends outputs only.
// Filled from DepthAI's ImgFrame type and owns the memory.
// Used mainly for ncvslideio operations.
class GAPI_EXPORTS OAKMediaAdapter final : public ncvslideio::MediaFrame::IAdapter {
public:
    OAKMediaAdapter() = default;
    OAKMediaAdapter(ncvslideio::Size sz, ncvslideio::MediaFormat fmt, std::vector<uint8_t>&& buffer);
    ncvslideio::GFrameDesc meta() const override;
    ncvslideio::MediaFrame::View access(ncvslideio::MediaFrame::Access) override;
    ~OAKMediaAdapter() = default;
private:
    ncvslideio::Size m_sz;
    ncvslideio::MediaFormat m_fmt;
    std::vector<uint8_t> m_buffer;
};

// Used for OAK backends outputs only.
// Filled from DepthAI's NNData type and owns the memory.
// Used only for infer operations.
class GAPI_EXPORTS OAKRMatAdapter final : public ncvslideio::RMat::Adapter {
public:
    OAKRMatAdapter() = default;
    OAKRMatAdapter(const ncvslideio::Size& size, int precision, std::vector<float>&& buffer);
    ncvslideio::GMatDesc desc() const override;
    ncvslideio::RMat::View access(ncvslideio::RMat::Access) override;
    ~OAKRMatAdapter() = default;
private:
    ncvslideio::Size m_size;
    int m_precision;
    std::vector<float> m_buffer;
    ncvslideio::GMatDesc m_desc;
    ncvslideio::Mat m_mat;
};

} // namespace oak
} // namespace gapi
} // namespace ncvslideio

#endif // OPENCV_GAPI_OAK_MEDIA_ADAPTER_HPP
