// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_RENDER_PRIV_HPP
#define OPENCV_RENDER_PRIV_HPP

#include <opencv2/gapi/render/render.hpp>

namespace ncvslideio
{
namespace gapi
{
namespace wip
{
namespace draw
{

// FIXME only for tests
GAPI_EXPORTS void cvtNV12ToYUV(const ncvslideio::Mat& y, const ncvslideio::Mat& uv, ncvslideio::Mat& yuv);
GAPI_EXPORTS void cvtYUVToNV12(const ncvslideio::Mat& yuv, ncvslideio::Mat& y, ncvslideio::Mat& uv);

} // namespace draw
} // namespace wip
} // namespace gapi
} // namespace ncvslideio

#endif // OPENCV_RENDER_PRIV_HPP
