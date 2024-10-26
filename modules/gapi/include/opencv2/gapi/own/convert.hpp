// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_CONVERT_HPP
#define OPENCV_GAPI_OWN_CONVERT_HPP

#if !defined(GAPI_STANDALONE)

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/own/mat.hpp>

namespace ncvslideio
{
    template<typename T>
    std::vector<T> to_own(const ncvslideio::MatSize &sz) {
        std::vector<T> result(sz.dims());
        for (int i = 0; i < sz.dims(); i++) {
            // Note: ncvslideio::MatSize is not iterable
            result[i] = static_cast<T>(sz[i]);
        }
        return result;
    }

    ncvslideio::gapi::own::Mat to_own(Mat&&) = delete;

    inline ncvslideio::gapi::own::Mat to_own(Mat const& m) {
        return (m.dims == 2)
            ?  ncvslideio::gapi::own::Mat{m.rows, m.cols, m.type(), m.data, m.step}
            :  ncvslideio::gapi::own::Mat{to_own<int>(m.size), m.type(), m.data};
    }

namespace gapi
{
namespace own
{

    inline ncvslideio::Mat to_ocv(Mat const& m) {
        return m.dims.empty()
            ? ncvslideio::Mat{m.rows, m.cols, m.type(), m.data, m.step}
            : ncvslideio::Mat{m.dims, m.type(), m.data};
    }

    ncvslideio::Mat to_ocv(Mat&&) = delete;

} // namespace own
} // namespace gapi
} // namespace ncvslideio

#endif // !defined(GAPI_STANDALONE)

#endif // OPENCV_GAPI_OWN_CONVERT_HPP
