// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_PERF_TESTS_HPP
#define OPENCV_GAPI_RENDER_PERF_TESTS_HPP


#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/render/render.hpp>

namespace opencv_test
{

using namespace perf;

class RenderTestFTexts : public TestPerfParams<tuple<std::wstring, ncvslideio::Size, ncvslideio::Point,
                                                     int, ncvslideio::Scalar, ncvslideio::GCompileArgs>> {};
class RenderTestTexts : public TestPerfParams<tuple<std::string, ncvslideio::Size, ncvslideio::Point,
                                                    int, ncvslideio::Scalar, int, int,
                                                    bool, ncvslideio::GCompileArgs>> {};
class RenderTestRects : public TestPerfParams<tuple<ncvslideio::Size, ncvslideio::Rect, ncvslideio::Scalar,
                                                    int, int, int, ncvslideio::GCompileArgs>> {};
class RenderTestCircles : public TestPerfParams<tuple<ncvslideio::Size, ncvslideio::Point, int,
                                                      ncvslideio::Scalar, int, int, int,
                                                      ncvslideio::GCompileArgs>> {};
class RenderTestLines : public TestPerfParams<tuple<ncvslideio::Size, ncvslideio::Point, ncvslideio::Point,
                                                    ncvslideio::Scalar, int, int, int,
                                                    ncvslideio::GCompileArgs>> {};
class RenderTestMosaics : public TestPerfParams<tuple<ncvslideio::Size, ncvslideio::Rect, int, int,
                                                      ncvslideio::GCompileArgs>> {};
class RenderTestImages : public TestPerfParams<tuple<ncvslideio::Size, ncvslideio::Rect, ncvslideio::Scalar, double,
                                                     ncvslideio::GCompileArgs>> {};
class RenderTestPolylines : public TestPerfParams<tuple<ncvslideio::Size, std::vector<ncvslideio::Point>,
                                                        ncvslideio::Scalar, int, int, int,
                                                        ncvslideio::GCompileArgs>> {};
class RenderTestPolyItems : public TestPerfParams<tuple<ncvslideio::Size, int, int, int, ncvslideio::GCompileArgs>> {};

}
#endif //OPENCV_GAPI_RENDER_PERF_TESTS_HPP
