// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_PERF_TESTS_HPP
#define OPENCV_GAPI_VIDEO_PERF_TESTS_HPP

#include "../../test/common/gapi_video_tests_common.hpp"

namespace opencv_test
{

using namespace perf;

//------------------------------------------------------------------------------

class BuildOptFlowPyramidPerfTest : public TestPerfParams<tuple<std::string,int,int,bool,int,int,
                                                                bool,GCompileArgs>> {};
class OptFlowLKPerfTest : public TestPerfParams<tuple<std::string,int,tuple<int,int>,int,
                                                      ncvslideio::TermCriteria,ncvslideio::GCompileArgs>> {};
class OptFlowLKForPyrPerfTest : public TestPerfParams<tuple<std::string,int,tuple<int,int>,int,
                                                            ncvslideio::TermCriteria,bool,
                                                            ncvslideio::GCompileArgs>> {};
class BuildPyr_CalcOptFlow_PipelinePerfTest : public TestPerfParams<tuple<std::string,int,int,bool,
                                                                          ncvslideio::GCompileArgs>> {};

class BackgroundSubtractorPerfTest:
    public TestPerfParams<tuple<ncvslideio::gapi::video::BackgroundSubtractorType, std::string,
                                bool, double, std::size_t, ncvslideio::GCompileArgs, CompareMats>> {};

class KalmanFilterControlPerfTest   :
    public TestPerfParams<tuple<MatType2, int, int, size_t, bool, ncvslideio::GCompileArgs>> {};
class KalmanFilterNoControlPerfTest :
    public TestPerfParams<tuple<MatType2, int, int, size_t, bool, ncvslideio::GCompileArgs>> {};

} // opencv_test

#endif // OPENCV_GAPI_VIDEO_PERF_TESTS_HPP
