// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_stereo_tests.hpp"

#include <opencv2/gapi/stereo.hpp> // For ::gapi::stereo::disparity/depth
#include <opencv2/gapi/cpu/stereo.hpp>

namespace
{
#define STEREO_CPU [] () { return ncvslideio::compile_args(ncvslideio::gapi::use_only{ncvslideio::gapi::calib3d::cpu::kernels()}); }
}  // anonymous namespace

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(CPU_Tests, TestGAPIStereo,
                        Combine(Values(CV_8UC1),
                                Values(ncvslideio::Size(1280, 720)),
                                Values(CV_32FC1),
                                Values(STEREO_CPU),
                                Values(ncvslideio::gapi::StereoOutputFormat::DEPTH_FLOAT16,
                                       ncvslideio::gapi::StereoOutputFormat::DEPTH_FLOAT32,
                                       ncvslideio::gapi::StereoOutputFormat::DISPARITY_FIXED16_12_4,
                                       ncvslideio::gapi::StereoOutputFormat::DEPTH_16F,
                                       ncvslideio::gapi::StereoOutputFormat::DEPTH_32F,
                                       ncvslideio::gapi::StereoOutputFormat::DISPARITY_16Q_11_4),
                                Values(16),
                                Values(43),
                                Values(63.5),
                                Values(3.6),
                                Values(AbsExact().to_compare_obj())));

} // opencv_test
