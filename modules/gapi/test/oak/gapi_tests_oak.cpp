// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test_precomp.hpp"

#ifdef HAVE_OAK

#include <opencv2/gapi/oak/oak.hpp>

namespace opencv_test
{

// FIXME: consider a better solution
TEST(OAK, Available)
{
    ncvslideio::GFrame in;
    auto out = ncvslideio::gapi::oak::encode(in, {});
    auto args = ncvslideio::compile_args(ncvslideio::gapi::oak::ColorCameraParams{}, ncvslideio::gapi::oak::kernels());
    auto pipeline = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out)).compileStreaming(std::move(args));
}
} // opencv_test

#endif // HAVE_OAK
