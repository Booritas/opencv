// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"

// These tests verify some parts of ncvslideio::gapi::infer<> API
// regardless of the backend used

namespace opencv_test {
namespace {
template<class A, class B> using Check = ncvslideio::detail::valid_infer2_types<A, B>;

TEST(Infer, ValidInfer2Types)
{
    // Compiled == passed!

    // Argument block 1
    static_assert(Check< std::tuple<ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::GMat> > // Call
                  ::value == true, "Must work");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::GMat, ncvslideio::GMat> > // Call
                  ::value == true, "Must work");

    // Argument block 2
    static_assert(Check< std::tuple<ncvslideio::GMat>             // Net
                       , std::tuple<ncvslideio::Rect> >           // Call
                  ::value == true, "Must work");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::Rect, ncvslideio::Rect> > // Call
                  ::value == true, "Must work");

    // Argument block 3 (mixed cases)
    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::GMat, ncvslideio::Rect> > // Call
                  ::value == true, "Must work");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::Rect, ncvslideio::GMat> > // Call
                  ::value == true, "Must work");

    // Argument block 4 (super-mixed)
    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::Rect, ncvslideio::GMat, ncvslideio::Rect> > // Call
                  ::value == true, "Must work");

    // Argument block 5 (mainly negative)
    static_assert(Check< std::tuple<ncvslideio::GMat>             // Net
                       , std::tuple<int> >                // Call
                  ::value == false, "This type(s) shouldn't pass");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<int, ncvslideio::Rect> >      // Call
                  ::value == false, "This type(s) shouldn't pass");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::Rect, ncvslideio::Point> >// Call
                  ::value == false, "This type(s) shouldn't pass");

    // Argument block 5 (wrong args length)
    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::GMat> >           // Call
                  ::value == false, "Should fail -- not enough args");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>   // Net
                       , std::tuple<ncvslideio::Rect> >           // Call
                  ::value == false, "Should fail -- not enough args");

    static_assert(Check< std::tuple<ncvslideio::GMat, ncvslideio::GMat>             // Net
                       , std::tuple<ncvslideio::Rect, ncvslideio::Rect, ncvslideio::GMat> > // Call
                  ::value == false, "Should fail -- too much args");
}
} // anonymous namespace
} // namespace opencv_test
