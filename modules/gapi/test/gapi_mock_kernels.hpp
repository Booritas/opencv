// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include "api/gbackend_priv.hpp" // directly instantiate GBackend::Priv

namespace opencv_test
{
namespace {
    // FIXME: Currently every Kernel implementation in this test file has
    // its own backend() method and it is incorrect! API classes should
    // provide it out of the box.

namespace I
{
    G_TYPED_KERNEL(Foo, <ncvslideio::GMat(ncvslideio::GMat)>, "test.kernels.foo")
    {
        static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in) { return in; }
    };

    G_TYPED_KERNEL(Bar, <ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GMat)>, "test.kernels.bar")
    {
        static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in, const ncvslideio::GMatDesc &) { return in; }
    };

    G_TYPED_KERNEL(Baz, <ncvslideio::GScalar(ncvslideio::GMat)>, "test.kernels.baz")
    {
        static ncvslideio::GScalarDesc outMeta(const ncvslideio::GMatDesc &) { return ncvslideio::empty_scalar_desc(); }
    };

    G_TYPED_KERNEL(Qux, <ncvslideio::GMat(ncvslideio::GMat, ncvslideio::GScalar)>, "test.kernels.qux")
    {
        static ncvslideio::GMatDesc outMeta(const ncvslideio::GMatDesc &in, const ncvslideio::GScalarDesc &) { return in; }
    };

    G_TYPED_KERNEL(Quux, <ncvslideio::GMat(ncvslideio::GScalar, ncvslideio::GMat)>, "test.kernels.quux")
    {
        static ncvslideio::GMatDesc outMeta(const ncvslideio::GScalarDesc &, const ncvslideio::GMatDesc& in) { return in; }
    };
}

// Kernel implementations for imaginary Jupiter device
namespace Jupiter
{
    namespace detail
    {
        static ncvslideio::gapi::GBackend backend(std::make_shared<ncvslideio::gapi::GBackend::Priv>());
    }

    inline ncvslideio::gapi::GBackend backend() { return detail::backend; }

    GAPI_OCV_KERNEL(Foo, I::Foo)
    {
        static void run(const ncvslideio::Mat &, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Bar, I::Bar)
    {
        static void run(const ncvslideio::Mat &, const ncvslideio::Mat &, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Baz, I::Baz)
    {
        static void run(const ncvslideio::Mat &, ncvslideio::Scalar &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Qux, I::Qux)
    {
        static void run(const ncvslideio::Mat &, const ncvslideio::Scalar&, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };

    GAPI_OCV_KERNEL(Quux, I::Quux)
    {
        static void run(const ncvslideio::Scalar&, const ncvslideio::Mat&, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
} // namespace Jupiter

// Kernel implementations for imaginary Saturn device
namespace Saturn
{
    namespace detail
    {
        static ncvslideio::gapi::GBackend backend(std::make_shared<ncvslideio::gapi::GBackend::Priv>());
    }

    inline ncvslideio::gapi::GBackend backend() { return detail::backend; }

    GAPI_OCV_KERNEL(Foo, I::Foo)
    {
        static void run(const ncvslideio::Mat &, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Bar, I::Bar)
    {
        static void run(const ncvslideio::Mat &, const ncvslideio::Mat &, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Baz, I::Baz)
    {
        static void run(const ncvslideio::Mat &, ncvslideio::Scalar &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Qux, I::Qux)
    {
        static void run(const ncvslideio::Mat &, const ncvslideio::Scalar&, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };

    GAPI_OCV_KERNEL(Quux, I::Quux)
    {
        static void run(const ncvslideio::Scalar&, const ncvslideio::Mat&, ncvslideio::Mat &) { /*Do nothing*/ }
        static ncvslideio::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
} // namespace Saturn
} // anonymous namespace
} // namespace opencv_test
