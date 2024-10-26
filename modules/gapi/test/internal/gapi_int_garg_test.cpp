// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

namespace opencv_test {
// Tests on T/Spec/Kind matching ///////////////////////////////////////////////
// {{

template<class T, ncvslideio::detail::ArgKind Exp>
struct Expected
{
    using type = T;
    static const constexpr ncvslideio::detail::ArgKind kind = Exp;
};

template<typename T>
struct GArgKind: public ::testing::Test
{
    using Type = typename T::type;
    const ncvslideio::detail::ArgKind Kind = T::kind;
};

// The reason here is to _manually_ list types and their kinds
// (and NOT reuse ncvslideio::detail::ArgKind::Traits<>, since it is a subject of testing)
using GArg_Test_Types = ::testing::Types
   <
  // G-API types
     Expected<ncvslideio::GMat,                 ncvslideio::detail::ArgKind::GMAT>
   , Expected<ncvslideio::GMatP,                ncvslideio::detail::ArgKind::GMATP>
   , Expected<ncvslideio::GFrame,               ncvslideio::detail::ArgKind::GFRAME>
   , Expected<ncvslideio::GScalar,              ncvslideio::detail::ArgKind::GSCALAR>
   , Expected<ncvslideio::GArray<int>,          ncvslideio::detail::ArgKind::GARRAY>
   , Expected<ncvslideio::GArray<float>,        ncvslideio::detail::ArgKind::GARRAY>
   , Expected<ncvslideio::GArray<ncvslideio::Point>,    ncvslideio::detail::ArgKind::GARRAY>
   , Expected<ncvslideio::GArray<ncvslideio::Rect>,     ncvslideio::detail::ArgKind::GARRAY>
   , Expected<ncvslideio::GOpaque<int>,         ncvslideio::detail::ArgKind::GOPAQUE>
   , Expected<ncvslideio::GOpaque<float>,       ncvslideio::detail::ArgKind::GOPAQUE>
   , Expected<ncvslideio::GOpaque<ncvslideio::Point>,   ncvslideio::detail::ArgKind::GOPAQUE>
   , Expected<ncvslideio::GOpaque<ncvslideio::Rect>,    ncvslideio::detail::ArgKind::GOPAQUE>

 // Built-in types
   , Expected<int,                      ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<float,                    ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<int*,                     ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<ncvslideio::Point,                ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<std::string,              ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<ncvslideio::Mat,                  ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<std::vector<int>,         ncvslideio::detail::ArgKind::OPAQUE_VAL>
   , Expected<std::vector<ncvslideio::Point>,   ncvslideio::detail::ArgKind::OPAQUE_VAL>
   >;

TYPED_TEST_CASE(GArgKind, GArg_Test_Types);

TYPED_TEST(GArgKind, LocalVar)
{
    typename TestFixture::Type val{};
    ncvslideio::GArg arg(val);
    EXPECT_EQ(TestFixture::Kind, arg.kind);
}

TYPED_TEST(GArgKind, ConstLocalVar)
{
    const typename TestFixture::Type val{};
    ncvslideio::GArg arg(val);
    EXPECT_EQ(TestFixture::Kind, arg.kind);
}

TYPED_TEST(GArgKind, RValue)
{
    ncvslideio::GArg arg = ncvslideio::GArg(typename TestFixture::Type());
    EXPECT_EQ(TestFixture::Kind, arg.kind);
}

////////////////////////////////////////////////////////////////////////////////

TEST(GArg, HasWrap)
{
    static_assert(!ncvslideio::detail::has_custom_wrap<ncvslideio::GMat>::value,
                  "GMat has no custom marshalling logic");
    static_assert(!ncvslideio::detail::has_custom_wrap<ncvslideio::GScalar>::value,
                  "GScalar has no custom marshalling logic");

    static_assert(ncvslideio::detail::has_custom_wrap<ncvslideio::GArray<int> >::value,
                  "GArray<int> has custom marshalling logic");
    static_assert(ncvslideio::detail::has_custom_wrap<ncvslideio::GArray<std::string> >::value,
                  "GArray<int> has custom marshalling logic");

    static_assert(ncvslideio::detail::has_custom_wrap<ncvslideio::GOpaque<int> >::value,
                  "GOpaque<int> has custom marshalling logic");
    static_assert(ncvslideio::detail::has_custom_wrap<ncvslideio::GOpaque<std::string> >::value,
                  "GOpaque<int> has custom marshalling logic");
}

TEST(GArg, GArrayU)
{
    // Placing a GArray<T> into GArg automatically strips it to GArrayU
    ncvslideio::GArg arg1 = ncvslideio::GArg(ncvslideio::GArray<int>());
    EXPECT_NO_THROW(arg1.get<ncvslideio::detail::GArrayU>());

    ncvslideio::GArg arg2 = ncvslideio::GArg(ncvslideio::GArray<ncvslideio::Point>());
    EXPECT_NO_THROW(arg2.get<ncvslideio::detail::GArrayU>());
}

TEST(GArg, GOpaqueU)
{
    // Placing a GOpaque<T> into GArg automatically strips it to GOpaqueU
    ncvslideio::GArg arg1 = ncvslideio::GArg(ncvslideio::GOpaque<int>());
    EXPECT_NO_THROW(arg1.get<ncvslideio::detail::GOpaqueU>());

    ncvslideio::GArg arg2 = ncvslideio::GArg(ncvslideio::GOpaque<ncvslideio::Point>());
    EXPECT_NO_THROW(arg2.get<ncvslideio::detail::GOpaqueU>());
}
} // namespace opencv_test
