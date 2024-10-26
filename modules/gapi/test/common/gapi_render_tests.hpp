// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_HPP
#define OPENCV_GAPI_RENDER_TESTS_HPP

#include "gapi_tests_common.hpp"

namespace opencv_test
{

template<typename ...SpecificParams>
struct RenderParams : public Params<SpecificParams...>
{
    using common_params_t = std::tuple<ncvslideio::Size>;
    using specific_params_t = std::tuple<SpecificParams...>;
    using params_t = std::tuple<ncvslideio::Size, SpecificParams...>;

    static constexpr const size_t common_params_size = std::tuple_size<common_params_t>::value;
    static constexpr const size_t specific_params_size = std::tuple_size<specific_params_t>::value;

    template<size_t I>
    static const typename std::tuple_element<I, common_params_t>::type&
    getCommon(const params_t& t)
    {
        static_assert(I < common_params_size, "Index out of range");
        return std::get<I>(t);
    }

    template<size_t I>
    static const typename std::tuple_element<I, specific_params_t>::type&
    getSpecific(const params_t& t)
    {
        static_assert(specific_params_size > 0,
            "Impossible to call this function: no specific parameters specified");
        static_assert(I < specific_params_size, "Index out of range");
        return std::get<common_params_size + I>(t);
    }
};

template<typename ...SpecificParams>
struct RenderTestBase : public TestWithParam<typename RenderParams<SpecificParams...>::params_t>
{
    using AllParams = RenderParams<SpecificParams...>;

    // Get common (pre-defined) parameter value by index
    template<size_t I>
    inline auto getCommonParam() const
        -> decltype(AllParams::template getCommon<I>(this->GetParam()))
    {
        return AllParams::template getCommon<I>(this->GetParam());
    }

    // Get specific (user-defined) parameter value by index
    template<size_t I>
    inline auto getSpecificParam() const
        -> decltype(AllParams::template getSpecific<I>(this->GetParam()))
    {
        return AllParams::template getSpecific<I>(this->GetParam());
    }

    ncvslideio::Size sz_ = getCommonParam<0>();
};

template <typename ...Args>
class RenderBGRTestBase : public RenderTestBase<Args...>
{
protected:
    void Init(const ncvslideio::Size& sz)
    {
        MatType type = CV_8UC3;

        ref_mat.create(sz, type);
        gapi_mat.create(sz, type);

        ncvslideio::randu(ref_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
        ref_mat.copyTo(gapi_mat);
    }

    ncvslideio::Mat gapi_mat, ref_mat;
};

template <typename ...Args>
class RenderNV12TestBase : public RenderTestBase<Args...>
{
protected:
    void Init(const ncvslideio::Size& sz)
    {
        auto create_rand_mats = [](const ncvslideio::Size& size, MatType type, ncvslideio::Mat& ref_mat, ncvslideio::Mat& gapi_mat) {
            ref_mat.create(size, type);
            ncvslideio::randu(ref_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
            ref_mat.copyTo(gapi_mat);
        };

        create_rand_mats(sz,     CV_8UC1, y_ref_mat  , y_gapi_mat);
        create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat , uv_gapi_mat);
    }

    ncvslideio::Mat y_ref_mat, uv_ref_mat, y_gapi_mat, uv_gapi_mat;
};

ncvslideio::Scalar cvtBGRToYUVC(const ncvslideio::Scalar& bgr);
void drawMosaicRef(const ncvslideio::Mat& mat, const ncvslideio::Rect &rect, int cellSz);
void blendImageRef(ncvslideio::Mat& mat,
                   const ncvslideio::Point& org,
                   const ncvslideio::Mat& img,
                   const ncvslideio::Mat& alpha);

#define GAPI_RENDER_TEST_FIXTURE_NV12(Fixture, API, Number, ...)  \
struct Fixture : public RenderNV12TestBase API {                  \
    __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__))   \
    Fixture() {                                                   \
        Init(sz_);                                                \
    }                                                             \
};

#define GAPI_RENDER_TEST_FIXTURE_BGR(Fixture, API, Number, ...)  \
struct Fixture : public RenderBGRTestBase API {                  \
    __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__))   \
    Fixture() {                                                   \
        Init(sz_);                                                \
    }                                                             \
};

#define GET_VA_ARGS(...) __VA_ARGS__
#define GAPI_RENDER_TEST_FIXTURES(Fixture, API, Number, ...)                    \
    GAPI_RENDER_TEST_FIXTURE_BGR(RenderBGR##Fixture,   GET_VA_ARGS(API), Number, __VA_ARGS__) \
    GAPI_RENDER_TEST_FIXTURE_NV12(RenderNV12##Fixture, GET_VA_ARGS(API), Number, __VA_ARGS__) \
    GAPI_RENDER_TEST_FIXTURE_NV12(RenderMFrame##Fixture, GET_VA_ARGS(API), Number, __VA_ARGS__) \


using Points = std::vector<ncvslideio::Point>;
GAPI_RENDER_TEST_FIXTURES(TestTexts,     FIXTURE_API(std::string, ncvslideio::Point, double, ncvslideio::Scalar), 4, text, org, fs, color)
GAPI_RENDER_TEST_FIXTURES(TestRects,     FIXTURE_API(ncvslideio::Rect, ncvslideio::Scalar, int),                  3, rect, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestCircles,   FIXTURE_API(ncvslideio::Point, int, ncvslideio::Scalar, int),            4, center, radius, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestLines,     FIXTURE_API(ncvslideio::Point, ncvslideio::Point, ncvslideio::Scalar, int),      4, pt1, pt2, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestMosaics,   FIXTURE_API(ncvslideio::Rect, int, int),                         3, mos, cellsz, decim)
GAPI_RENDER_TEST_FIXTURES(TestImages,    FIXTURE_API(ncvslideio::Rect, ncvslideio::Scalar, double),               3, rect, color, transparency)
GAPI_RENDER_TEST_FIXTURES(TestPolylines, FIXTURE_API(Points, ncvslideio::Scalar, int),                    3, points, color, thick)

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_HPP
