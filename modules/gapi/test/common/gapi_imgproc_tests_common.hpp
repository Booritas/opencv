// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_IMGPROC_TESTS_COMMON_HPP
#define OPENCV_GAPI_IMGPROC_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/imgproc.hpp"

#include <opencv2/imgproc.hpp>

namespace opencv_test
{
// Draw random ellipses on given ncvslideio::Mat of given size and type
static void initMatForFindingContours(ncvslideio::Mat& mat, const ncvslideio::Size& sz, const int type)
{
    ncvslideio::RNG& rng = theRNG();
    mat = ncvslideio::Mat(sz, type, ncvslideio::Scalar::all(0));
    const size_t numEllipses = rng.uniform(1, 10);

    for( size_t i = 0; i < numEllipses; i++ )
    {
        ncvslideio::Point center;
        ncvslideio::Size  axes;
        center.x    = rng.uniform(0, sz.width);
        center.y    = rng.uniform(0, sz.height);
        axes.width  = rng.uniform(2, sz.width);
        axes.height = rng.uniform(2, sz.height);
        const int    color = rng.uniform(1, 256);
        const double angle = rng.uniform(0., 180.);
        ncvslideio::ellipse(mat, center, axes, angle, 0., 360., color, 1, FILLED);
    }
}

enum OptionalFindContoursOutput {NONE, HIERARCHY};

template<OptionalFindContoursOutput optional = NONE>
ncvslideio::GComputation findContoursTestGAPI(const ncvslideio::Mat& in, const ncvslideio::RetrievalModes mode,
                                      const ncvslideio::ContourApproximationModes method,
                                      ncvslideio::GCompileArgs&& args,
                                      std::vector<std::vector<ncvslideio::Point>>& out_cnts_gapi,
                                      std::vector<ncvslideio::Vec4i>& /*out_hier_gapi*/,
                                      const ncvslideio::Point& offset = ncvslideio::Point())
{
    ncvslideio::GMat g_in;
    ncvslideio::GOpaque<ncvslideio::Point> gOffset;
    ncvslideio::GArray<ncvslideio::GArray<ncvslideio::Point>> outCts;
    outCts = ncvslideio::gapi::findContours(g_in, mode, method, gOffset);
    ncvslideio::GComputation c(GIn(g_in, gOffset), GOut(outCts));
    c.apply(gin(in, offset), gout(out_cnts_gapi), std::move(args));
    return c;
}

template<> ncvslideio::GComputation findContoursTestGAPI<HIERARCHY> (
    const ncvslideio::Mat& in, const ncvslideio::RetrievalModes mode, const ncvslideio::ContourApproximationModes method,
    ncvslideio::GCompileArgs&& args, std::vector<std::vector<ncvslideio::Point>>& out_cnts_gapi,
    std::vector<ncvslideio::Vec4i>& out_hier_gapi, const ncvslideio::Point& offset)
{
    ncvslideio::GMat g_in;
    ncvslideio::GOpaque<ncvslideio::Point> gOffset;
    ncvslideio::GArray<ncvslideio::GArray<ncvslideio::Point>> outCts;
    ncvslideio::GArray<ncvslideio::Vec4i> outHier;
    std::tie(outCts, outHier) = ncvslideio::gapi::findContoursH(g_in, mode, method, gOffset);
    ncvslideio::GComputation c(GIn(g_in, gOffset), GOut(outCts, outHier));
    c.apply(gin(in, offset), gout(out_cnts_gapi, out_hier_gapi), std::move(args));
    return c;
}

template<OptionalFindContoursOutput optional = NONE>
void findContoursTestOpenCVCompare(const ncvslideio::Mat& in, const ncvslideio::RetrievalModes mode,
                                   const ncvslideio::ContourApproximationModes method,
                                   const std::vector<std::vector<ncvslideio::Point>>& out_cnts_gapi,
                                   const std::vector<ncvslideio::Vec4i>&              out_hier_gapi,
                                   const CompareMats& cmpF, const ncvslideio::Point& offset = ncvslideio::Point())
{
    // OpenCV code /////////////////////////////////////////////////////////////
    std::vector<std::vector<ncvslideio::Point>> out_cnts_ocv;
    std::vector<ncvslideio::Vec4i>              out_hier_ocv;
    ncvslideio::findContours(in, out_cnts_ocv, out_hier_ocv, mode, method, offset);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(out_cnts_gapi.size() == out_cnts_ocv.size());

    ncvslideio::Mat out_mat_ocv  = ncvslideio::Mat(ncvslideio::Size{ in.cols, in.rows }, in.type(), ncvslideio::Scalar::all(0));
    ncvslideio::Mat out_mat_gapi = ncvslideio::Mat(ncvslideio::Size{ in.cols, in.rows }, in.type(), ncvslideio::Scalar::all(0));
    ncvslideio::fillPoly(out_mat_ocv,  out_cnts_ocv,  ncvslideio::Scalar::all(1));
    ncvslideio::fillPoly(out_mat_gapi, out_cnts_gapi, ncvslideio::Scalar::all(1));
    EXPECT_TRUE(cmpF(out_mat_ocv, out_mat_gapi));
    if (optional == HIERARCHY)
    {
        EXPECT_TRUE(out_hier_ocv.size() == out_hier_gapi.size());
        EXPECT_TRUE(AbsExactVector<ncvslideio::Vec4i>().to_compare_f()(out_hier_ocv, out_hier_gapi));
    }
}

template<OptionalFindContoursOutput optional = NONE>
void findContoursTestBody(const ncvslideio::Size& sz, const MatType2& type, const ncvslideio::RetrievalModes mode,
                          const ncvslideio::ContourApproximationModes method, const CompareMats& cmpF,
                          ncvslideio::GCompileArgs&& args, const ncvslideio::Point& offset = ncvslideio::Point())
{
    ncvslideio::Mat in;
    initMatForFindingContours(in, sz, type);

    std::vector<std::vector<ncvslideio::Point>> out_cnts_gapi;
    std::vector<ncvslideio::Vec4i>              out_hier_gapi;
    findContoursTestGAPI<optional>(in, mode, method, std::move(args), out_cnts_gapi, out_hier_gapi,
                                   offset);
    findContoursTestOpenCVCompare<optional>(in, mode, method, out_cnts_gapi, out_hier_gapi, cmpF,
                                            offset);
}

//-------------------------------------------------------------------------------------------------

template<typename In>
static ncvslideio::GComputation boundingRectTestGAPI(const In& in, ncvslideio::GCompileArgs&& args,
                                             ncvslideio::Rect& out_rect_gapi)
{
    ncvslideio::detail::g_type_of_t<In> g_in;
    auto out = ncvslideio::gapi::boundingRect(g_in);
    ncvslideio::GComputation c(ncvslideio::GIn(g_in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in), ncvslideio::gout(out_rect_gapi), std::move(args));
    return c;
}

template<typename In>
static void boundingRectTestOpenCVCompare(const In& in, const ncvslideio::Rect& out_rect_gapi,
                                          const CompareRects& cmpF)
{
    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::Rect out_rect_ocv = ncvslideio::boundingRect(in);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_rect_gapi, out_rect_ocv));
}

template<typename In>
static void boundingRectTestBody(const In& in, const CompareRects& cmpF, ncvslideio::GCompileArgs&& args)
{
    ncvslideio::Rect out_rect_gapi;
    boundingRectTestGAPI(in, std::move(args), out_rect_gapi);
    boundingRectTestOpenCVCompare(in, out_rect_gapi, cmpF);
}

//-------------------------------------------------------------------------------------------------

template<typename In>
static ncvslideio::GComputation fitLineTestGAPI(const In& in, const ncvslideio::DistanceTypes distType,
                                        ncvslideio::GCompileArgs&& args, ncvslideio::Vec4f& out_vec_gapi)
{
    const double paramDefault = 0., repsDefault = 0., aepsDefault = 0.;

    ncvslideio::detail::g_type_of_t<In> g_in;
    auto out = ncvslideio::gapi::fitLine2D(g_in, distType, paramDefault, repsDefault, aepsDefault);
    ncvslideio::GComputation c(ncvslideio::GIn(g_in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in), ncvslideio::gout(out_vec_gapi), std::move(args));
    return c;
}

template<typename In>
static ncvslideio::GComputation fitLineTestGAPI(const In& in, const ncvslideio::DistanceTypes distType,
                                        ncvslideio::GCompileArgs&& args, ncvslideio::Vec6f& out_vec_gapi)
{
    const double paramDefault = 0., repsDefault = 0., aepsDefault = 0.;

    ncvslideio::detail::g_type_of_t<In> g_in;
    auto out = ncvslideio::gapi::fitLine3D(g_in, distType, paramDefault, repsDefault, aepsDefault);
    ncvslideio::GComputation c(ncvslideio::GIn(g_in), ncvslideio::GOut(out));
    c.apply(ncvslideio::gin(in), ncvslideio::gout(out_vec_gapi), std::move(args));
    return c;
}

template<typename In, int dim>
static void fitLineTestOpenCVCompare(const In& in, const ncvslideio::DistanceTypes distType,
                                     const ncvslideio::Vec<float, dim>& out_vec_gapi,
                                     const CompareVecs<float, dim>& cmpF)
{
    const double paramDefault = 0., repsDefault = 0., aepsDefault = 0.;

    // OpenCV code /////////////////////////////////////////////////////////////
    ncvslideio::Vec<float, dim> out_vec_ocv;
    ncvslideio::fitLine(in, out_vec_ocv, distType, paramDefault, repsDefault, aepsDefault);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(cmpF(out_vec_gapi, out_vec_ocv));
}

template<typename In, int dim>
static void fitLineTestBody(const In& in, const ncvslideio::DistanceTypes distType,
                            const CompareVecs<float, dim>& cmpF, ncvslideio::GCompileArgs&& args)
{
    ncvslideio::Vec<float, dim> out_vec_gapi;
    fitLineTestGAPI(in, distType, std::move(args), out_vec_gapi);
    fitLineTestOpenCVCompare(in, distType, out_vec_gapi, cmpF);
}
} // namespace opencv_test

#endif // OPENCV_GAPI_IMGPROC_TESTS_COMMON_HPP
