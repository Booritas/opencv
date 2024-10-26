// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_CORE_TESTS_COMMON_HPP
#define OPENCV_GAPI_CORE_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include "../../include/opencv2/gapi/core.hpp"

#include <opencv2/core.hpp>

namespace opencv_test
{
namespace
{
template <typename Elem, typename CmpF>
inline bool compareKMeansOutputs(const std::vector<Elem>& outGAPI,
                                 const std::vector<Elem>& outOCV,
                                 const CmpF& = AbsExact().to_compare_obj())
{
    return AbsExactVector<Elem>().to_compare_f()(outGAPI, outOCV);
}

inline bool compareKMeansOutputs(const ncvslideio::Mat& outGAPI,
                                 const ncvslideio::Mat& outOCV,
                                 const CompareMats& cmpF)
{
    return cmpF(outGAPI, outOCV);
}
}

// Overload with initializing the labels
template<typename Labels, typename In>
ncvslideio::GComputation kmeansTestGAPI(const In& in, const Labels& bestLabels, const int K,
                                const ncvslideio::KmeansFlags flags, ncvslideio::GCompileArgs&& args,
                                double& compact_gapi, Labels& labels_gapi, In& centers_gapi)
{
    const ncvslideio::TermCriteria criteria(ncvslideio::TermCriteria::MAX_ITER + ncvslideio::TermCriteria::EPS, 30, 0);
    const int attempts = 1;

    ncvslideio::detail::g_type_of_t<In> gIn, centers;
    ncvslideio::GOpaque<double> compactness;
    ncvslideio::detail::g_type_of_t<Labels> inLabels, outLabels;
    std::tie(compactness, outLabels, centers) =
        ncvslideio::gapi::kmeans(gIn, K, inLabels, criteria, attempts, flags);
    ncvslideio::GComputation c(ncvslideio::GIn(gIn, inLabels), ncvslideio::GOut(compactness, outLabels, centers));
    c.apply(ncvslideio::gin(in, bestLabels), ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi),
            std::move(args));
    return c;
}

// Overload for vector<Point> tests w/o initializing the labels
template<typename Pt>
ncvslideio::GComputation kmeansTestGAPI(const std::vector<Pt>& in, const int K,
                                const ncvslideio::KmeansFlags flags, ncvslideio::GCompileArgs&& args,
                                double& compact_gapi, std::vector<int>& labels_gapi,
                                std::vector<Pt>& centers_gapi)
{
    const ncvslideio::TermCriteria criteria(ncvslideio::TermCriteria::MAX_ITER + ncvslideio::TermCriteria::EPS, 30, 0);
    const int attempts = 1;

    ncvslideio::GArray<Pt> gIn, centers;
    ncvslideio::GOpaque<double> compactness;
    ncvslideio::GArray<int> inLabels(std::vector<int>{}), outLabels;
    std::tie(compactness, outLabels, centers) =
        ncvslideio::gapi::kmeans(gIn, K, inLabels, criteria, attempts, flags);
    ncvslideio::GComputation c(ncvslideio::GIn(gIn), ncvslideio::GOut(compactness, outLabels, centers));
    c.apply(ncvslideio::gin(in), ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi), std::move(args));
    return c;
}

// Overload for Mat tests w/o initializing the labels
static ncvslideio::GComputation kmeansTestGAPI(const ncvslideio::Mat& in, const int K,
                                       const ncvslideio::KmeansFlags flags, ncvslideio::GCompileArgs&& args,
                                       double& compact_gapi, ncvslideio::Mat& labels_gapi,
                                       ncvslideio::Mat& centers_gapi)
{
    const ncvslideio::TermCriteria criteria(ncvslideio::TermCriteria::MAX_ITER + ncvslideio::TermCriteria::EPS, 30, 0);
    const int attempts = 1;

    ncvslideio::GMat gIn, centers, labels;
    ncvslideio::GOpaque<double> compactness;
    std::tie(compactness, labels, centers) = ncvslideio::gapi::kmeans(gIn, K, criteria, attempts, flags);
    ncvslideio::GComputation c(ncvslideio::GIn(gIn), ncvslideio::GOut(compactness, labels, centers));
    c.apply(ncvslideio::gin(in), ncvslideio::gout(compact_gapi, labels_gapi, centers_gapi), std::move(args));
    return c;
}

template<typename Pt>
void kmeansTestValidate(const ncvslideio::Size& sz, const MatType2&, const int K,
                        const double compact_gapi, const std::vector<int>& labels_gapi,
                        const std::vector<Pt>& centers_gapi)
{
    const int amount = sz.height;
    // Validation
    EXPECT_GE(compact_gapi, 0.);
    EXPECT_EQ(labels_gapi.size(), static_cast<size_t>(amount));
    EXPECT_EQ(centers_gapi.size(), static_cast<size_t>(K));
}

static void kmeansTestValidate(const ncvslideio::Size& sz, const MatType2& type, const int K,
                               const double compact_gapi, const ncvslideio::Mat& labels_gapi,
                               const ncvslideio::Mat& centers_gapi)
{
    const int chan   = (type >> CV_CN_SHIFT) + 1;
    const int amount = sz.height != 1 ? sz.height : sz.width;
    const int dim    = sz.height != 1 ? sz.width * chan : chan;
    // Validation
    EXPECT_GE(compact_gapi, 0.);
    EXPECT_FALSE(labels_gapi.empty());
    EXPECT_FALSE(centers_gapi.empty());
    EXPECT_EQ(labels_gapi.rows, amount);
    EXPECT_EQ(labels_gapi.cols, 1);
    EXPECT_EQ(centers_gapi.rows, K);
    EXPECT_EQ(centers_gapi.cols, dim);
}

template<typename Labels, typename In>
void kmeansTestOpenCVCompare(const In& in, const Labels& bestLabels, const int K,
                             const ncvslideio::KmeansFlags flags, const double compact_gapi,
                             const Labels& labels_gapi, const In& centers_gapi,
                             const CompareMats& cmpF = AbsExact().to_compare_obj())
{
    const ncvslideio::TermCriteria criteria(ncvslideio::TermCriteria::MAX_ITER + ncvslideio::TermCriteria::EPS, 30, 0);
    const int attempts = 1;
    Labels labels_ocv;
    In centers_ocv;
    { // step to generalize ncvslideio::Mat & std::vector cases of bestLabels' types
        ncvslideio::Mat bestLabelsMat(bestLabels);
        bestLabelsMat.copyTo(labels_ocv);
    }
    // OpenCV code /////////////////////////////////////////////////////////////
    double compact_ocv = ncvslideio::kmeans(in, K, labels_ocv, criteria, attempts, flags, centers_ocv);
    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(compact_gapi == compact_ocv);
    EXPECT_TRUE(compareKMeansOutputs(labels_gapi, labels_ocv, cmpF));
    EXPECT_TRUE(compareKMeansOutputs(centers_gapi, centers_ocv, cmpF));
}

// If an input type is ncvslideio::Mat, labels' type is also ncvslideio::Mat;
// in other cases, their type has to be std::vector<int>
template<typename In>
using KMeansLabelType = typename std::conditional<std::is_same<In, ncvslideio::Mat>::value,
                                                  ncvslideio::Mat,
                                                  std::vector<int>
                                                 >::type;
template<typename In, typename Labels = KMeansLabelType<In> >
void kmeansTestBody(const In& in, const ncvslideio::Size& sz, const MatType2& type, const int K,
                    const ncvslideio::KmeansFlags flags, ncvslideio::GCompileArgs&& args,
                    const CompareMats& cmpF = AbsExact().to_compare_obj())
{
    double compact_gapi = -1.;
    Labels labels_gapi;
    In centers_gapi;
    if (flags & ncvslideio::KMEANS_USE_INITIAL_LABELS)
    {
        Labels bestLabels;
        { // step to generalize ncvslideio::Mat & std::vector cases of bestLabels' types
            const int amount = (sz.height != 1 || sz.width == -1) ? sz.height : sz.width;
            ncvslideio::Mat bestLabelsMat(ncvslideio::Size{1, amount}, CV_32SC1);
            ncvslideio::randu(bestLabelsMat, 0, K);
            bestLabelsMat.copyTo(bestLabels);
        }
        kmeansTestGAPI(in, bestLabels, K, flags, std::move(args), compact_gapi, labels_gapi,
                       centers_gapi);
        kmeansTestOpenCVCompare(in, bestLabels, K, flags, compact_gapi, labels_gapi,
                                centers_gapi, cmpF);
    }
    else
    {
        kmeansTestGAPI(in, K, flags, std::move(args), compact_gapi, labels_gapi, centers_gapi);
        kmeansTestValidate(sz, type, K, compact_gapi, labels_gapi, centers_gapi);
    }
}
} // namespace opencv_test

#endif // OPENCV_GAPI_CORE_TESTS_COMMON_HPP
