// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#include "../perf_precomp.hpp"
#include "../common/gapi_core_perf_tests.hpp"
#include <opencv2/gapi/cpu/core.hpp>

#define CORE_CPU ncvslideio::gapi::core::cpu::kernels()

namespace opencv_test
{

INSTANTIATE_TEST_CASE_P(PhasePerfTestFluid, PhasePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_32FC1, CV_64FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SqrtPerfTestFluid, SqrtPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_32FC1, CV_64FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AddPerfTestCPU, AddPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AddCPerfTestCPU, AddCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SubPerfTestCPU, SubPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SubCPerfTestCPU, SubCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SubRCPerfTestCPU, SubRCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MulPerfTestCPU, MulPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(2.0),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MulDoublePerfTestCPU, MulDoublePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MulCPerfTestCPU, MulCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivPerfTestCPU, DivPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_16S, CV_32F),
            Values(2.3),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivCPerfTestCPU, DivCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(1.0),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(DivRCPerfTestCPU, DivRCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(1.0),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MaskPerfTestCPU, MaskPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_16UC1, CV_16SC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MeanPerfTestCPU, MeanPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Polar2CartPerfTestCPU, Polar2CartPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Cart2PolarPerfTestCPU, Cart2PolarPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CmpPerfTestCPU, CmpPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CmpWithScalarPerfTestCPU, CmpWithScalarPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(BitwisePerfTestCPU, BitwisePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(AND, OR, XOR),
            testing::Bool(),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(BitwiseNotPerfTestCPU, BitwiseNotPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SelectPerfTestCPU, SelectPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MinPerfTestCPU, MinPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(MaxPerfTestCPU, MaxPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffPerfTestCPU, AbsDiffPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AbsDiffCPerfTestCPU, AbsDiffCPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SumPerfTestCPU, SumPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CountNonZeroPerfTestCPU, CountNonZeroPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(AddWeightedPerfTestCPU, AddWeightedPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(-1, CV_8U, CV_16U, CV_32F),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(NormPerfTestCPU, NormPerfTest,
    Combine(Values(AbsToleranceScalar(0.0).to_compare_f()),
            Values(NORM_INF, NORM_L1, NORM_L2),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(IntegralPerfTestCPU, IntegralPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestCPU, ThresholdPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::THRESH_BINARY, ncvslideio::THRESH_BINARY_INV, ncvslideio::THRESH_TRUNC, ncvslideio::THRESH_TOZERO, ncvslideio::THRESH_TOZERO_INV),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ThresholdPerfTestCPU, ThresholdOTPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1),
            Values(ncvslideio::THRESH_OTSU, ncvslideio::THRESH_TRIANGLE),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(InRangePerfTestCPU, InRangePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Split3PerfTestCPU, Split3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Split4PerfTestCPU, Split4PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Merge3PerfTestCPU, Merge3PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8U),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(Merge4PerfTestCPU, Merge4PerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(RemapPerfTestCPU, RemapPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(FlipPerfTestCPU, FlipPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(0, 1, -1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CropPerfTestCPU, CropPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::Rect(10, 8, 20, 35), ncvslideio::Rect(4, 10, 37, 50)),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(CopyPerfTestCPU, CopyPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorPerfTestCPU, ConcatHorPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatHorVecPerfTestCPU, ConcatHorVecPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertPerfTestCPU, ConcatVertPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConcatVertVecPerfTestCPU, ConcatVertVecPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestCPU, LUTPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC1, CV_8UC3),
            Values(CV_8UC1),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(LUTPerfTestCustomCPU, LUTPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC3),
            Values(CV_8UC3),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ConvertToPerfTestCPU, ConvertToPerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(CV_8UC3, CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1),
            Values(CV_8U, CV_16U, CV_16S, CV_32F),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(2.5, 1.0),
            Values(0.0),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(KMeansNDPerfTestCPU, KMeansNDPerfTest,
    Combine(Values(ncvslideio::Size(1, 20),
                    ncvslideio::Size(16, 4096)),
            Values(AbsTolerance(0.01).to_compare_obj()),
            Values(5, 15),
            Values(ncvslideio::KMEANS_RANDOM_CENTERS,
                    ncvslideio::KMEANS_PP_CENTERS,
                    ncvslideio::KMEANS_RANDOM_CENTERS | ncvslideio::KMEANS_USE_INITIAL_LABELS,
                    ncvslideio::KMEANS_PP_CENTERS     | ncvslideio::KMEANS_USE_INITIAL_LABELS),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(KMeans2DPerfTestCPU, KMeans2DPerfTest,
    Combine(Values(20, 4096),
            Values(5, 15),
            Values(ncvslideio::KMEANS_RANDOM_CENTERS,
                    ncvslideio::KMEANS_PP_CENTERS,
                    ncvslideio::KMEANS_RANDOM_CENTERS | ncvslideio::KMEANS_USE_INITIAL_LABELS,
                    ncvslideio::KMEANS_PP_CENTERS     | ncvslideio::KMEANS_USE_INITIAL_LABELS),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(KMeans3DPerfTestCPU, KMeans3DPerfTest,
    Combine(Values(20, 4096),
            Values(5, 15),
            Values(ncvslideio::KMEANS_RANDOM_CENTERS,
                    ncvslideio::KMEANS_PP_CENTERS,
                    ncvslideio::KMEANS_RANDOM_CENTERS | ncvslideio::KMEANS_USE_INITIAL_LABELS,
                    ncvslideio::KMEANS_PP_CENTERS     | ncvslideio::KMEANS_USE_INITIAL_LABELS),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(TransposePerfTestCPU, TransposePerfTest,
    Combine(Values(AbsExact().to_compare_f()),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1,
                    CV_8UC2, CV_16UC2, CV_16SC2, CV_32FC2,
                    CV_8UC3, CV_16UC3, CV_16SC3, CV_32FC3),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ParseSSDBLPerfTestCPU, ParseSSDBLPerfTest,
    Combine(Values(sz720p, sz1080p),
            Values(0.3f, 0.7f),
            Values(0, 1),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ParseSSDPerfTestCPU, ParseSSDPerfTest,
    Combine(Values(sz720p, sz1080p),
            Values(0.3f, 0.7f),
            testing::Bool(),
            testing::Bool(),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(ParseYoloPerfTestCPU, ParseYoloPerfTest,
    Combine(Values(sz720p, sz1080p),
            Values(0.3f, 0.7f),
            Values(0.5),
            Values(7, 80),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SizePerfTestCPU, SizePerfTest,
    Combine(Values(CV_8UC1, CV_8UC3, CV_32FC1),
            Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));

INSTANTIATE_TEST_CASE_P(SizeRPerfTestCPU, SizeRPerfTest,
    Combine(Values(szSmall128, szVGA, sz720p, sz1080p),
            Values(ncvslideio::compile_args(CORE_CPU))));
} // opencv_test
