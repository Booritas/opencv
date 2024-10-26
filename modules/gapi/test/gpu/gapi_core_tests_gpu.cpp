// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "../test_precomp.hpp"
#include "../common/gapi_core_tests.hpp"

namespace
{
#define CORE_GPU [] () { return ncvslideio::compile_args(ncvslideio::gapi::use_only{ncvslideio::gapi::core::gpu::kernels()}); }
    const std::vector <ncvslideio::Size> in_sizes{ ncvslideio::Size(1280, 720), ncvslideio::Size(128, 128) };
}  // anonymous namespace

namespace opencv_test
{

// FIXME: Wut? See MulTestGPU/MathOpTest below (duplicate?)
INSTANTIATE_TEST_CASE_P(AddTestGPU, MathOpTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU),
                                Values(ADD, MUL),
                                testing::Bool(),
                                Values(1.0),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(MulTestGPU, MathOpTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU),
                                Values(MUL),
                                testing::Bool(),
                                Values(1.0, 0.5, 2.0),
                                Values(false)));

INSTANTIATE_TEST_CASE_P(SubTestGPU, MathOpTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU),
                                Values(SUB),
                                testing::Bool(),
                                Values (1.0),
                                testing::Bool()));

// FIXME: Accuracy test for DIV math operation fails on CV_8UC3 HD input ncvslideio::Mat, double-presicion
//        input ncvslideio::Scalar and CV_16U output ncvslideio::Mat when we also test reverse operation on Mac.
//        Accuracy test for DIV math operation fails on CV_8UC3 VGA input ncvslideio::Mat, double-presicion
//        input ncvslideio::Scalar and output ncvslideio::Mat having the SAME depth as input one when we also test
//        reverse operation on Mac.
//        It is oddly, but test doesn't fail if we have VGA CV_8UC3 input ncvslideio::Mat, double-precision
//        input ncvslideio::Scalar and output ncvslideio::Mat having explicitly specified CV_8U depth when we also
//        test reverse operation on Mac.
//        As failures are sporadic, disabling all instantiation cases for DIV operation.
//        Github ticket: https://github.com/opencv/opencv/issues/18373.
INSTANTIATE_TEST_CASE_P(DISABLED_DivTestGPU, MathOpTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU),
                                Values(DIV),
                                testing::Bool(),
                                Values (1.0, 0.5, 2.0),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(MulTestGPU, MulDoubleTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(DivTestGPU, DivTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(DivCTestGPU, DivCTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(MeanTestGPU, MeanTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

//TODO: mask test doesn't work
INSTANTIATE_TEST_CASE_P(DISABLED_MaskTestGPU, MaskTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(SelectTestGPU, SelectTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(Polar2CartGPU, Polar2CartTest,
                        Combine(Values(CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(CV_32FC1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(Cart2PolarGPU, Cart2PolarTest,
                        Combine(Values(CV_32FC1),
                                ValuesIn(in_sizes),
                                Values(CV_32FC1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(CompareTestGPU, CmpTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(CV_8U),
                                Values(CORE_GPU),
                                Values(CMP_EQ, CMP_GE, CMP_NE, CMP_GT, CMP_LT, CMP_LE),
                                testing::Bool(),
                                Values(AbsExact().to_compare_obj())));

INSTANTIATE_TEST_CASE_P(BitwiseTestGPU, BitwiseTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AND, OR, XOR),
                                testing::Bool()));

INSTANTIATE_TEST_CASE_P(BitwiseNotTestGPU, NotTest,
                        Combine(Values(CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(DISABLED_MinTestGPU, MinTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(DISABLED_MaxTestGPU, MaxTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(SumTestGPU, SumTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AbsToleranceScalar(1e-5).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(CountNonZeroTestGPU, CountNonZeroTest,
                        Combine(Values( CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AbsToleranceScalar(1e-5).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(AbsDiffTestGPU, AbsDiffTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(AbsDiffCTestGPU, AbsDiffCTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(AddWeightedTestGPU, AddWeightedTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values( -1, CV_8U, CV_16U, CV_32F ),
                                Values(CORE_GPU),
                                Values(Tolerance_FloatRel_IntAbs(1e-6, 1).to_compare_obj())));

INSTANTIATE_TEST_CASE_P(NormTestGPU, NormTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AbsToleranceScalar(1e-3).to_compare_obj()), //TODO: too relaxed?
                                Values(NORM_INF, NORM_L1, NORM_L2)));

INSTANTIATE_TEST_CASE_P(IntegralTestGPU, IntegralTest,
                        Combine(Values( CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(ThresholdTestGPU, ThresholdTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(ncvslideio::THRESH_BINARY, ncvslideio::THRESH_BINARY_INV, ncvslideio::THRESH_TRUNC,
                                       ncvslideio::THRESH_TOZERO, ncvslideio::THRESH_TOZERO_INV),
                                Values(ncvslideio::Scalar(0, 0, 0, 0),
                                       ncvslideio::Scalar(100, 100, 100, 100),
                                       ncvslideio::Scalar(255, 255, 255, 255))));

INSTANTIATE_TEST_CASE_P(ThresholdTestGPU, ThresholdOTTest,
                        Combine(Values(CV_8UC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(ncvslideio::THRESH_OTSU, ncvslideio::THRESH_TRIANGLE)));


INSTANTIATE_TEST_CASE_P(InRangeTestGPU, InRangeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(Split3TestGPU, Split3Test,
                        Combine(Values(CV_8UC3),
                                ValuesIn(in_sizes),
                                Values(CV_8UC1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(Split4TestGPU, Split4Test,
                        Combine(Values(CV_8UC4),
                                ValuesIn(in_sizes),
                                Values(CV_8UC1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(Merge3TestGPU, Merge3Test,
                        Combine(Values(CV_8UC1),
                                ValuesIn(in_sizes),
                                Values(CV_8UC3),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(Merge4TestGPU, Merge4Test,
                        Combine(Values(CV_8UC1),
                                ValuesIn(in_sizes),
                                Values(CV_8UC4),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(RemapTestGPU, RemapTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(FlipTestGPU, FlipTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(0,1,-1)));

INSTANTIATE_TEST_CASE_P(CropTestGPU, CropTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(ncvslideio::Rect(10, 8, 20, 35), ncvslideio::Rect(4, 10, 37, 50))));

INSTANTIATE_TEST_CASE_P(LUTTestGPU, LUTTest,
                        Combine(Values(CV_8UC1, CV_8UC3),
                                ValuesIn(in_sizes),
                                Values(CV_8UC1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(LUTTestCustomGPU, LUTTest,
                        Combine(Values(CV_8UC3),
                                ValuesIn(in_sizes),
                                Values(CV_8UC3),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(ConvertToGPU, ConvertToTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(CV_8U, CV_16U, CV_16S, CV_32F),
                                Values(CORE_GPU),
                                Values(AbsExact().to_compare_obj()),
                                Values(2.5, 1.0, -1.0),
                                Values(250.0, 0.0, -128.0)));

INSTANTIATE_TEST_CASE_P(ConcatHorTestGPU, ConcatHorTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(ConcatVertTestGPU, ConcatVertTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(TransposeTestGPU, TransposeTest,
                        Combine(Values(CV_8UC1, CV_16UC1, CV_16SC1, CV_32FC1,
                                       CV_8UC2, CV_16UC2, CV_16SC2, CV_32FC2,
                                       CV_8UC3, CV_16UC3, CV_16SC3, CV_32FC3),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(AbsExact().to_compare_obj())));
// PLEASE DO NOT PUT NEW ACCURACY TESTS BELOW THIS POINT! //////////////////////

INSTANTIATE_TEST_CASE_P(BackendOutputAllocationTestGPU, BackendOutputAllocationTest,
                        Combine(Values(CV_8UC3, CV_16SC2, CV_32FC1),
                                Values(ncvslideio::Size(50, 50)),
                                Values(-1),
                                Values(CORE_GPU)));

// FIXME: there's an issue in OCL backend with matrix reallocation that shouldn't happen
INSTANTIATE_TEST_CASE_P(DISABLED_BackendOutputAllocationLargeSizeWithCorrectSubmatrixTestGPU,
                        BackendOutputAllocationLargeSizeWithCorrectSubmatrixTest,
                        Combine(Values(CV_8UC3, CV_16SC2, CV_32FC1),
                                Values(ncvslideio::Size(50, 50)),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(ReInitOutTestGPU, ReInitOutTest,
                        Combine(Values(CV_8UC3, CV_16SC4, CV_32FC1),
                                Values(ncvslideio::Size(640, 480)),
                                Values(-1),
                                Values(CORE_GPU),
                                Values(ncvslideio::Size(640, 400),
                                       ncvslideio::Size(10, 480))));

//TODO: fix this backend to allow ConcatVertVec ConcatHorVec
INSTANTIATE_TEST_CASE_P(DISABLED_ConcatVertVecTestGPU, ConcatVertVecTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));

INSTANTIATE_TEST_CASE_P(DISABLED_ConcatHorVecTestGPU, ConcatHorVecTest,
                        Combine(Values( CV_8UC1, CV_8UC3, CV_16UC1, CV_16SC1, CV_32FC1 ),
                                ValuesIn(in_sizes),
                                Values(-1),
                                Values(CORE_GPU)));
}
