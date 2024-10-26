// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2021 Intel Corporation


#ifndef OPENCV_GAPI_CORE_PERF_TESTS_HPP
#define OPENCV_GAPI_CORE_PERF_TESTS_HPP


#include "../../test/common/gapi_tests_common.hpp"
#include "../../test/common/gapi_parsers_tests_common.hpp"
#include <opencv2/gapi/core.hpp>

namespace opencv_test
{
  using namespace perf;

  enum bitwiseOp
  {
      AND = 0,
      OR = 1,
      XOR = 2,
      NOT = 3
  };

//------------------------------------------------------------------------------
    class PhasePerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class SqrtPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class AddPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class AddCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class SubPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class SubCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class SubRCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class MulPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, double, ncvslideio::GCompileArgs>> {};
    class MulDoublePerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class MulCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class DivPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, double, ncvslideio::GCompileArgs>> {};
    class DivCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, double, ncvslideio::GCompileArgs>> {};
    class DivRCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, double, ncvslideio::GCompileArgs>> {};
    class MaskPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class MeanPerfTest : public TestPerfParams<tuple<compare_scalar_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class Polar2CartPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class Cart2PolarPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class CmpPerfTest : public TestPerfParams<tuple<compare_f, CmpTypes, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class CmpWithScalarPerfTest : public TestPerfParams<tuple<compare_f, CmpTypes, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class BitwisePerfTest : public TestPerfParams<tuple<compare_f, bitwiseOp, bool, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class BitwiseNotPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class SelectPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class MinPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class MaxPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class AbsDiffPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class AbsDiffCPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class SumPerfTest : public TestPerfParams<tuple<compare_scalar_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class CountNonZeroPerfTest : public TestPerfParams<tuple<compare_scalar_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class AddWeightedPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class NormPerfTest : public TestPerfParams<tuple<compare_scalar_f, NormTypes, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class IntegralPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class ThresholdPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class ThresholdOTPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class InRangePerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class Split3PerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class Split4PerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class Merge3PerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class Merge4PerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class RemapPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class FlipPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, int, ncvslideio::GCompileArgs>> {};
    class CropPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::Rect, ncvslideio::GCompileArgs>> {};
    class CopyPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class ConcatHorPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class ConcatHorVecPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class ConcatVertPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class ConcatVertVecPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class LUTPerfTest : public TestPerfParams<tuple<compare_f, MatType, MatType, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class ConvertToPerfTest : public TestPerfParams<tuple<compare_f, MatType, int, ncvslideio::Size, double, double, ncvslideio::GCompileArgs>> {};
    class KMeansNDPerfTest : public TestPerfParams<tuple<ncvslideio::Size, CompareMats, int, ncvslideio::KmeansFlags, ncvslideio::GCompileArgs>> {};
    class KMeans2DPerfTest : public TestPerfParams<tuple<int, int, ncvslideio::KmeansFlags, ncvslideio::GCompileArgs>> {};
    class KMeans3DPerfTest : public TestPerfParams<tuple<int, int, ncvslideio::KmeansFlags, ncvslideio::GCompileArgs>> {};
    class TransposePerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, MatType, ncvslideio::GCompileArgs>> {};
    class ParseSSDBLPerfTest : public TestPerfParams<tuple<ncvslideio::Size, float, int, ncvslideio::GCompileArgs>>, public ParserSSDTest {};
    class ParseSSDPerfTest   : public TestPerfParams<tuple<ncvslideio::Size, float, bool, bool, ncvslideio::GCompileArgs>>, public ParserSSDTest {};
    class ParseYoloPerfTest  : public TestPerfParams<tuple<ncvslideio::Size, float, float, int, ncvslideio::GCompileArgs>>, public ParserYoloTest {};
    class SizePerfTest       : public TestPerfParams<tuple<MatType, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
    class SizeRPerfTest      : public TestPerfParams<tuple<ncvslideio::Size, ncvslideio::GCompileArgs>> {};
}
#endif // OPENCV_GAPI_CORE_PERF_TESTS_HPP
