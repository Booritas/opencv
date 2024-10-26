// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#ifndef OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP
#define OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP



#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/imgproc.hpp>

namespace opencv_test
{

  using namespace perf;

  //------------------------------------------------------------------------------

class SepFilterPerfTest       : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int, ncvslideio::GCompileArgs>> {};
class Filter2DPerfTest        : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int,int, ncvslideio::GCompileArgs>> {};
class BoxFilterPerfTest       : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int,int, ncvslideio::GCompileArgs>> {};
class BlurPerfTest            : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int, ncvslideio::GCompileArgs>> {};
class GaussianBlurPerfTest    : public TestPerfParams<tuple<compare_f, MatType,int, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class MedianBlurPerfTest      : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class ErodePerfTest           : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int, ncvslideio::GCompileArgs>> {};
class Erode3x3PerfTest        : public TestPerfParams<tuple<compare_f, MatType,ncvslideio::Size,int, ncvslideio::GCompileArgs>> {};
class DilatePerfTest          : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int, ncvslideio::GCompileArgs>> {};
class Dilate3x3PerfTest       : public TestPerfParams<tuple<compare_f, MatType,ncvslideio::Size,int, ncvslideio::GCompileArgs>> {};
class MorphologyExPerfTest    : public TestPerfParams<tuple<compare_f,MatType,ncvslideio::Size,
                                                            ncvslideio::MorphTypes,ncvslideio::GCompileArgs>> {};
class SobelPerfTest           : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int,int,int, ncvslideio::GCompileArgs>> {};
class SobelXYPerfTest         : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int,int, ncvslideio::GCompileArgs>> {};
class LaplacianPerfTest       : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int,
                                                            ncvslideio::GCompileArgs>> {};
class BilateralFilterPerfTest : public TestPerfParams<tuple<compare_f, MatType,int,ncvslideio::Size,int, double,double,
                                                            ncvslideio::GCompileArgs>> {};
class CannyPerfTest           : public TestPerfParams<tuple<compare_f, MatType,ncvslideio::Size,double,double,int,bool,
                                                            ncvslideio::GCompileArgs>> {};
class GoodFeaturesPerfTest    : public TestPerfParams<tuple<compare_vector_f<ncvslideio::Point2f>, std::string,
                                                            int,int,double,double,int,bool,
                                                            ncvslideio::GCompileArgs>> {};
class FindContoursPerfTest    : public TestPerfParams<tuple<CompareMats, MatType,ncvslideio::Size,
                                                            ncvslideio::RetrievalModes,
                                                            ncvslideio::ContourApproximationModes,
                                                            ncvslideio::GCompileArgs>> {};
class FindContoursHPerfTest   : public TestPerfParams<tuple<CompareMats, MatType,ncvslideio::Size,
                                                            ncvslideio::RetrievalModes,
                                                            ncvslideio::ContourApproximationModes,
                                                            ncvslideio::GCompileArgs>> {};
class BoundingRectMatPerfTest       :
    public TestPerfParams<tuple<CompareRects, MatType,ncvslideio::Size,bool, ncvslideio::GCompileArgs>> {};
class BoundingRectVector32SPerfTest :
    public TestPerfParams<tuple<CompareRects, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BoundingRectVector32FPerfTest :
    public TestPerfParams<tuple<CompareRects, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class FitLine2DMatVectorPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               MatType,ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine2DVector32SPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine2DVector32FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine2DVector64FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 4>,
                                                               ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine3DMatVectorPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               MatType,ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine3DVector32SPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine3DVector32FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class FitLine3DVector64FPerfTest : public TestPerfParams<tuple<CompareVecs<float, 6>,
                                                               ncvslideio::Size,ncvslideio::DistanceTypes,
                                                               ncvslideio::GCompileArgs>> {};
class EqHistPerfTest      : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BGR2RGBPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class RGB2GrayPerfTest    : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BGR2GrayPerfTest    : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class RGB2YUVPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class YUV2RGBPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BGR2I420PerfTest    : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class RGB2I420PerfTest    : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class I4202BGRPerfTest    : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class I4202RGBPerfTest    : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class RGB2LabPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BGR2LUVPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class LUV2BGRPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BGR2YUVPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class YUV2BGRPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class RGB2HSVPerfTest     : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class BayerGR2RGBPerfTest : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class RGB2YUV422PerfTest  : public TestPerfParams<tuple<compare_f, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class ResizePerfTest      : public TestPerfParams<tuple<compare_f, MatType, int, ncvslideio::Size, ncvslideio::Size, ncvslideio::GCompileArgs>> {};
class ResizeFxFyPerfTest  : public TestPerfParams<tuple<compare_f, MatType, int, ncvslideio::Size, double, double, ncvslideio::GCompileArgs>> {};
class ResizeInSimpleGraphPerfTest : public TestPerfParams<tuple<compare_f, MatType, ncvslideio::Size, double, double,  ncvslideio::GCompileArgs>> {};
class BottleneckKernelsConstInputPerfTest : public TestPerfParams<tuple<compare_f, std::string, ncvslideio::GCompileArgs>> {};
} // opencv_test

#endif //OPENCV_GAPI_IMGPROC_PERF_TESTS_HPP
