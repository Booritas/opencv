// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_PERF_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_PERF_TESTS_INL_HPP

#include <iostream>

#include "gapi_video_perf_tests.hpp"

namespace opencv_test
{

  using namespace perf;

//------------------------------------------------------------------------------

PERF_TEST_P_(BuildOptFlowPyramidPerfTest, TestPerformance)
{
    std::vector<Mat> outPyrOCV,          outPyrGAPI;
    int              outMaxLevelOCV = 0, outMaxLevelGAPI = 0;
    Scalar           outMaxLevelSc;

    BuildOpticalFlowPyramidTestParams params;
    std::tie(params.fileName, params.winSize,
             params.maxLevel, params.withDerivatives,
             params.pyrBorder, params.derivBorder,
             params.tryReuseInputImage, params.compileArgs) = GetParam();

    BuildOpticalFlowPyramidTestOutput outOCV  { outPyrOCV,  outMaxLevelOCV };
    BuildOpticalFlowPyramidTestOutput outGAPI { outPyrGAPI, outMaxLevelGAPI };

    GComputation c = runOCVnGAPIBuildOptFlowPyramid(*this, params, outOCV, outGAPI);

    declare.in(in_mat1).out(outPyrGAPI);

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1), ncvslideio::gout(outPyrGAPI, outMaxLevelSc));
    }
    outMaxLevelGAPI = static_cast<int>(outMaxLevelSc[0]);

    // Comparison //////////////////////////////////////////////////////////////
    compareOutputPyramids(outGAPI, outOCV);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(OptFlowLKPerfTest, TestPerformance)
{
    std::vector<ncvslideio::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params;
    std::tie(params.fileNamePattern, params.channels,
             params.pointsNum, params.winSize, params.criteria,
             params.compileArgs) = GetParam();

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    ncvslideio::GComputation c = runOCVnGAPIOptFlowLK(*this, inPts, params, outOCV, outGAPI);

    declare.in(in_mat1, in_mat2, inPts).out(outPtsGAPI, outStatusGAPI, outErrGAPI);

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1, in_mat2, inPts, std::vector<ncvslideio::Point2f>{ }),
                ncvslideio::gout(outPtsGAPI, outStatusGAPI, outErrGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    compareOutputsOptFlow(outGAPI, outOCV);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

PERF_TEST_P_(OptFlowLKForPyrPerfTest, TestPerformance)
{
    std::vector<ncvslideio::Mat>     inPyr1, inPyr2;
    std::vector<ncvslideio::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    bool withDeriv = false;
    OptFlowLKTestParams params;
    std::tie(params.fileNamePattern, params.channels,
             params.pointsNum, params.winSize, params.criteria,
             withDeriv, params.compileArgs) = GetParam();

    OptFlowLKTestInput<std::vector<ncvslideio::Mat>> in { inPyr1, inPyr2, inPts };
    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    ncvslideio::GComputation c = runOCVnGAPIOptFlowLKForPyr(*this, in, params, withDeriv, outOCV, outGAPI);

    declare.in(inPyr1, inPyr2, inPts).out(outPtsGAPI, outStatusGAPI, outErrGAPI);

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(inPyr1, inPyr2, inPts, std::vector<ncvslideio::Point2f>{ }),
                ncvslideio::gout(outPtsGAPI, outStatusGAPI, outErrGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    compareOutputsOptFlow(outGAPI, outOCV);

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(BuildPyr_CalcOptFlow_PipelinePerfTest, TestPerformance)
{
    std::vector<Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>   outStatusOCV, outStatusGAPI;
    std::vector<float>   outErrOCV,    outErrGAPI;

    BuildOpticalFlowPyramidTestParams params;
    params.pyrBorder          = BORDER_DEFAULT;
    params.derivBorder        = BORDER_DEFAULT;
    params.tryReuseInputImage = true;
    std::tie(params.fileName, params.winSize,
             params.maxLevel, params.withDerivatives,
             params.compileArgs) = GetParam();

    auto customKernel  = gapi::kernels<GCPUMinScalar>();
    auto kernels       = gapi::combine(customKernel,
                                       params.compileArgs[0].get<GKernelPackage>());
    params.compileArgs = compile_args(kernels);

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    ncvslideio::GComputation c = runOCVnGAPIOptFlowPipeline(*this, params, outOCV, outGAPI, inPts);

    declare.in(in_mat1, in_mat2, inPts).out(outPtsGAPI, outStatusGAPI, outErrGAPI);

    TEST_CYCLE()
    {
        c.apply(ncvslideio::gin(in_mat1, in_mat2, inPts, std::vector<ncvslideio::Point2f>{ }),
                ncvslideio::gout(outPtsGAPI, outStatusGAPI, outErrGAPI));
    }

    // Comparison //////////////////////////////////////////////////////////////
    compareOutputsOptFlow(outGAPI, outOCV);

    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

#ifdef HAVE_OPENCV_VIDEO

PERF_TEST_P_(BackgroundSubtractorPerfTest, TestPerformance)
{
    namespace gvideo = ncvslideio::gapi::video;

    gvideo::BackgroundSubtractorType opType;
    std::string filePath = "";
    bool detectShadows = false;
    double learningRate = -1.;
    std::size_t testNumFrames = 0;
    ncvslideio::GCompileArgs compileArgs;
    CompareMats cmpF;

    std::tie(opType, filePath, detectShadows, learningRate, testNumFrames,
             compileArgs, cmpF) = GetParam();

    const int histLength = 500;
    double thr = -1;
    switch (opType)
    {
        case gvideo::TYPE_BS_MOG2:
        {
            thr = 16.;
            break;
        }
        case gvideo::TYPE_BS_KNN:
        {
            thr = 400.;
            break;
        }
        default:
            FAIL() << "unsupported type of BackgroundSubtractor";
    }
    const gvideo::BackgroundSubtractorParams bsp(opType, histLength, thr, detectShadows,
                                                 learningRate);

    // Retrieving frames
    std::vector<ncvslideio::Mat> frames;
    frames.reserve(testNumFrames);
    {
        ncvslideio::Mat frame;
        ncvslideio::VideoCapture cap;
        if (!cap.open(findDataFile(filePath)))
            throw SkipTestException("Video file can not be opened");
        for (std::size_t i = 0; i < testNumFrames && cap.read(frame); i++)
        {
            frames.push_back(frame);
        }
    }
    GAPI_Assert(testNumFrames == frames.size() && "Can't read required number of frames");

    // G-API graph declaration
    ncvslideio::GMat in;
    ncvslideio::GMat out = ncvslideio::gapi::BackgroundSubtractor(in, bsp);
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(out));
    auto cc = c.compile(ncvslideio::descr_of(frames[0]), std::move(compileArgs));

    ncvslideio::Mat gapiForeground;
    TEST_CYCLE()
    {
        cc.prepareForNewStream();
        for (size_t i = 0; i < testNumFrames; i++)
        {
            cc(ncvslideio::gin(frames[i]), ncvslideio::gout(gapiForeground));
        }
    }

    // OpenCV Background Subtractor declaration
    ncvslideio::Ptr<ncvslideio::BackgroundSubtractor> pOCVBackSub;
    if (opType == gvideo::TYPE_BS_MOG2)
        pOCVBackSub = ncvslideio::createBackgroundSubtractorMOG2(histLength, thr, detectShadows);
    else if (opType == gvideo::TYPE_BS_KNN)
        pOCVBackSub = ncvslideio::createBackgroundSubtractorKNN(histLength, thr, detectShadows);
    ncvslideio::Mat ocvForeground;
    for (size_t i = 0; i < testNumFrames; i++)
    {
        pOCVBackSub->apply(frames[i], ocvForeground, learningRate);
    }
    // Validation
    EXPECT_TRUE(cmpF(gapiForeground, ocvForeground));
    SANITY_CHECK_NOTHING();
}

//------------------------------------------------------------------------------

inline void generateInputKalman(const int mDim, const MatType2& type,
                                const size_t testNumMeasurements, const bool receiveRandMeas,
                                std::vector<bool>&    haveMeasurements,
                                std::vector<ncvslideio::Mat>& measurements)
{
    ncvslideio::RNG& rng = ncvslideio::theRNG();
    measurements.clear();
    haveMeasurements = std::vector<bool>(testNumMeasurements, true);
    for (size_t i = 0; i < testNumMeasurements; i++)
    {
        if (receiveRandMeas)
        {
            haveMeasurements[i] = rng(2u) == 1; // returns 0 or 1 - whether we have measurement
                                                // at this iteration or not
        } // if not - testing the slowest case in which we have measurements at every iteration

        ncvslideio::Mat measurement = ncvslideio::Mat::zeros(mDim, 1, type);
        if (haveMeasurements[i])
        {
            ncvslideio::randu(measurement, ncvslideio::Scalar::all(-1), ncvslideio::Scalar::all(1));
        }
        measurements.push_back(measurement.clone());
    }
}

inline void generateInputKalman(const int mDim, const int cDim, const MatType2& type,
                                const size_t testNumMeasurements, const bool receiveRandMeas,
                                std::vector<bool>&    haveMeasurements,
                                std::vector<ncvslideio::Mat>& measurements,
                                std::vector<ncvslideio::Mat>& ctrls)
{
    generateInputKalman(mDim, type, testNumMeasurements, receiveRandMeas,
                        haveMeasurements, measurements);
    ctrls.clear();
    ncvslideio::Mat ctrl(cDim, 1, type);
    for (size_t i = 0; i < testNumMeasurements; i++)
    {
        ncvslideio::randu(ctrl, ncvslideio::Scalar::all(-1), ncvslideio::Scalar::all(1));
        ctrls.push_back(ctrl.clone());
    }
}

PERF_TEST_P_(KalmanFilterControlPerfTest, TestPerformance)
{
    MatType2 type = -1;
    int dDim = -1, mDim = -1;
    size_t testNumMeasurements = 0;
    bool receiveRandMeas = true;
    ncvslideio::GCompileArgs compileArgs;
    std::tie(type, dDim, mDim, testNumMeasurements, receiveRandMeas, compileArgs) = GetParam();

    const int cDim = 2;
    ncvslideio::gapi::KalmanParams kp;
    initKalmanParams(type, dDim, mDim, cDim, kp);

    // Generating input
    std::vector<bool> haveMeasurements;
    std::vector<ncvslideio::Mat> measurements, ctrls;
    generateInputKalman(mDim, cDim, type, testNumMeasurements, receiveRandMeas,
                        haveMeasurements, measurements, ctrls);

    // G-API graph declaration
    ncvslideio::GMat m, ctrl;
    ncvslideio::GOpaque<bool> have_m;
    ncvslideio::GMat out = ncvslideio::gapi::KalmanFilter(m, have_m, ctrl, kp);
    ncvslideio::GComputation c(ncvslideio::GIn(m, have_m, ctrl), ncvslideio::GOut(out));
    auto cc = c.compile(
        ncvslideio::descr_of(ncvslideio::gin(ncvslideio::Mat(mDim, 1, type), true, ncvslideio::Mat(cDim, 1, type))),
        std::move(compileArgs));

    ncvslideio::Mat gapiKState(dDim, 1, type);
    TEST_CYCLE()
    {
        cc.prepareForNewStream();
        for (size_t i = 0; i < testNumMeasurements; i++)
        {
            bool hvMeas = haveMeasurements[i];
            cc(ncvslideio::gin(measurements[i], hvMeas, ctrls[i]), ncvslideio::gout(gapiKState));
        }
    }

    // OpenCV reference KalmanFilter initialization
    ncvslideio::KalmanFilter ocvKalman(dDim, mDim, cDim, type);
    initKalmanFilter(kp, true, ocvKalman);

    ncvslideio::Mat ocvKState(dDim, 1, type);
    for (size_t i = 0; i < testNumMeasurements; i++)
    {
        ocvKState = ocvKalman.predict(ctrls[i]);
        if (haveMeasurements[i])
            ocvKState = ocvKalman.correct(measurements[i]);
    }
    // Validation
    EXPECT_TRUE(AbsExact().to_compare_f()(gapiKState, ocvKState));
    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(KalmanFilterNoControlPerfTest, TestPerformance)
{
    MatType2 type = -1;
    int dDim = -1, mDim = -1;
    size_t testNumMeasurements = 0;
    bool receiveRandMeas = true;
    ncvslideio::GCompileArgs compileArgs;
    std::tie(type, dDim, mDim, testNumMeasurements, receiveRandMeas, compileArgs) = GetParam();

    const int cDim = 0;
    ncvslideio::gapi::KalmanParams kp;
    initKalmanParams(type, dDim, mDim, cDim, kp);

    // Generating input
    std::vector<bool> haveMeasurements;
    std::vector<ncvslideio::Mat> measurements;
    generateInputKalman(mDim, type, testNumMeasurements, receiveRandMeas,
                        haveMeasurements, measurements);

    // G-API graph declaration
    ncvslideio::GMat m;
    ncvslideio::GOpaque<bool> have_m;
    ncvslideio::GMat out = ncvslideio::gapi::KalmanFilter(m, have_m, kp);
    ncvslideio::GComputation c(ncvslideio::GIn(m, have_m), ncvslideio::GOut(out));
    auto cc = c.compile(ncvslideio::descr_of(ncvslideio::gin(ncvslideio::Mat(mDim, 1, type), true)),
                        std::move(compileArgs));

    ncvslideio::Mat gapiKState(dDim, 1, type);
    TEST_CYCLE()
    {
        cc.prepareForNewStream();
        for (size_t i = 0; i < testNumMeasurements; i++)
        {
            bool hvMeas = haveMeasurements[i];
            cc(ncvslideio::gin(measurements[i], hvMeas), ncvslideio::gout(gapiKState));
        }
    }

    // OpenCV reference KalmanFilter declaration
    ncvslideio::KalmanFilter ocvKalman(dDim, mDim, cDim, type);
    initKalmanFilter(kp, false, ocvKalman);

    ncvslideio::Mat ocvKState(dDim, 1, type);
    for (size_t i = 0; i < testNumMeasurements; i++)
    {
        ocvKState = ocvKalman.predict();
        if (haveMeasurements[i])
            ocvKState = ocvKalman.correct(measurements[i]);
    }
    // Validation
    EXPECT_TRUE(AbsExact().to_compare_f()(gapiKState, ocvKState));
    SANITY_CHECK_NOTHING();
}
#endif // HAVE_OPENCV_VIDEO

} // opencv_test

#endif // OPENCV_GAPI_VIDEO_PERF_TESTS_INL_HPP
