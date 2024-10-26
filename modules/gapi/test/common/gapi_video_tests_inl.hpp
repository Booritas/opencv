// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_VIDEO_TESTS_INL_HPP
#define OPENCV_GAPI_VIDEO_TESTS_INL_HPP

#include "gapi_video_tests.hpp"
#include <opencv2/gapi/streaming/cap.hpp>

namespace opencv_test
{

TEST_P(BuildOptFlowPyramidTest, AccuracyTest)
{
    std::vector<Mat> outPyrOCV,          outPyrGAPI;
    int              outMaxLevelOCV = 0, outMaxLevelGAPI = 0;

    BuildOpticalFlowPyramidTestParams params { fileName, winSize, maxLevel,
                                              withDerivatives, pyrBorder, derivBorder,
                                              tryReuseInputImage, getCompileArgs() };

    BuildOpticalFlowPyramidTestOutput outOCV  { outPyrOCV,  outMaxLevelOCV };
    BuildOpticalFlowPyramidTestOutput outGAPI { outPyrGAPI, outMaxLevelGAPI };

    runOCVnGAPIBuildOptFlowPyramid(*this, params, outOCV, outGAPI);

    compareOutputPyramids(outGAPI, outOCV);
}

TEST_P(OptFlowLKTest, AccuracyTest)
{
    std::vector<ncvslideio::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params { fileNamePattern, channels, pointsNum,
                                 winSize, criteria, getCompileArgs() };

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowLK(*this, inPts, params, outOCV, outGAPI);

    compareOutputsOptFlow(outGAPI, outOCV);
}

TEST_P(OptFlowLKTestForPyr, AccuracyTest)
{
    std::vector<ncvslideio::Mat>     inPyr1, inPyr2;
    std::vector<ncvslideio::Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>       outStatusOCV, outStatusGAPI;
    std::vector<float>       outErrOCV,    outErrGAPI;

    OptFlowLKTestParams params { fileNamePattern, channels, pointsNum,
                                 winSize, criteria, getCompileArgs() };

    OptFlowLKTestInput<std::vector<ncvslideio::Mat>> in { inPyr1, inPyr2, inPts };
    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowLKForPyr(*this, in, params, withDeriv, outOCV, outGAPI);

    compareOutputsOptFlow(outGAPI, outOCV);
}

TEST_P(BuildPyr_CalcOptFlow_PipelineTest, AccuracyTest)
{
    std::vector<Point2f> outPtsOCV,    outPtsGAPI,    inPts;
    std::vector<uchar>   outStatusOCV, outStatusGAPI;
    std::vector<float>   outErrOCV,    outErrGAPI;

    BuildOpticalFlowPyramidTestParams params { fileNamePattern, winSize, maxLevel,
                                              withDerivatives, BORDER_DEFAULT, BORDER_DEFAULT,
                                              true, getCompileArgs() };

    auto customKernel  = gapi::kernels<GCPUMinScalar>();
    auto kernels       = gapi::combine(customKernel,
                                       params.compileArgs[0].get<GKernelPackage>());
    params.compileArgs = compile_args(kernels);

    OptFlowLKTestOutput outOCV  { outPtsOCV,  outStatusOCV,  outErrOCV };
    OptFlowLKTestOutput outGAPI { outPtsGAPI, outStatusGAPI, outErrGAPI };

    runOCVnGAPIOptFlowPipeline(*this, params, outOCV, outGAPI, inPts);

    compareOutputsOptFlow(outGAPI, outOCV);
}

#ifdef HAVE_OPENCV_VIDEO
TEST_P(BackgroundSubtractorTest, AccuracyTest)
{
    ncvslideio::gapi::video::BackgroundSubtractorType opType;
    double thr = -1;
    std::tie(opType, thr) = typeAndThreshold;

    ncvslideio::gapi::video::BackgroundSubtractorParams bsp(opType, histLength, thr,
                                                    detectShadows, learningRate);

    // G-API graph declaration
    ncvslideio::GMat in;
    ncvslideio::GMat out = ncvslideio::gapi::BackgroundSubtractor(in, bsp);
    // Preserving 'in' in output to have possibility to compare with OpenCV reference
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(ncvslideio::gapi::copy(in), out));

    // G-API compilation of graph for streaming mode
    auto gapiBackSub = c.compileStreaming(getCompileArgs());

    // Testing G-API Background Substractor in streaming mode
    const auto path = findDataFile(filePath);
    try
    {
        gapiBackSub.setSource(gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(path));
    }
    catch (...)
    { throw SkipTestException("Video file can't be opened."); }

    ncvslideio::Ptr<ncvslideio::BackgroundSubtractor> pOCVBackSub;

    if (opType == ncvslideio::gapi::video::TYPE_BS_MOG2)
        pOCVBackSub = ncvslideio::createBackgroundSubtractorMOG2(histLength, thr,
                                                         detectShadows);
    else if (opType == ncvslideio::gapi::video::TYPE_BS_KNN)
        pOCVBackSub = ncvslideio::createBackgroundSubtractorKNN(histLength, thr,
                                                        detectShadows);

    // Allowing 1% difference of all pixels between G-API and reference OpenCV results
    testBackgroundSubtractorStreaming(gapiBackSub, pOCVBackSub, 1, 1, learningRate, testNumFrames);
}

TEST_P(KalmanFilterTest, AccuracyTest)
{
    ncvslideio::gapi::KalmanParams kp;
    initKalmanParams(type, dDim, mDim, cDim, kp);

    // OpenCV reference KalmanFilter initialization
    ncvslideio::KalmanFilter ocvKalman(dDim, mDim, cDim, type);
    initKalmanFilter(kp, true, ocvKalman);

    // measurement vector
    ncvslideio::Mat measure_vec(mDim, 1, type);

    // control vector
    ncvslideio::Mat ctrl_vec = Mat::zeros(cDim > 0 ? cDim : 2, 1, type);

    // G-API Kalman's output state
    ncvslideio::Mat gapiKState(dDim, 1, type);
    // OCV Kalman's output state
    ncvslideio::Mat ocvKState(dDim, 1, type);

    // G-API graph initialization
    ncvslideio::GMat m, ctrl;
    ncvslideio::GOpaque<bool> have_m;
    ncvslideio::GMat out = ncvslideio::gapi::KalmanFilter(m, have_m, ctrl, kp);
    ncvslideio::GComputation comp(ncvslideio::GIn(m, have_m, ctrl), ncvslideio::GOut(out));

    ncvslideio::RNG& rng = ncvslideio::theRNG();
    bool haveMeasure;

    for (int i = 0; i < numIter; i++)
    {
        haveMeasure = (rng(2u) == 1); // returns 0 or 1 - whether we have measurement at this iteration or not

        if (haveMeasure)
            ncvslideio::randu(measure_vec, Scalar::all(-1), Scalar::all(1));
        if (cDim > 0)
            ncvslideio::randu(ctrl_vec, Scalar::all(-1), Scalar::all(1));

        // G-API KalmanFilter call
        comp.apply(ncvslideio::gin(measure_vec, haveMeasure, ctrl_vec), ncvslideio::gout(gapiKState));
        // OpenCV KalmanFilter call
        ocvKState = cDim > 0 ? ocvKalman.predict(ctrl_vec) : ocvKalman.predict();
        if (haveMeasure)
            ocvKState = ocvKalman.correct(measure_vec);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(AbsExact().to_compare_f()(gapiKState, ocvKState));
    }
}

TEST_P(KalmanFilterNoControlTest, AccuracyTest)
{
    ncvslideio::gapi::KalmanParams kp;
    initKalmanParams(type, dDim, mDim, 0, kp);

    // OpenCV reference KalmanFilter initialization
    ncvslideio::KalmanFilter ocvKalman(dDim, mDim, 0, type);
    initKalmanFilter(kp, false, ocvKalman);

    // measurement vector
    ncvslideio::Mat measure_vec(mDim, 1, type);

    // G-API Kalman's output state
    ncvslideio::Mat gapiKState(dDim, 1, type);
    // OCV Kalman's output state
    ncvslideio::Mat ocvKState(dDim, 1, type);

    // G-API graph initialization
    ncvslideio::GMat m;
    ncvslideio::GOpaque<bool> have_m;
    ncvslideio::GMat out = ncvslideio::gapi::KalmanFilter(m, have_m, kp);
    ncvslideio::GComputation comp(ncvslideio::GIn(m, have_m), ncvslideio::GOut(out));

    ncvslideio::RNG& rng = ncvslideio::theRNG();
    bool haveMeasure;

    for (int i = 0; i < numIter; i++)
    {
        haveMeasure = (rng(2u) == 1); // returns 0 or 1 - whether we have measurement at this iteration or not

        if (haveMeasure)
            ncvslideio::randu(measure_vec, Scalar::all(-1), Scalar::all(1));

        // G-API
        comp.apply(ncvslideio::gin(measure_vec, haveMeasure), ncvslideio::gout(gapiKState));

        // OpenCV
        ocvKState = ocvKalman.predict();
        if (haveMeasure)
            ocvKState = ocvKalman.correct(measure_vec);
    }

    // Comparison //////////////////////////////////////////////////////////////
    {
            EXPECT_TRUE(AbsExact().to_compare_f()(gapiKState, ocvKState));
    }
}

TEST_P(KalmanFilterCircleSampleTest, AccuracyTest)
{
    // auxiliary variables
    ncvslideio::Mat processNoise(2, 1, type);

    // Input measurement
    ncvslideio::Mat measurement = Mat::zeros(1, 1, type);
    // Angle and it's delta(phi, delta_phi)
    ncvslideio::Mat state(2, 1, type);

    // G-API graph initialization
    ncvslideio::gapi::KalmanParams kp;

    kp.state = Mat::zeros(2, 1, type);
    ncvslideio::randn(kp.state, Scalar::all(0), Scalar::all(0.1));

    kp.errorCov = Mat::eye(2, 2, type);

    if (type == CV_32F)
        kp.transitionMatrix = (Mat_<float>(2, 2) << 1, 1, 0, 1);
    else
        kp.transitionMatrix = (Mat_<double>(2, 2) << 1, 1, 0, 1);

    kp.processNoiseCov = Mat::eye(2, 2, type) * (1e-5);
    kp.measurementMatrix = Mat::eye(1, 2, type);
    kp.measurementNoiseCov = Mat::eye(1, 1, type) * (1e-1);

    ncvslideio::GMat m;
    ncvslideio::GOpaque<bool> have_measure;
    ncvslideio::GMat out = ncvslideio::gapi::KalmanFilter(m, have_measure, kp);
    ncvslideio::GComputation comp(ncvslideio::GIn(m, have_measure), ncvslideio::GOut(out));

    // OCV Kalman initialization
    ncvslideio::KalmanFilter KF(2, 1, 0);
    initKalmanFilter(kp, false, KF);

    ncvslideio::randn(state, Scalar::all(0), Scalar::all(0.1));

    // GAPI Corrected state
    ncvslideio::Mat gapiState(2, 1, type);
    // OCV Corrected state
    ncvslideio::Mat ocvCorrState(2, 1, type);
    // OCV Predicted state
    ncvslideio::Mat ocvPreState(2, 1, type);

    bool haveMeasure;

    for (int i = 0; i < numIter; ++i)
    {
        // Get OCV Prediction
        ocvPreState = KF.predict();

        GAPI_DbgAssert(ncvslideio::norm(kp.measurementNoiseCov, KF.measurementNoiseCov, ncvslideio::NORM_INF) == 0);
        // generation measurement
        ncvslideio::randn(measurement, Scalar::all(0), Scalar::all((type == CV_32FC1) ?
                  kp.measurementNoiseCov.at<float>(0) : kp.measurementNoiseCov.at<double>(0)));

        GAPI_DbgAssert(ncvslideio::norm(kp.measurementMatrix, KF.measurementMatrix, ncvslideio::NORM_INF) == 0);
        measurement += kp.measurementMatrix*state;

        if (ncvslideio::theRNG().uniform(0, 4) != 0)
        {
            haveMeasure = true;
            ocvCorrState = KF.correct(measurement);
            comp.apply(ncvslideio::gin(measurement, haveMeasure), ncvslideio::gout(gapiState));
            EXPECT_TRUE(AbsExact().to_compare_f()(gapiState, ocvCorrState));
        }
        else
        {
            // Get GAPI Prediction
            haveMeasure = false;
            comp.apply(ncvslideio::gin(measurement, haveMeasure), ncvslideio::gout(gapiState));
            EXPECT_TRUE(AbsExact().to_compare_f()(gapiState, ocvPreState));
        }

        GAPI_DbgAssert(ncvslideio::norm(kp.processNoiseCov, KF.processNoiseCov, ncvslideio::NORM_INF) == 0);
        ncvslideio::randn(processNoise, Scalar(0), Scalar::all(sqrt(type == CV_32FC1 ?
                                                       kp.processNoiseCov.at<float>(0, 0):
                                                       kp.processNoiseCov.at<double>(0, 0))));

        GAPI_DbgAssert(ncvslideio::norm(kp.transitionMatrix, KF.transitionMatrix, ncvslideio::NORM_INF) == 0);
        state = kp.transitionMatrix*state + processNoise;
    }
}

#endif
} // opencv_test

#endif // OPENCV_GAPI_VIDEO_TESTS_INL_HPP
