// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/video.hpp>
#include <opencv2/gapi/cpu/video.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>

#ifdef HAVE_OPENCV_VIDEO
#include <opencv2/video.hpp>
#endif // HAVE_OPENCV_VIDEO

#ifdef HAVE_OPENCV_VIDEO

GAPI_OCV_KERNEL(GCPUBuildOptFlowPyramid, ncvslideio::gapi::video::GBuildOptFlowPyramid)
{
    static void run(const ncvslideio::Mat              &img,
                    const ncvslideio::Size             &winSize,
                    const ncvslideio::Scalar           &maxLevel,
                          bool                  withDerivatives,
                          int                   pyrBorder,
                          int                   derivBorder,
                          bool                  tryReuseInputImage,
                          std::vector<ncvslideio::Mat> &outPyr,
                          ncvslideio::Scalar           &outMaxLevel)
    {
        outMaxLevel = ncvslideio::buildOpticalFlowPyramid(img, outPyr, winSize,
                                                  static_cast<int>(maxLevel[0]),
                                                  withDerivatives, pyrBorder,
                                                  derivBorder, tryReuseInputImage);
    }
};

GAPI_OCV_KERNEL(GCPUCalcOptFlowLK, ncvslideio::gapi::video::GCalcOptFlowLK)
{
    static void run(const ncvslideio::Mat                  &prevImg,
                    const ncvslideio::Mat                  &nextImg,
                    const std::vector<ncvslideio::Point2f> &prevPts,
                    const std::vector<ncvslideio::Point2f> &predPts,
                    const ncvslideio::Size                 &winSize,
                    const ncvslideio::Scalar               &maxLevel,
                    const ncvslideio::TermCriteria         &criteria,
                          int                       flags,
                          double                    minEigThresh,
                          std::vector<ncvslideio::Point2f> &outPts,
                          std::vector<uchar>       &status,
                          std::vector<float>       &err)
    {
        if (flags & ncvslideio::OPTFLOW_USE_INITIAL_FLOW)
            outPts = predPts;
        ncvslideio::calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, outPts, status, err, winSize,
                                 static_cast<int>(maxLevel[0]), criteria, flags, minEigThresh);
    }
};

GAPI_OCV_KERNEL(GCPUCalcOptFlowLKForPyr, ncvslideio::gapi::video::GCalcOptFlowLKForPyr)
{
    static void run(const std::vector<ncvslideio::Mat>     &prevPyr,
                    const std::vector<ncvslideio::Mat>     &nextPyr,
                    const std::vector<ncvslideio::Point2f> &prevPts,
                    const std::vector<ncvslideio::Point2f> &predPts,
                    const ncvslideio::Size                 &winSize,
                    const ncvslideio::Scalar               &maxLevel,
                    const ncvslideio::TermCriteria         &criteria,
                          int                       flags,
                          double                    minEigThresh,
                          std::vector<ncvslideio::Point2f> &outPts,
                          std::vector<uchar>       &status,
                          std::vector<float>       &err)
    {
        if (flags & ncvslideio::OPTFLOW_USE_INITIAL_FLOW)
            outPts = predPts;
        ncvslideio::calcOpticalFlowPyrLK(prevPyr, nextPyr, prevPts, outPts, status, err, winSize,
                                 static_cast<int>(maxLevel[0]), criteria, flags, minEigThresh);
    }
};

GAPI_OCV_KERNEL_ST(GCPUBackgroundSubtractor,
                   ncvslideio::gapi::video::GBackgroundSubtractor,
                   ncvslideio::BackgroundSubtractor)
{
    static void setup(const ncvslideio::GMatDesc&, const ncvslideio::gapi::video::BackgroundSubtractorParams& bsParams,
                      std::shared_ptr<ncvslideio::BackgroundSubtractor>& state,
                      const ncvslideio::GCompileArgs&)
    {
        if (bsParams.operation == ncvslideio::gapi::video::TYPE_BS_MOG2)
            state = ncvslideio::createBackgroundSubtractorMOG2(bsParams.history,
                                                       bsParams.threshold,
                                                       bsParams.detectShadows);
        else if (bsParams.operation == ncvslideio::gapi::video::TYPE_BS_KNN)
            state = ncvslideio::createBackgroundSubtractorKNN(bsParams.history,
                                                      bsParams.threshold,
                                                      bsParams.detectShadows);

        GAPI_Assert(state);
    }

    static void run(const ncvslideio::Mat& in, const ncvslideio::gapi::video::BackgroundSubtractorParams& bsParams,
                    ncvslideio::Mat &out, ncvslideio::BackgroundSubtractor& state)
    {
        state.apply(in, out, bsParams.learningRate);
    }
};

GAPI_OCV_KERNEL_ST(GCPUKalmanFilter, ncvslideio::gapi::video::GKalmanFilter, ncvslideio::KalmanFilter)
{
    static void setup(const ncvslideio::GMatDesc&, const ncvslideio::GOpaqueDesc&,
                      const ncvslideio::GMatDesc&, const ncvslideio::gapi::KalmanParams& kfParams,
                      std::shared_ptr<ncvslideio::KalmanFilter> &state, const ncvslideio::GCompileArgs&)
    {
        state = std::make_shared<ncvslideio::KalmanFilter>(kfParams.transitionMatrix.rows, kfParams.measurementMatrix.rows,
                                                   kfParams.controlMatrix.cols, kfParams.transitionMatrix.type());

        // initial state
        kfParams.state.copyTo(state->statePost);
        kfParams.errorCov.copyTo(state->errorCovPost);

        // dynamic system initialization
        kfParams.controlMatrix.copyTo(state->controlMatrix);
        kfParams.measurementMatrix.copyTo(state->measurementMatrix);
        kfParams.transitionMatrix.copyTo(state->transitionMatrix);
        kfParams.processNoiseCov.copyTo(state->processNoiseCov);
        kfParams.measurementNoiseCov.copyTo(state->measurementNoiseCov);
    }

    static void run(const ncvslideio::Mat& measurements, bool haveMeasurement,
                    const ncvslideio::Mat& control, const ncvslideio::gapi::KalmanParams&,
                    ncvslideio::Mat &out, ncvslideio::KalmanFilter& state)
    {
        ncvslideio::Mat pre = state.predict(control);

        if (haveMeasurement)
            state.correct(measurements).copyTo(out);
        else
            pre.copyTo(out);
    }
};

GAPI_OCV_KERNEL_ST(GCPUKalmanFilterNoControl, ncvslideio::gapi::video::GKalmanFilterNoControl, ncvslideio::KalmanFilter)
{
    static void setup(const ncvslideio::GMatDesc&, const ncvslideio::GOpaqueDesc&,
                      const ncvslideio::gapi::KalmanParams& kfParams,
                      std::shared_ptr<ncvslideio::KalmanFilter> &state,
                      const ncvslideio::GCompileArgs&)
    {
        state = std::make_shared<ncvslideio::KalmanFilter>(kfParams.transitionMatrix.rows, kfParams.measurementMatrix.rows,
                                                   0, kfParams.transitionMatrix.type());
        // initial state
        kfParams.state.copyTo(state->statePost);
        kfParams.errorCov.copyTo(state->errorCovPost);

        // dynamic system initialization
        kfParams.measurementMatrix.copyTo(state->measurementMatrix);
        kfParams.transitionMatrix.copyTo(state->transitionMatrix);
        kfParams.processNoiseCov.copyTo(state->processNoiseCov);
        kfParams.measurementNoiseCov.copyTo(state->measurementNoiseCov);
    }

    static void run(const ncvslideio::Mat& measurements, bool haveMeasurement,
                    const ncvslideio::gapi::KalmanParams&, ncvslideio::Mat &out,
                    ncvslideio::KalmanFilter& state)
    {
        ncvslideio::Mat pre = state.predict();

        if (haveMeasurement)
            state.correct(measurements).copyTo(out);
        else
            pre.copyTo(out);
    }
};

ncvslideio::GKernelPackage ncvslideio::gapi::video::cpu::kernels()
{
    static auto pkg = ncvslideio::gapi::kernels
        < GCPUBuildOptFlowPyramid
        , GCPUCalcOptFlowLK
        , GCPUCalcOptFlowLKForPyr
        , GCPUBackgroundSubtractor
        , GCPUKalmanFilter
        , GCPUKalmanFilterNoControl
        >();
    return pkg;
}

#else

ncvslideio::GKernelPackage ncvslideio::gapi::video::cpu::kernels()
{
    return GKernelPackage();
}

#endif // HAVE_OPENCV_VIDEO
