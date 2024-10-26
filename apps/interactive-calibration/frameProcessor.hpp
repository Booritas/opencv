// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect.hpp>

#include "calibCommon.hpp"
#include "calibController.hpp"

namespace calib
{
class FrameProcessor
{
protected:

public:
    virtual ~FrameProcessor();
    virtual ncvslideio::Mat processFrame(const ncvslideio::Mat& frame) = 0;
    virtual bool isProcessed() const = 0;
    virtual void resetState() = 0;
};

class CalibProcessor : public FrameProcessor
{
protected:
    ncvslideio::Ptr<calibrationData> mCalibData;
    TemplateType mBoardType;
    ncvslideio::Size mBoardSizeUnits;
    ncvslideio::Size mBoardSizeInnerCorners;
    std::vector<ncvslideio::Point2f> mTemplateLocations;
    std::vector<ncvslideio::Point2f> mCurrentImagePoints;
    ncvslideio::Mat mCurrentCharucoCorners;
    ncvslideio::Mat mCurrentCharucoIds;

    ncvslideio::Ptr<ncvslideio::SimpleBlobDetector> mBlobDetectorPtr;
    ncvslideio::aruco::Dictionary mArucoDictionary;
    ncvslideio::Ptr<ncvslideio::aruco::CharucoBoard> mCharucoBoard;
    ncvslideio::Ptr<ncvslideio::aruco::CharucoDetector> detector;

    int mNeededFramesNum;
    unsigned mDelayBetweenCaptures;
    int mCapuredFrames;
    double mMaxTemplateOffset;
    float mSquareSize;
    float mTemplDist;
    bool mSaveFrames;
    float mZoom;

    bool detectAndParseChessboard(const ncvslideio::Mat& frame);
    bool detectAndParseChAruco(const ncvslideio::Mat& frame);
    bool detectAndParseCircles(const ncvslideio::Mat& frame);
    bool detectAndParseACircles(const ncvslideio::Mat& frame);
    bool detectAndParseDualACircles(const ncvslideio::Mat& frame);
    void saveFrameData();
    void showCaptureMessage(const ncvslideio::Mat &frame, const std::string& message);
    bool checkLastFrame();

public:
    CalibProcessor(ncvslideio::Ptr<calibrationData> data, captureParameters& capParams);
    virtual ncvslideio::Mat processFrame(const ncvslideio::Mat& frame) CV_OVERRIDE;
    virtual bool isProcessed() const CV_OVERRIDE;
    virtual void resetState() CV_OVERRIDE;
    ~CalibProcessor() CV_OVERRIDE;
};

enum visualisationMode {Grid, Window};

class ShowProcessor : public FrameProcessor
{
protected:
    ncvslideio::Ptr<calibrationData> mCalibdata;
    ncvslideio::Ptr<calibController> mController;
    TemplateType mBoardType;
    visualisationMode mVisMode;
    bool mNeedUndistort;
    double mGridViewScale;
    double mTextSize;

    void drawBoard(ncvslideio::Mat& img, ncvslideio::InputArray points);
    void drawGridPoints(const ncvslideio::Mat& frame);
public:
    ShowProcessor(ncvslideio::Ptr<calibrationData> data, ncvslideio::Ptr<calibController> controller, TemplateType board);
    virtual ncvslideio::Mat processFrame(const ncvslideio::Mat& frame) CV_OVERRIDE;
    virtual bool isProcessed() const CV_OVERRIDE;
    virtual void resetState() CV_OVERRIDE;

    void setVisualizationMode(visualisationMode mode);
    void switchVisualizationMode();
    void clearBoardsView();
    void updateBoardsView();

    void switchUndistort();
    void setUndistort(bool isEnabled);
    ~ShowProcessor() CV_OVERRIDE;
};

}


#endif
