// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "frameProcessor.hpp"
#include "rotationConverters.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>
#include <string>
#include <limits>

using namespace calib;

#define VIDEO_TEXT_SIZE 4
#define POINT_SIZE 5

static ncvslideio::SimpleBlobDetector::Params getDetectorParams()
{
    ncvslideio::SimpleBlobDetector::Params detectorParams;

    detectorParams.thresholdStep = 40;
    detectorParams.minThreshold = 20;
    detectorParams.maxThreshold = 500;
    detectorParams.minRepeatability = 2;
    detectorParams.minDistBetweenBlobs = 5;

    detectorParams.filterByColor = true;
    detectorParams.blobColor = 0;

    detectorParams.filterByArea = true;
    detectorParams.minArea = 5;
    detectorParams.maxArea = 5000;

    detectorParams.filterByCircularity = false;
    detectorParams.minCircularity = 0.8f;
    detectorParams.maxCircularity = std::numeric_limits<float>::max();

    detectorParams.filterByInertia = true;
    detectorParams.minInertiaRatio = 0.1f;
    detectorParams.maxInertiaRatio = std::numeric_limits<float>::max();

    detectorParams.filterByConvexity = true;
    detectorParams.minConvexity = 0.8f;
    detectorParams.maxConvexity = std::numeric_limits<float>::max();

    return detectorParams;
}

FrameProcessor::~FrameProcessor()
{

}

bool CalibProcessor::detectAndParseChessboard(const ncvslideio::Mat &frame)
{
    int chessBoardFlags = ncvslideio::CALIB_CB_ADAPTIVE_THRESH | ncvslideio::CALIB_CB_NORMALIZE_IMAGE | ncvslideio::CALIB_CB_FAST_CHECK;
    bool isTemplateFound = ncvslideio::findChessboardCorners(frame, mBoardSizeInnerCorners, mCurrentImagePoints, chessBoardFlags);

    if (isTemplateFound) {
        ncvslideio::Mat viewGray;
        ncvslideio::cvtColor(frame, viewGray, ncvslideio::COLOR_BGR2GRAY);
        ncvslideio::cornerSubPix(viewGray, mCurrentImagePoints, ncvslideio::Size(11,11),
            ncvslideio::Size(-1,-1), ncvslideio::TermCriteria( ncvslideio::TermCriteria::EPS+ncvslideio::TermCriteria::COUNT, 30, 0.1 ));
        ncvslideio::drawChessboardCorners(frame, mBoardSizeInnerCorners, ncvslideio::Mat(mCurrentImagePoints), isTemplateFound);
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseChAruco(const ncvslideio::Mat &frame)
{
    ncvslideio::Ptr<ncvslideio::aruco::Board> board = mCharucoBoard.staticCast<ncvslideio::aruco::Board>();

    std::vector<std::vector<ncvslideio::Point2f> > corners;
    std::vector<int> ids;
    ncvslideio::Mat currentCharucoCorners, currentCharucoIds;
    detector->detectBoard(frame, currentCharucoCorners, currentCharucoIds, corners, ids);
    if(ids.size() > 0) ncvslideio::aruco::drawDetectedMarkers(frame, corners);

    if(currentCharucoCorners.total() > 3) {
        float centerX = 0, centerY = 0;
        for (int i = 0; i < currentCharucoCorners.size[0]; i++) {
            centerX += currentCharucoCorners.at<float>(i, 0);
            centerY += currentCharucoCorners.at<float>(i, 1);
        }
        centerX /= currentCharucoCorners.size[0];
        centerY /= currentCharucoCorners.size[0];

        mTemplateLocations.insert(mTemplateLocations.begin(), ncvslideio::Point2f(centerX, centerY));
        ncvslideio::aruco::drawDetectedCornersCharuco(frame, currentCharucoCorners, currentCharucoIds);
        mCurrentCharucoCorners = currentCharucoCorners;
        mCurrentCharucoIds = currentCharucoIds;
        return true;
    }
    return false;
}

bool CalibProcessor::detectAndParseCircles(const ncvslideio::Mat &frame)
{
    bool isTemplateFound = findCirclesGrid(frame, mBoardSizeUnits, mCurrentImagePoints, ncvslideio::CALIB_CB_SYMMETRIC_GRID, mBlobDetectorPtr);
    if(isTemplateFound) {
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
        ncvslideio::drawChessboardCorners(frame, mBoardSizeUnits, ncvslideio::Mat(mCurrentImagePoints), isTemplateFound);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseACircles(const ncvslideio::Mat &frame)
{
    bool isTemplateFound = findCirclesGrid(frame, mBoardSizeUnits, mCurrentImagePoints, ncvslideio::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);
    if(isTemplateFound) {
        mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);
        ncvslideio::drawChessboardCorners(frame, mBoardSizeUnits, ncvslideio::Mat(mCurrentImagePoints), isTemplateFound);
    }
    return isTemplateFound;
}

bool CalibProcessor::detectAndParseDualACircles(const ncvslideio::Mat &frame)
{
    std::vector<ncvslideio::Point2f> blackPointbuf;

    ncvslideio::Mat invertedView;
    ncvslideio::bitwise_not(frame, invertedView);
    bool isWhiteGridFound = ncvslideio::findCirclesGrid(frame, mBoardSizeUnits, mCurrentImagePoints, ncvslideio::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);
    if(!isWhiteGridFound)
        return false;
    bool isBlackGridFound = ncvslideio::findCirclesGrid(invertedView, mBoardSizeUnits, blackPointbuf, ncvslideio::CALIB_CB_ASYMMETRIC_GRID, mBlobDetectorPtr);

    if(!isBlackGridFound)
    {
        mCurrentImagePoints.clear();
        return false;
    }
    ncvslideio::drawChessboardCorners(frame, mBoardSizeUnits, ncvslideio::Mat(mCurrentImagePoints), isWhiteGridFound);
    ncvslideio::drawChessboardCorners(frame, mBoardSizeUnits, ncvslideio::Mat(blackPointbuf), isBlackGridFound);
    mCurrentImagePoints.insert(mCurrentImagePoints.end(), blackPointbuf.begin(), blackPointbuf.end());
    mTemplateLocations.insert(mTemplateLocations.begin(), mCurrentImagePoints[0]);

    return true;
}

void CalibProcessor::saveFrameData()
{
    std::vector<ncvslideio::Point3f> objectPoints;
    std::vector<ncvslideio::Point2f> imagePoints;

    switch(mBoardType)
    {
    case Chessboard:
        objectPoints.reserve(mBoardSizeInnerCorners.height*mBoardSizeInnerCorners.width);
        for( int i = 0; i < mBoardSizeInnerCorners.height; ++i )
            for( int j = 0; j < mBoardSizeInnerCorners.width; ++j )
                objectPoints.push_back(ncvslideio::Point3f(j*mSquareSize, i*mSquareSize, 0));
        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case ChArUco:
        mCalibData->allCharucoCorners.push_back(mCurrentCharucoCorners);
        mCalibData->allCharucoIds.push_back(mCurrentCharucoIds);

        mCharucoBoard->matchImagePoints(mCurrentCharucoCorners, mCurrentCharucoIds, objectPoints, imagePoints);
        CV_Assert(mCurrentCharucoIds.total() == imagePoints.size());
        mCalibData->imagePoints.push_back(imagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case CirclesGrid:
        objectPoints.reserve(mBoardSizeUnits.height*mBoardSizeUnits.width);
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(ncvslideio::Point3f(j*mSquareSize, i*mSquareSize, 0));
        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case AcirclesGrid:
        objectPoints.reserve(mBoardSizeUnits.height*mBoardSizeUnits.width);
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(ncvslideio::Point3f((2*j + i % 2)*mSquareSize, i*mSquareSize, 0));
        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
        break;
    case DoubleAcirclesGrid:
    {
        float gridCenterX = (2*((float)mBoardSizeUnits.width - 1) + 1)*mSquareSize + mTemplDist / 2;
        float gridCenterY = (mBoardSizeUnits.height - 1)*mSquareSize / 2;
        objectPoints.reserve(2*mBoardSizeUnits.height*mBoardSizeUnits.width);

        //white part
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(
                            ncvslideio::Point3f(-float((2*j + i % 2)*mSquareSize + mTemplDist +
                                               (2*(mBoardSizeUnits.width - 1) + 1)*mSquareSize - gridCenterX),
                                        -float(i*mSquareSize) - gridCenterY,
                                        0));
        //black part
        for( int i = 0; i < mBoardSizeUnits.height; i++ )
            for( int j = 0; j < mBoardSizeUnits.width; j++ )
                objectPoints.push_back(ncvslideio::Point3f(-float((2*j + i % 2)*mSquareSize - gridCenterX),
                                          -float(i*mSquareSize) - gridCenterY, 0));

        mCalibData->imagePoints.push_back(mCurrentImagePoints);
        mCalibData->objectPoints.push_back(objectPoints);
    }
        break;
    }
}

void CalibProcessor::showCaptureMessage(const ncvslideio::Mat& frame, const std::string &message)
{
    ncvslideio::Point textOrigin(100, 100);
    double textSize = VIDEO_TEXT_SIZE * frame.cols / (double) IMAGE_MAX_WIDTH;
    ncvslideio::bitwise_not(frame, frame);
    ncvslideio::putText(frame, message, textOrigin, 1, textSize, ncvslideio::Scalar(0,0,255), 2, ncvslideio::LINE_AA);
    ncvslideio::Mat resized;
    if (std::fabs(mZoom - 1.) > 0.001f)
    {
        ncvslideio::resize(frame, resized, ncvslideio::Size(), mZoom, mZoom);
    }
    else
    {
        resized = frame;
    }
    ncvslideio::imshow(mainWindowName, resized);
    ncvslideio::waitKey(300);
}

bool CalibProcessor::checkLastFrame()
{
    bool isFrameBad = false;
    ncvslideio::Mat tmpCamMatrix;
    const double badAngleThresh = 40;

    if(!mCalibData->cameraMatrix.total()) {
        tmpCamMatrix = ncvslideio::Mat::eye(3, 3, CV_64F);
        tmpCamMatrix.at<double>(0,0) = 20000;
        tmpCamMatrix.at<double>(1,1) = 20000;
        tmpCamMatrix.at<double>(0,2) = mCalibData->imageSize.height/2;
        tmpCamMatrix.at<double>(1,2) = mCalibData->imageSize.width/2;
    }
    else
        mCalibData->cameraMatrix.copyTo(tmpCamMatrix);

    ncvslideio::Mat r, t, angles;
    ncvslideio::solvePnP(mCalibData->objectPoints.back(), mCalibData->imagePoints.back(), tmpCamMatrix, mCalibData->distCoeffs, r, t);
    RodriguesToEuler(r, angles, CALIB_DEGREES);
    if(fabs(angles.at<double>(0)) > badAngleThresh || fabs(angles.at<double>(1)) > badAngleThresh) {
        mCalibData->objectPoints.pop_back();
        mCalibData->imagePoints.pop_back();
        if (mCalibData->allCharucoCorners.size()) {
            mCalibData->allCharucoCorners.pop_back();
            mCalibData->allCharucoIds.pop_back();
        }
        isFrameBad = true;
    }
    return isFrameBad;
}

CalibProcessor::CalibProcessor(ncvslideio::Ptr<calibrationData> data, captureParameters &capParams) :
    mCalibData(data), mBoardType(capParams.board), mBoardSizeUnits(capParams.boardSizeUnits),
    mBoardSizeInnerCorners(capParams.boardSizeInnerCorners)
{
    mCapuredFrames = 0;
    mNeededFramesNum = capParams.calibrationStep;
    mDelayBetweenCaptures = static_cast<int>(capParams.captureDelay * capParams.fps);
    mMaxTemplateOffset = std::sqrt(static_cast<float>(mCalibData->imageSize.height * mCalibData->imageSize.height) +
                                   static_cast<float>(mCalibData->imageSize.width * mCalibData->imageSize.width)) / 20.0;
    mSquareSize = capParams.squareSize;
    mTemplDist = capParams.templDst;
    mSaveFrames = capParams.saveFrames;
    mZoom = capParams.zoom;
    ncvslideio::aruco::CharucoParameters charucoParameters;
    charucoParameters.tryRefineMarkers = true;

    switch(mBoardType)
    {
    case ChArUco:
        if (capParams.charucoDictFile != "None") {
            std::string filename = capParams.charucoDictFile;
            ncvslideio::FileStorage dict_file(filename, ncvslideio::FileStorage::Mode::READ);
            ncvslideio::FileNode fn(dict_file.root());
            mArucoDictionary.readDictionary(fn);
        }
        else {
            mArucoDictionary = ncvslideio::aruco::getPredefinedDictionary(ncvslideio::aruco::PredefinedDictionaryType(capParams.charucoDictName));
        }
        mCharucoBoard = ncvslideio::makePtr<ncvslideio::aruco::CharucoBoard>(ncvslideio::Size(mBoardSizeUnits.width, mBoardSizeUnits.height), capParams.charucoSquareLength,
                                capParams.charucoMarkerSize, mArucoDictionary);
        detector = ncvslideio::makePtr<ncvslideio::aruco::CharucoDetector>(ncvslideio::aruco::CharucoDetector(*mCharucoBoard, charucoParameters));
        break;
    case CirclesGrid:
    case AcirclesGrid:
        mBlobDetectorPtr = ncvslideio::SimpleBlobDetector::create();
        break;
    case DoubleAcirclesGrid:
        mBlobDetectorPtr = ncvslideio::SimpleBlobDetector::create(getDetectorParams());
        break;
    case Chessboard:
        break;
    }
}

ncvslideio::Mat CalibProcessor::processFrame(const ncvslideio::Mat &frame)
{
    ncvslideio::Mat frameCopy;
    ncvslideio::Mat frameCopyToSave;
    frame.copyTo(frameCopy);
    bool isTemplateFound = false;
    mCurrentImagePoints.clear();

    if(mSaveFrames)
        frame.copyTo(frameCopyToSave);

    switch(mBoardType)
    {
    case Chessboard:
        isTemplateFound = detectAndParseChessboard(frameCopy);
        break;
    case ChArUco:
        isTemplateFound = detectAndParseChAruco(frameCopy);
        break;
    case CirclesGrid:
        isTemplateFound = detectAndParseCircles(frameCopy);
        break;
    case AcirclesGrid:
        isTemplateFound = detectAndParseACircles(frameCopy);
        break;
    case DoubleAcirclesGrid:
        isTemplateFound = detectAndParseDualACircles(frameCopy);
        break;
    }

    if(mTemplateLocations.size() > mDelayBetweenCaptures)
        mTemplateLocations.pop_back();
    if(mTemplateLocations.size() == mDelayBetweenCaptures && isTemplateFound) {
        if(ncvslideio::norm(mTemplateLocations.front() - mTemplateLocations.back()) < mMaxTemplateOffset) {
            saveFrameData();
            bool isFrameBad = checkLastFrame();
            if (!isFrameBad) {
                std::string displayMessage = ncvslideio::format("Frame # %zu captured", std::max(mCalibData->imagePoints.size(),
                                                                                        mCalibData->allCharucoCorners.size()));
                if(!showOverlayMessage(displayMessage))
                    showCaptureMessage(frame, displayMessage);

                if(mSaveFrames)
                    mCalibData->allFrames.push_back(frameCopyToSave);

                mCapuredFrames++;
            }
            else {
                std::string displayMessage = "Frame rejected";
                if(!showOverlayMessage(displayMessage))
                    showCaptureMessage(frame, displayMessage);
            }
            mTemplateLocations.clear();
            mTemplateLocations.reserve(mDelayBetweenCaptures);
        }
    }

    return frameCopy;
}

bool CalibProcessor::isProcessed() const
{
    if(mCapuredFrames < mNeededFramesNum)
        return false;
    else
        return true;
}

void CalibProcessor::resetState()
{
    mCapuredFrames = 0;
    mTemplateLocations.clear();
}

CalibProcessor::~CalibProcessor()
{

}

////////////////////////////////////////////

void ShowProcessor::drawBoard(ncvslideio::Mat &img, ncvslideio::InputArray points)
{
    ncvslideio::Mat tmpView = ncvslideio::Mat::zeros(img.rows, img.cols, CV_8UC3);
    std::vector<ncvslideio::Point2f> templateHull;
    std::vector<ncvslideio::Point> poly;
    ncvslideio::convexHull(points, templateHull);
    poly.resize(templateHull.size());
    for(size_t i=0; i<templateHull.size();i++)
        poly[i] = ncvslideio::Point((int)(templateHull[i].x*mGridViewScale), (int)(templateHull[i].y*mGridViewScale));
    ncvslideio::fillConvexPoly(tmpView, poly, ncvslideio::Scalar(0, 255, 0), ncvslideio::LINE_AA);
    ncvslideio::addWeighted(tmpView, .2, img, 1, 0, img);
}

void ShowProcessor::drawGridPoints(const ncvslideio::Mat &frame)
{
    if(mBoardType != ChArUco)
        for(std::vector<std::vector<ncvslideio::Point2f> >::iterator it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it)
            for(std::vector<ncvslideio::Point2f>::iterator pointIt = (*it).begin(); pointIt != (*it).end(); ++pointIt)
                ncvslideio::circle(frame, *pointIt, POINT_SIZE, ncvslideio::Scalar(0, 255, 0), 1, ncvslideio::LINE_AA);
    else
        for(std::vector<ncvslideio::Mat>::iterator it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
            for(int i = 0; i < (*it).size[0]; i++)
                ncvslideio::circle(frame, ncvslideio::Point((int)(*it).at<float>(i, 0), (int)(*it).at<float>(i, 1)),
                           POINT_SIZE, ncvslideio::Scalar(0, 255, 0), 1, ncvslideio::LINE_AA);
}

ShowProcessor::ShowProcessor(ncvslideio::Ptr<calibrationData> data, ncvslideio::Ptr<calibController> controller, TemplateType board) :
    mCalibdata(data), mController(controller), mBoardType(board)
{
    mNeedUndistort = true;
    mVisMode = Grid;
    mGridViewScale = 0.5;
    mTextSize = VIDEO_TEXT_SIZE;
}

ncvslideio::Mat ShowProcessor::processFrame(const ncvslideio::Mat &frame)
{
    if (!mCalibdata->cameraMatrix.empty() && !mCalibdata->distCoeffs.empty())
    {
        mTextSize = VIDEO_TEXT_SIZE * (double) frame.cols / IMAGE_MAX_WIDTH;
        ncvslideio::Scalar textColor = ncvslideio::Scalar(0,0,255);
        ncvslideio::Mat frameCopy;

        if (mNeedUndistort && mController->getFramesNumberState()) {
            if(mVisMode == Grid)
                drawGridPoints(frame);
            ncvslideio::remap(frame, frameCopy, mCalibdata->undistMap1, mCalibdata->undistMap2, ncvslideio::INTER_LINEAR);
            int baseLine = 100;
            ncvslideio::Size textSize = ncvslideio::getTextSize("Undistorted view", 1, mTextSize, 2, &baseLine);
            ncvslideio::Point textOrigin(baseLine, frame.rows - (int)(2.5*textSize.height));
            ncvslideio::putText(frameCopy, "Undistorted view", textOrigin, 1, mTextSize, textColor, 2, ncvslideio::LINE_AA);
        }
        else {
            frame.copyTo(frameCopy);
            if(mVisMode == Grid)
                drawGridPoints(frameCopy);
        }
        std::string displayMessage;
        if(mCalibdata->stdDeviations.at<double>(0) == 0)
            displayMessage = ncvslideio::format("F = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0,0), mCalibdata->totalAvgErr);
        else
            displayMessage = ncvslideio::format("Fx = %d Fy = %d RMS = %.3f", (int)mCalibdata->cameraMatrix.at<double>(0,0),
                                            (int)mCalibdata->cameraMatrix.at<double>(1,1), mCalibdata->totalAvgErr);
        if(mController->getRMSState() && mController->getFramesNumberState())
            displayMessage.append(" OK");

        int baseLine = 100;
        ncvslideio::Size textSize = ncvslideio::getTextSize(displayMessage, 1, mTextSize - 1, 2, &baseLine);
        ncvslideio::Point textOrigin = ncvslideio::Point(baseLine, 2*textSize.height);
        ncvslideio::putText(frameCopy, displayMessage, textOrigin, 1, mTextSize - 1, textColor, 2, ncvslideio::LINE_AA);

        if(mCalibdata->stdDeviations.at<double>(0) == 0)
            displayMessage = ncvslideio::format("DF = %.2f", mCalibdata->stdDeviations.at<double>(1)*sigmaMult);
        else
            displayMessage = ncvslideio::format("DFx = %.2f DFy = %.2f", mCalibdata->stdDeviations.at<double>(0)*sigmaMult,
                                                    mCalibdata->stdDeviations.at<double>(1)*sigmaMult);
        if(mController->getConfidenceIntrervalsState() && mController->getFramesNumberState())
            displayMessage.append(" OK");
        ncvslideio::putText(frameCopy, displayMessage, ncvslideio::Point(baseLine, 4*textSize.height), 1, mTextSize - 1, textColor, 2, ncvslideio::LINE_AA);

        if(mController->getCommonCalibrationState()) {
            displayMessage = ncvslideio::format("Calibration is done");
            ncvslideio::putText(frameCopy, displayMessage, ncvslideio::Point(baseLine, 6*textSize.height), 1, mTextSize - 1, textColor, 2, ncvslideio::LINE_AA);
        }
        int calibFlags = mController->getNewFlags();
        displayMessage = "";
        if(!(calibFlags & ncvslideio::CALIB_FIX_ASPECT_RATIO))
            displayMessage.append(ncvslideio::format("AR=%.3f ", mCalibdata->cameraMatrix.at<double>(0,0)/mCalibdata->cameraMatrix.at<double>(1,1)));
        if(calibFlags & ncvslideio::CALIB_ZERO_TANGENT_DIST)
            displayMessage.append("TD=0 ");
        displayMessage.append(ncvslideio::format("K1=%.2f K2=%.2f K3=%.2f", mCalibdata->distCoeffs.at<double>(0), mCalibdata->distCoeffs.at<double>(1),
                                         mCalibdata->distCoeffs.at<double>(4)));
        ncvslideio::putText(frameCopy, displayMessage, ncvslideio::Point(baseLine, frameCopy.rows - (int)(1.5*textSize.height)),
                    1, mTextSize - 1, textColor, 2, ncvslideio::LINE_AA);
        return frameCopy;
    }

    return frame;
}

bool ShowProcessor::isProcessed() const
{
    return false;
}

void ShowProcessor::resetState()
{

}

void ShowProcessor::setVisualizationMode(visualisationMode mode)
{
    mVisMode = mode;
}

void ShowProcessor::switchVisualizationMode()
{
    if(mVisMode == Grid) {
        mVisMode = Window;
        updateBoardsView();
    }
    else {
        mVisMode = Grid;
        ncvslideio::destroyWindow(gridWindowName);
    }
}

void ShowProcessor::clearBoardsView()
{
    ncvslideio::imshow(gridWindowName, ncvslideio::Mat());
}

void ShowProcessor::updateBoardsView()
{
    if(mVisMode == Window) {
        ncvslideio::Size originSize = mCalibdata->imageSize;
        ncvslideio::Mat altGridView = ncvslideio::Mat::zeros((int)(originSize.height*mGridViewScale), (int)(originSize.width*mGridViewScale), CV_8UC3);
        if(mBoardType != ChArUco)
            for(std::vector<std::vector<ncvslideio::Point2f> >::iterator it = mCalibdata->imagePoints.begin(); it != mCalibdata->imagePoints.end(); ++it)
                if(mBoardType != DoubleAcirclesGrid)
                    drawBoard(altGridView, *it);
                else {
                    size_t pointsNum = (*it).size()/2;
                    std::vector<ncvslideio::Point2f> points(pointsNum);
                    std::copy((*it).begin(), (*it).begin() + pointsNum, points.begin());
                    drawBoard(altGridView, points);
                    std::copy((*it).begin() + pointsNum, (*it).begin() + 2*pointsNum, points.begin());
                    drawBoard(altGridView, points);
                }
        else
            for(std::vector<ncvslideio::Mat>::iterator it = mCalibdata->allCharucoCorners.begin(); it != mCalibdata->allCharucoCorners.end(); ++it)
                drawBoard(altGridView, *it);
        ncvslideio::imshow(gridWindowName, altGridView);
    }
}

void ShowProcessor::switchUndistort()
{
    mNeedUndistort = !mNeedUndistort;
}

void ShowProcessor::setUndistort(bool isEnabled)
{
    mNeedUndistort = isEnabled;
}

ShowProcessor::~ShowProcessor()
{

}
