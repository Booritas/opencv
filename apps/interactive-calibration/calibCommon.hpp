// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef CALIB_COMMON_HPP
#define CALIB_COMMON_HPP

#include <opencv2/core.hpp>

#include <memory>
#include <vector>
#include <string>

namespace calib
{
    #define OVERLAY_DELAY 1000
    #define IMAGE_MAX_WIDTH 1280
    #define IMAGE_MAX_HEIGHT 960

    bool showOverlayMessage(const std::string& message);

    enum InputType { Video, Pictures };
    enum InputVideoSource { Camera, File };
    enum TemplateType { AcirclesGrid, Chessboard, ChArUco, DoubleAcirclesGrid, CirclesGrid };

    static const std::string mainWindowName = "Calibration";
    static const std::string gridWindowName = "Board locations";
    static const std::string consoleHelp = "Hot keys:\nesc - exit application\n"
                              "s - save current data to .xml file\n"
                              "r - delete last frame\n"
                              "u - enable/disable applying undistortion\n"
                              "d - delete all frames\n"
                              "v - switch visualization";

    static const double sigmaMult = 1.96;

    struct calibrationData
    {
        ncvslideio::Mat cameraMatrix;
        ncvslideio::Mat distCoeffs;
        ncvslideio::Mat stdDeviations;
        ncvslideio::Mat perViewErrors;
        std::vector<ncvslideio::Mat> rvecs;
        std::vector<ncvslideio::Mat> tvecs;
        double totalAvgErr;
        ncvslideio::Size imageSize;

        std::vector<ncvslideio::Mat> allFrames;

        std::vector<std::vector<ncvslideio::Point2f> > imagePoints;
        std::vector< std::vector<ncvslideio::Point3f> > objectPoints;

        std::vector<ncvslideio::Mat> allCharucoCorners;
        std::vector<ncvslideio::Mat> allCharucoIds;

        ncvslideio::Mat undistMap1, undistMap2;

        calibrationData()
        {
            imageSize = ncvslideio::Size(IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT);
        }
    };

    struct cameraParameters
    {
        ncvslideio::Mat cameraMatrix;
        ncvslideio::Mat distCoeffs;
        ncvslideio::Mat stdDeviations;
        double avgError;

        cameraParameters(){}
        cameraParameters(ncvslideio::Mat& _cameraMatrix, ncvslideio::Mat& _distCoeffs, ncvslideio::Mat& _stdDeviations, double _avgError = 0) :
            cameraMatrix(_cameraMatrix), distCoeffs(_distCoeffs), stdDeviations(_stdDeviations), avgError(_avgError)
        {}
    };

    struct captureParameters
    {
        InputType captureMethod;
        InputVideoSource source;
        TemplateType board;
        ncvslideio::Size inputBoardSize;
        ncvslideio::Size boardSizeInnerCorners; // board size in inner corners for chessboard
        ncvslideio::Size boardSizeUnits; // board size in squares, circles, etc.
        int charucoDictName;
        std::string charucoDictFile;
        int calibrationStep;
        float charucoSquareLength, charucoMarkerSize;
        float captureDelay;
        float squareSize;
        float templDst;
        std::string videoFileName;
        bool flipVertical;
        int camID;
        int fps;
        ncvslideio::Size cameraResolution;
        int maxFramesNum;
        int minFramesNum;
        bool saveFrames;
        float zoom;
        bool forceReopen;

        captureParameters()
        {
            calibrationStep = 1;
            captureDelay = 500.f;
            maxFramesNum = 30;
            minFramesNum = 10;
            fps = 30;
            cameraResolution = ncvslideio::Size(IMAGE_MAX_WIDTH, IMAGE_MAX_HEIGHT);
            saveFrames = false;
        }
    };

    struct internalParameters
    {
        double solverEps;
        int solverMaxIters;
        bool fastSolving;
        double filterAlpha;

        internalParameters()
        {
            solverEps = 1e-7;
            solverMaxIters = 30;
            fastSolving = false;
            filterAlpha = 0.1;
        }
    };
}

#endif
