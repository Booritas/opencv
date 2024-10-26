#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"

#include <iostream>
#include <cctype>

static ncvslideio::UMat image;
static bool backprojMode = false;
static bool selectObject = false;
static int trackObject = 0;
static bool showHist = true;
static ncvslideio::Rect selection;
static int vmin = 10, vmax = 256, smin = 30;

static void onMouse(int event, int x, int y, int, void*)
{
    static ncvslideio::Point origin;

    if (selectObject)
    {
        selection.x = std::min(x, origin.x);
        selection.y = std::min(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);

        selection &= ncvslideio::Rect(0, 0, image.cols, image.rows);
    }

    switch (event)
    {
    case ncvslideio::EVENT_LBUTTONDOWN:
        origin = ncvslideio::Point(x, y);
        selection = ncvslideio::Rect(x, y, 0, 0);
        selectObject = true;
        break;
    case ncvslideio::EVENT_LBUTTONUP:
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            trackObject = -1;
        break;
    default:
        break;
    }
}

static void help()
{
    std::cout << "\nThis is a demo that shows mean-shift based tracking using Transparent API\n"
            "You select a color objects such as your face and it tracks it.\n"
            "This reads from video camera (0 by default, or the camera number the user enters\n"
            "Usage: \n"
            "   ./camshiftdemo [camera number]\n";

    std::cout << "\n\nHot keys: \n"
            "\tESC - quit the program\n"
            "\ts - stop the tracking\n"
            "\tb - switch to/from backprojection view\n"
            "\th - show/hide object histogram\n"
            "\tp - pause video\n"
            "\tc - use OpenCL or not\n"
            "To initialize tracking, select the object with mouse\n";
}

int main(int argc, const char ** argv)
{
    help();

    ncvslideio::VideoCapture cap;
    ncvslideio::Rect trackWindow;
    int hsize = 16;
    float hranges[2] = { 0, 180 };

    const char * const keys = { "{@camera_number| 0 | camera number}" };
    ncvslideio::CommandLineParser parser(argc, argv, keys);
    int camNum = parser.get<int>(0);

    cap.open(camNum);

    if (!cap.isOpened())
    {
        help();

        std::cout << "***Could not initialize capturing...***\n";
        std::cout << "Current parameter's value: \n";
        parser.printMessage();

        return EXIT_FAILURE;
    }

    ncvslideio::namedWindow("Histogram", ncvslideio::WINDOW_NORMAL);
    ncvslideio::namedWindow("CamShift Demo", ncvslideio::WINDOW_NORMAL);
    ncvslideio::setMouseCallback("CamShift Demo", onMouse);
    ncvslideio::createTrackbar("Vmin", "CamShift Demo", &vmin, 256);
    ncvslideio::createTrackbar("Vmax", "CamShift Demo", &vmax, 256);
    ncvslideio::createTrackbar("Smin", "CamShift Demo", &smin, 256);

    ncvslideio::Mat frame, histimg(200, 320, CV_8UC3, ncvslideio::Scalar::all(0));
    ncvslideio::UMat hsv, hist, hue, mask, backproj;
    bool paused = false;

    for ( ; ; )
    {
        if (!paused)
        {
            cap >> frame;
            if (frame.empty())
                break;
        }

        frame.copyTo(image);

        if (!paused)
        {
            ncvslideio::cvtColor(image, hsv, ncvslideio::COLOR_BGR2HSV);

            if (trackObject)
            {
                int _vmin = vmin, _vmax = vmax;

                ncvslideio::inRange(hsv, ncvslideio::Scalar(0, smin, std::min(_vmin, _vmax)),
                        ncvslideio::Scalar(180, 256, std::max(_vmin, _vmax)), mask);

                int fromTo[2] = { 0,0 };
                hue.create(hsv.size(), hsv.depth());
                ncvslideio::mixChannels(std::vector<ncvslideio::UMat>(1, hsv), std::vector<ncvslideio::UMat>(1, hue), fromTo, 1);

                if (trackObject < 0)
                {
                    ncvslideio::UMat roi(hue, selection), maskroi(mask, selection);
                    ncvslideio::calcHist(std::vector<ncvslideio::Mat>(1, roi.getMat(ncvslideio::ACCESS_READ)), std::vector<int>(1, 0),
                                 maskroi, hist, std::vector<int>(1, hsize), std::vector<float>(hranges, hranges + 2));
                    ncvslideio::normalize(hist, hist, 0, 255, ncvslideio::NORM_MINMAX);

                    trackWindow = selection;
                    trackObject = 1;

                    histimg = ncvslideio::Scalar::all(0);
                    int binW = histimg.cols / hsize;
                    ncvslideio::Mat buf (1, hsize, CV_8UC3);
                    for (int i = 0; i < hsize; i++)
                        buf.at<ncvslideio::Vec3b>(i) = ncvslideio::Vec3b(ncvslideio::saturate_cast<uchar>(i*180./hsize), 255, 255);
                    ncvslideio::cvtColor(buf, buf, ncvslideio::COLOR_HSV2BGR);

                    {
                        ncvslideio::Mat _hist = hist.getMat(ncvslideio::ACCESS_READ);
                        for (int i = 0; i < hsize; i++)
                        {
                            int val = ncvslideio::saturate_cast<int>(_hist.at<float>(i)*histimg.rows/255);
                            ncvslideio::rectangle(histimg, ncvslideio::Point(i*binW, histimg.rows),
                                       ncvslideio::Point((i+1)*binW, histimg.rows - val),
                                       ncvslideio::Scalar(buf.at<ncvslideio::Vec3b>(i)), -1, 8);
                        }
                    }
                }

                ncvslideio::calcBackProject(std::vector<ncvslideio::UMat>(1, hue), std::vector<int>(1, 0), hist, backproj,
                                    std::vector<float>(hranges, hranges + 2), 1.0);
                ncvslideio::bitwise_and(backproj, mask, backproj);

                ncvslideio::RotatedRect trackBox = ncvslideio::CamShift(backproj, trackWindow,
                                    ncvslideio::TermCriteria(ncvslideio::TermCriteria::EPS | ncvslideio::TermCriteria::COUNT, 10, 1));
                if (trackWindow.area() <= 1)
                {
                    int cols = backproj.cols, rows = backproj.rows, r = (std::min(cols, rows) + 5)/6;
                    trackWindow = ncvslideio::Rect(trackWindow.x - r, trackWindow.y - r,
                                       trackWindow.x + r, trackWindow.y + r) &
                                  ncvslideio::Rect(0, 0, cols, rows);
                }

                if (backprojMode)
                    ncvslideio::cvtColor(backproj, image, ncvslideio::COLOR_GRAY2BGR);

                {
                    ncvslideio::Mat _image = image.getMat(ncvslideio::ACCESS_RW);
                    ncvslideio::ellipse(_image, trackBox, ncvslideio::Scalar(0, 0, 255), 3, ncvslideio::LINE_AA);
                }
            }
        }
        else if (trackObject < 0)
            paused = false;

        if (selectObject && selection.width > 0 && selection.height > 0)
        {
            ncvslideio::UMat roi(image, selection);
            ncvslideio::bitwise_not(roi, roi);
        }

        ncvslideio::imshow("CamShift Demo", image);
        if (showHist)
            ncvslideio::imshow("Histogram", histimg);

        char c = (char)ncvslideio::waitKey(10);
        if (c == 27)
            break;

        switch(c)
        {
        case 'b':
            backprojMode = !backprojMode;
            break;
        case 't':
            trackObject = 0;
            histimg = ncvslideio::Scalar::all(0);
            break;
        case 'h':
            showHist = !showHist;
            if (!showHist)
                ncvslideio::destroyWindow("Histogram");
            else
                ncvslideio::namedWindow("Histogram", ncvslideio::WINDOW_AUTOSIZE);
            break;
        case 'p':
            paused = !paused;
            break;
        case 'c':
            ncvslideio::ocl::setUseOpenCL(!ncvslideio::ocl::useOpenCL());
        default:
            break;
        }
    }

    return EXIT_SUCCESS;
}
