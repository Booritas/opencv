// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019 Intel Corporation

#include "opencv2/opencv_modules.hpp"
#if defined(HAVE_OPENCV_GAPI)

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/infer.hpp>
#include <opencv2/gapi/infer/ie.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#include <opencv2/highgui.hpp> // windows

namespace config
{
constexpr char       kWinFaceBeautification[] = "FaceBeautificator";
constexpr char       kWinInput[]              = "Input";
constexpr char       kParserAbout[]           =
        "Use this script to run the face beautification algorithm with G-API.";
constexpr char       kParserOptions[]         =
"{ help         h ||      print the help message. }"

"{ facepath     f ||      a path to a Face detection model file (.xml).}"
"{ facedevice     |GPU|   the face detection computation device.}"

"{ landmpath    l ||      a path to a Landmarks detection model file (.xml).}"
"{ landmdevice    |CPU|   the landmarks detection computation device.}"

"{ input        i ||      a path to an input. Skip to capture from a camera.}"
"{ boxes        b |false| set true to draw face Boxes in the \"Input\" window.}"
"{ landmarks    m |false| set true to draw landMarks in the \"Input\" window.}"
"{ streaming    s |true|  set false to disable stream pipelining.}"
"{ performance  p |false| set true to disable output displaying.}";

const     ncvslideio::Scalar kClrWhite (255, 255, 255);
const     ncvslideio::Scalar kClrGreen (  0, 255,   0);
const     ncvslideio::Scalar kClrYellow(  0, 255, 255);

constexpr float      kConfThresh   = 0.7f;

const     ncvslideio::Size   kGKernelSize(5, 5);
constexpr double     kGSigma       = 0.0;
constexpr int        kBSize        = 9;
constexpr double     kBSigmaCol    = 30.0;
constexpr double     kBSigmaSp     = 30.0;
constexpr int        kUnshSigma    = 3;
constexpr float      kUnshStrength = 0.7f;
constexpr int        kAngDelta     = 1;
constexpr bool       kClosedLine   = true;
} // namespace config

namespace
{
//! [vec_ROI]
using VectorROI = std::vector<ncvslideio::Rect>;
//! [vec_ROI]
using GArrayROI = ncvslideio::GArray<ncvslideio::Rect>;
using Contour   = std::vector<ncvslideio::Point>;
using Landmarks = std::vector<ncvslideio::Point>;


// Wrapper function
template<typename Tp> inline int toIntRounded(const Tp x)
{
    return static_cast<int>(std::lround(x));
}

//! [toDbl]
template<typename Tp> inline double toDouble(const Tp x)
{
    return static_cast<double>(x);
}
//! [toDbl]

struct Avg {
       struct Elapsed {
           explicit Elapsed(double ms) : ss(ms / 1000.),
                                         mm(toIntRounded(ss / 60)) {}
           const double ss;
           const int    mm;
       };

       using MS = std::chrono::duration<double, std::ratio<1, 1000>>;
       using TS = std::chrono::time_point<std::chrono::high_resolution_clock>;
       TS started;

       void    start() { started = now(); }
       TS      now() const { return std::chrono::high_resolution_clock::now(); }
       double  tick() const { return std::chrono::duration_cast<MS>(now() - started).count(); }
       Elapsed elapsed() const { return Elapsed{tick()}; }
       double  fps(std::size_t n) const { return static_cast<double>(n) / (tick() / 1000.); }
   };
std::ostream& operator<<(std::ostream &os, const Avg::Elapsed &e) {
   os << e.mm << ':' << (e.ss - 60*e.mm);
   return os;
}

std::string getWeightsPath(const std::string &mdlXMLPath) // mdlXMLPath =
                                                          // "The/Full/Path.xml"
{
    size_t size = mdlXMLPath.size();
    CV_Assert(mdlXMLPath.substr(size - 4, size)           // The last 4 symbols
                  == ".xml");                             // must be ".xml"
    std::string mdlBinPath(mdlXMLPath);
    return mdlBinPath.replace(size - 3, 3, "bin");        // return
                                                          // "The/Full/Path.bin"
}
} // anonymous namespace



namespace custom
{
using TplPtsFaceElements_Jaw = std::tuple<ncvslideio::GArray<Landmarks>,
                                          ncvslideio::GArray<Contour>>;

// Wrapper-functions
inline int getLineInclinationAngleDegrees(const ncvslideio::Point &ptLeft,
                                          const ncvslideio::Point &ptRight);
inline Contour getForeheadEllipse(const ncvslideio::Point &ptJawLeft,
                                  const ncvslideio::Point &ptJawRight,
                                  const ncvslideio::Point &ptJawMiddle);
inline Contour getEyeEllipse(const ncvslideio::Point &ptLeft,
                             const ncvslideio::Point &ptRight);
inline Contour getPatchedEllipse(const ncvslideio::Point &ptLeft,
                                 const ncvslideio::Point &ptRight,
                                 const ncvslideio::Point &ptUp,
                                 const ncvslideio::Point &ptDown);

// Networks
//! [net_decl]
G_API_NET(FaceDetector,  <ncvslideio::GMat(ncvslideio::GMat)>, "face_detector");
G_API_NET(LandmDetector, <ncvslideio::GMat(ncvslideio::GMat)>, "landm_detector");
//! [net_decl]

// Function kernels
G_TYPED_KERNEL(GBilatFilter, <ncvslideio::GMat(ncvslideio::GMat,int,double,double)>,
               "custom.faceb12n.bilateralFilter")
{
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in, int,double,double)
    {
        return in;
    }
};

G_TYPED_KERNEL(GLaplacian, <ncvslideio::GMat(ncvslideio::GMat,int)>,
               "custom.faceb12n.Laplacian")
{
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in, int)
    {
        return in;
    }
};

G_TYPED_KERNEL(GFillPolyGContours, <ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GArray<Contour>)>,
               "custom.faceb12n.fillPolyGContours")
{
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in, ncvslideio::GArrayDesc)
    {
        return in.withType(CV_8U, 1);
    }
};

G_TYPED_KERNEL(GPolyLines, <ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GArray<Contour>,bool,
                                     ncvslideio::Scalar)>,
               "custom.faceb12n.polyLines")
{
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in, ncvslideio::GArrayDesc,bool,ncvslideio::Scalar)
    {
        return in;
    }
};

G_TYPED_KERNEL(GRectangle, <ncvslideio::GMat(ncvslideio::GMat,GArrayROI,ncvslideio::Scalar)>,
               "custom.faceb12n.rectangle")
{
    static ncvslideio::GMatDesc outMeta(ncvslideio::GMatDesc in, ncvslideio::GArrayDesc,ncvslideio::Scalar)
    {
        return in;
    }
};

G_TYPED_KERNEL(GFacePostProc, <GArrayROI(ncvslideio::GMat,ncvslideio::GMat,float)>,
               "custom.faceb12n.faceDetectPostProc")
{
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc&,const ncvslideio::GMatDesc&,float)
    {
        return ncvslideio::empty_array_desc();
    }
};

G_TYPED_KERNEL_M(GLandmPostProc, <TplPtsFaceElements_Jaw(ncvslideio::GArray<ncvslideio::GMat>,
                                                         GArrayROI)>,
                 "custom.faceb12n.landmDetectPostProc")
{
    static std::tuple<ncvslideio::GArrayDesc,ncvslideio::GArrayDesc> outMeta(
                const ncvslideio::GArrayDesc&,const ncvslideio::GArrayDesc&)
    {
        return std::make_tuple(ncvslideio::empty_array_desc(), ncvslideio::empty_array_desc());
    }
};

//! [kern_m_decl]
using TplFaces_FaceElements  = std::tuple<ncvslideio::GArray<Contour>, ncvslideio::GArray<Contour>>;
G_TYPED_KERNEL_M(GGetContours, <TplFaces_FaceElements (ncvslideio::GArray<Landmarks>, ncvslideio::GArray<Contour>)>,
                 "custom.faceb12n.getContours")
{
    static std::tuple<ncvslideio::GArrayDesc,ncvslideio::GArrayDesc> outMeta(const ncvslideio::GArrayDesc&,const ncvslideio::GArrayDesc&)
    {
        return std::make_tuple(ncvslideio::empty_array_desc(), ncvslideio::empty_array_desc());
    }
};
//! [kern_m_decl]


// OCV_Kernels
// This kernel applies Bilateral filter to an input src with default
//  "ncvslideio::bilateralFilter" border argument
GAPI_OCV_KERNEL(GCPUBilateralFilter, custom::GBilatFilter)
{
    static void run(const ncvslideio::Mat &src,
                    const int      diameter,
                    const double   sigmaColor,
                    const double   sigmaSpace,
                          ncvslideio::Mat &out)
    {
        ncvslideio::bilateralFilter(src, out, diameter, sigmaColor, sigmaSpace);
    }
};

// This kernel applies Laplace operator to an input src with default
//  "ncvslideio::Laplacian" arguments
GAPI_OCV_KERNEL(GCPULaplacian, custom::GLaplacian)
{
    static void run(const ncvslideio::Mat &src,
                    const int      ddepth,
                          ncvslideio::Mat &out)
    {
        ncvslideio::Laplacian(src, out, ddepth);
    }
};

// This kernel draws given white filled contours "cnts" on a clear Mat "out"
//  (defined by a Scalar(0)) with standard "ncvslideio::fillPoly" arguments.
//  It should be used to create a mask.
// The input Mat seems unused inside the function "run", but it is used deeper
//  in the kernel to define an output size.
GAPI_OCV_KERNEL(GCPUFillPolyGContours, custom::GFillPolyGContours)
{
    static void run(const ncvslideio::Mat              &,
                    const std::vector<Contour> &cnts,
                          ncvslideio::Mat              &out)
    {
        out = ncvslideio::Scalar(0);
        ncvslideio::fillPoly(out, cnts, config::kClrWhite);
    }
};

// This kernel draws given contours on an input src with default "ncvslideio::polylines"
//  arguments
GAPI_OCV_KERNEL(GCPUPolyLines, custom::GPolyLines)
{
    static void run(const ncvslideio::Mat              &src,
                    const std::vector<Contour> &cnts,
                    const bool                  isClosed,
                    const ncvslideio::Scalar           &color,
                          ncvslideio::Mat              &out)
    {
        src.copyTo(out);
        ncvslideio::polylines(out, cnts, isClosed, color);
    }
};

// This kernel draws given rectangles on an input src with default
//  "ncvslideio::rectangle" arguments
GAPI_OCV_KERNEL(GCPURectangle, custom::GRectangle)
{
    static void run(const ncvslideio::Mat    &src,
                    const VectorROI  &vctFaceBoxes,
                    const ncvslideio::Scalar &color,
                          ncvslideio::Mat    &out)
    {
        src.copyTo(out);
        for (const ncvslideio::Rect &box : vctFaceBoxes)
        {
            ncvslideio::rectangle(out, box, color);
        }
    }
};

// A face detector outputs a blob with the shape: [1, 1, N, 7], where N is
//  the number of detected bounding boxes. Structure of an output for every
//  detected face is the following:
//  [image_id, label, conf, x_min, y_min, x_max, y_max], all the seven elements
//  are floating point. For more details please visit:
// https://github.com/opencv/open_model_zoo/blob/master/intel_models/face-detection-adas-0001
// This kernel is the face detection output blob parsing that returns a vector
//  of detected faces' rects:
//! [fd_pp]
GAPI_OCV_KERNEL(GCPUFacePostProc, GFacePostProc)
{
    static void run(const ncvslideio::Mat   &inDetectResult,
                    const ncvslideio::Mat   &inFrame,
                    const float      faceConfThreshold,
                          VectorROI &outFaces)
    {
        const int kObjectSize  = 7;
        const int imgCols = inFrame.size().width;
        const int imgRows = inFrame.size().height;
        const ncvslideio::Rect borders({0, 0}, inFrame.size());
        outFaces.clear();
        const int    numOfDetections = inDetectResult.size[2];
        const float *data            = inDetectResult.ptr<float>();
        for (int i = 0; i < numOfDetections; i++)
        {
            const float faceId         = data[i * kObjectSize + 0];
            if (faceId < 0.f)  // indicates the end of detections
            {
                break;
            }
            const float faceConfidence = data[i * kObjectSize + 2];
            // We can cut detections by the `conf` field
            //  to avoid mistakes of the detector.
            if (faceConfidence > faceConfThreshold)
            {
                const float left   = data[i * kObjectSize + 3];
                const float top    = data[i * kObjectSize + 4];
                const float right  = data[i * kObjectSize + 5];
                const float bottom = data[i * kObjectSize + 6];
                // These are normalized coordinates and are between 0 and 1;
                //  to get the real pixel coordinates we should multiply it by
                //  the image sizes respectively to the directions:
                ncvslideio::Point tl(toIntRounded(left   * imgCols),
                             toIntRounded(top    * imgRows));
                ncvslideio::Point br(toIntRounded(right  * imgCols),
                             toIntRounded(bottom * imgRows));
                outFaces.push_back(ncvslideio::Rect(tl, br) & borders);
            }
        }
    }
};
//! [fd_pp]

// This kernel is the facial landmarks detection output Mat parsing for every
//  detected face; returns a tuple containing a vector of vectors of
//  face elements' Points and a vector of vectors of jaw's Points:
// There are 35 landmarks given by the default detector for each face
//  in a frame; the first 18 of them are face elements (eyes, eyebrows,
//  a nose, a mouth) and the last 17 - a jaw contour. The detector gives
//  floating point values for landmarks' normed coordinates relatively
//  to an input ROI (not the original frame).
//  For more details please visit:
// https://github.com/opencv/open_model_zoo/blob/master/intel_models/facial-landmarks-35-adas-0002
GAPI_OCV_KERNEL(GCPULandmPostProc, GLandmPostProc)
{
    static void run(const std::vector<ncvslideio::Mat>   &vctDetectResults,
                    const VectorROI              &vctRects,
                          std::vector<Landmarks> &vctPtsFaceElems,
                          std::vector<Contour>   &vctCntJaw)
    {
        static constexpr int kNumFaceElems = 18;
        static constexpr int kNumTotal     = 35;
        const size_t numFaces = vctRects.size();
        CV_Assert(vctPtsFaceElems.size() == 0ul);
        CV_Assert(vctCntJaw.size()       == 0ul);
        vctPtsFaceElems.reserve(numFaces);
        vctCntJaw.reserve(numFaces);

        Landmarks ptsFaceElems;
        Contour   cntJaw;
        ptsFaceElems.reserve(kNumFaceElems);
        cntJaw.reserve(kNumTotal - kNumFaceElems);

        for (size_t i = 0; i < numFaces; i++)
        {
            const float *data = vctDetectResults[i].ptr<float>();
            // The face elements points:
            ptsFaceElems.clear();
            for (int j = 0; j < kNumFaceElems * 2; j += 2)
            {
                ncvslideio::Point pt = ncvslideio::Point(toIntRounded(data[j]   * vctRects[i].width),
                                         toIntRounded(data[j+1] * vctRects[i].height)) + vctRects[i].tl();
                ptsFaceElems.push_back(pt);
            }
            vctPtsFaceElems.push_back(ptsFaceElems);

            // The jaw contour points:
            cntJaw.clear();
            for(int j = kNumFaceElems * 2; j < kNumTotal * 2; j += 2)
            {
                ncvslideio::Point pt = ncvslideio::Point(toIntRounded(data[j]   * vctRects[i].width),
                                         toIntRounded(data[j+1] * vctRects[i].height)) + vctRects[i].tl();
                cntJaw.push_back(pt);
            }
            vctCntJaw.push_back(cntJaw);
        }
    }
};

// This kernel is the facial landmarks detection post-processing for every face
//  detected before; output is a tuple of vectors of detected face contours and
//  facial elements contours:
//! [ld_pp_cnts]
//! [kern_m_impl]
GAPI_OCV_KERNEL(GCPUGetContours, GGetContours)
{
    static void run(const std::vector<Landmarks> &vctPtsFaceElems,  // 18 landmarks of the facial elements
                    const std::vector<Contour>   &vctCntJaw,        // 17 landmarks of a jaw
                          std::vector<Contour>   &vctElemsContours,
                          std::vector<Contour>   &vctFaceContours)
    {
//! [kern_m_impl]
        size_t numFaces = vctCntJaw.size();
        CV_Assert(numFaces == vctPtsFaceElems.size());
        CV_Assert(vctElemsContours.size() == 0ul);
        CV_Assert(vctFaceContours.size()  == 0ul);
        // vctFaceElemsContours will store all the face elements' contours found
        //  in an input image, namely 4 elements (two eyes, nose, mouth) for every detected face:
        vctElemsContours.reserve(numFaces * 4);
        // vctFaceElemsContours will store all the faces' contours found in an input image:
        vctFaceContours.reserve(numFaces);

        Contour cntFace, cntLeftEye, cntRightEye, cntNose, cntMouth;
        cntNose.reserve(4);

        for (size_t i = 0ul; i < numFaces; i++)
        {
            // The face elements contours

            // A left eye:
            // Approximating the lower eye contour by half-ellipse (using eye points) and storing in cntLeftEye:
            cntLeftEye = getEyeEllipse(vctPtsFaceElems[i][1], vctPtsFaceElems[i][0]);
            // Pushing the left eyebrow clock-wise:
            cntLeftEye.insert(cntLeftEye.end(), {vctPtsFaceElems[i][12], vctPtsFaceElems[i][13],
                                                 vctPtsFaceElems[i][14]});

            // A right eye:
            // Approximating the lower eye contour by half-ellipse (using eye points) and storing in vctRightEye:
            cntRightEye = getEyeEllipse(vctPtsFaceElems[i][2], vctPtsFaceElems[i][3]);
            // Pushing the right eyebrow clock-wise:
            cntRightEye.insert(cntRightEye.end(), {vctPtsFaceElems[i][15], vctPtsFaceElems[i][16],
                                                   vctPtsFaceElems[i][17]});

            // A nose:
            // Storing the nose points clock-wise
            cntNose.clear();
            cntNose.insert(cntNose.end(), {vctPtsFaceElems[i][4], vctPtsFaceElems[i][7],
                                           vctPtsFaceElems[i][5], vctPtsFaceElems[i][6]});

            // A mouth:
            // Approximating the mouth contour by two half-ellipses (using mouth points) and storing in vctMouth:
            cntMouth = getPatchedEllipse(vctPtsFaceElems[i][8], vctPtsFaceElems[i][9],
                                         vctPtsFaceElems[i][10], vctPtsFaceElems[i][11]);

            // Storing all the elements in a vector:
            vctElemsContours.insert(vctElemsContours.end(), {cntLeftEye, cntRightEye, cntNose, cntMouth});

            // The face contour:
            // Approximating the forehead contour by half-ellipse (using jaw points) and storing in vctFace:
            cntFace = getForeheadEllipse(vctCntJaw[i][0], vctCntJaw[i][16], vctCntJaw[i][8]);
            // The ellipse is drawn clock-wise, but jaw contour points goes vice versa, so it's necessary to push
            //  cntJaw from the end to the begin using a reverse iterator:
            std::copy(vctCntJaw[i].crbegin(), vctCntJaw[i].crend(), std::back_inserter(cntFace));
            // Storing the face contour in another vector:
            vctFaceContours.push_back(cntFace);
        }
    }
};
//! [ld_pp_cnts]

// GAPI subgraph functions
inline ncvslideio::GMat unsharpMask(const ncvslideio::GMat &src,
                            const int       sigma,
                            const float     strength);
inline ncvslideio::GMat mask3C(const ncvslideio::GMat &src,
                       const ncvslideio::GMat &mask);
} // namespace custom


// Functions implementation:
// Returns an angle (in degrees) between a line given by two Points and
//  the horison. Note that the result depends on the arguments order:
//! [ld_pp_incl]
inline int custom::getLineInclinationAngleDegrees(const ncvslideio::Point &ptLeft, const ncvslideio::Point &ptRight)
{
    const ncvslideio::Point residual = ptRight - ptLeft;
    if (residual.y == 0 && residual.x == 0)
        return 0;
    else
        return toIntRounded(atan2(toDouble(residual.y), toDouble(residual.x)) * 180.0 / CV_PI);
}
//! [ld_pp_incl]

// Approximates a forehead by half-ellipse using jaw points and some geometry
//  and then returns points of the contour; "capacity" is used to reserve enough
//  memory as there will be other points inserted.
//! [ld_pp_fhd]
inline Contour custom::getForeheadEllipse(const ncvslideio::Point &ptJawLeft,
                                          const ncvslideio::Point &ptJawRight,
                                          const ncvslideio::Point &ptJawLower)
{
    Contour cntForehead;
    // The point amid the top two points of a jaw:
    const ncvslideio::Point ptFaceCenter((ptJawLeft + ptJawRight) / 2);
    // This will be the center of the ellipse.

    // The angle between the jaw and the vertical:
    const int angFace = getLineInclinationAngleDegrees(ptJawLeft, ptJawRight);
    // This will be the inclination of the ellipse

    // Counting the half-axis of the ellipse:
    const double jawWidth  = ncvslideio::norm(ptJawLeft - ptJawRight);
    // A forehead width equals the jaw width, and we need a half-axis:
    const int axisX        = toIntRounded(jawWidth / 2.0);

    const double jawHeight = ncvslideio::norm(ptFaceCenter - ptJawLower);
    // According to research, in average a forehead is approximately 2/3 of
    //  a jaw:
    const int axisY        = toIntRounded(jawHeight * 2 / 3.0);

    // We need the upper part of an ellipse:
    static constexpr int kAngForeheadStart = 180;
    static constexpr int kAngForeheadEnd   = 360;
    ncvslideio::ellipse2Poly(ptFaceCenter, ncvslideio::Size(axisX, axisY), angFace, kAngForeheadStart, kAngForeheadEnd,
                     config::kAngDelta, cntForehead);
    return cntForehead;
}
//! [ld_pp_fhd]

// Approximates the lower eye contour by half-ellipse using eye points and some
//  geometry and then returns points of the contour.
//! [ld_pp_eye]
inline Contour custom::getEyeEllipse(const ncvslideio::Point &ptLeft, const ncvslideio::Point &ptRight)
{
    Contour cntEyeBottom;
    const ncvslideio::Point ptEyeCenter((ptRight + ptLeft) / 2);
    const int angle = getLineInclinationAngleDegrees(ptLeft, ptRight);
    const int axisX = toIntRounded(ncvslideio::norm(ptRight - ptLeft) / 2.0);
    // According to research, in average a Y axis of an eye is approximately
    //  1/3 of an X one.
    const int axisY = axisX / 3;
    // We need the lower part of an ellipse:
    static constexpr int kAngEyeStart = 0;
    static constexpr int kAngEyeEnd   = 180;
    ncvslideio::ellipse2Poly(ptEyeCenter, ncvslideio::Size(axisX, axisY), angle, kAngEyeStart, kAngEyeEnd, config::kAngDelta,
                     cntEyeBottom);
    return cntEyeBottom;
}
//! [ld_pp_eye]

//This function approximates an object (a mouth) by two half-ellipses using
//  4 points of the axes' ends and then returns points of the contour:
inline Contour custom::getPatchedEllipse(const ncvslideio::Point &ptLeft,
                                         const ncvslideio::Point &ptRight,
                                         const ncvslideio::Point &ptUp,
                                         const ncvslideio::Point &ptDown)
{
    // Shared characteristics for both half-ellipses:
    const ncvslideio::Point ptMouthCenter((ptLeft + ptRight) / 2);
    const int angMouth = getLineInclinationAngleDegrees(ptLeft, ptRight);
    const int axisX    = toIntRounded(ncvslideio::norm(ptRight - ptLeft) / 2.0);

    // The top half-ellipse:
    Contour cntMouthTop;
    const int axisYTop = toIntRounded(ncvslideio::norm(ptMouthCenter - ptUp));
    // We need the upper part of an ellipse:
    static constexpr int angTopStart = 180;
    static constexpr int angTopEnd   = 360;
    ncvslideio::ellipse2Poly(ptMouthCenter, ncvslideio::Size(axisX, axisYTop), angMouth, angTopStart, angTopEnd, config::kAngDelta, cntMouthTop);

    // The bottom half-ellipse:
    Contour cntMouth;
    const int axisYBot = toIntRounded(ncvslideio::norm(ptMouthCenter - ptDown));
    // We need the lower part of an ellipse:
    static constexpr int angBotStart = 0;
    static constexpr int angBotEnd   = 180;
    ncvslideio::ellipse2Poly(ptMouthCenter, ncvslideio::Size(axisX, axisYBot), angMouth, angBotStart, angBotEnd, config::kAngDelta, cntMouth);

    // Pushing the upper part to vctOut
    std::copy(cntMouthTop.cbegin(), cntMouthTop.cend(), std::back_inserter(cntMouth));
    return cntMouth;
}

//! [unsh]
inline ncvslideio::GMat custom::unsharpMask(const ncvslideio::GMat &src,
                                    const int       sigma,
                                    const float     strength)
{
    ncvslideio::GMat blurred   = ncvslideio::gapi::medianBlur(src, sigma);
    ncvslideio::GMat laplacian = custom::GLaplacian::on(blurred, CV_8U);
    return (src - (laplacian * strength));
}
//! [unsh]

inline ncvslideio::GMat custom::mask3C(const ncvslideio::GMat &src,
                               const ncvslideio::GMat &mask)
{
    std::tuple<ncvslideio::GMat,ncvslideio::GMat,ncvslideio::GMat> tplIn = ncvslideio::gapi::split3(src);
    ncvslideio::GMat masked0 = ncvslideio::gapi::mask(std::get<0>(tplIn), mask);
    ncvslideio::GMat masked1 = ncvslideio::gapi::mask(std::get<1>(tplIn), mask);
    ncvslideio::GMat masked2 = ncvslideio::gapi::mask(std::get<2>(tplIn), mask);
    return ncvslideio::gapi::merge3(masked0, masked1, masked2);
}


int main(int argc, char** argv)
{
    ncvslideio::namedWindow(config::kWinFaceBeautification, ncvslideio::WINDOW_NORMAL);
    ncvslideio::namedWindow(config::kWinInput,              ncvslideio::WINDOW_NORMAL);

    ncvslideio::CommandLineParser parser(argc, argv, config::kParserOptions);
    parser.about(config::kParserAbout);
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    // Parsing input arguments
    const std::string faceXmlPath = parser.get<std::string>("facepath");
    const std::string faceBinPath = getWeightsPath(faceXmlPath);
    const std::string faceDevice  = parser.get<std::string>("facedevice");

    const std::string landmXmlPath = parser.get<std::string>("landmpath");
    const std::string landmBinPath = getWeightsPath(landmXmlPath);
    const std::string landmDevice  = parser.get<std::string>("landmdevice");

    // Declaring a graph
    // The version of a pipeline expression with a lambda-based
    //  constructor is used to keep all temporary objects in a dedicated scope.
//! [ppl]
    ncvslideio::GComputation pipeline([=]()
    {
//! [net_usg_fd]
        ncvslideio::GMat  gimgIn;                                                                           // input

        ncvslideio::GMat  faceOut  = ncvslideio::gapi::infer<custom::FaceDetector>(gimgIn);
//! [net_usg_fd]
        GArrayROI garRects = custom::GFacePostProc::on(faceOut, gimgIn, config::kConfThresh);       // post-proc

//! [net_usg_ld]
        ncvslideio::GArray<ncvslideio::GMat> landmOut  = ncvslideio::gapi::infer<custom::LandmDetector>(garRects, gimgIn);
//! [net_usg_ld]
        ncvslideio::GArray<Landmarks> garElems;                                                             // |
        ncvslideio::GArray<Contour>   garJaws;                                                              // |output arrays
        std::tie(garElems, garJaws)    = custom::GLandmPostProc::on(landmOut, garRects);            // post-proc
        ncvslideio::GArray<Contour> garElsConts;                                                            // face elements
        ncvslideio::GArray<Contour> garFaceConts;                                                           // whole faces
        std::tie(garElsConts, garFaceConts) = custom::GGetContours::on(garElems, garJaws);          // interpolation

//! [msk_ppline]
        ncvslideio::GMat mskSharp        = custom::GFillPolyGContours::on(gimgIn, garElsConts);             // |
        ncvslideio::GMat mskSharpG       = ncvslideio::gapi::gaussianBlur(mskSharp, config::kGKernelSize,           // |
                                                          config::kGSigma);                         // |
        ncvslideio::GMat mskBlur         = custom::GFillPolyGContours::on(gimgIn, garFaceConts);            // |
        ncvslideio::GMat mskBlurG        = ncvslideio::gapi::gaussianBlur(mskBlur, config::kGKernelSize,            // |
                                                          config::kGSigma);                         // |draw masks
        // The first argument in mask() is Blur as we want to subtract from                         // |
        // BlurG the next step:                                                                     // |
        ncvslideio::GMat mskBlurFinal    = mskBlurG - ncvslideio::gapi::mask(mskBlurG, mskSharpG);                  // |
        ncvslideio::GMat mskFacesGaussed = mskBlurFinal + mskSharpG;                                        // |
        ncvslideio::GMat mskFacesWhite   = ncvslideio::gapi::threshold(mskFacesGaussed, 0, 255, ncvslideio::THRESH_BINARY); // |
        ncvslideio::GMat mskNoFaces      = ncvslideio::gapi::bitwise_not(mskFacesWhite);                            // |
//! [msk_ppline]

        ncvslideio::GMat gimgBilat       = custom::GBilatFilter::on(gimgIn, config::kBSize,
                                                            config::kBSigmaCol, config::kBSigmaSp);
        ncvslideio::GMat gimgSharp       = custom::unsharpMask(gimgIn, config::kUnshSigma,
                                                       config::kUnshStrength);
        // Applying the masks
        // Custom function mask3C() should be used instead of just gapi::mask()
        //  as mask() provides CV_8UC1 source only (and we have CV_8U3C)
        ncvslideio::GMat gimgBilatMasked = custom::mask3C(gimgBilat, mskBlurFinal);
        ncvslideio::GMat gimgSharpMasked = custom::mask3C(gimgSharp, mskSharpG);
        ncvslideio::GMat gimgInMasked    = custom::mask3C(gimgIn,    mskNoFaces);
        ncvslideio::GMat gimgBeautif = gimgBilatMasked + gimgSharpMasked + gimgInMasked;
        return ncvslideio::GComputation(ncvslideio::GIn(gimgIn), ncvslideio::GOut(gimgBeautif,
                                                          ncvslideio::gapi::copy(gimgIn),
                                                          garFaceConts,
                                                          garElsConts,
                                                          garRects));
    });
//! [ppl]
    // Declaring IE params for networks
//! [net_param]
    auto faceParams  = ncvslideio::gapi::ie::Params<custom::FaceDetector>
    {
        /*std::string*/ faceXmlPath,
        /*std::string*/ faceBinPath,
        /*std::string*/ faceDevice
    };
    auto landmParams = ncvslideio::gapi::ie::Params<custom::LandmDetector>
    {
        /*std::string*/ landmXmlPath,
        /*std::string*/ landmBinPath,
        /*std::string*/ landmDevice
    };
//! [net_param]
//! [netw]
    auto networks      = ncvslideio::gapi::networks(faceParams, landmParams);
//! [netw]
    // Declaring custom and fluid kernels have been used:
//! [kern_pass_1]
    auto customKernels = ncvslideio::gapi::kernels<custom::GCPUBilateralFilter,
                                           custom::GCPULaplacian,
                                           custom::GCPUFillPolyGContours,
                                           custom::GCPUPolyLines,
                                           custom::GCPURectangle,
                                           custom::GCPUFacePostProc,
                                           custom::GCPULandmPostProc,
                                           custom::GCPUGetContours>();
    auto kernels       = ncvslideio::gapi::combine(ncvslideio::gapi::core::fluid::kernels(),
                                           customKernels);
//! [kern_pass_1]

    Avg avg;
    size_t frames = 0;

    // The flags for drawing/not drawing face boxes or/and landmarks in the
    //  \"Input\" window:
    const bool flgBoxes     = parser.get<bool>("boxes");
    const bool flgLandmarks = parser.get<bool>("landmarks");
    // The flag to involve stream pipelining:
    const bool flgStreaming = parser.get<bool>("streaming");
    // The flag to display the output images or not:
    const bool flgPerformance = parser.get<bool>("performance");
    // Now we are ready to compile the pipeline to a stream with specified
    //  kernels, networks and image format expected to process
    if (flgStreaming == true)
    {
//! [str_comp]
        ncvslideio::GStreamingCompiled stream = pipeline.compileStreaming(ncvslideio::compile_args(kernels, networks));
//! [str_comp]
        // Setting the source for the stream:
//! [str_src]
        if (parser.has("input"))
        {
            stream.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(parser.get<ncvslideio::String>("input")));
        }
//! [str_src]
        else
        {
            stream.setSource(ncvslideio::gapi::wip::make_src<ncvslideio::gapi::wip::GCaptureSource>(0));
        }
        // Declaring output variables
        // Streaming:
        ncvslideio::Mat imgShow;
        ncvslideio::Mat imgBeautif;
        std::vector<Contour> vctFaceConts, vctElsConts;
        VectorROI vctRects;
        if (flgPerformance == true)
        {
            auto out_vector = ncvslideio::gout(imgBeautif, imgShow, vctFaceConts,
                                       vctElsConts, vctRects);
            stream.start();
            avg.start();
            while (stream.running())
            {
                stream.pull(std::move(out_vector));
                frames++;
            }
        }
        else // flgPerformance == false
        {
//! [str_loop]
            auto out_vector = ncvslideio::gout(imgBeautif, imgShow, vctFaceConts,
                                       vctElsConts, vctRects);
            stream.start();
            avg.start();
            while (stream.running())
            {
                if (!stream.try_pull(std::move(out_vector)))
                {
                    // Use a try_pull() to obtain data.
                    // If there's no data, let UI refresh (and handle keypress)
                    if (ncvslideio::waitKey(1) >= 0) break;
                    else continue;
                }
                frames++;
                // Drawing face boxes and landmarks if necessary:
                if (flgLandmarks == true)
                {
                    ncvslideio::polylines(imgShow, vctFaceConts, config::kClosedLine,
                                  config::kClrYellow);
                    ncvslideio::polylines(imgShow, vctElsConts, config::kClosedLine,
                                  config::kClrYellow);
                }
                if (flgBoxes == true)
                    for (auto rect : vctRects)
                        ncvslideio::rectangle(imgShow, rect, config::kClrGreen);
                ncvslideio::imshow(config::kWinInput,              imgShow);
                ncvslideio::imshow(config::kWinFaceBeautification, imgBeautif);
            }
//! [str_loop]
        }
        std::cout << "Processed " << frames << " frames in " << avg.elapsed()
                  << " (" << avg.fps(frames) << " FPS)" << std::endl;
    }
    else // serial mode:
    {
//! [bef_cap]
#include <opencv2/videoio.hpp>
        ncvslideio::GCompiled cc;
        ncvslideio::VideoCapture cap;
        if (parser.has("input"))
        {
            cap.open(parser.get<ncvslideio::String>("input"));
        }
//! [bef_cap]
        else if (!cap.open(0))
        {
            std::cout << "No input available" << std::endl;
            return 1;
        }
        if (flgPerformance == true)
        {
            while (true)
            {
                ncvslideio::Mat img;
                ncvslideio::Mat imgShow;
                ncvslideio::Mat imgBeautif;
                std::vector<Contour> vctFaceConts, vctElsConts;
                VectorROI vctRects;
                cap >> img;
                if (img.empty())
                {
                   break;
                }
                frames++;
                if (!cc)
                {
                    cc = pipeline.compile(ncvslideio::descr_of(img), ncvslideio::compile_args(kernels, networks));
                    avg.start();
                }
                cc(ncvslideio::gin(img), ncvslideio::gout(imgBeautif, imgShow, vctFaceConts,
                                          vctElsConts, vctRects));
            }
        }
        else // flgPerformance == false
        {
//! [bef_loop]
            while (ncvslideio::waitKey(1) < 0)
            {
                ncvslideio::Mat img;
                ncvslideio::Mat imgShow;
                ncvslideio::Mat imgBeautif;
                std::vector<Contour> vctFaceConts, vctElsConts;
                VectorROI vctRects;
                cap >> img;
                if (img.empty())
                {
                   ncvslideio::waitKey();
                   break;
                }
                frames++;
//! [apply]
                pipeline.apply(ncvslideio::gin(img), ncvslideio::gout(imgBeautif, imgShow,
                                                      vctFaceConts,
                                                      vctElsConts, vctRects),
                               ncvslideio::compile_args(kernels, networks));
//! [apply]
                if (frames == 1)
                {
                    // Start timer only after 1st frame processed -- compilation
                    // happens on-the-fly here
                    avg.start();
                }
                // Drawing face boxes and landmarks if necessary:
                if (flgLandmarks == true)
                {
                    ncvslideio::polylines(imgShow, vctFaceConts, config::kClosedLine,
                                  config::kClrYellow);
                    ncvslideio::polylines(imgShow, vctElsConts, config::kClosedLine,
                                  config::kClrYellow);
                }
                if (flgBoxes == true)
                    for (auto rect : vctRects)
                        ncvslideio::rectangle(imgShow, rect, config::kClrGreen);
                ncvslideio::imshow(config::kWinInput,              imgShow);
                ncvslideio::imshow(config::kWinFaceBeautification, imgBeautif);
            }
        }
//! [bef_loop]
        std::cout << "Processed " << frames << " frames in " << avg.elapsed()
                  << " (" << avg.fps(frames) << " FPS)" << std::endl;
    }
    return 0;
}
#else
#include <iostream>
int main()
{
    std::cerr << "This tutorial code requires G-API module "
                 "with Inference Engine backend to run"
              << std::endl;
    return 1;
}
#endif  // HAVE_OPECV_GAPI
