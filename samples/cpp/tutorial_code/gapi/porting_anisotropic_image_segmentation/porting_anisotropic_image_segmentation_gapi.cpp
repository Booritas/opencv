/**
* @brief You will learn how port an existing algorithm to G-API
* @author Dmitry Matveev, dmitry.matveev@intel.com, based
*    on sample by Karpushin Vladislav, karpushin@ngs.ru
*/
#include "opencv2/opencv_modules.hpp"
#ifdef HAVE_OPENCV_GAPI

//! [full_sample]
#include <iostream>
#include <utility>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/gapi.hpp"
#include "opencv2/gapi/core.hpp"
#include "opencv2/gapi/imgproc.hpp"

//! [calcGST_proto]
void calcGST(const ncvslideio::GMat& inputImg, ncvslideio::GMat& imgCoherencyOut, ncvslideio::GMat& imgOrientationOut, int w);
//! [calcGST_proto]

int main()
{
    int W = 52;             // window size is WxW
    double C_Thr = 0.43;    // threshold for coherency
    int LowThr = 35;        // threshold1 for orientation, it ranges from 0 to 180
    int HighThr = 57;       // threshold2 for orientation, it ranges from 0 to 180

    ncvslideio::Mat imgIn = ncvslideio::imread("input.jpg", ncvslideio::IMREAD_GRAYSCALE);
    if (imgIn.empty()) //check whether the image is loaded or not
    {
        std::cout << "ERROR : Image cannot be loaded..!!" << std::endl;
        return -1;
    }

    //! [main]
    // Calculate Gradient Structure Tensor and post-process it for output with G-API
    ncvslideio::GMat in;
    ncvslideio::GMat imgCoherency, imgOrientation;
    calcGST(in, imgCoherency, imgOrientation, W);

    ncvslideio::GMat imgCoherencyBin = imgCoherency > C_Thr;
    ncvslideio::GMat imgOrientationBin = ncvslideio::gapi::inRange(imgOrientation, LowThr, HighThr);
    ncvslideio::GMat imgBin = imgCoherencyBin & imgOrientationBin;
    ncvslideio::GMat out = ncvslideio::gapi::addWeighted(in, 0.5, imgBin, 0.5, 0.0);

    // Normalize extra outputs
    ncvslideio::GMat imgCoherencyNorm = ncvslideio::gapi::normalize(imgCoherency, 0, 255, ncvslideio::NORM_MINMAX);
    ncvslideio::GMat imgOrientationNorm = ncvslideio::gapi::normalize(imgOrientation, 0, 255, ncvslideio::NORM_MINMAX);

    // Capture the graph into object segm
    ncvslideio::GComputation segm(ncvslideio::GIn(in), ncvslideio::GOut(out, imgCoherencyNorm, imgOrientationNorm));

    // Define ncvslideio::Mats for output data
    ncvslideio::Mat imgOut, imgOutCoherency, imgOutOrientation;

    // Run the graph
    segm.apply(ncvslideio::gin(imgIn), ncvslideio::gout(imgOut, imgOutCoherency, imgOutOrientation));

    ncvslideio::imwrite("result.jpg", imgOut);
    ncvslideio::imwrite("Coherency.jpg", imgOutCoherency);
    ncvslideio::imwrite("Orientation.jpg", imgOutOrientation);
    //! [main]

    return 0;
}
//! [calcGST]
//! [calcGST_header]
void calcGST(const ncvslideio::GMat& inputImg, ncvslideio::GMat& imgCoherencyOut, ncvslideio::GMat& imgOrientationOut, int w)
{
    auto img = ncvslideio::gapi::convertTo(inputImg, CV_32F);
    auto imgDiffX = ncvslideio::gapi::Sobel(img, CV_32F, 1, 0, 3);
    auto imgDiffY = ncvslideio::gapi::Sobel(img, CV_32F, 0, 1, 3);
    auto imgDiffXY = ncvslideio::gapi::mul(imgDiffX, imgDiffY);
    //! [calcGST_header]

    auto imgDiffXX = ncvslideio::gapi::mul(imgDiffX, imgDiffX);
    auto imgDiffYY = ncvslideio::gapi::mul(imgDiffY, imgDiffY);

    auto J11 = ncvslideio::gapi::boxFilter(imgDiffXX, CV_32F, ncvslideio::Size(w, w));
    auto J22 = ncvslideio::gapi::boxFilter(imgDiffYY, CV_32F, ncvslideio::Size(w, w));
    auto J12 = ncvslideio::gapi::boxFilter(imgDiffXY, CV_32F, ncvslideio::Size(w, w));

    auto tmp1 = J11 + J22;
    auto tmp2 = J11 - J22;
    auto tmp22 = ncvslideio::gapi::mul(tmp2, tmp2);
    auto tmp3 = ncvslideio::gapi::mul(J12, J12);
    auto tmp4 = ncvslideio::gapi::sqrt(tmp22 + 4.0*tmp3);

    auto lambda1 = tmp1 + tmp4;
    auto lambda2 = tmp1 - tmp4;

    imgCoherencyOut = (lambda1 - lambda2) / (lambda1 + lambda2);
    imgOrientationOut = 0.5*ncvslideio::gapi::phase(J22 - J11, 2.0*J12, true);
}
//! [calcGST]

//! [full_sample]

#else
#include <iostream>
int main()
{
    std::cerr << "This tutorial code requires G-API module to run" << std::endl;
}
#endif  // HAVE_OPECV_GAPI
