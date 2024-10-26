// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"

#ifndef OPENCV_EXCLUDE_C_API

CV_IMPL CvScalar cvSum( const CvArr* srcarr )
{
    ncvslideio::Scalar sum = ncvslideio::sum(ncvslideio::cvarrToMat(srcarr, false, true, 1));
    if( CV_IS_IMAGE(srcarr) )
    {
        int coi = cvGetImageCOI((IplImage*)srcarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            sum = ncvslideio::Scalar(sum[coi-1]);
        }
    }
    return cvScalar(sum);
}

CV_IMPL int cvCountNonZero( const CvArr* imgarr )
{
    ncvslideio::Mat img = ncvslideio::cvarrToMat(imgarr, false, true, 1);
    if( img.channels() > 1 )
        ncvslideio::extractImageCOI(imgarr, img);
    return countNonZero(img);
}


CV_IMPL  CvScalar
cvAvg( const void* imgarr, const void* maskarr )
{
    ncvslideio::Mat img = ncvslideio::cvarrToMat(imgarr, false, true, 1);
    ncvslideio::Scalar mean = !maskarr ? ncvslideio::mean(img) : ncvslideio::mean(img, ncvslideio::cvarrToMat(maskarr));
    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = ncvslideio::Scalar(mean[coi-1]);
        }
    }
    return cvScalar(mean);
}


CV_IMPL  void
cvAvgSdv( const CvArr* imgarr, CvScalar* _mean, CvScalar* _sdv, const void* maskarr )
{
    ncvslideio::Scalar mean, sdv;

    ncvslideio::Mat mask;
    if( maskarr )
        mask = ncvslideio::cvarrToMat(maskarr);

    ncvslideio::meanStdDev(ncvslideio::cvarrToMat(imgarr, false, true, 1), mean, sdv, mask );

    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = ncvslideio::Scalar(mean[coi-1]);
            sdv = ncvslideio::Scalar(sdv[coi-1]);
        }
    }

    if( _mean )
        *(ncvslideio::Scalar*)_mean = mean;
    if( _sdv )
        *(ncvslideio::Scalar*)_sdv = sdv;
}


CV_IMPL void
cvMinMaxLoc( const void* imgarr, double* _minVal, double* _maxVal,
             CvPoint* _minLoc, CvPoint* _maxLoc, const void* maskarr )
{
    ncvslideio::Mat mask, img = ncvslideio::cvarrToMat(imgarr, false, true, 1);
    if( maskarr )
        mask = ncvslideio::cvarrToMat(maskarr);
    if( img.channels() > 1 )
        ncvslideio::extractImageCOI(imgarr, img);

    ncvslideio::minMaxLoc( img, _minVal, _maxVal,
                   (ncvslideio::Point*)_minLoc, (ncvslideio::Point*)_maxLoc, mask );
}


CV_IMPL  double
cvNorm( const void* imgA, const void* imgB, int normType, const void* maskarr )
{
    ncvslideio::Mat a, mask;
    if( !imgA )
    {
        imgA = imgB;
        imgB = 0;
    }

    a = ncvslideio::cvarrToMat(imgA, false, true, 1);
    if( maskarr )
        mask = ncvslideio::cvarrToMat(maskarr);

    if( a.channels() > 1 && CV_IS_IMAGE(imgA) && cvGetImageCOI((const IplImage*)imgA) > 0 )
        ncvslideio::extractImageCOI(imgA, a);

    if( !imgB )
        return !maskarr ? ncvslideio::norm(a, normType) : ncvslideio::norm(a, normType, mask);

    ncvslideio::Mat b = ncvslideio::cvarrToMat(imgB, false, true, 1);
    if( b.channels() > 1 && CV_IS_IMAGE(imgB) && cvGetImageCOI((const IplImage*)imgB) > 0 )
        ncvslideio::extractImageCOI(imgB, b);

    return !maskarr ? ncvslideio::norm(a, b, normType) : ncvslideio::norm(a, b, normType, mask);
}

#endif  // OPENCV_EXCLUDE_C_API
