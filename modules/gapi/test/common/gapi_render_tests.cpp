// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "../test_precomp.hpp"
#include "gapi_render_tests.hpp"

namespace opencv_test
{

ncvslideio::Scalar cvtBGRToYUVC(const ncvslideio::Scalar& bgr)
{
    double y = bgr[2] *  0.299000 + bgr[1] *  0.587000 + bgr[0] *  0.114000;
    double u = bgr[2] * -0.168736 + bgr[1] * -0.331264 + bgr[0] *  0.500000 + 128;
    double v = bgr[2] *  0.500000 + bgr[1] * -0.418688 + bgr[0] * -0.081312 + 128;
    return {y, u, v};
}

void drawMosaicRef(const ncvslideio::Mat& mat, const ncvslideio::Rect &rect, int cellSz)
{
    ncvslideio::Rect mat_rect(0, 0, mat.cols, mat.rows);
    auto intersection = mat_rect & rect;

    ncvslideio::Mat msc_roi = mat(intersection);

    bool has_crop_x = false;
    bool has_crop_y = false;

    int cols = msc_roi.cols;
    int rows = msc_roi.rows;

    if (msc_roi.cols % cellSz != 0)
    {
        has_crop_x = true;
        cols -= msc_roi.cols % cellSz;
    }

    if (msc_roi.rows % cellSz != 0)
    {
        has_crop_y = true;
        rows -= msc_roi.rows % cellSz;
    }

    ncvslideio::Mat cell_roi;
    for(int i = 0; i < rows; i += cellSz )
    {
        for(int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(ncvslideio::Rect(j, i, cellSz, cellSz));
            cell_roi = ncvslideio::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(ncvslideio::Rect(cols, i, msc_roi.cols - cols, cellSz));
            cell_roi = ncvslideio::mean(cell_roi);
        }
    }

    if (has_crop_y)
    {
        for(int j = 0; j < cols; j += cellSz)
        {
            cell_roi = msc_roi(ncvslideio::Rect(j, rows, cellSz, msc_roi.rows - rows));
            cell_roi = ncvslideio::mean(cell_roi);
        }
        if (has_crop_x)
        {
            cell_roi = msc_roi(ncvslideio::Rect(cols, rows, msc_roi.cols - cols, msc_roi.rows - rows));
            cell_roi = ncvslideio::mean(cell_roi);
        }
    }
}

void blendImageRef(ncvslideio::Mat& mat, const ncvslideio::Point& org, const ncvslideio::Mat& img, const ncvslideio::Mat& alpha)
{
    auto roi = mat(ncvslideio::Rect(org, img.size()));
    ncvslideio::Mat img32f_w;
    ncvslideio::merge(std::vector<ncvslideio::Mat>(3, alpha), img32f_w);

    ncvslideio::Mat roi32f_w(roi.size(), CV_32FC3, ncvslideio::Scalar::all(1.0));
    roi32f_w -= img32f_w;

    ncvslideio::Mat img32f, roi32f;
    img.convertTo(img32f, CV_32F, 1.0/255);
    roi.convertTo(roi32f, CV_32F, 1.0/255);

    ncvslideio::multiply(img32f, img32f_w, img32f);
    ncvslideio::multiply(roi32f, roi32f_w, roi32f);
    roi32f += img32f;

    roi32f.convertTo(roi, CV_8U, 255.0);
}

} // namespace opencv_test
