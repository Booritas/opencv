// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <opencv2/gapi/infer/parsers.hpp>

#ifndef OPENCV_NNPARSERS_OCV_HPP
#define OPENCV_NNPARSERS_OCV_HPP

namespace ncvslideio
{
void ParseSSD(const ncvslideio::Mat&  in_ssd_result,
              const ncvslideio::Size& in_size,
              const float     confidence_threshold,
              const int       filter_label,
              const bool      alignment_to_square,
              const bool      filter_out_of_bounds,
              std::vector<ncvslideio::Rect>& out_boxes,
              std::vector<int>&      out_labels);

void parseYolo(const ncvslideio::Mat&  in_yolo_result,
               const ncvslideio::Size& in_size,
               const float     confidence_threshold,
               const float     nms_threshold,
               const std::vector<float>& anchors,
               std::vector<ncvslideio::Rect>& out_boxes,
               std::vector<int>&      out_labels);
}
#endif // OPENCV_NNPARSERS_OCV_HPP
