// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef ROTATION_CONVERTERS_HPP
#define ROTATION_CONVERTERS_HPP

#include <opencv2/core.hpp>

namespace calib
{
#define CALIB_RADIANS 0
#define CALIB_DEGREES 1

    void Euler(const ncvslideio::Mat& src, ncvslideio::Mat& dst, int argType = CALIB_RADIANS);
    void RodriguesToEuler(const ncvslideio::Mat& src, ncvslideio::Mat& dst, int argType = CALIB_RADIANS);
    void EulerToRodrigues(const ncvslideio::Mat& src, ncvslideio::Mat& dst, int argType = CALIB_RADIANS);

}
#endif
