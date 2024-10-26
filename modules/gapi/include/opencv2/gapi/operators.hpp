// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPERATORS_HPP
#define OPENCV_GAPI_OPERATORS_HPP

#include <opencv2/gapi/gmat.hpp>
#include <opencv2/gapi/gscalar.hpp>

namespace ncvslideio
{
GAPI_EXPORTS ncvslideio::GMat operator+(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator+(const ncvslideio::GMat&    lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator+(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator-(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator-(const ncvslideio::GMat&    lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator-(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator*(const ncvslideio::GMat&    lhs, float              rhs);
GAPI_EXPORTS ncvslideio::GMat operator*(float              lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator*(const ncvslideio::GMat&    lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator*(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator/(const ncvslideio::GMat&    lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator/(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator/(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator&(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator|(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator^(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator~(const ncvslideio::GMat&    lhs);

GAPI_EXPORTS ncvslideio::GMat operator&(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator|(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator^(const ncvslideio::GScalar& lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator&(const ncvslideio::GMat& lhs, const ncvslideio::GScalar&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator|(const ncvslideio::GMat& lhs, const ncvslideio::GScalar&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator^(const ncvslideio::GMat& lhs, const ncvslideio::GScalar&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator>(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator>=(const ncvslideio::GMat&   lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator<(const ncvslideio::GMat&    lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator<=(const ncvslideio::GMat&   lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator==(const ncvslideio::GMat&   lhs, const ncvslideio::GMat&    rhs);
GAPI_EXPORTS ncvslideio::GMat operator!=(const ncvslideio::GMat&   lhs, const ncvslideio::GMat&    rhs);

GAPI_EXPORTS ncvslideio::GMat operator>(const ncvslideio::GMat&    lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator>=(const ncvslideio::GMat&   lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator<(const ncvslideio::GMat&    lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator<=(const ncvslideio::GMat&   lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator==(const ncvslideio::GMat&   lhs, const ncvslideio::GScalar& rhs);
GAPI_EXPORTS ncvslideio::GMat operator!=(const ncvslideio::GMat&   lhs, const ncvslideio::GScalar& rhs);

GAPI_EXPORTS ncvslideio::GMat operator>(const ncvslideio::GScalar&    lhs, const ncvslideio::GMat& rhs);
GAPI_EXPORTS ncvslideio::GMat operator>=(const ncvslideio::GScalar&   lhs, const ncvslideio::GMat& rhs);
GAPI_EXPORTS ncvslideio::GMat operator<(const ncvslideio::GScalar&    lhs, const ncvslideio::GMat& rhs);
GAPI_EXPORTS ncvslideio::GMat operator<=(const ncvslideio::GScalar&   lhs, const ncvslideio::GMat& rhs);
GAPI_EXPORTS ncvslideio::GMat operator==(const ncvslideio::GScalar&   lhs, const ncvslideio::GMat& rhs);
GAPI_EXPORTS ncvslideio::GMat operator!=(const ncvslideio::GScalar&   lhs, const ncvslideio::GMat& rhs);
} // cv

#endif // OPENCV_GAPI_OPERATORS_HPP
