// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/gscalar.hpp>
#include <opencv2/gapi/operators.hpp>

namespace ncvslideio
{
ncvslideio::GMat operator+(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::add(lhs, rhs);
}

ncvslideio::GMat operator+(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::addC(lhs, rhs);
}

ncvslideio::GMat operator+(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::addC(rhs, lhs);
}

ncvslideio::GMat operator-(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::sub(lhs, rhs);
}

ncvslideio::GMat operator-(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::subC(lhs, rhs);
}

ncvslideio::GMat operator-(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::subRC(lhs, rhs);
}

ncvslideio::GMat operator*(const ncvslideio::GMat& lhs, float rhs)
{
    return ncvslideio::gapi::mulC(lhs, static_cast<double>(rhs));
}

ncvslideio::GMat operator*(float lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::mulC(rhs, static_cast<double>(lhs));
}

ncvslideio::GMat operator*(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::mulC(lhs, rhs);
}

ncvslideio::GMat operator*(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::mulC(rhs, lhs);
}

ncvslideio::GMat operator/(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::divC(lhs, rhs, 1.0);
}

ncvslideio::GMat operator/(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::div(lhs, rhs, 1.0);
}

ncvslideio::GMat operator/(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::divRC(lhs, rhs, 1.0);
}

ncvslideio::GMat operator&(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::bitwise_and(lhs, rhs);
}

ncvslideio::GMat operator&(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::bitwise_and(lhs, rhs);
}

ncvslideio::GMat operator&(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::bitwise_and(rhs, lhs);
}

ncvslideio::GMat operator|(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::bitwise_or(lhs, rhs);
}

ncvslideio::GMat operator|(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::bitwise_or(lhs, rhs);
}

ncvslideio::GMat operator|(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::bitwise_or(rhs, lhs);
}

ncvslideio::GMat operator^(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::bitwise_xor(lhs, rhs);
}

ncvslideio::GMat operator^(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::bitwise_xor(lhs, rhs);
}

ncvslideio::GMat operator^(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::bitwise_xor(rhs, lhs);
}

ncvslideio::GMat operator~(const ncvslideio::GMat& lhs)
{
    return ncvslideio::gapi::bitwise_not(lhs);
}

ncvslideio::GMat operator>(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpGT(lhs, rhs);
}

ncvslideio::GMat operator>=(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpGE(lhs, rhs);
}

ncvslideio::GMat operator<(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpLT(lhs, rhs);
}

ncvslideio::GMat operator<=(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpLE(lhs, rhs);
}

ncvslideio::GMat operator==(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpEQ(lhs, rhs);
}

ncvslideio::GMat operator!=(const ncvslideio::GMat& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpNE(lhs, rhs);
}

ncvslideio::GMat operator>(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::cmpGT(lhs, rhs);
}

ncvslideio::GMat operator>=(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::cmpGE(lhs, rhs);
}

ncvslideio::GMat operator<(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::cmpLT(lhs, rhs);
}

ncvslideio::GMat operator<=(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::cmpLE(lhs, rhs);
}

ncvslideio::GMat operator==(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::cmpEQ(lhs, rhs);
}

ncvslideio::GMat operator!=(const ncvslideio::GMat& lhs, const ncvslideio::GScalar& rhs)
{
    return ncvslideio::gapi::cmpNE(lhs, rhs);
}

ncvslideio::GMat operator>(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpLT(rhs, lhs);
}
ncvslideio::GMat operator>=(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpLE(rhs, lhs);
}
ncvslideio::GMat operator<(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpGT(rhs, lhs);
}
ncvslideio::GMat operator<=(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpGE(rhs, lhs);
}
ncvslideio::GMat operator==(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpEQ(rhs, lhs);
}
ncvslideio::GMat operator!=(const ncvslideio::GScalar& lhs, const ncvslideio::GMat& rhs)
{
    return ncvslideio::gapi::cmpNE(rhs, lhs);
}
} // cv
