// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include <cassert>

#include <opencv2/gapi/ocl/goclkernel.hpp>

const ncvslideio::UMat& ncvslideio::GOCLContext::inMat(int input)
{
    return (inArg<ncvslideio::UMat>(input));
}

ncvslideio::UMat& ncvslideio::GOCLContext::outMatR(int output)
{
    return (*(util::get<ncvslideio::UMat*>(m_results.at(output))));
}

const ncvslideio::Scalar& ncvslideio::GOCLContext::inVal(int input)
{
    return inArg<ncvslideio::Scalar>(input);
}

ncvslideio::Scalar& ncvslideio::GOCLContext::outValR(int output)
{
    return *util::get<ncvslideio::Scalar*>(m_results.at(output));
}

ncvslideio::detail::VectorRef& ncvslideio::GOCLContext::outVecRef(int output)
{
    return util::get<ncvslideio::detail::VectorRef>(m_results.at(output));
}

ncvslideio::detail::OpaqueRef& ncvslideio::GOCLContext::outOpaqueRef(int output)
{
    return util::get<ncvslideio::detail::OpaqueRef>(m_results.at(output));
}

ncvslideio::GOCLKernel::GOCLKernel()
{
}

ncvslideio::GOCLKernel::GOCLKernel(const GOCLKernel::F &f)
    : m_f(f)
{
}

void ncvslideio::GOCLKernel::apply(GOCLContext &ctx)
{
    CV_Assert(m_f);
    m_f(ctx);
}
