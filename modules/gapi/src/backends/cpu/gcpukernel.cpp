// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <cassert>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

const ncvslideio::Mat& ncvslideio::GCPUContext::inMat(int input)
{
    return inArg<ncvslideio::Mat>(input);
}

ncvslideio::Mat&  ncvslideio::GCPUContext::outMatR(int output)
{
    return *util::get<ncvslideio::Mat*>(m_results.at(output));
}

const ncvslideio::Scalar& ncvslideio::GCPUContext::inVal(int input)
{
    return inArg<ncvslideio::Scalar>(input);
}

ncvslideio::Scalar& ncvslideio::GCPUContext::outValR(int output)
{
    return *util::get<ncvslideio::Scalar*>(m_results.at(output));
}

ncvslideio::detail::VectorRef& ncvslideio::GCPUContext::outVecRef(int output)
{
    return util::get<ncvslideio::detail::VectorRef>(m_results.at(output));
}

ncvslideio::detail::OpaqueRef& ncvslideio::GCPUContext::outOpaqueRef(int output)
{
    return util::get<ncvslideio::detail::OpaqueRef>(m_results.at(output));
}

ncvslideio::MediaFrame& ncvslideio::GCPUContext::outFrame(int output)
{
    return *util::get<ncvslideio::MediaFrame*>(m_results.at(output));
}

ncvslideio::GCPUKernel::GCPUKernel()
{
}

ncvslideio::GCPUKernel::GCPUKernel(const GCPUKernel::RunF &runF, const GCPUKernel::SetupF &setupF)
    : m_runF(runF), m_setupF(setupF), m_isStateful(m_setupF != nullptr)
{
}
