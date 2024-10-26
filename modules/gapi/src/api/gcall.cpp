// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <cassert>
#include <opencv2/gapi/gcall.hpp>
#include "api/gcall_priv.hpp"

// GCall private implementation ////////////////////////////////////////////////
ncvslideio::GCall::Priv::Priv(const ncvslideio::GKernel &k)
    : m_k(k)
{
}

// GCall public implementation /////////////////////////////////////////////////

ncvslideio::GCall::GCall(const ncvslideio::GKernel &k)
    : m_priv(new Priv(k))
{
    // Here we have a reference to GNode,
    // and GNode has a reference to us. Cycle! Now see destructor.
    m_priv->m_node = GNode::Call(*this);
}

ncvslideio::GCall::~GCall()
{
    // FIXME: current behavior of the destructor can cause troubles in a threaded environment. GCall
    // is not supposed to be accessed for modification within multiple threads. There should be a
    // way to ensure somehow that no problem occurs in future. For now, this is a reminder that
    // GCall is not supposed to be copied inside a code block that is executed in parallel.

    // When a GCall object is destroyed (and GCall::Priv is likely still alive,
    // as there might be other references), reset m_node to break cycle.
    m_priv->m_node = GNode();
}

void ncvslideio::GCall::setArgs(std::vector<GArg> &&args)
{
    // FIXME: Check if argument number is matching kernel prototype
    m_priv->m_args = std::move(args);
}

ncvslideio::GMat ncvslideio::GCall::yield(int output)
{
    return ncvslideio::GMat(m_priv->m_node, output);
}

ncvslideio::GMatP ncvslideio::GCall::yieldP(int output)
{
    return ncvslideio::GMatP(m_priv->m_node, output);
}

ncvslideio::GScalar ncvslideio::GCall::yieldScalar(int output)
{
    return ncvslideio::GScalar(m_priv->m_node, output);
}

ncvslideio::detail::GArrayU ncvslideio::GCall::yieldArray(int output)
{
    return ncvslideio::detail::GArrayU(m_priv->m_node, output);
}

ncvslideio::detail::GOpaqueU ncvslideio::GCall::yieldOpaque(int output)
{
    return ncvslideio::detail::GOpaqueU(m_priv->m_node, output);
}

ncvslideio::GFrame ncvslideio::GCall::yieldFrame(int output)
{
    return ncvslideio::GFrame(m_priv->m_node, output);
}

ncvslideio::GCall::Priv& ncvslideio::GCall::priv()
{
    return *m_priv;
}

const ncvslideio::GCall::Priv& ncvslideio::GCall::priv() const
{
    return *m_priv;
}

ncvslideio::GKernel& ncvslideio::GCall::kernel()
{
    return m_priv->m_k;
}

ncvslideio::util::any& ncvslideio::GCall::params()
{
    return m_priv->m_params;
}
