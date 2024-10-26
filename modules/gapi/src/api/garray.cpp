// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"
#include <opencv2/gapi/garray.hpp>
#include "api/gorigin.hpp"

// ncvslideio::detail::GArrayU public implementation ///////////////////////////////////
ncvslideio::detail::GArrayU::GArrayU()
    : m_priv(new GOrigin(GShape::GARRAY, ncvslideio::GNode::Param()))
{
}

ncvslideio::detail::GArrayU::GArrayU(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GARRAY, n, out))
{
}

ncvslideio::detail::GArrayU::GArrayU(const detail::VectorRef& vref)
    : m_priv(new GOrigin(GShape::GARRAY, ncvslideio::gimpl::ConstVal(vref)))
{
}

ncvslideio::GOrigin& ncvslideio::detail::GArrayU::priv()
{
    return *m_priv;
}

const ncvslideio::GOrigin& ncvslideio::detail::GArrayU::priv() const
{
    return *m_priv;
}

void ncvslideio::detail::GArrayU::setConstructFcn(ConstructVec &&ncvslideio)
{
    m_priv->ctor = std::move(ncvslideio);
}

void ncvslideio::detail::GArrayU::setKind(ncvslideio::detail::OpaqueKind kind)
{
    m_priv->kind = kind;
}

namespace ncvslideio {
std::ostream& operator<<(std::ostream& os, const ncvslideio::GArrayDesc &)
{
    // FIXME: add type information here
    os << "(array)";
    return os;
}
}
