// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation


#include "precomp.hpp"
#include <opencv2/gapi/gopaque.hpp>
#include "api/gorigin.hpp"

// ncvslideio::detail::GOpaqueU public implementation ///////////////////////////////////
ncvslideio::detail::GOpaqueU::GOpaqueU()
    : m_priv(new GOrigin(GShape::GOPAQUE, ncvslideio::GNode::Param()))
{
}

ncvslideio::detail::GOpaqueU::GOpaqueU(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GOPAQUE, n, out))
{
}

ncvslideio::GOrigin& ncvslideio::detail::GOpaqueU::priv()
{
    return *m_priv;
}

const ncvslideio::GOrigin& ncvslideio::detail::GOpaqueU::priv() const
{
    return *m_priv;
}

void ncvslideio::detail::GOpaqueU::setConstructFcn(ConstructOpaque &&co)
{
    m_priv->ctor = std::move(co);
}

void ncvslideio::detail::GOpaqueU::setKind(ncvslideio::detail::OpaqueKind kind)
{
    m_priv->kind = kind;
}

namespace ncvslideio {
std::ostream& operator<<(std::ostream& os, const ncvslideio::GOpaqueDesc &)
{
    // FIXME: add type information here
    os << "(Opaque)";
    return os;
}
}
