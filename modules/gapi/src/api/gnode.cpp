// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <cassert>

#include "api/gnode.hpp"
#include "api/gnode_priv.hpp"

// GNode private implementation
ncvslideio::GNode::Priv::Priv()
    : m_shape(NodeShape::EMPTY)
{
}

ncvslideio::GNode::Priv::Priv(GCall c)
    : m_shape(NodeShape::CALL), m_spec(c)
{
}

ncvslideio::GNode::Priv::Priv(ParamTag)
    : m_shape(NodeShape::PARAM)
{
}

ncvslideio::GNode::Priv::Priv(ConstTag)
    : m_shape(NodeShape::CONST_BOUNDED)
{
}

// GNode public implementation
ncvslideio::GNode::GNode()
    : m_priv(new Priv())
{
}

ncvslideio::GNode::GNode(const GCall &c)
    : m_priv(new Priv(c))
{
}

ncvslideio::GNode::GNode(ParamTag)
    : m_priv(new Priv(Priv::ParamTag()))
{
}

ncvslideio::GNode::GNode(ConstTag)
    : m_priv(new Priv(Priv::ConstTag()))
{
}

ncvslideio::GNode ncvslideio::GNode::Call(const GCall &c)
{
    return GNode(c);
}

ncvslideio::GNode ncvslideio::GNode::Param()
{
    return GNode(ParamTag());
}

ncvslideio::GNode ncvslideio::GNode::Const()
{
    return GNode(ConstTag());
}

ncvslideio::GNode::Priv& ncvslideio::GNode::priv()
{
    return *m_priv;
}

const ncvslideio::GNode::Priv& ncvslideio::GNode::priv() const
{
    return *m_priv;
}

const ncvslideio::GNode::NodeShape& ncvslideio::GNode::shape() const
{
    return m_priv->m_shape;
}

const ncvslideio::GCall& ncvslideio::GNode::call()  const
{
    return util::get<GCall>(m_priv->m_spec);
}
