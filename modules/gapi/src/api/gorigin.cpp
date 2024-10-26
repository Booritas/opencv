// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"
#include <ade/util/assert.hpp>

#include "api/gorigin.hpp"
#include "api/gnode_priv.hpp"

ncvslideio::GOrigin::GOrigin(GShape s,
                    const ncvslideio::GNode& n,
                    std::size_t p,
                    const ncvslideio::gimpl::HostCtor c,
                    ncvslideio::detail::OpaqueKind k)
    : shape(s), node(n), port(p), ctor(c), kind(k)
{
}

ncvslideio::GOrigin::GOrigin(GShape s, ncvslideio::gimpl::ConstVal v)
    : shape(s), node(ncvslideio::GNode::Const()), value(v), port(INVALID_PORT),
      kind(util::holds_alternative<detail::VectorRef>(v)
               ? util::get<detail::VectorRef>(v).getKind()
               : ncvslideio::detail::OpaqueKind::CV_UNKNOWN)
{
}

bool ncvslideio::detail::GOriginCmp::operator() (const ncvslideio::GOrigin &lhs,
                                         const ncvslideio::GOrigin &rhs) const
{
    const GNode::Priv* lhs_p = &lhs.node.priv();
    const GNode::Priv* rhs_p = &rhs.node.priv();
    if (lhs_p == rhs_p)
    {
        if (lhs.port == rhs.port)
        {
            // A data Origin is uniquely identified by {node/port} pair.
            // The situation when there're two Origins with same {node/port}s
            // but with different shapes (data formats) is illegal!
            GAPI_Assert(lhs.shape == rhs.shape);
        }
        return lhs.port < rhs.port;
    }
    else return lhs_p < rhs_p;
}
