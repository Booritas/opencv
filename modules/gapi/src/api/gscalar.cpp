// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <opencv2/gapi/gscalar.hpp>
#include "api/gorigin.hpp"

// ncvslideio::GScalar public implementation ///////////////////////////////////////////
ncvslideio::GScalar::GScalar()
    : m_priv(new GOrigin(GShape::GSCALAR, ncvslideio::GNode::Param()))
{
}

ncvslideio::GScalar::GScalar(const GNode &n, std::size_t out)
    : m_priv(new GOrigin(GShape::GSCALAR, n, out))
{
}

ncvslideio::GScalar::GScalar(const ncvslideio::Scalar& s)
    : m_priv(new GOrigin(GShape::GSCALAR, ncvslideio::gimpl::ConstVal(s)))
{
}

ncvslideio::GScalar::GScalar(ncvslideio::Scalar&& s)
    : m_priv(new GOrigin(GShape::GSCALAR, ncvslideio::gimpl::ConstVal(std::move(s))))
{
}

ncvslideio::GScalar::GScalar(double v0)
    : m_priv(new GOrigin(GShape::GSCALAR, ncvslideio::gimpl::ConstVal(ncvslideio::Scalar(v0))))
{
}

ncvslideio::GOrigin& ncvslideio::GScalar::priv()
{
    return *m_priv;
}

const ncvslideio::GOrigin& ncvslideio::GScalar::priv() const
{
    return *m_priv;
}

//N.B. if we ever need more complicated logic for desc_of(ncvslideio::(gapi::own::)Scalar)
//dispatching should be done in the same way as for ncvslideio::(gapi::own)::Mat
ncvslideio::GScalarDesc ncvslideio::descr_of(const ncvslideio::Scalar &)
{
    return empty_scalar_desc();
}

namespace ncvslideio {
std::ostream& operator<<(std::ostream& os, const ncvslideio::GScalarDesc &)
{
    os << "(scalar)";
    return os;
}
}
