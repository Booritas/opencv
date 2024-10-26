// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_GCOMPUTATION_PRIV_HPP
#define OPENCV_GAPI_GCOMPUTATION_PRIV_HPP

#include <ade/graph.hpp>

#include "opencv2/gapi/util/variant.hpp"

#include "opencv2/gapi.hpp"
#include "opencv2/gapi/gcall.hpp"

#include "opencv2/gapi/util/variant.hpp"

#include "backends/common/serialization.hpp"

namespace ncvslideio {

struct GraphInfo
{
    using Ptr = std::shared_ptr<GraphInfo>;
    ncvslideio::GTypesInfo inputs;
    ncvslideio::GTypesInfo outputs;
};

class GComputation::Priv
{
public:
    struct Expr {
        ncvslideio::GProtoArgs m_ins;
        ncvslideio::GProtoArgs m_outs;
    };

    using Dump = ncvslideio::gapi::s11n::GSerialized;

    using Shape = ncvslideio::util::variant
        < Expr    // An expression-based graph
        , Dump    // A deserialized graph
        >;

    GCompiled      m_lastCompiled;
    GMetaArgs      m_lastMetas; // TODO: make GCompiled remember its metas?
    Shape          m_shape;
    GraphInfo::Ptr m_info;      // NB: Used by python bridge
};

}

#endif // OPENCV_GAPI_GCOMPUTATION_PRIV_HPP
