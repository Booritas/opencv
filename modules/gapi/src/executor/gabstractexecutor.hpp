// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation


#ifndef OPENCV_GAPI_GABSTRACT_EXECUTOR_HPP
#define OPENCV_GAPI_GABSTRACT_EXECUTOR_HPP

#include <memory> // unique_ptr, shared_ptr

#include <utility> // tuple, required by magazine
#include <unordered_map> // required by magazine

#include <ade/graph.hpp>

#include "backends/common/gbackend.hpp"

namespace ncvslideio {
namespace gimpl {

// Graph-level executor interface.
//
// This class specifies API for a "super-executor" which orchestrates
// the overall Island graph execution.
//
// Every Island (subgraph) execution is delegated to a particular
// backend and is done opaquely to the GExecutor.
//
// Inputs to a GExecutor instance are:
// - GIslandModel - a high-level graph model which may be seen as a
//   "procedure" to execute.
//   - GModel - a low-level graph of operations (from which a GIslandModel
//     is projected)
// - GComputation runtime arguments - vectors of input/output objects
//
// Every GExecutor is responsible for
// a. Maintaining non-island (intermediate) data objects within graph
// b. Providing GIslandExecutables with input/output data according to
//    their protocols
// c. Triggering execution of GIslandExecutables when task/data dependencies
//    are met.
//
// By default G-API stores all data on host, and cross-Island
// exchange happens via host buffers (and ncvslideio data objects).
//
// Today's exchange data objects are:
// - ncvslideio::Mat, ncvslideio::RMat     - for image buffers
// - ncvslideio::Scalar            - for single values (with up to four components inside)
// - ncvslideio::detail::VectorRef - an untyped wrapper over std::vector<T>
// - ncvslideio::detail::OpaqueRef - an untyped wrapper over T
// - ncvslideio::MediaFrame        - for image textures and surfaces (e.g. in planar format)

class GAbstractExecutor
{
protected:
    std::unique_ptr<ade::Graph> m_orig_graph;
    std::shared_ptr<ade::Graph> m_island_graph;

    ncvslideio::gimpl::GModel::Graph       m_gm;  // FIXME: make const?
    ncvslideio::gimpl::GIslandModel::Graph m_gim; // FIXME: make const?

public:
    explicit GAbstractExecutor(std::unique_ptr<ade::Graph> &&g_model);
    virtual ~GAbstractExecutor() = default;
    virtual void run(ncvslideio::gimpl::GRuntimeArgs &&args) = 0;

    virtual bool canReshape() const = 0;
    virtual void reshape(const GMetaArgs& inMetas, const GCompileArgs& args) = 0;

    virtual void prepareForNewStream() = 0;

    const GModel::Graph& model() const; // FIXME: make it ConstGraph?
};

} // namespace gimpl
} // namespace ncvslideio

#endif // OPENCV_GAPI_GABSTRACT_EXECUTOR_HPP
