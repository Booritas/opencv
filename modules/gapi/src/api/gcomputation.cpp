// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"
#include <algorithm> // remove_if
#include <cctype>    // isspace (non-locale version)
#include <ade/util/algorithm.hpp>
#include <ade/util/zip_range.hpp>   // util::indexed

#include "logger.hpp" // GAPI_LOG

#include <opencv2/gapi/gcomputation.hpp>
#include <opencv2/gapi/gkernel.hpp>

#include "api/gcomputation_priv.hpp"
#include "api/gcall_priv.hpp"
#include "api/gnode_priv.hpp"

#include "compiler/gmodelbuilder.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gcompiled_priv.hpp"
#include "compiler/gstreaming_priv.hpp"

static ncvslideio::GTypesInfo collectInfo(const ncvslideio::gimpl::GModel::ConstGraph& g,
                                  const std::vector<ade::NodeHandle>& nhs) {
    ncvslideio::GTypesInfo info;
    info.reserve(nhs.size());

    ade::util::transform(nhs, std::back_inserter(info), [&g](const ade::NodeHandle& nh) {
        const auto& data = g.metadata(nh).get<ncvslideio::gimpl::Data>();
        return ncvslideio::GTypeInfo{data.shape, data.kind, data.ctor};
    });

    return info;
}

// NB: This function is used to collect graph input/output info.
// Needed for python bridge to unpack inputs and constructs outputs properly.
static ncvslideio::GraphInfo::Ptr collectGraphInfo(const ncvslideio::GComputation::Priv& priv)
{
    auto g = ncvslideio::gimpl::GCompiler::makeGraph(priv);
    ncvslideio::gimpl::GModel::ConstGraph cgr(*g);
    auto in_info  = collectInfo(cgr, cgr.metadata().get<ncvslideio::gimpl::Protocol>().in_nhs);
    auto out_info = collectInfo(cgr, cgr.metadata().get<ncvslideio::gimpl::Protocol>().out_nhs);
    return ncvslideio::GraphInfo::Ptr(new ncvslideio::GraphInfo{std::move(in_info), std::move(out_info)});
}

// ncvslideio::GComputation private implementation /////////////////////////////////////
// <none>

// ncvslideio::GComputation public implementation //////////////////////////////////////
ncvslideio::GComputation::GComputation(const Generator& gen)
    : m_priv(gen().m_priv)
{
}

ncvslideio::GComputation::GComputation(GMat in, GMat out)
    : ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
{
}


ncvslideio::GComputation::GComputation(GMat in, GScalar out)
    : ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out))
{
}

ncvslideio::GComputation::GComputation(GMat in1, GMat in2, GMat out)
    : ncvslideio::GComputation(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out))
{
}

ncvslideio::GComputation::GComputation(GMat in1, GMat in2, GScalar out)
    : ncvslideio::GComputation(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out))
{
}

ncvslideio::GComputation::GComputation(const std::vector<GMat> &ins,
                               const std::vector<GMat> &outs)
    : m_priv(new Priv())
{
    Priv::Expr e;
    const auto wrap = [](ncvslideio::GMat m) { return GProtoArg(m); };
    ade::util::transform(ins,  std::back_inserter(e.m_ins),  wrap);
    ade::util::transform(outs, std::back_inserter(e.m_outs), wrap);
    m_priv->m_shape = std::move(e);
}

ncvslideio::GComputation::GComputation(ncvslideio::GProtoInputArgs &&ins,
                               ncvslideio::GProtoOutputArgs &&outs)
    : m_priv(new Priv())
{
    m_priv->m_shape = Priv::Expr{
          std::move(ins.m_args)
        , std::move(outs.m_args)
    };
}

ncvslideio::GComputation::GComputation(ncvslideio::gapi::s11n::IIStream &is)
    : m_priv(new Priv())
{
    m_priv->m_shape = gapi::s11n::deserialize(is);
}

void ncvslideio::GComputation::serialize(ncvslideio::gapi::s11n::IOStream &os) const
{
    // Build a basic GModel and write the whole thing to the stream
    auto pG = ncvslideio::gimpl::GCompiler::makeGraph(*m_priv);
    std::vector<ade::NodeHandle> nhs(pG->nodes().begin(), pG->nodes().end());
    gapi::s11n::serialize(os, *pG, nhs);
}


ncvslideio::GCompiled ncvslideio::GComputation::compile(GMetaArgs &&metas, GCompileArgs &&args)
{
    // FIXME: Cache gcompiled per parameters here?
    ncvslideio::gimpl::GCompiler comp(*this, std::move(metas), std::move(args));
    return comp.compile();
}

ncvslideio::GStreamingCompiled ncvslideio::GComputation::compileStreaming(GMetaArgs &&metas, GCompileArgs &&args)
{
    ncvslideio::gimpl::GCompiler comp(*this, std::move(metas), std::move(args));
    return comp.compileStreaming();
}

ncvslideio::GStreamingCompiled ncvslideio::GComputation::compileStreaming(GCompileArgs &&args)
{
    // NB: Used by python bridge
    if (!m_priv->m_info)
    {
        m_priv->m_info = collectGraphInfo(*m_priv);
    }

    ncvslideio::gimpl::GCompiler comp(*this, {}, std::move(args));
    auto compiled = comp.compileStreaming();

    compiled.priv().setInInfo(m_priv->m_info->inputs);
    compiled.priv().setOutInfo(m_priv->m_info->outputs);

    return compiled;
}

ncvslideio::GStreamingCompiled ncvslideio::GComputation::compileStreaming(const ncvslideio::detail::ExtractMetaCallback &callback,
                                                                GCompileArgs                   &&args)
{
    // NB: Used by python bridge
    if (!m_priv->m_info)
    {
        m_priv->m_info = collectGraphInfo(*m_priv);
    }

    auto ins = callback(m_priv->m_info->inputs);
    ncvslideio::gimpl::GCompiler comp(*this, std::move(ins), std::move(args));
    auto compiled = comp.compileStreaming();
    compiled.priv().setInInfo(m_priv->m_info->inputs);
    compiled.priv().setOutInfo(m_priv->m_info->outputs);

    return compiled;
}

// FIXME: Introduce similar query/test method for GMetaArgs as a building block
// for functions like this?
static bool formats_are_same(const ncvslideio::GMetaArgs& metas1, const ncvslideio::GMetaArgs& metas2)
{
    return std::equal(metas1.cbegin(), metas1.cend(), metas2.cbegin(),
                      [](const ncvslideio::GMetaArg& meta1, const ncvslideio::GMetaArg& meta2) {
                          if (meta1.index() == meta2.index() && meta1.index() == ncvslideio::GMetaArg::index_of<ncvslideio::GMatDesc>())
                          {
                              const auto& desc1 = ncvslideio::util::get<ncvslideio::GMatDesc>(meta1);
                              const auto& desc2 = ncvslideio::util::get<ncvslideio::GMatDesc>(meta2);

                              // comparison by size is omitted
                              return (desc1.chan  == desc2.chan &&
                                      desc1.depth == desc2.depth);
                          }
                          else
                          {
                              return meta1 == meta2;
                          }
                     });
}

void ncvslideio::GComputation::recompile(GMetaArgs&& in_metas, GCompileArgs &&args)
{
    // FIXME Graph should be recompiled when GCompileArgs have changed
    if (m_priv->m_lastMetas != in_metas)
    {
        if (m_priv->m_lastCompiled &&
            m_priv->m_lastCompiled.canReshape() &&
            formats_are_same(m_priv->m_lastMetas, in_metas))
        {
            m_priv->m_lastCompiled.reshape(in_metas, args);
        }
        else
        {
            // FIXME: Had to construct temporary object as compile() takes && (r-value)
            m_priv->m_lastCompiled = compile(GMetaArgs(in_metas), std::move(args));
        }
        m_priv->m_lastMetas = in_metas;
    }
    else if (in_metas.size() == 0) {
        // Happens when the graph is head-less (e.g. starts with const-vals only)
        // always compile ad-hoc
        m_priv->m_lastCompiled = compile(GMetaArgs(in_metas), std::move(args));
    }
}

void ncvslideio::GComputation::apply(GRunArgs &&ins, GRunArgsP &&outs, GCompileArgs &&args)
{
    recompile(descr_of(ins), std::move(args));
    m_priv->m_lastCompiled(std::move(ins), std::move(outs));
}

void ncvslideio::GComputation::apply(const std::vector<ncvslideio::Mat> &ins,
                             const std::vector<ncvslideio::Mat> &outs,
                             GCompileArgs &&args)
{
    GRunArgs call_ins;
    GRunArgsP call_outs;

    auto tmp = outs;
    for (const ncvslideio::Mat &m : ins) { call_ins.emplace_back(m);   }
    for (      ncvslideio::Mat &m : tmp) { call_outs.emplace_back(&m); }

    apply(std::move(call_ins), std::move(call_outs), std::move(args));
}

// NB: This overload is called from python code
ncvslideio::GRunArgs ncvslideio::GComputation::apply(const ncvslideio::detail::ExtractArgsCallback &callback,
                                           GCompileArgs                   &&args)
{
    // NB: Used by python bridge
    if (!m_priv->m_info)
    {
        m_priv->m_info = collectGraphInfo(*m_priv);
    }

    auto ins = callback(m_priv->m_info->inputs);
    recompile(descr_of(ins), std::move(args));

    GRunArgs run_args;
    GRunArgsP outs;
    run_args.reserve(m_priv->m_info->outputs.size());
    outs.reserve(m_priv->m_info->outputs.size());

    ncvslideio::detail::constructGraphOutputs(m_priv->m_info->outputs, run_args, outs);

    m_priv->m_lastCompiled(std::move(ins), std::move(outs));
    return run_args;
}

#if !defined(GAPI_STANDALONE)
void ncvslideio::GComputation::apply(ncvslideio::Mat in, ncvslideio::Mat &out, GCompileArgs &&args)
{
    apply(ncvslideio::gin(in), ncvslideio::gout(out), std::move(args));
    // FIXME: The following doesn't work!
    // Operation result is not replicated into user's object
    // apply({GRunArg(in)}, {GRunArg(out)});
}

void ncvslideio::GComputation::apply(ncvslideio::Mat in, ncvslideio::Scalar &out, GCompileArgs &&args)
{
    apply(ncvslideio::gin(in), ncvslideio::gout(out), std::move(args));
}

void ncvslideio::GComputation::apply(ncvslideio::Mat in1, ncvslideio::Mat in2, ncvslideio::Mat &out, GCompileArgs &&args)
{
    apply(ncvslideio::gin(in1, in2), ncvslideio::gout(out), std::move(args));
}

void ncvslideio::GComputation::apply(ncvslideio::Mat in1, ncvslideio::Mat in2, ncvslideio::Scalar &out, GCompileArgs &&args)
{
    apply(ncvslideio::gin(in1, in2), ncvslideio::gout(out), std::move(args));
}

void ncvslideio::GComputation::apply(const std::vector<ncvslideio::Mat> &ins,
                                   std::vector<ncvslideio::Mat> &outs,
                             GCompileArgs &&args)
{
    GRunArgs call_ins;
    GRunArgsP call_outs;

    for (const ncvslideio::Mat &m : ins)  { call_ins.emplace_back(m);   }
    for (      ncvslideio::Mat &m : outs) { call_outs.emplace_back(&m); }

    apply(std::move(call_ins), std::move(call_outs), std::move(args));
}
#endif // !defined(GAPI_STANDALONE)

ncvslideio::GComputation::Priv& ncvslideio::GComputation::priv()
{
    return *m_priv;
}

const ncvslideio::GComputation::Priv& ncvslideio::GComputation::priv() const
{
    return *m_priv;
}

// Islands /////////////////////////////////////////////////////////////////////

void ncvslideio::gapi::island(const std::string       &name,
                            GProtoInputArgs  &&ins,
                            GProtoOutputArgs &&outs)
{
    {
        // Island must have a printable name.
        // Forbid names which contain only spaces.
        GAPI_Assert(!name.empty());
        const auto first_printable_it = std::find_if_not(name.begin(), name.end(), isspace);
        const bool likely_printable   = first_printable_it != name.end();
        GAPI_Assert(likely_printable);
    }
    // Even if the name contains spaces, keep it unmodified as user will
    // then use this string to assign affinity, etc.

    // First, set island tags on all operations from `ins` to `outs`
    auto island = ncvslideio::gimpl::unrollExpr(ins.m_args, outs.m_args);
    if (island.all_ops.empty())
    {
        util::throw_error(std::logic_error("Operation range is empty"));
    }
    for (auto &op_expr_node : island.all_ops)
    {
        auto &op_expr_node_p = op_expr_node.priv();

        GAPI_Assert(op_expr_node.shape() == GNode::NodeShape::CALL);
        const GCall&       call   = op_expr_node.call();
        const GCall::Priv& call_p = call.priv();

        if (!op_expr_node_p.m_island.empty())
        {
            util::throw_error(std::logic_error
                              (  "Operation " + call_p.m_k.name
                               + " is already assigned to island \""
                               + op_expr_node_p.m_island + "\""));
        }
        else
        {
            op_expr_node_p.m_island = name;
            GAPI_LOG_INFO(NULL,
                          "Assigned " << call_p.m_k.name << "_" << &call_p <<
                          " to island \"" << name << "\"");
        }
    }

    // Note - this function only sets islands to all operations in
    // expression tree, it is just a first step.
    // The second step is assigning intermediate data objects to Islands,
    // see passes::initIslands for details.
}
