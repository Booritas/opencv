// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"

#include <ade/graph.hpp>

#include <opencv2/gapi/gproto.hpp> // can_describe
#include <opencv2/gapi/gcompiled.hpp>

#include "compiler/gcompiled_priv.hpp"
#include "backends/common/gbackend.hpp"
#include "executor/gexecutor.hpp"

// GCompiled private implementation ////////////////////////////////////////////
void ncvslideio::GCompiled::Priv::setup(const GMetaArgs &_metaArgs,
                                const GMetaArgs &_outMetas,
                                std::unique_ptr<ncvslideio::gimpl::GAbstractExecutor> &&_pE)
{
    m_metas    = _metaArgs;
    m_outMetas = _outMetas;
    m_exec     = std::move(_pE);
}

bool ncvslideio::GCompiled::Priv::isEmpty() const
{
    return !m_exec;
}

void ncvslideio::GCompiled::Priv::run(ncvslideio::gimpl::GRuntimeArgs &&args)
{
    // Strip away types since ADE knows nothing about that
    // args will be taken by specific GBackendExecutables
    checkArgs(args);
    m_exec->run(std::move(args));
}

const ncvslideio::GMetaArgs& ncvslideio::GCompiled::Priv::metas() const
{
    return m_metas;
}

const ncvslideio::GMetaArgs& ncvslideio::GCompiled::Priv::outMetas() const
{
    return m_outMetas;
}

void ncvslideio::GCompiled::Priv::checkArgs(const ncvslideio::gimpl::GRuntimeArgs &args) const
{
    if (!can_describe(m_metas, args.inObjs))
    {
        util::throw_error(std::logic_error("This object was compiled "
                                           "for different metadata!"));
        // FIXME: Add details on what is actually wrong
    }
    validate_input_args(args.inObjs);
    // FIXME: Actually, the passed parameter vector is never checked
    // against its shapes - so if you compile with GScalarDesc passed
    // for GMat argument, you will get your compilation right (!!)
    // Probably it was there but somehow that olds checks (if they
    // exist) are bypassed now.
}

bool ncvslideio::GCompiled::Priv::canReshape() const
{
    GAPI_Assert(m_exec);
    return m_exec->canReshape();
}

void ncvslideio::GCompiled::Priv::reshape(const GMetaArgs& inMetas, const GCompileArgs& args)
{
    GAPI_Assert(m_exec);
    m_exec->reshape(inMetas, args);
    m_metas = inMetas;
}

void ncvslideio::GCompiled::Priv::prepareForNewStream()
{
    GAPI_Assert(m_exec);
    m_exec->prepareForNewStream();
}

const ncvslideio::gimpl::GModel::Graph& ncvslideio::GCompiled::Priv::model() const
{
    GAPI_Assert(nullptr != m_exec);
    return m_exec->model();
}

// GCompiled public implementation /////////////////////////////////////////////
ncvslideio::GCompiled::GCompiled()
    : m_priv(new Priv())
{
}

ncvslideio::GCompiled::operator bool() const
{
    return !m_priv->isEmpty();
}

void ncvslideio::GCompiled::operator() (GRunArgs &&ins, GRunArgsP &&outs)
{
    // FIXME: Check that <ins> matches the protocol!!!
    // FIXME: Check that <outs> matches the protocol
    m_priv->run(ncvslideio::gimpl::GRuntimeArgs{std::move(ins),std::move(outs)});
}

#if !defined(GAPI_STANDALONE)
void ncvslideio::GCompiled::operator ()(ncvslideio::Mat in, ncvslideio::Mat &out)
{
    (*this)(ncvslideio::gin(in), ncvslideio::gout(out));
}

void ncvslideio::GCompiled::operator() (ncvslideio::Mat in, ncvslideio::Scalar &out)
{
    (*this)(ncvslideio::gin(in), ncvslideio::gout(out));
}

void ncvslideio::GCompiled::operator() (ncvslideio::Mat in1, ncvslideio::Mat in2, ncvslideio::Mat &out)
{
    (*this)(ncvslideio::gin(in1, in2), ncvslideio::gout(out));
}

void ncvslideio::GCompiled::operator() (ncvslideio::Mat in1, ncvslideio::Mat in2, ncvslideio::Scalar &out)
{
    (*this)(ncvslideio::gin(in1, in2), ncvslideio::gout(out));
}

void ncvslideio::GCompiled::operator ()(const std::vector<ncvslideio::Mat> &ins,
                                const std::vector<ncvslideio::Mat> &outs)
{
    GRunArgs call_ins;
    GRunArgsP call_outs;

    // Make a temporary copy of vector outs - ncvslideio::Mats are copies anyway
    auto tmp = outs;
    for (const ncvslideio::Mat &m : ins) { call_ins.emplace_back(m);   }
    for (      ncvslideio::Mat &m : tmp) { call_outs.emplace_back(&m); }

    (*this)(std::move(call_ins), std::move(call_outs));
}
#endif // !defined(GAPI_STANDALONE)

const ncvslideio::GMetaArgs& ncvslideio::GCompiled::metas() const
{
    return m_priv->metas();
}

const ncvslideio::GMetaArgs& ncvslideio::GCompiled::outMetas() const
{
    return m_priv->outMetas();
}

ncvslideio::GCompiled::Priv& ncvslideio::GCompiled::priv()
{
    return *m_priv;
}

bool ncvslideio::GCompiled::canReshape() const
{
    return m_priv->canReshape();
}

void ncvslideio::GCompiled::reshape(const GMetaArgs& inMetas, const GCompileArgs& args)
{
    m_priv->reshape(inMetas, args);
}

void ncvslideio::GCompiled::prepareForNewStream()
{
    m_priv->prepareForNewStream();
}
