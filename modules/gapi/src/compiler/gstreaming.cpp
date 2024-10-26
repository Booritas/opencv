// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "precomp.hpp"

#include <ade/graph.hpp>
#include <ade/util/zip_range.hpp>   // util::indexed

#include <opencv2/gapi/gproto.hpp> // can_describe
#include <opencv2/gapi/gcompiled.hpp>

#include "compiler/gstreaming_priv.hpp"
#include "backends/common/gbackend.hpp"

// GStreamingCompiled private implementation ///////////////////////////////////
void ncvslideio::GStreamingCompiled::Priv::setup(const GMetaArgs &_metaArgs,
                                         const GMetaArgs &_outMetas,
                                         std::unique_ptr<ncvslideio::gimpl::GAbstractStreamingExecutor> &&_pE)
{
    m_metas    = _metaArgs;
    m_outMetas = _outMetas;
    m_exec     = std::move(_pE);
}

void ncvslideio::GStreamingCompiled::Priv::setup(std::unique_ptr<ncvslideio::gimpl::GAbstractStreamingExecutor> &&_pE)
{
    m_exec = std::move(_pE);
}

bool ncvslideio::GStreamingCompiled::Priv::isEmpty() const
{
    return !m_exec;
}

const ncvslideio::GMetaArgs& ncvslideio::GStreamingCompiled::Priv::metas() const
{
    return m_metas;
}

const ncvslideio::GMetaArgs& ncvslideio::GStreamingCompiled::Priv::outMetas() const
{
    return m_outMetas;
}

// FIXME: What is the reason in having Priv here if Priv actually dispatches
// everything to the underlying executable?? May be this executable may become
// the G*Compiled's priv?
void ncvslideio::GStreamingCompiled::Priv::setSource(ncvslideio::GRunArgs &&args)
{
    if (!m_metas.empty() && !can_describe(m_metas, args))
    {
        util::throw_error(std::logic_error("This object was compiled "
                                           "for different metadata!"));
    }
    GAPI_Assert(m_exec != nullptr);
    m_exec->setSource(std::move(args));
}

void ncvslideio::GStreamingCompiled::Priv::start()
{
    m_exec->start();
}

bool ncvslideio::GStreamingCompiled::Priv::pull(ncvslideio::GRunArgsP &&outs)
{
    return m_exec->pull(std::move(outs));
}

bool ncvslideio::GStreamingCompiled::Priv::pull(ncvslideio::GOptRunArgsP &&outs)
{
    return m_exec->pull(std::move(outs));
}

std::tuple<bool, ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>> ncvslideio::GStreamingCompiled::Priv::pull()
{
    return m_exec->pull();
}

bool ncvslideio::GStreamingCompiled::Priv::try_pull(ncvslideio::GRunArgsP &&outs)
{
    return m_exec->try_pull(std::move(outs));
}

void ncvslideio::GStreamingCompiled::Priv::stop()
{
    m_exec->stop();
}

bool ncvslideio::GStreamingCompiled::Priv::running() const
{
    return m_exec->running();
}

// GStreamingCompiled public implementation ////////////////////////////////////
ncvslideio::GStreamingCompiled::GStreamingCompiled()
    : m_priv(new Priv())
{
}

// NB: This overload is called from python code
void ncvslideio::GStreamingCompiled::setSource(const ncvslideio::detail::ExtractArgsCallback& callback)
{
    setSource(callback(m_priv->inInfo()));
}

void ncvslideio::GStreamingCompiled::setSource(GRunArgs &&ins)
{
    // FIXME: verify these input parameters according to the graph input meta
    m_priv->setSource(std::move(ins));
}

void ncvslideio::GStreamingCompiled::setSource(const ncvslideio::gapi::wip::IStreamSource::Ptr &s)
{
    setSource(ncvslideio::gin(s));
}

void ncvslideio::GStreamingCompiled::start()
{
    m_priv->start();
}

bool ncvslideio::GStreamingCompiled::pull(ncvslideio::GRunArgsP &&outs)
{
    return m_priv->pull(std::move(outs));
}

std::tuple<bool, ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>> ncvslideio::GStreamingCompiled::pull()
{
    return m_priv->pull();
}

bool ncvslideio::GStreamingCompiled::pull(ncvslideio::GOptRunArgsP &&outs)
{
    return m_priv->pull(std::move(outs));
}

bool ncvslideio::GStreamingCompiled::try_pull(ncvslideio::GRunArgsP &&outs)
{
    return m_priv->try_pull(std::move(outs));
}

void ncvslideio::GStreamingCompiled::stop()
{
    m_priv->stop();
}

bool ncvslideio::GStreamingCompiled::running() const
{
    return m_priv->running();
}

ncvslideio::GStreamingCompiled::operator bool() const
{
    return !m_priv->isEmpty();
}

const ncvslideio::GMetaArgs& ncvslideio::GStreamingCompiled::metas() const
{
    return m_priv->metas();
}

const ncvslideio::GMetaArgs& ncvslideio::GStreamingCompiled::outMetas() const
{
    return m_priv->outMetas();
}

ncvslideio::GStreamingCompiled::Priv& ncvslideio::GStreamingCompiled::priv()
{
    return *m_priv;
}
