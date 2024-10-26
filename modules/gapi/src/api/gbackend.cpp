// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2020 Intel Corporation


#include "precomp.hpp"
#include <memory> // unique_ptr
#include <functional> // multiplies

#include <opencv2/gapi/gkernel.hpp>

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"
#include "compiler/gobjref.hpp"
#include "compiler/gislandmodel.hpp"

// GBackend private implementation /////////////////////////////////////////////
void ncvslideio::gapi::GBackend::Priv::unpackKernel(ade::Graph             & /*graph  */ ,
                                            const ade::NodeHandle  & /*op_node*/ ,
                                            const GKernelImpl      & /*impl   */ )
{
    // Default implementation is still there as Priv
    // is instantiated by some tests.
    // Priv is even instantiated as a mock object in a number of tests
    // as a backend and this method is called for mock objects (doing nothing).
    // FIXME: add a warning message here
    // FIXME: Do something with this! Ideally this function should be "=0";
}

std::unique_ptr<ncvslideio::gimpl::GIslandExecutable>
ncvslideio::gapi::GBackend::Priv::compile(const ade::Graph&,
                                  const GCompileArgs&,
                                  const std::vector<ade::NodeHandle> &) const
{
    // ...and this method is here for the same reason!
    GAPI_Error("InternalError");
}

std::unique_ptr<ncvslideio::gimpl::GIslandExecutable>
ncvslideio::gapi::GBackend::Priv::compile(const ade::Graph& graph,
                                  const GCompileArgs& args,
                                  const std::vector<ade::NodeHandle>& nodes,
                                  const std::vector<ncvslideio::gimpl::Data>&,
                                  const std::vector<ncvslideio::gimpl::Data>&) const
{
    return compile(graph, args, nodes);
}

void ncvslideio::gapi::GBackend::Priv::addBackendPasses(ade::ExecutionEngineSetupContext &)
{
    // Do nothing by default, plugins may override this to
    // add custom (backend-specific) graph transformations
}

void ncvslideio::gapi::GBackend::Priv::addMetaSensitiveBackendPasses(ade::ExecutionEngineSetupContext &)
{
    // Do nothing by default, plugins may override this to
    // add custom (backend-specific) graph transformations
    // which are sensitive to metadata
}

ncvslideio::GKernelPackage ncvslideio::gapi::GBackend::Priv::auxiliaryKernels() const
{
    return {};
}

bool ncvslideio::gapi::GBackend::Priv::controlsMerge() const
{
    return false;
}

bool ncvslideio::gapi::GBackend::Priv::allowsMerge(const ncvslideio::gimpl::GIslandModel::Graph &,
                                           const ade::NodeHandle &,
                                           const ade::NodeHandle &,
                                           const ade::NodeHandle &) const
{
    GAPI_Assert(controlsMerge());
    return true;
}

bool ncvslideio::gapi::GBackend::Priv::supportsConst(ncvslideio::GShape) const {
    return false;
}

// GBackend public implementation //////////////////////////////////////////////
ncvslideio::gapi::GBackend::GBackend()
{
}

ncvslideio::gapi::GBackend::GBackend(std::shared_ptr<ncvslideio::gapi::GBackend::Priv> &&p)
    : m_priv(std::move(p))
{
}

ncvslideio::gapi::GBackend::Priv& ncvslideio::gapi::GBackend::priv()
{
    return *m_priv;
}

const ncvslideio::gapi::GBackend::Priv& ncvslideio::gapi::GBackend::priv() const
{
    return *m_priv;
}

std::size_t ncvslideio::gapi::GBackend::hash() const
{
    return std::hash<const ncvslideio::gapi::GBackend::Priv*>{}(m_priv.get());
}

bool ncvslideio::gapi::GBackend::operator== (const ncvslideio::gapi::GBackend &rhs) const
{
    return m_priv == rhs.m_priv;
}

// Abstract Host-side data manipulation ////////////////////////////////////////
// Reused between CPU backend and more generic GExecutor
namespace ncvslideio {
namespace gimpl {
namespace magazine {

namespace {
// Utility function, used in both bindInArg and bindOutArg,
// implements default RMat bind behaviour (if backend doesn't handle RMats in specific way):
// view + wrapped ncvslideio::Mat are placed into the magazine
void bindRMat(Mag& mag, const RcDesc& rc, const ncvslideio::RMat& rmat, RMat::Access a)
{
    auto& matv = mag.template slot<RMat::View>()[rc.id];
    matv = rmat.access(a);
    mag.template slot<ncvslideio::Mat>()[rc.id] = asMat(matv);
}
} // anonymous namespace

// FIXME implement the below functions with visit()?
void bindInArg(Mag& mag, const RcDesc &rc, const GRunArg &arg, HandleRMat handleRMat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        // In case of handleRMat == SKIP
        // We assume that backend can work with some device-specific RMats
        // and will handle them in some specific way, so just return
        if (handleRMat == HandleRMat::SKIP) return;
        GAPI_Assert(arg.index() == GRunArg::index_of<ncvslideio::RMat>());
        bindRMat(mag, rc, util::get<ncvslideio::RMat>(arg), RMat::Access::R);

        // FIXME: Here meta may^WWILL be copied multiple times!
        // Replace it is reference-counted object?
        mag.meta<ncvslideio::RMat>()[rc.id] = arg.meta;
        mag.meta<ncvslideio::Mat>()[rc.id] = arg.meta;
#if !defined(GAPI_STANDALONE)
        mag.meta<ncvslideio::UMat>()[rc.id] = arg.meta;
#endif
        break;
    }

    case GShape::GSCALAR:
    {
        auto& mag_scalar = mag.template slot<ncvslideio::Scalar>()[rc.id];
        switch (arg.index())
        {
        case GRunArg::index_of<ncvslideio::Scalar>() : mag_scalar = util::get<ncvslideio::Scalar>(arg);    break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        mag.meta<ncvslideio::Scalar>()[rc.id] = arg.meta;
        break;
    }

    case GShape::GARRAY:
        mag.slot<ncvslideio::detail::VectorRef>()[rc.id] = util::get<ncvslideio::detail::VectorRef>(arg);
        mag.meta<ncvslideio::detail::VectorRef>()[rc.id] = arg.meta;
        break;

    case GShape::GOPAQUE:
        mag.slot<ncvslideio::detail::OpaqueRef>()[rc.id] = util::get<ncvslideio::detail::OpaqueRef>(arg);
        mag.meta<ncvslideio::detail::OpaqueRef>()[rc.id] = arg.meta;
        break;

    case GShape::GFRAME:
        mag.slot<ncvslideio::MediaFrame>()[rc.id] = util::get<ncvslideio::MediaFrame>(arg);
        mag.meta<ncvslideio::MediaFrame>()[rc.id] = arg.meta;
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void bindOutArg(Mag& mag, const RcDesc &rc, const GRunArgP &arg, HandleRMat handleRMat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        // In case of handleRMat == SKIP
        // We assume that backend can work with some device-specific RMats
        // and will handle them in some specific way, so just return
        if (handleRMat == HandleRMat::SKIP) return;
        GAPI_Assert(arg.index() == GRunArgP::index_of<ncvslideio::RMat*>());
        bindRMat(mag, rc, *util::get<ncvslideio::RMat*>(arg), RMat::Access::W);
        break;
    }

    case GShape::GSCALAR:
    {
        auto& mag_scalar = mag.template slot<ncvslideio::Scalar>()[rc.id];
        switch (arg.index())
        {
        case GRunArgP::index_of<ncvslideio::Scalar*>() : mag_scalar = *util::get<ncvslideio::Scalar*>(arg); break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }
    case GShape::GFRAME:
        mag.template slot<ncvslideio::MediaFrame>()[rc.id] = *util::get<ncvslideio::MediaFrame*>(arg);
        break;
    case GShape::GARRAY:
        mag.template slot<ncvslideio::detail::VectorRef>()[rc.id] = util::get<ncvslideio::detail::VectorRef>(arg);
        break;

    case GShape::GOPAQUE:
        mag.template slot<ncvslideio::detail::OpaqueRef>()[rc.id] = util::get<ncvslideio::detail::OpaqueRef>(arg);
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void resetInternalData(Mag& mag, const Data &d)
{
    if (d.storage != Data::Storage::INTERNAL)
        return;

    switch (d.shape)
    {
    case GShape::GARRAY:
        util::get<ncvslideio::detail::ConstructVec>(d.ctor)
            (mag.template slot<ncvslideio::detail::VectorRef>()[d.rc]);
        break;

    case GShape::GOPAQUE:
        util::get<ncvslideio::detail::ConstructOpaque>(d.ctor)
            (mag.template slot<ncvslideio::detail::OpaqueRef>()[d.rc]);
        break;

    case GShape::GSCALAR:
        mag.template slot<ncvslideio::Scalar>()[d.rc] = ncvslideio::Scalar();
        break;

    case GShape::GMAT:
    case GShape::GFRAME:
        // Do nothing here - FIXME unify with initInternalData?
        break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

ncvslideio::GRunArg getArg(const Mag& mag, const RcDesc &ref)
{
    // Wrap associated CPU object (either host or an internal one)
    switch (ref.shape)
    {
    case GShape::GMAT:
        return GRunArg(mag.slot<ncvslideio::RMat>().at(ref.id),
                       mag.meta<ncvslideio::RMat>().at(ref.id));
    case GShape::GSCALAR:
        return GRunArg(mag.slot<ncvslideio::Scalar>().at(ref.id),
                       mag.meta<ncvslideio::Scalar>().at(ref.id));
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case GShape::GARRAY:
        return GRunArg(mag.slot<ncvslideio::detail::VectorRef>().at(ref.id),
                       mag.meta<ncvslideio::detail::VectorRef>().at(ref.id));
    case GShape::GOPAQUE:
        return GRunArg(mag.slot<ncvslideio::detail::OpaqueRef>().at(ref.id),
                       mag.meta<ncvslideio::detail::OpaqueRef>().at(ref.id));
    case GShape::GFRAME:
        return GRunArg(mag.slot<ncvslideio::MediaFrame>().at(ref.id),
                       mag.meta<ncvslideio::MediaFrame>().at(ref.id));
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

ncvslideio::GRunArgP getObjPtr(Mag& mag, const RcDesc &rc, bool is_umat)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
        if (is_umat)
        {
#if !defined(GAPI_STANDALONE)
            return GRunArgP(&mag.template slot<ncvslideio::UMat>()[rc.id]);
#else
            util::throw_error(std::logic_error("UMat is not supported in standalone build"));
#endif //  !defined(GAPI_STANDALONE)
        }
        else
            return GRunArgP(&mag.template slot<ncvslideio::Mat>()[rc.id]);
    case GShape::GSCALAR: return GRunArgP(&mag.template slot<ncvslideio::Scalar>()[rc.id]);
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructor by either bindIn/Out or resetInternal)
    case GShape::GARRAY:
        // FIXME(DM): For some absolutely unknown to me reason, move
        // semantics is involved here without const_cast to const (and
        // value from map is moved into return value GRunArgP, leaving
        // map with broken value I've spent few late Friday hours
        // debugging this!!!1
        return GRunArgP(const_cast<const Mag&>(mag)
                        .template slot<ncvslideio::detail::VectorRef>().at(rc.id));
    case GShape::GOPAQUE:
        // FIXME(DM): For some absolutely unknown to me reason, move
        // semantics is involved here without const_cast to const (and
        // value from map is moved into return value GRunArgP, leaving
        // map with broken value I've spent few late Friday hours
        // debugging this!!!1
        return GRunArgP(const_cast<const Mag&>(mag)
                        .template slot<ncvslideio::detail::OpaqueRef>().at(rc.id));
    case GShape::GFRAME:
        return GRunArgP(&mag.template slot<ncvslideio::MediaFrame>()[rc.id]);

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void writeBack(const Mag& mag, const RcDesc &rc, GRunArgP &g_arg)
{
    switch (rc.shape)
    {
    case GShape::GARRAY:
    case GShape::GMAT:
    case GShape::GOPAQUE:
        // Do nothing - should we really do anything here?
        break;

    case GShape::GSCALAR:
    {
        switch (g_arg.index())
        {
        case GRunArgP::index_of<ncvslideio::Scalar*>() : *util::get<ncvslideio::Scalar*>(g_arg) = mag.template slot<ncvslideio::Scalar>().at(rc.id); break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
        break;
    }

    case GShape::GFRAME:
    {
        *util::get<ncvslideio::MediaFrame*>(g_arg) = mag.template slot<ncvslideio::MediaFrame>().at(rc.id);
        break;
    }

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void unbind(Mag& mag, const RcDesc &rc)
{
    switch (rc.shape)
    {
    case GShape::GARRAY:
    case GShape::GOPAQUE:
    case GShape::GSCALAR:
        // TODO: Do nothing - should we really do anything here?
        break;

    case GShape::GMAT:
        // Clean-up everything - a ncvslideio::Mat, ncvslideio::RMat::View, a ncvslideio::UMat, and ncvslideio::RMat
        // if applicable
        mag.slot<ncvslideio::Mat>().erase(rc.id);
#if !defined(GAPI_STANDALONE)
        mag.slot<ncvslideio::UMat>().erase(rc.id);
#endif
        mag.slot<ncvslideio::RMat::View>().erase(rc.id);
        mag.slot<ncvslideio::RMat>().erase(rc.id);
        break;

    case GShape::GFRAME:
        // MediaFrame can also be associated with external memory,
        // so requires a special handling here.
        mag.slot<ncvslideio::MediaFrame>().erase(rc.id);
        break;

    default:
        GAPI_Error("InternalError");
    }
}

} // namespace magazine

void createMat(const ncvslideio::GMatDesc &desc, ncvslideio::Mat& mat)
{
    // FIXME: Refactor (probably start supporting N-Dimensional blobs natively
    if (desc.dims.empty())
    {
        const auto type = desc.planar ? desc.depth : CV_MAKETYPE(desc.depth, desc.chan);
        const auto size = desc.planar ? ncvslideio::Size{desc.size.width, desc.size.height*desc.chan}
                                      : desc.size;
        mat.create(size, type);
    }
    else
    {
        GAPI_Assert(!desc.planar);
        mat.create(desc.dims, desc.depth);
#if !defined(GAPI_STANDALONE)
        // NB: WA for 1D mats.
        if (desc.dims.size() == 1u) {
            mat.dims = 1;
        }
#endif
    }
}

} // namespace gimpl
} // namespace ncvslideio
