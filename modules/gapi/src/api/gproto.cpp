// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "precomp.hpp"

#include <ade/util/algorithm.hpp>
#include <opencv2/gapi/util/throw.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gproto.hpp>

#include "api/gorigin.hpp"
#include "api/gproto_priv.hpp"

// FIXME: it should be a visitor!
// FIXME: Reimplement with traits?

const ncvslideio::GOrigin& ncvslideio::gimpl::proto::origin_of(const ncvslideio::GProtoArg &arg)
{
    switch (arg.index())
    {
    case ncvslideio::GProtoArg::index_of<ncvslideio::GMat>():
        return util::get<ncvslideio::GMat>(arg).priv();

    case ncvslideio::GProtoArg::index_of<ncvslideio::GMatP>():
        return util::get<ncvslideio::GMatP>(arg).priv();

    case ncvslideio::GProtoArg::index_of<ncvslideio::GFrame>():
        return util::get<ncvslideio::GFrame>(arg).priv();

    case ncvslideio::GProtoArg::index_of<ncvslideio::GScalar>():
        return util::get<ncvslideio::GScalar>(arg).priv();

    case ncvslideio::GProtoArg::index_of<ncvslideio::detail::GArrayU>():
        return util::get<ncvslideio::detail::GArrayU>(arg).priv();

    case ncvslideio::GProtoArg::index_of<ncvslideio::detail::GOpaqueU>():
        return util::get<ncvslideio::detail::GOpaqueU>(arg).priv();

    default:
        util::throw_error(std::logic_error("Unsupported GProtoArg type"));
    }
}

const ncvslideio::GOrigin& ncvslideio::gimpl::proto::origin_of(const ncvslideio::GArg &arg)
{
    // Generic, but not very efficient implementation
    // FIXME: Walking a thin line here!!! Here we rely that GArg and
    // GProtoArg share the same object and this is true while objects
    // are reference-counted, so return value is not a reference to a tmp.
    return origin_of(rewrap(arg));
}

bool ncvslideio::gimpl::proto::is_dynamic(const ncvslideio::GArg& arg)
{
    // FIXME: refactor this method to be auto-generated from
    // - GProtoArg variant parameter pack, and
    // - traits over every type
    switch (arg.kind)
    {
    case detail::ArgKind::GMAT:
    case detail::ArgKind::GMATP:
    case detail::ArgKind::GFRAME:
    case detail::ArgKind::GSCALAR:
    case detail::ArgKind::GARRAY:
    case detail::ArgKind::GOPAQUE:
        return true;

    default:
        return false;
    }
}

ncvslideio::GRunArg ncvslideio::value_of(const ncvslideio::GOrigin &origin)
{
    switch (origin.shape)
    {
    case GShape::GSCALAR: return GRunArg(util::get<ncvslideio::Scalar>(origin.value));
    case GShape::GARRAY:  return GRunArg(util::get<ncvslideio::detail::VectorRef>(origin.value));
    case GShape::GMAT:    return GRunArg(util::get<ncvslideio::Mat>(origin.value));
    default: util::throw_error(std::logic_error("Unsupported shape for constant"));
    }
}

ncvslideio::GProtoArg ncvslideio::gimpl::proto::rewrap(const ncvslideio::GArg &arg)
{
    // FIXME: replace with a more generic any->variant
    // (or variant<T> -> variant<U>) conversion?
    switch (arg.kind)
    {
    case detail::ArgKind::GMAT:    return GProtoArg(arg.get<ncvslideio::GMat>());
    case detail::ArgKind::GMATP:   return GProtoArg(arg.get<ncvslideio::GMatP>());
    case detail::ArgKind::GFRAME:  return GProtoArg(arg.get<ncvslideio::GFrame>());
    case detail::ArgKind::GSCALAR: return GProtoArg(arg.get<ncvslideio::GScalar>());
    case detail::ArgKind::GARRAY:  return GProtoArg(arg.get<ncvslideio::detail::GArrayU>());
    case detail::ArgKind::GOPAQUE: return GProtoArg(arg.get<ncvslideio::detail::GOpaqueU>());
    default: util::throw_error(std::logic_error("Unsupported GArg type"));
    }
}

ncvslideio::GMetaArg ncvslideio::descr_of(const ncvslideio::GRunArg &arg)
{
    switch (arg.index())
    {
        case GRunArg::index_of<ncvslideio::Mat>():
            return ncvslideio::GMetaArg(ncvslideio::descr_of(util::get<ncvslideio::Mat>(arg)));

        case GRunArg::index_of<ncvslideio::Scalar>():
            return ncvslideio::GMetaArg(descr_of(util::get<ncvslideio::Scalar>(arg)));

        case GRunArg::index_of<ncvslideio::detail::VectorRef>():
            return ncvslideio::GMetaArg(util::get<ncvslideio::detail::VectorRef>(arg).descr_of());

        case GRunArg::index_of<ncvslideio::detail::OpaqueRef>():
            return ncvslideio::GMetaArg(util::get<ncvslideio::detail::OpaqueRef>(arg).descr_of());

        case GRunArg::index_of<ncvslideio::gapi::wip::IStreamSource::Ptr>():
            return ncvslideio::util::get<ncvslideio::gapi::wip::IStreamSource::Ptr>(arg)->descr_of();

        case GRunArg::index_of<ncvslideio::RMat>():
            return ncvslideio::GMetaArg(ncvslideio::util::get<ncvslideio::RMat>(arg).desc());

        case GRunArg::index_of<ncvslideio::MediaFrame>():
            return ncvslideio::GMetaArg(ncvslideio::util::get<ncvslideio::MediaFrame>(arg).desc());

        default: util::throw_error(std::logic_error("Unsupported GRunArg type"));
    }
}

ncvslideio::GMetaArgs ncvslideio::descr_of(const ncvslideio::GRunArgs &args)
{
    ncvslideio::GMetaArgs metas;
    ade::util::transform(args, std::back_inserter(metas), [](const ncvslideio::GRunArg &arg){ return descr_of(arg); });
    return metas;
}

// FIXME: Is it tested for all types?
ncvslideio::GMetaArg ncvslideio::descr_of(const ncvslideio::GRunArgP &argp)
{
    switch (argp.index())
    {
#if !defined(GAPI_STANDALONE)
    case GRunArgP::index_of<ncvslideio::UMat*>():              return GMetaArg(ncvslideio::descr_of(*util::get<ncvslideio::UMat*>(argp)));
#endif //  !defined(GAPI_STANDALONE)
    case GRunArgP::index_of<ncvslideio::Mat*>():               return GMetaArg(ncvslideio::descr_of(*util::get<ncvslideio::Mat*>(argp)));
    case GRunArgP::index_of<ncvslideio::Scalar*>():            return GMetaArg(descr_of(*util::get<ncvslideio::Scalar*>(argp)));
    case GRunArgP::index_of<ncvslideio::MediaFrame*>():        return GMetaArg(descr_of(*util::get<ncvslideio::MediaFrame*>(argp)));
    case GRunArgP::index_of<ncvslideio::detail::VectorRef>():  return GMetaArg(util::get<ncvslideio::detail::VectorRef>(argp).descr_of());
    case GRunArgP::index_of<ncvslideio::detail::OpaqueRef>():  return GMetaArg(util::get<ncvslideio::detail::OpaqueRef>(argp).descr_of());
    default: util::throw_error(std::logic_error("Unsupported GRunArgP type"));
    }
}

// FIXME: Is it tested for all types??
bool ncvslideio::can_describe(const GMetaArg& meta, const GRunArgP& argp)
{
    switch (argp.index())
    {
#if !defined(GAPI_STANDALONE)
    case GRunArgP::index_of<ncvslideio::UMat*>():              return meta == GMetaArg(ncvslideio::descr_of(*util::get<ncvslideio::UMat*>(argp)));
#endif //  !defined(GAPI_STANDALONE)
    case GRunArgP::index_of<ncvslideio::Mat*>():               return util::holds_alternative<GMatDesc>(meta) &&
                                                              util::get<GMatDesc>(meta).canDescribe(*util::get<ncvslideio::Mat*>(argp));
    case GRunArgP::index_of<ncvslideio::Scalar*>():            return meta == GMetaArg(ncvslideio::descr_of(*util::get<ncvslideio::Scalar*>(argp)));
    case GRunArgP::index_of<ncvslideio::MediaFrame*>():        return meta == GMetaArg(ncvslideio::descr_of(*util::get<ncvslideio::MediaFrame*>(argp)));
    case GRunArgP::index_of<ncvslideio::detail::VectorRef>():  return meta == GMetaArg(util::get<ncvslideio::detail::VectorRef>(argp).descr_of());
    case GRunArgP::index_of<ncvslideio::detail::OpaqueRef>():  return meta == GMetaArg(util::get<ncvslideio::detail::OpaqueRef>(argp).descr_of());
    default: util::throw_error(std::logic_error("Unsupported GRunArgP type"));
    }
}

// FIXME: Is it tested for all types??
bool ncvslideio::can_describe(const GMetaArg& meta, const GRunArg& arg)
{
    switch (arg.index())
    {
#if !defined(GAPI_STANDALONE)
    case GRunArg::index_of<ncvslideio::UMat>():              return meta == ncvslideio::GMetaArg(descr_of(util::get<ncvslideio::UMat>(arg)));
#endif //  !defined(GAPI_STANDALONE)
    case GRunArg::index_of<ncvslideio::Mat>():               return util::holds_alternative<GMatDesc>(meta) &&
                                                            util::get<GMatDesc>(meta).canDescribe(util::get<ncvslideio::Mat>(arg));
    case GRunArg::index_of<ncvslideio::Scalar>():            return meta == ncvslideio::GMetaArg(descr_of(util::get<ncvslideio::Scalar>(arg)));
    case GRunArg::index_of<ncvslideio::detail::VectorRef>(): return meta == ncvslideio::GMetaArg(util::get<ncvslideio::detail::VectorRef>(arg).descr_of());
    case GRunArg::index_of<ncvslideio::detail::OpaqueRef>(): return meta == ncvslideio::GMetaArg(util::get<ncvslideio::detail::OpaqueRef>(arg).descr_of());
    case GRunArg::index_of<ncvslideio::gapi::wip::IStreamSource::Ptr>(): return util::holds_alternative<GMatDesc>(meta); // FIXME(?) may be not the best option
    case GRunArg::index_of<ncvslideio::RMat>():              return util::holds_alternative<GMatDesc>(meta) &&
                                                            util::get<GMatDesc>(meta).canDescribe(ncvslideio::util::get<ncvslideio::RMat>(arg));
    case GRunArg::index_of<ncvslideio::MediaFrame>():        return meta == ncvslideio::GMetaArg(util::get<ncvslideio::MediaFrame>(arg).desc());
    default: util::throw_error(std::logic_error("Unsupported GRunArg type"));
    }
}

bool ncvslideio::can_describe(const GMetaArgs &metas, const GRunArgs &args)
{
    return metas.size() == args.size() &&
           std::equal(metas.begin(), metas.end(), args.begin(),
                     [](const GMetaArg& meta, const GRunArg& arg) {
                         return can_describe(meta, arg);
                     });
}

void ncvslideio::gimpl::proto::validate_input_meta_arg(const ncvslideio::GMetaArg& meta)
{
    switch (meta.index())
    {
        case ncvslideio::GMetaArg::index_of<ncvslideio::GMatDesc>():
        {
            ncvslideio::gimpl::proto::validate_input_meta(ncvslideio::util::get<GMatDesc>(meta)); //may throw
            break;
        }
        default:
            break;
    }
}

void ncvslideio::gimpl::proto::validate_input_meta(const ncvslideio::GMatDesc& meta)
{
    if (meta.dims.empty())
    {
        if (!(meta.size.height > 0 && meta.size.width > 0))
        {
            ncvslideio::util::throw_error
                (std::logic_error(
                 "Image format is invalid. Size must contain positive values"
                 ", got width: " + std::to_string(meta.size.width ) +
                 (", height: ") + std::to_string(meta.size.height)));
        }

        if (!(meta.chan > 0))
        {
            ncvslideio::util::throw_error
                (std::logic_error(
                 "Image format is invalid. Channel mustn't be negative value, got channel: " +
                 std::to_string(meta.chan)));
        }
    }

    if (!(meta.depth >= 0))
    {
        ncvslideio::util::throw_error
            (std::logic_error(
             "Image format is invalid. Depth must be positive value, got depth: " +
             std::to_string(meta.depth)));
    }
    // All checks are ok
}

// FIXME: Is it tested for all types?
// FIXME: Where does this validation happen??
void ncvslideio::validate_input_arg(const GRunArg& arg)
{
    // FIXME: It checks only Mat argument
    switch (arg.index())
    {
#if !defined(GAPI_STANDALONE)
    case GRunArg::index_of<ncvslideio::UMat>():
    {
        const auto desc = ncvslideio::descr_of(util::get<ncvslideio::UMat>(arg));
        ncvslideio::gimpl::proto::validate_input_meta(desc); //may throw
        break;
    }
#endif //  !defined(GAPI_STANDALONE)
    case GRunArg::index_of<ncvslideio::Mat>():
    {
        const auto desc = ncvslideio::descr_of(util::get<ncvslideio::Mat>(arg));
        ncvslideio::gimpl::proto::validate_input_meta(desc); //may throw
        break;
    }
    default:
        // No extra handling
        break;
    }
}

void ncvslideio::validate_input_args(const GRunArgs& args)
{
    for (const auto& arg : args)
    {
        validate_input_arg(arg);
    }
}

namespace ncvslideio {
std::ostream& operator<<(std::ostream& os, const ncvslideio::GMetaArg &arg)
{
    // FIXME: Implement via variant visitor
    switch (arg.index())
    {
    case ncvslideio::GMetaArg::index_of<util::monostate>():
        os << "(unresolved)";
        break;

    case ncvslideio::GMetaArg::index_of<ncvslideio::GMatDesc>():
        os << util::get<ncvslideio::GMatDesc>(arg);
        break;

    case ncvslideio::GMetaArg::index_of<ncvslideio::GScalarDesc>():
        os << util::get<ncvslideio::GScalarDesc>(arg);
        break;

    case ncvslideio::GMetaArg::index_of<ncvslideio::GArrayDesc>():
        os << util::get<ncvslideio::GArrayDesc>(arg);
        break;

    case ncvslideio::GMetaArg::index_of<ncvslideio::GOpaqueDesc>():
        os << util::get<ncvslideio::GOpaqueDesc>(arg);
        break;

    case ncvslideio::GMetaArg::index_of<ncvslideio::GFrameDesc>():
        os << util::get<ncvslideio::GFrameDesc>(arg);
        break;

    default:
        GAPI_Error("InternalError");
    }

    return os;
}
} // namespace ncvslideio

const void* ncvslideio::gimpl::proto::ptr(const GRunArgP &arg)
{
    switch (arg.index())
    {
#if !defined(GAPI_STANDALONE)
    case GRunArgP::index_of<ncvslideio::UMat*>():
        return static_cast<const void*>(ncvslideio::util::get<ncvslideio::UMat*>(arg));
#endif
    case GRunArgP::index_of<ncvslideio::Mat*>():
        return static_cast<const void*>(ncvslideio::util::get<ncvslideio::Mat*>(arg));
    case GRunArgP::index_of<ncvslideio::Scalar*>():
        return static_cast<const void*>(ncvslideio::util::get<ncvslideio::Scalar*>(arg));
    case GRunArgP::index_of<ncvslideio::RMat*>():
        return static_cast<const void*>(ncvslideio::util::get<ncvslideio::RMat*>(arg));
    case GRunArgP::index_of<ncvslideio::detail::VectorRef>():
        return ncvslideio::util::get<ncvslideio::detail::VectorRef>(arg).ptr();
    case GRunArgP::index_of<ncvslideio::detail::OpaqueRef>():
        return ncvslideio::util::get<ncvslideio::detail::OpaqueRef>(arg).ptr();
    case GRunArgP::index_of<ncvslideio::MediaFrame*>():
        return static_cast<const void*>(ncvslideio::util::get<ncvslideio::MediaFrame*>(arg));
    default:
        util::throw_error(std::logic_error("Unknown GRunArgP type!"));
    }
}
