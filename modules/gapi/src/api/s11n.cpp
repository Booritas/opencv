// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020-2021 Intel Corporation

#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>

#include "backends/common/serialization.hpp"

std::vector<char> ncvslideio::gapi::serialize(const ncvslideio::GComputation &c) {
    ncvslideio::gapi::s11n::ByteMemoryOutStream os;
    c.serialize(os);
    return os.data();
}

ncvslideio::GComputation ncvslideio::gapi::detail::getGraph(const std::vector<char> &p) {
    ncvslideio::gapi::s11n::ByteMemoryInStream is(p);
    return ncvslideio::GComputation(is);
}

ncvslideio::GMetaArgs ncvslideio::gapi::detail::getMetaArgs(const std::vector<char> &p) {
    ncvslideio::gapi::s11n::ByteMemoryInStream is(p);
    return meta_args_deserialize(is);
}

ncvslideio::GRunArgs ncvslideio::gapi::detail::getRunArgs(const std::vector<char> &p) {
    ncvslideio::gapi::s11n::ByteMemoryInStream is(p);
    return run_args_deserialize(is);
}

std::vector<std::string> ncvslideio::gapi::detail::getVectorOfStrings(const std::vector<char> &p) {
    ncvslideio::gapi::s11n::ByteMemoryInStream is(p);
    return vector_of_strings_deserialize(is);
}

std::vector<char> ncvslideio::gapi::serialize(const ncvslideio::GMetaArgs& ma)
{
    ncvslideio::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, ma);
    return os.data();
}

std::vector<char> ncvslideio::gapi::serialize(const ncvslideio::GRunArgs& ra)
{
    ncvslideio::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, ra);
    return os.data();
}

std::vector<char> ncvslideio::gapi::serialize(const ncvslideio::GCompileArgs& ca)
{
    ncvslideio::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, ca);
    return os.data();
}

std::vector<char> ncvslideio::gapi::serialize(const std::vector<std::string>& vs)
{
    ncvslideio::gapi::s11n::ByteMemoryOutStream os;
    serialize(os, vs);
    return os.data();
}

// FIXME: This function should move from S11N to GRunArg-related entities.
// it has nothing to do with the S11N as it is
ncvslideio::GRunArgsP ncvslideio::gapi::bind(ncvslideio::GRunArgs &out_args)
{
    ncvslideio::GRunArgsP outputs;
    outputs.reserve(out_args.size());
    for (ncvslideio::GRunArg &res_obj : out_args)
    {
        using T = ncvslideio::GRunArg;
        switch (res_obj.index())
        {
#if !defined(GAPI_STANDALONE)
        case T::index_of<ncvslideio::UMat>() :
            outputs.emplace_back(&(ncvslideio::util::get<ncvslideio::UMat>(res_obj)));
            break;
#endif
        case ncvslideio::GRunArg::index_of<ncvslideio::Mat>() :
            outputs.emplace_back(&(ncvslideio::util::get<ncvslideio::Mat>(res_obj)));
            break;
        case ncvslideio::GRunArg::index_of<ncvslideio::Scalar>() :
            outputs.emplace_back(&(ncvslideio::util::get<ncvslideio::Scalar>(res_obj)));
            break;
        case T::index_of<ncvslideio::detail::VectorRef>() :
            outputs.emplace_back(ncvslideio::util::get<ncvslideio::detail::VectorRef>(res_obj));
            break;
        case T::index_of<ncvslideio::detail::OpaqueRef>() :
            outputs.emplace_back(ncvslideio::util::get<ncvslideio::detail::OpaqueRef>(res_obj));
            break;
        case ncvslideio::GRunArg::index_of<ncvslideio::RMat>() :
            outputs.emplace_back(&(ncvslideio::util::get<ncvslideio::RMat>(res_obj)));
            break;
        case ncvslideio::GRunArg::index_of<ncvslideio::MediaFrame>() :
            outputs.emplace_back(&(ncvslideio::util::get<ncvslideio::MediaFrame>(res_obj)));
            break;
        default:
            GAPI_Error("This value type is not supported!"); // ...maybe because of STANDALONE mode.
            break;
        }
    }
    return outputs;
}

// FIXME: move it out of s11n to api/
// FIXME: don't we have such function already?
ncvslideio::GRunArg ncvslideio::gapi::bind(ncvslideio::GRunArgP &out)
{
    using T = ncvslideio::GRunArgP;
    switch (out.index())
    {
#if !defined(GAPI_STANDALONE)
    case T::index_of<ncvslideio::UMat*>() :
        GAPI_Error("Please implement this!");
        break;
#endif

    case T::index_of<ncvslideio::detail::VectorRef>() :
        return ncvslideio::GRunArg(ncvslideio::util::get<ncvslideio::detail::VectorRef>(out));

    case T::index_of<ncvslideio::detail::OpaqueRef>() :
        return ncvslideio::GRunArg(ncvslideio::util::get<ncvslideio::detail::OpaqueRef>(out));

    case T::index_of<ncvslideio::Mat*>() :
        return ncvslideio::GRunArg(*ncvslideio::util::get<ncvslideio::Mat*>(out));

    case T::index_of<ncvslideio::Scalar*>() :
        return ncvslideio::GRunArg(*ncvslideio::util::get<ncvslideio::Scalar*>(out));

    case T::index_of<ncvslideio::RMat*>() :
        return ncvslideio::GRunArg(*ncvslideio::util::get<ncvslideio::RMat*>(out));

    case T::index_of<ncvslideio::MediaFrame*>() :
        return ncvslideio::GRunArg(*ncvslideio::util::get<ncvslideio::MediaFrame*>(out));

    default:
        // ...maybe our types were extended
        GAPI_Error("This value type is UNKNOWN!");
        break;
    }
    return ncvslideio::GRunArg();
}
