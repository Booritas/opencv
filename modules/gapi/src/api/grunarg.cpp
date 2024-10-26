// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "precomp.hpp"
#include <opencv2/gapi/garg.hpp>

ncvslideio::GRunArg::GRunArg() {
}

ncvslideio::GRunArg::GRunArg(const ncvslideio::GRunArg &arg)
    : ncvslideio::GRunArgBase(static_cast<const ncvslideio::GRunArgBase&>(arg))
    , meta(arg.meta) {
}

ncvslideio::GRunArg::GRunArg(ncvslideio::GRunArg &&arg)
    : ncvslideio::GRunArgBase(std::move(static_cast<const ncvslideio::GRunArgBase&>(arg)))
    , meta(std::move(arg.meta)) {
}

ncvslideio::GRunArg& ncvslideio::GRunArg::operator= (const ncvslideio::GRunArg &arg) {
    ncvslideio::GRunArgBase::operator=(static_cast<const ncvslideio::GRunArgBase&>(arg));
    meta = arg.meta;
    return *this;
}

ncvslideio::GRunArg& ncvslideio::GRunArg::operator= (ncvslideio::GRunArg &&arg) {
    ncvslideio::GRunArgBase::operator=(std::move(static_cast<const ncvslideio::GRunArgBase&>(arg)));
    meta = std::move(arg.meta);
    return *this;
}

// NB: Construct GRunArgsP based on passed info and store the memory in passed ncvslideio::GRunArgs.
// Needed for python bridge, because in case python user doesn't pass output arguments to apply.
void ncvslideio::detail::constructGraphOutputs(const ncvslideio::GTypesInfo &out_info,
                                       ncvslideio::GRunArgs         &args,
                                       ncvslideio::GRunArgsP        &outs)
{
    for (auto&& info : out_info)
    {
        switch (info.shape)
        {
            case ncvslideio::GShape::GMAT:
            {
                args.emplace_back(ncvslideio::Mat{});
                outs.emplace_back(&ncvslideio::util::get<ncvslideio::Mat>(args.back()));
                break;
            }
            case ncvslideio::GShape::GSCALAR:
            {
                args.emplace_back(ncvslideio::Scalar{});
                outs.emplace_back(&ncvslideio::util::get<ncvslideio::Scalar>(args.back()));
                break;
            }
            case ncvslideio::GShape::GARRAY:
            {
                ncvslideio::detail::VectorRef ref;
                util::get<ncvslideio::detail::ConstructVec>(info.ctor)(ref);
                args.emplace_back(ref);
                outs.emplace_back(ncvslideio::util::get<ncvslideio::detail::VectorRef>(args.back()));
                break;
            }
            case ncvslideio::GShape::GOPAQUE:
            {
                ncvslideio::detail::OpaqueRef ref;
                util::get<ncvslideio::detail::ConstructOpaque>(info.ctor)(ref);
                args.emplace_back(ref);
                outs.emplace_back(ref);
                break;
            }

            default:
                util::throw_error(std::logic_error("Unsupported output shape for python"));
        }
    }
}
