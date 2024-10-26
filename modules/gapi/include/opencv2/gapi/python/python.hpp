// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation


#ifndef OPENCV_GAPI_PYTHON_API_HPP
#define OPENCV_GAPI_PYTHON_API_HPP

#include <opencv2/gapi/gkernel.hpp>     // GKernelPackage
#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS

namespace ncvslideio {
namespace gapi {

/**
 * @brief This namespace contains G-API Python backend functions,
 * structures, and symbols.
 *
 * This functionality is required to enable G-API custom operations
 * and kernels when using G-API from Python, no need to use it in the
 * C++ form.
 */
namespace python {

GAPI_EXPORTS ncvslideio::gapi::GBackend backend();

struct GPythonContext
{
    const ncvslideio::GArgs      &ins;
    const ncvslideio::GMetaArgs  &in_metas;
    const ncvslideio::GTypesInfo &out_info;

    ncvslideio::optional<ncvslideio::GArg> m_state;
};

using Impl = std::function<ncvslideio::GRunArgs(const GPythonContext&)>;
using Setup = std::function<ncvslideio::GArg(const GMetaArgs&, const GArgs&)>;

class GAPI_EXPORTS GPythonKernel
{
public:
    GPythonKernel() = default;
    GPythonKernel(Impl run, Setup setup);

    Impl  run;
    Setup setup       = nullptr;
    bool  is_stateful = false;
};

class GAPI_EXPORTS GPythonFunctor : public ncvslideio::gapi::GFunctor
{
public:
    using Meta = ncvslideio::GKernel::M;

    GPythonFunctor(const char* id, const Meta& meta, const Impl& impl,
                   const Setup& setup = nullptr);

    GKernelImpl    impl()    const override;
    gapi::GBackend backend() const override;

private:
    GKernelImpl impl_;
};

} // namespace python
} // namespace gapi
} // namespace ncvslideio

#endif // OPENCV_GAPI_PYTHON_API_HPP
