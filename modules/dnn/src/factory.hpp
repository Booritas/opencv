// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_FACTORY_HPP
#define OPENCV_DNN_FACTORY_HPP

#include "backend.hpp"

namespace ncvslideio { namespace dnn_backend {

class IDNNBackendFactory
{
public:
    virtual ~IDNNBackendFactory() {}
    virtual std::shared_ptr<ncvslideio::dnn_backend::NetworkBackend> createNetworkBackend() const = 0;
};

//
// PluginDNNBackendFactory is implemented in plugin_wrapper
//

std::shared_ptr<IDNNBackendFactory> createPluginDNNBackendFactory(const std::string& baseName);

/// @brief Returns createPluginDNNBackendFactory()->createNetworkBackend()
ncvslideio::dnn_backend::NetworkBackend& createPluginDNNNetworkBackend(const std::string& baseName);

}}  // namespace

#endif  // OPENCV_DNN_FACTORY_HPP
