// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include <opencv2/gapi/infer/bindings_onnx.hpp>

ncvslideio::gapi::onnx::PyParams::PyParams(const std::string& tag,
                                   const std::string& model_path)
    : m_priv(std::make_shared<Params<ncvslideio::gapi::Generic>>(tag, model_path)) {}

ncvslideio::gapi::onnx::PyParams& ncvslideio::gapi::onnx::PyParams::cfgMeanStd(const std::string &layer_name,
                                                               const ncvslideio::Scalar &m,
                                                               const ncvslideio::Scalar &s) {
    m_priv->cfgMeanStdDev(layer_name, m, s);
    return *this;
}

ncvslideio::gapi::onnx::PyParams& ncvslideio::gapi::onnx::PyParams::cfgNormalize(const std::string &layer_name,
                                                                 bool flag) {
    m_priv->cfgNormalize(layer_name, flag);
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgAddExecutionProvider(ncvslideio::gapi::onnx::ep::OpenVINO ep) {
    m_priv->cfgAddExecutionProvider(std::move(ep));
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgAddExecutionProvider(ncvslideio::gapi::onnx::ep::DirectML ep) {
    m_priv->cfgAddExecutionProvider(std::move(ep));
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgAddExecutionProvider(ncvslideio::gapi::onnx::ep::CoreML ep) {
    m_priv->cfgAddExecutionProvider(std::move(ep));
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgAddExecutionProvider(ncvslideio::gapi::onnx::ep::CUDA ep) {
    m_priv->cfgAddExecutionProvider(std::move(ep));
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgAddExecutionProvider(ncvslideio::gapi::onnx::ep::TensorRT ep) {
    m_priv->cfgAddExecutionProvider(std::move(ep));
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgDisableMemPattern() {
    m_priv->cfgDisableMemPattern();
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgSessionOptions(const std::map<std::string, std::string>& options) {
    m_priv->cfgSessionOptions(options);
    return *this;
}

ncvslideio::gapi::onnx::PyParams&
ncvslideio::gapi::onnx::PyParams::cfgOptLevel(const int opt_level) {
    m_priv->cfgOptLevel(opt_level);
    return *this;
}

ncvslideio::gapi::GBackend ncvslideio::gapi::onnx::PyParams::backend() const {
    return m_priv->backend();
}

std::string ncvslideio::gapi::onnx::PyParams::tag() const { return m_priv->tag(); }

ncvslideio::util::any ncvslideio::gapi::onnx::PyParams::params() const {
    return m_priv->params();
}

ncvslideio::gapi::onnx::PyParams ncvslideio::gapi::onnx::params(
    const std::string& tag, const std::string& model_path) {
    return {tag, model_path};
}
