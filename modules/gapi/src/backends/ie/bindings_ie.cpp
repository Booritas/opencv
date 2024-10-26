#include <opencv2/gapi/infer/bindings_ie.hpp>

ncvslideio::gapi::ie::PyParams::PyParams(const std::string &tag,
                                 const std::string &model,
                                 const std::string &weights,
                                 const std::string &device)
    : m_priv(std::make_shared<Params<ncvslideio::gapi::Generic>>(tag, model, weights, device)) {
}

ncvslideio::gapi::ie::PyParams::PyParams(const std::string &tag,
                                 const std::string &model,
                                 const std::string &device)
    : m_priv(std::make_shared<Params<ncvslideio::gapi::Generic>>(tag, model, device)) {
}

ncvslideio::gapi::GBackend ncvslideio::gapi::ie::PyParams::backend() const {
    return m_priv->backend();
}

std::string ncvslideio::gapi::ie::PyParams::tag() const {
    return m_priv->tag();
}

ncvslideio::util::any ncvslideio::gapi::ie::PyParams::params() const {
    return m_priv->params();
}

ncvslideio::gapi::ie::PyParams ncvslideio::gapi::ie::params(const std::string &tag,
                                            const std::string &model,
                                            const std::string &weights,
                                            const std::string &device) {
    return {tag, model, weights, device};
}

ncvslideio::gapi::ie::PyParams ncvslideio::gapi::ie::params(const std::string &tag,
                                            const std::string &model,
                                            const std::string &device) {
    return {tag, model, device};
}

ncvslideio::gapi::ie::PyParams& ncvslideio::gapi::ie::PyParams::constInput(const std::string &layer_name,
                                                           const ncvslideio::Mat &data,
                                                           TraitAs hint) {
    m_priv->constInput(layer_name, data, hint);
    return *this;
}

ncvslideio::gapi::ie::PyParams& ncvslideio::gapi::ie::PyParams::cfgNumRequests(size_t nireq) {
    m_priv->cfgNumRequests(nireq);
    return *this;
}

ncvslideio::gapi::ie::PyParams&
ncvslideio::gapi::ie::PyParams::cfgBatchSize(const size_t size) {
    m_priv->cfgBatchSize(size);
    return *this;
}
