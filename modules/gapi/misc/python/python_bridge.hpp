// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_PYTHON_BRIDGE_HPP
#define OPENCV_GAPI_PYTHON_BRIDGE_HPP

#include <opencv2/gapi.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gopaque.hpp>
#include <opencv2/gapi/render/render_types.hpp> // Prim

#define ID(T, E)  T
#define ID_(T, E) ID(T, E),

#define WRAP_ARGS(T, E, G) \
    G(T, E)

#define SWITCH(type, LIST_G, HC) \
    switch(type) { \
        LIST_G(HC, HC)  \
        default: \
            GAPI_Error("Unsupported type"); \
    }

using ncvslideio::gapi::wip::draw::Prim;

#define GARRAY_TYPE_LIST_G(G, G2) \
WRAP_ARGS(bool        , ncvslideio::gapi::ArgType::CV_BOOL,      G)  \
WRAP_ARGS(int         , ncvslideio::gapi::ArgType::CV_INT,       G)  \
WRAP_ARGS(int64_t     , ncvslideio::gapi::ArgType::CV_INT64,     G)  \
WRAP_ARGS(uint64_t    , ncvslideio::gapi::ArgType::CV_UINT64,    G)  \
WRAP_ARGS(double      , ncvslideio::gapi::ArgType::CV_DOUBLE,    G)  \
WRAP_ARGS(float       , ncvslideio::gapi::ArgType::CV_FLOAT,     G)  \
WRAP_ARGS(std::string , ncvslideio::gapi::ArgType::CV_STRING,    G)  \
WRAP_ARGS(ncvslideio::Point   , ncvslideio::gapi::ArgType::CV_POINT,     G)  \
WRAP_ARGS(ncvslideio::Point2f , ncvslideio::gapi::ArgType::CV_POINT2F,   G)  \
WRAP_ARGS(ncvslideio::Point3f , ncvslideio::gapi::ArgType::CV_POINT3F,   G)  \
WRAP_ARGS(ncvslideio::Size    , ncvslideio::gapi::ArgType::CV_SIZE,      G)  \
WRAP_ARGS(ncvslideio::Rect    , ncvslideio::gapi::ArgType::CV_RECT,      G)  \
WRAP_ARGS(ncvslideio::Scalar  , ncvslideio::gapi::ArgType::CV_SCALAR,    G)  \
WRAP_ARGS(ncvslideio::Mat     , ncvslideio::gapi::ArgType::CV_MAT,       G)  \
WRAP_ARGS(Prim        , ncvslideio::gapi::ArgType::CV_DRAW_PRIM, G)  \
WRAP_ARGS(ncvslideio::GArg    , ncvslideio::gapi::ArgType::CV_ANY,       G)  \
WRAP_ARGS(ncvslideio::GMat    , ncvslideio::gapi::ArgType::CV_GMAT,      G2) \

#define GOPAQUE_TYPE_LIST_G(G, G2) \
WRAP_ARGS(bool        , ncvslideio::gapi::ArgType::CV_BOOL,    G)  \
WRAP_ARGS(int         , ncvslideio::gapi::ArgType::CV_INT,     G)  \
WRAP_ARGS(int64_t     , ncvslideio::gapi::ArgType::CV_INT64,   G)  \
WRAP_ARGS(uint64_t    , ncvslideio::gapi::ArgType::CV_UINT64,  G)  \
WRAP_ARGS(double      , ncvslideio::gapi::ArgType::CV_DOUBLE,  G)  \
WRAP_ARGS(float       , ncvslideio::gapi::ArgType::CV_FLOAT,   G)  \
WRAP_ARGS(std::string , ncvslideio::gapi::ArgType::CV_STRING,  G)  \
WRAP_ARGS(ncvslideio::Point   , ncvslideio::gapi::ArgType::CV_POINT,   G)  \
WRAP_ARGS(ncvslideio::Point2f , ncvslideio::gapi::ArgType::CV_POINT2F, G)  \
WRAP_ARGS(ncvslideio::Point3f , ncvslideio::gapi::ArgType::CV_POINT3F, G)  \
WRAP_ARGS(ncvslideio::Size    , ncvslideio::gapi::ArgType::CV_SIZE,    G)  \
WRAP_ARGS(ncvslideio::GArg    , ncvslideio::gapi::ArgType::CV_ANY,     G)  \
WRAP_ARGS(ncvslideio::Rect    , ncvslideio::gapi::ArgType::CV_RECT,    G2) \

namespace ncvslideio {
namespace gapi {

// NB: ncvslideio.gapi.CV_BOOL in python
enum ArgType {
    CV_BOOL,
    CV_INT,
    CV_INT64,
    CV_UINT64,
    CV_DOUBLE,
    CV_FLOAT,
    CV_STRING,
    CV_POINT,
    CV_POINT2F,
    CV_POINT3F,
    CV_SIZE,
    CV_RECT,
    CV_SCALAR,
    CV_MAT,
    CV_GMAT,
    CV_DRAW_PRIM,
    CV_ANY,
};

GAPI_EXPORTS_W inline ncvslideio::GInferOutputs infer(const String& name, const ncvslideio::GInferInputs& inputs)
{
    return infer<Generic>(name, inputs);
}

GAPI_EXPORTS_W inline GInferOutputs infer(const std::string& name,
                                          const ncvslideio::GOpaque<ncvslideio::Rect>& roi,
                                          const GInferInputs& inputs)
{
    return infer<Generic>(name, roi, inputs);
}

GAPI_EXPORTS_W inline GInferListOutputs infer(const std::string& name,
                                              const ncvslideio::GArray<ncvslideio::Rect>& rois,
                                              const GInferInputs& inputs)
{
    return infer<Generic>(name, rois, inputs);
}

GAPI_EXPORTS_W inline GInferListOutputs infer2(const std::string& name,
                                               const ncvslideio::GMat in,
                                               const GInferListInputs& inputs)
{
    return infer2<Generic>(name, in, inputs);
}

} // namespace gapi

namespace detail {

template <template <typename> class Wrapper, typename T>
struct WrapType { using type = Wrapper<T>; };

template <template <typename> class T, typename... Types>
using MakeVariantType = ncvslideio::util::variant<typename WrapType<T, Types>::type...>;

template<typename T> struct ArgTypeTraits;

#define DEFINE_TYPE_TRAITS(T, E) \
template <> \
struct ArgTypeTraits<T> { \
    static constexpr const ncvslideio::gapi::ArgType type = E; \
}; \

GARRAY_TYPE_LIST_G(DEFINE_TYPE_TRAITS, DEFINE_TYPE_TRAITS)

} // namespace detail

class GAPI_EXPORTS_W_SIMPLE GOpaqueT
{
public:
    GOpaqueT() = default;
    using Storage = ncvslideio::detail::MakeVariantType<ncvslideio::GOpaque, GOPAQUE_TYPE_LIST_G(ID_, ID)>;

    template<typename T>
    GOpaqueT(ncvslideio::GOpaque<T> arg) : m_type(ncvslideio::detail::ArgTypeTraits<T>::type), m_arg(arg) { }

    GAPI_WRAP GOpaqueT(gapi::ArgType type) : m_type(type)
    {

#define HC(T, K) case K: \
        m_arg = ncvslideio::GOpaque<T>(); \
        break;

        SWITCH(type, GOPAQUE_TYPE_LIST_G, HC)
#undef HC
    }

    ncvslideio::detail::GOpaqueU strip() {
#define HC(T, K) case Storage:: index_of<ncvslideio::GOpaque<T>>(): \
        return ncvslideio::util::get<ncvslideio::GOpaque<T>>(m_arg).strip(); \

        SWITCH(m_arg.index(), GOPAQUE_TYPE_LIST_G, HC)
#undef HC

            GAPI_Error("InternalError");
    }

    GAPI_WRAP gapi::ArgType type() { return m_type; }
    const Storage& arg() const     { return m_arg;  }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

class GAPI_EXPORTS_W_SIMPLE GArrayT
{
public:
    GArrayT() = default;
    using Storage = ncvslideio::detail::MakeVariantType<ncvslideio::GArray, GARRAY_TYPE_LIST_G(ID_, ID)>;

    template<typename T>
    GArrayT(ncvslideio::GArray<T> arg) : m_type(ncvslideio::detail::ArgTypeTraits<T>::type), m_arg(arg) { }

    GAPI_WRAP GArrayT(gapi::ArgType type) : m_type(type)
    {

#define HC(T, K) case K: \
        m_arg = ncvslideio::GArray<T>(); \
        break;

        SWITCH(type, GARRAY_TYPE_LIST_G, HC)
#undef HC
    }

    ncvslideio::detail::GArrayU strip() {
#define HC(T, K) case Storage:: index_of<ncvslideio::GArray<T>>(): \
        return ncvslideio::util::get<ncvslideio::GArray<T>>(m_arg).strip(); \

        SWITCH(m_arg.index(), GARRAY_TYPE_LIST_G, HC)
#undef HC

        GAPI_Error("InternalError");
    }

    GAPI_WRAP gapi::ArgType type() { return m_type; }
    const Storage& arg() const     { return m_arg;  }

private:
    gapi::ArgType m_type;
    Storage m_arg;
};

namespace gapi {
namespace wip {

class GAPI_EXPORTS_W_SIMPLE GOutputs
{
public:
    GOutputs() = default;
    GOutputs(const std::string& id, ncvslideio::GKernel::M outMeta, ncvslideio::GArgs &&ins);

    GAPI_WRAP ncvslideio::GMat     getGMat();
    GAPI_WRAP ncvslideio::GScalar  getGScalar();
    GAPI_WRAP ncvslideio::GArrayT  getGArray(ncvslideio::gapi::ArgType type);
    GAPI_WRAP ncvslideio::GOpaqueT getGOpaque(ncvslideio::gapi::ArgType type);

private:
    class Priv;
    std::shared_ptr<Priv> m_priv;
};

GOutputs op(const std::string& id, ncvslideio::GKernel::M outMeta, ncvslideio::GArgs&& args);

template <typename... T>
GOutputs op(const std::string& id, ncvslideio::GKernel::M outMeta, T&&... args)
{
    return op(id, outMeta, ncvslideio::GArgs{ncvslideio::GArg(std::forward<T>(args))... });
}

} // namespace wip
} // namespace gapi
} // namespace ncvslideio

ncvslideio::gapi::wip::GOutputs ncvslideio::gapi::wip::op(const std::string& id,
                                          ncvslideio::GKernel::M outMeta,
                                          ncvslideio::GArgs&& args)
{
    ncvslideio::gapi::wip::GOutputs outputs{id, outMeta, std::move(args)};
    return outputs;
}

class ncvslideio::gapi::wip::GOutputs::Priv
{
public:
    Priv(const std::string& id, ncvslideio::GKernel::M outMeta, ncvslideio::GArgs &&ins);

    ncvslideio::GMat     getGMat();
    ncvslideio::GScalar  getGScalar();
    ncvslideio::GArrayT  getGArray(ncvslideio::gapi::ArgType);
    ncvslideio::GOpaqueT getGOpaque(ncvslideio::gapi::ArgType);

private:
    int output = 0;
    std::unique_ptr<ncvslideio::GCall> m_call;
};

ncvslideio::gapi::wip::GOutputs::Priv::Priv(const std::string& id, ncvslideio::GKernel::M outMeta, ncvslideio::GArgs &&args)
{
    ncvslideio::GKinds kinds;
    kinds.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(kinds),
            [](const ncvslideio::GArg& arg) { return arg.opaque_kind; });

    m_call.reset(new ncvslideio::GCall{ncvslideio::GKernel{id, {}, outMeta, {}, std::move(kinds), {}, {}}});
    m_call->setArgs(std::move(args));
}

ncvslideio::GMat ncvslideio::gapi::wip::GOutputs::Priv::getGMat()
{
    m_call->kernel().outShapes.push_back(ncvslideio::GShape::GMAT);
    m_call->kernel().outKinds.push_back(ncvslideio::detail::OpaqueKind::CV_UNKNOWN);
    // ...so _empty_ constructor is passed here.
    m_call->kernel().outCtors.emplace_back(ncvslideio::util::monostate{});
    return m_call->yield(output++);
}

ncvslideio::GScalar ncvslideio::gapi::wip::GOutputs::Priv::getGScalar()
{
    m_call->kernel().outShapes.push_back(ncvslideio::GShape::GSCALAR);
    m_call->kernel().outKinds.push_back(ncvslideio::detail::OpaqueKind::CV_UNKNOWN);
    // ...so _empty_ constructor is passed here.
    m_call->kernel().outCtors.emplace_back(ncvslideio::util::monostate{});
    return m_call->yieldScalar(output++);
}

ncvslideio::GArrayT ncvslideio::gapi::wip::GOutputs::Priv::getGArray(ncvslideio::gapi::ArgType type)
{
    m_call->kernel().outShapes.push_back(ncvslideio::GShape::GARRAY);

#define HC(T, K)                                                                                 \
    case K: {                                                                                    \
        const auto kind = ncvslideio::detail::GTypeTraits<ncvslideio::GArray<T>>::op_kind;                       \
        m_call->kernel().outKinds.emplace_back(kind);                                            \
        m_call->kernel().outCtors.emplace_back(ncvslideio::detail::GObtainCtor<ncvslideio::GArray<T>>::get());   \
        return ncvslideio::GArrayT(m_call->yieldArray<T>(output++));                                     \
    }

    SWITCH(type, GARRAY_TYPE_LIST_G, HC)
#undef HC
}

ncvslideio::GOpaqueT ncvslideio::gapi::wip::GOutputs::Priv::getGOpaque(ncvslideio::gapi::ArgType type)
{
    m_call->kernel().outShapes.push_back(ncvslideio::GShape::GOPAQUE);
#define HC(T, K)                                                                                  \
    case K: {                                                                                     \
        const auto kind = ncvslideio::detail::GTypeTraits<ncvslideio::GOpaque<T>>::op_kind;                       \
        m_call->kernel().outKinds.emplace_back(kind);                                             \
        m_call->kernel().outCtors.emplace_back(ncvslideio::detail::GObtainCtor<ncvslideio::GOpaque<T>>::get());   \
        return ncvslideio::GOpaqueT(m_call->yieldOpaque<T>(output++));                                    \
    }

    SWITCH(type, GOPAQUE_TYPE_LIST_G, HC)
#undef HC
}

ncvslideio::gapi::wip::GOutputs::GOutputs(const std::string& id,
                                  ncvslideio::GKernel::M outMeta,
                                  ncvslideio::GArgs &&ins) :
    m_priv(new ncvslideio::gapi::wip::GOutputs::Priv(id, outMeta, std::move(ins)))
{
}

ncvslideio::GMat ncvslideio::gapi::wip::GOutputs::getGMat()
{
    return m_priv->getGMat();
}

ncvslideio::GScalar ncvslideio::gapi::wip::GOutputs::getGScalar()
{
    return m_priv->getGScalar();
}

ncvslideio::GArrayT ncvslideio::gapi::wip::GOutputs::getGArray(ncvslideio::gapi::ArgType type)
{
    return m_priv->getGArray(type);
}

ncvslideio::GOpaqueT ncvslideio::gapi::wip::GOutputs::getGOpaque(ncvslideio::gapi::ArgType type)
{
    return m_priv->getGOpaque(type);
}

#endif // OPENCV_GAPI_PYTHON_BRIDGE_HPP
