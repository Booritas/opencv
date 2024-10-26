#ifndef OPENCV_GAPI_PYOPENCV_GAPI_HPP
#define OPENCV_GAPI_PYOPENCV_GAPI_HPP

#ifdef HAVE_OPENCV_GAPI

#ifdef _MSC_VER
#pragma warning(disable: 4503)  // "decorated name length exceeded"
#endif

#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/python/python.hpp>

// NB: Python wrapper replaces :: with _ for classes
using gapi_GKernelPackage           = ncvslideio::GKernelPackage;
using gapi_GNetPackage              = ncvslideio::gapi::GNetPackage;
using gapi_ie_PyParams              = ncvslideio::gapi::ie::PyParams;
using gapi_onnx_PyParams            = ncvslideio::gapi::onnx::PyParams;
using gapi_ov_PyParams              = ncvslideio::gapi::ov::PyParams;
using gapi_wip_IStreamSource_Ptr    = ncvslideio::Ptr<ncvslideio::gapi::wip::IStreamSource>;
using detail_ExtractArgsCallback    = ncvslideio::detail::ExtractArgsCallback;
using detail_ExtractMetaCallback    = ncvslideio::detail::ExtractMetaCallback;
using vector_GNetParam              = std::vector<ncvslideio::gapi::GNetParam>;
using vector_GMat                   = std::vector<ncvslideio::GMat>;
using gapi_streaming_queue_capacity = ncvslideio::gapi::streaming::queue_capacity;
using GStreamerSource_OutputType    = ncvslideio::gapi::wip::GStreamerSource::OutputType;
using map_string_and_int            = std::map<std::string, int>;
using map_string_and_string         = std::map<std::string, std::string>;
using map_string_and_string         = std::map<std::string, std::string>;
using map_string_and_vector_size_t  = std::map<std::string, std::vector<size_t>>;
using map_string_and_vector_float   = std::map<std::string, std::vector<float>>;
using map_int_and_double            = std::map<int, double>;
using ep_OpenVINO                   = ncvslideio::gapi::onnx::ep::OpenVINO;
using ep_DirectML                   = ncvslideio::gapi::onnx::ep::DirectML;
using ep_CoreML                     = ncvslideio::gapi::onnx::ep::CoreML;
using ep_CUDA                       = ncvslideio::gapi::onnx::ep::CUDA;
using ep_TensorRT                   = ncvslideio::gapi::onnx::ep::TensorRT;

// NB: Python wrapper generate T_U for T<U>
// This behavior is only observed for inputs
using GOpaque_bool    = ncvslideio::GOpaque<bool>;
using GOpaque_int     = ncvslideio::GOpaque<int>;
using GOpaque_double  = ncvslideio::GOpaque<double>;
using GOpaque_float   = ncvslideio::GOpaque<double>;
using GOpaque_string  = ncvslideio::GOpaque<std::string>;
using GOpaque_Point2i = ncvslideio::GOpaque<ncvslideio::Point>;
using GOpaque_Point2f = ncvslideio::GOpaque<ncvslideio::Point2f>;
using GOpaque_Size    = ncvslideio::GOpaque<ncvslideio::Size>;
using GOpaque_Rect    = ncvslideio::GOpaque<ncvslideio::Rect>;

using GArray_bool    = ncvslideio::GArray<bool>;
using GArray_int     = ncvslideio::GArray<int>;
using GArray_double  = ncvslideio::GArray<double>;
using GArray_float   = ncvslideio::GArray<double>;
using GArray_string  = ncvslideio::GArray<std::string>;
using GArray_Point2i = ncvslideio::GArray<ncvslideio::Point>;
using GArray_Point2f = ncvslideio::GArray<ncvslideio::Point2f>;
using GArray_Point3f = ncvslideio::GArray<ncvslideio::Point3f>;
using GArray_Size    = ncvslideio::GArray<ncvslideio::Size>;
using GArray_Rect    = ncvslideio::GArray<ncvslideio::Rect>;
using GArray_Scalar  = ncvslideio::GArray<ncvslideio::Scalar>;
using GArray_Mat     = ncvslideio::GArray<ncvslideio::Mat>;
using GArray_GMat    = ncvslideio::GArray<ncvslideio::GMat>;
using GArray_Prim    = ncvslideio::GArray<ncvslideio::gapi::wip::draw::Prim>;

// FIXME: Python wrapper generate code without namespace std,
// so it cause error: "string wasn't declared"
// WA: Create using
using std::string;

namespace ncvslideio
{
namespace detail
{

class PyObjectHolder
{
public:
    PyObjectHolder(PyObject* o, bool owner = true);
    PyObject* get() const;

private:
    class Impl;
    std::shared_ptr<Impl> m_impl;
};

} // namespace detail
} // namespace ncvslideio

class ncvslideio::detail::PyObjectHolder::Impl
{
public:
    Impl(PyObject* object, bool owner);
    PyObject* get() const;
    ~Impl();

private:
    PyObject* m_object;
};

ncvslideio::detail::PyObjectHolder::Impl::Impl(PyObject* object, bool owner)
    : m_object(object)
{
    // NB: Become an owner of that PyObject.
    // Need to store this and get access
    // after the caller which provide the object is out of range.
    if (owner)
    {
        // NB: Impossible take ownership if object is NULL.
        GAPI_Assert(object);
        Py_INCREF(m_object);
    }
}

ncvslideio::detail::PyObjectHolder::Impl::~Impl()
{
    // NB: If NULL was set, don't decrease counter.
    if (m_object)
    {
        Py_DECREF(m_object);
    }
}

PyObject* ncvslideio::detail::PyObjectHolder::Impl::get() const
{
    return m_object;
}

ncvslideio::detail::PyObjectHolder::PyObjectHolder(PyObject* object, bool owner)
        : m_impl(new ncvslideio::detail::PyObjectHolder::Impl{object, owner})
{
}

PyObject* ncvslideio::detail::PyObjectHolder::get() const
{
    return m_impl->get();
}

template<>
PyObject* pyopencv_from(const ncvslideio::detail::PyObjectHolder& v)
{
    PyObject* o = ncvslideio::util::any_cast<ncvslideio::detail::PyObjectHolder>(v).get();
    Py_INCREF(o);
    return o;
}

// #FIXME: Is it possible to implement pyopencv_from/pyopencv_to for generic
// ncvslideio::variant<Types...> ?
template <>
PyObject* pyopencv_from(const ncvslideio::gapi::wip::draw::Prim& prim)
{
    switch (prim.index())
    {
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Rect>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Rect>(prim));
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Text>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Text>(prim));
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Circle>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Circle>(prim));
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Line>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Line>(prim));
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Poly>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Poly>(prim));
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Mosaic>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Mosaic>(prim));
        case ncvslideio::gapi::wip::draw::Prim::index_of<ncvslideio::gapi::wip::draw::Image>():
            return pyopencv_from(ncvslideio::util::get<ncvslideio::gapi::wip::draw::Image>(prim));
    }

    util::throw_error(std::logic_error("Unsupported draw primitive type"));
}

template <>
PyObject* pyopencv_from(const ncvslideio::gapi::wip::draw::Prims& value)
{
    return pyopencv_from_generic_vec(value);
}

template<>
bool pyopencv_to(PyObject* obj, ncvslideio::gapi::wip::draw::Prim& value, const ArgInfo&)
{
#define TRY_EXTRACT(Prim)                                                                                  \
    if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(pyopencv_gapi_wip_draw_##Prim##_TypePtr))) \
    {                                                                                                      \
        value = reinterpret_cast<pyopencv_gapi_wip_draw_##Prim##_t*>(obj)->v;                              \
        return true;                                                                                       \
    }                                                                                                      \

    TRY_EXTRACT(Rect)
    TRY_EXTRACT(Text)
    TRY_EXTRACT(Circle)
    TRY_EXTRACT(Line)
    TRY_EXTRACT(Mosaic)
    TRY_EXTRACT(Image)
    TRY_EXTRACT(Poly)
#undef TRY_EXTRACT

    failmsg("Unsupported primitive type");
    return false;
}

template <>
bool pyopencv_to(PyObject* obj, ncvslideio::gapi::wip::draw::Prims& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template <>
bool pyopencv_to(PyObject* obj, ncvslideio::GMetaArg& value, const ArgInfo&)
{
#define TRY_EXTRACT(Meta)                                                    \
    if (PyObject_TypeCheck(obj,                                              \
                reinterpret_cast<PyTypeObject*>(pyopencv_##Meta##_TypePtr))) \
    {                                                                        \
        value = reinterpret_cast<pyopencv_##Meta##_t*>(obj)->v;              \
        return true;                                                         \
    }                                                                        \

    TRY_EXTRACT(GMatDesc)
    TRY_EXTRACT(GScalarDesc)
    TRY_EXTRACT(GArrayDesc)
    TRY_EXTRACT(GOpaqueDesc)
#undef TRY_EXTRACT

    failmsg("Unsupported ncvslideio::GMetaArg type");
    return false;
}

template <>
bool pyopencv_to(PyObject* obj, ncvslideio::GMetaArgs& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}


template<>
PyObject* pyopencv_from(const ncvslideio::GArg& value)
{
    GAPI_Assert(value.kind != ncvslideio::detail::ArgKind::GOBJREF);
#define HANDLE_CASE(T, O) case ncvslideio::detail::OpaqueKind::CV_##T:  \
    {                                                           \
        return pyopencv_from(value.get<O>());                   \
    }

#define UNSUPPORTED(T) case ncvslideio::detail::OpaqueKind::CV_##T: break
    switch (value.opaque_kind)
    {
        HANDLE_CASE(BOOL,      bool);
        HANDLE_CASE(INT,       int);
        HANDLE_CASE(INT64,     int64_t);
        HANDLE_CASE(UINT64,    uint64_t);
        HANDLE_CASE(DOUBLE,    double);
        HANDLE_CASE(FLOAT,     float);
        HANDLE_CASE(STRING,    std::string);
        HANDLE_CASE(POINT,     ncvslideio::Point);
        HANDLE_CASE(POINT2F,   ncvslideio::Point2f);
        HANDLE_CASE(POINT3F,   ncvslideio::Point3f);
        HANDLE_CASE(SIZE,      ncvslideio::Size);
        HANDLE_CASE(RECT,      ncvslideio::Rect);
        HANDLE_CASE(SCALAR,    ncvslideio::Scalar);
        HANDLE_CASE(MAT,       ncvslideio::Mat);
        HANDLE_CASE(UNKNOWN,   ncvslideio::detail::PyObjectHolder);
        HANDLE_CASE(DRAW_PRIM, ncvslideio::gapi::wip::draw::Prim);
#undef HANDLE_CASE
#undef UNSUPPORTED
    }
    util::throw_error(std::logic_error("Unsupported kernel input type"));
}

template<>
bool pyopencv_to(PyObject* obj, ncvslideio::GArg& value, const ArgInfo& info)
{
    value = ncvslideio::GArg(ncvslideio::detail::PyObjectHolder(obj));
    return true;
}

template <>
bool pyopencv_to(PyObject* obj, std::vector<ncvslideio::gapi::GNetParam>& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template <>
PyObject* pyopencv_from(const std::vector<ncvslideio::gapi::GNetParam>& value)
{
    return pyopencv_from_generic_vec(value);
}

template <>
bool pyopencv_to(PyObject* obj, std::vector<GCompileArg>& value, const ArgInfo& info)
{
    return pyopencv_to_generic_vec(obj, value, info);
}

template <>
PyObject* pyopencv_from(const std::vector<GCompileArg>& value)
{
    return pyopencv_from_generic_vec(value);
}

template<>
PyObject* pyopencv_from(const ncvslideio::detail::OpaqueRef& o)
{
    switch (o.getKind())
    {
        case ncvslideio::detail::OpaqueKind::CV_BOOL      : return pyopencv_from(o.rref<bool>());
        case ncvslideio::detail::OpaqueKind::CV_INT       : return pyopencv_from(o.rref<int>());
        case ncvslideio::detail::OpaqueKind::CV_INT64     : return pyopencv_from(o.rref<int64_t>());
        case ncvslideio::detail::OpaqueKind::CV_UINT64    : return pyopencv_from(o.rref<uint64_t>());
        case ncvslideio::detail::OpaqueKind::CV_DOUBLE    : return pyopencv_from(o.rref<double>());
        case ncvslideio::detail::OpaqueKind::CV_FLOAT     : return pyopencv_from(o.rref<float>());
        case ncvslideio::detail::OpaqueKind::CV_STRING    : return pyopencv_from(o.rref<std::string>());
        case ncvslideio::detail::OpaqueKind::CV_POINT     : return pyopencv_from(o.rref<ncvslideio::Point>());
        case ncvslideio::detail::OpaqueKind::CV_POINT2F   : return pyopencv_from(o.rref<ncvslideio::Point2f>());
        case ncvslideio::detail::OpaqueKind::CV_POINT3F   : return pyopencv_from(o.rref<ncvslideio::Point3f>());
        case ncvslideio::detail::OpaqueKind::CV_SIZE      : return pyopencv_from(o.rref<ncvslideio::Size>());
        case ncvslideio::detail::OpaqueKind::CV_RECT      : return pyopencv_from(o.rref<ncvslideio::Rect>());
        case ncvslideio::detail::OpaqueKind::CV_UNKNOWN   : return pyopencv_from(o.rref<ncvslideio::GArg>());
        case ncvslideio::detail::OpaqueKind::CV_DRAW_PRIM : return pyopencv_from(o.rref<ncvslideio::gapi::wip::draw::Prim>());
        case ncvslideio::detail::OpaqueKind::CV_SCALAR    : break;
        case ncvslideio::detail::OpaqueKind::CV_MAT       : break;
    }

    PyErr_SetString(PyExc_TypeError, "Unsupported GOpaque type");
    return NULL;
}

template <>
PyObject* pyopencv_from(const ncvslideio::detail::VectorRef& v)
{
    switch (v.getKind())
    {
        case ncvslideio::detail::OpaqueKind::CV_BOOL      : return pyopencv_from_generic_vec(v.rref<bool>());
        case ncvslideio::detail::OpaqueKind::CV_INT       : return pyopencv_from_generic_vec(v.rref<int>());
        case ncvslideio::detail::OpaqueKind::CV_INT64     : return pyopencv_from_generic_vec(v.rref<int64_t>());
        case ncvslideio::detail::OpaqueKind::CV_UINT64    : return pyopencv_from_generic_vec(v.rref<uint64_t>());
        case ncvslideio::detail::OpaqueKind::CV_DOUBLE    : return pyopencv_from_generic_vec(v.rref<double>());
        case ncvslideio::detail::OpaqueKind::CV_FLOAT     : return pyopencv_from_generic_vec(v.rref<float>());
        case ncvslideio::detail::OpaqueKind::CV_STRING    : return pyopencv_from_generic_vec(v.rref<std::string>());
        case ncvslideio::detail::OpaqueKind::CV_POINT     : return pyopencv_from_generic_vec(v.rref<ncvslideio::Point>());
        case ncvslideio::detail::OpaqueKind::CV_POINT2F   : return pyopencv_from_generic_vec(v.rref<ncvslideio::Point2f>());
        case ncvslideio::detail::OpaqueKind::CV_POINT3F   : return pyopencv_from_generic_vec(v.rref<ncvslideio::Point3f>());
        case ncvslideio::detail::OpaqueKind::CV_SIZE      : return pyopencv_from_generic_vec(v.rref<ncvslideio::Size>());
        case ncvslideio::detail::OpaqueKind::CV_RECT      : return pyopencv_from_generic_vec(v.rref<ncvslideio::Rect>());
        case ncvslideio::detail::OpaqueKind::CV_SCALAR    : return pyopencv_from_generic_vec(v.rref<ncvslideio::Scalar>());
        case ncvslideio::detail::OpaqueKind::CV_MAT       : return pyopencv_from_generic_vec(v.rref<ncvslideio::Mat>());
        case ncvslideio::detail::OpaqueKind::CV_UNKNOWN   : return pyopencv_from_generic_vec(v.rref<ncvslideio::GArg>());
        case ncvslideio::detail::OpaqueKind::CV_DRAW_PRIM : return pyopencv_from_generic_vec(v.rref<ncvslideio::gapi::wip::draw::Prim>());
    }

    PyErr_SetString(PyExc_TypeError, "Unsupported GArray type");
    return NULL;
}

template <>
PyObject* pyopencv_from(const GRunArg& v)
{
    switch (v.index())
    {
        case GRunArg::index_of<ncvslideio::Mat>():
            return pyopencv_from(util::get<ncvslideio::Mat>(v));

        case GRunArg::index_of<ncvslideio::Scalar>():
            return pyopencv_from(util::get<ncvslideio::Scalar>(v));

        case GRunArg::index_of<ncvslideio::detail::VectorRef>():
            return pyopencv_from(util::get<ncvslideio::detail::VectorRef>(v));

        case GRunArg::index_of<ncvslideio::detail::OpaqueRef>():
            return pyopencv_from(util::get<ncvslideio::detail::OpaqueRef>(v));
    }

    PyErr_SetString(PyExc_TypeError, "Failed to unpack GRunArgs. Index of variant is unknown");
    return NULL;
}

template <typename T>
PyObject* pyopencv_from(const ncvslideio::optional<T>& opt)
{
    if (!opt.has_value())
    {
        Py_RETURN_NONE;
    }
    return pyopencv_from(*opt);
}

template <>
PyObject* pyopencv_from(const GOptRunArg& v)
{
    switch (v.index())
    {
        case GOptRunArg::index_of<ncvslideio::optional<ncvslideio::Mat>>():
            return pyopencv_from(util::get<ncvslideio::optional<ncvslideio::Mat>>(v));

        case GOptRunArg::index_of<ncvslideio::optional<ncvslideio::Scalar>>():
            return pyopencv_from(util::get<ncvslideio::optional<ncvslideio::Scalar>>(v));

        case GOptRunArg::index_of<optional<ncvslideio::detail::VectorRef>>():
            return pyopencv_from(util::get<optional<ncvslideio::detail::VectorRef>>(v));

        case GOptRunArg::index_of<optional<ncvslideio::detail::OpaqueRef>>():
            return pyopencv_from(util::get<optional<ncvslideio::detail::OpaqueRef>>(v));
    }

    PyErr_SetString(PyExc_TypeError, "Failed to unpack GOptRunArg. Index of variant is unknown");
    return NULL;
}

template<>
PyObject* pyopencv_from(const GRunArgs& value)
{
     return value.size() == 1 ? pyopencv_from(value[0]) : pyopencv_from_generic_vec(value);
}

template<>
PyObject* pyopencv_from(const GOptRunArgs& value)
{
    return value.size() == 1 ? pyopencv_from(value[0]) : pyopencv_from_generic_vec(value);
}

// FIXME: ncvslideio::variant should be wrapped once for all types.
template <>
PyObject* pyopencv_from(const ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>& v)
{
    using RunArgs = ncvslideio::util::variant<ncvslideio::GRunArgs, ncvslideio::GOptRunArgs>;
    switch (v.index())
    {
        case RunArgs::index_of<ncvslideio::GRunArgs>():
            return pyopencv_from(util::get<ncvslideio::GRunArgs>(v));
        case RunArgs::index_of<ncvslideio::GOptRunArgs>():
            return pyopencv_from(util::get<ncvslideio::GOptRunArgs>(v));
    }

    PyErr_SetString(PyExc_TypeError, "Failed to recognize kind of RunArgs. Index of variant is unknown");
    return NULL;
}

template <typename T>
void pyopencv_to_with_check(PyObject* from, T& to, const std::string& msg = "")
{
    if (!pyopencv_to(from, to, ArgInfo("", false)))
    {
        ncvslideio::util::throw_error(std::logic_error(msg));
    }
}

template <typename T>
void pyopencv_to_generic_vec_with_check(PyObject* from,
                                        std::vector<T>& to,
                                        const std::string& msg = "")
{
    if (!pyopencv_to_generic_vec(from, to, ArgInfo("", false)))
    {
        ncvslideio::util::throw_error(std::logic_error(msg));
    }
}

template <typename T>
static T extract_proto_args(PyObject* py_args)
{
    using namespace ncvslideio;

    GProtoArgs args;
    Py_ssize_t size = PyList_Size(py_args);
    args.reserve(size);
    for (int i = 0; i < size; ++i)
    {
        PyObject* item = PyList_GetItem(py_args, i);
        if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GScalar_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GScalar_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GMat_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GMat_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GOpaqueT_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GOpaqueT_t*>(item)->v.strip());
        }
        else if (PyObject_TypeCheck(item, reinterpret_cast<PyTypeObject*>(pyopencv_GArrayT_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GArrayT_t*>(item)->v.strip());
        }
        else
        {
            util::throw_error(std::logic_error("Unsupported type for GProtoArgs"));
        }
    }

    return T(std::move(args));
}

static ncvslideio::detail::OpaqueRef extract_opaque_ref(PyObject* from, ncvslideio::detail::OpaqueKind kind)
{
#define HANDLE_CASE(T, O) case ncvslideio::detail::OpaqueKind::CV_##T:  \
{                                                               \
    O obj{};                                                    \
    pyopencv_to_with_check(from, obj, "Failed to obtain " # O); \
    return ncvslideio::detail::OpaqueRef{std::move(obj)};               \
}
#define UNSUPPORTED(T) case ncvslideio::detail::OpaqueKind::CV_##T: break
    switch (kind)
    {
        HANDLE_CASE(BOOL,    bool);
        HANDLE_CASE(INT,     int);
        HANDLE_CASE(INT64,   int64_t);
        HANDLE_CASE(UINT64,  uint64_t);
        HANDLE_CASE(DOUBLE,  double);
        HANDLE_CASE(FLOAT,   float);
        HANDLE_CASE(STRING,  std::string);
        HANDLE_CASE(POINT,   ncvslideio::Point);
        HANDLE_CASE(POINT2F, ncvslideio::Point2f);
        HANDLE_CASE(POINT3F, ncvslideio::Point3f);
        HANDLE_CASE(SIZE,    ncvslideio::Size);
        HANDLE_CASE(RECT,    ncvslideio::Rect);
        HANDLE_CASE(UNKNOWN, ncvslideio::GArg);
        UNSUPPORTED(SCALAR);
        UNSUPPORTED(MAT);
        UNSUPPORTED(DRAW_PRIM);
#undef HANDLE_CASE
#undef UNSUPPORTED
    }
    util::throw_error(std::logic_error("Unsupported type for GOpaqueT"));
}

static ncvslideio::detail::VectorRef extract_vector_ref(PyObject* from, ncvslideio::detail::OpaqueKind kind)
{
#define HANDLE_CASE(T, O) case ncvslideio::detail::OpaqueKind::CV_##T:                        \
{                                                                                     \
    std::vector<O> obj;                                                               \
    pyopencv_to_generic_vec_with_check(from, obj, "Failed to obtain vector of " # O); \
    return ncvslideio::detail::VectorRef{std::move(obj)};                                     \
}
#define UNSUPPORTED(T) case ncvslideio::detail::OpaqueKind::CV_##T: break
    switch (kind)
    {
        HANDLE_CASE(BOOL,      bool);
        HANDLE_CASE(INT,       int);
        HANDLE_CASE(INT64,     int64_t);
        HANDLE_CASE(UINT64,    uint64_t);
        HANDLE_CASE(DOUBLE,    double);
        HANDLE_CASE(FLOAT,     float);
        HANDLE_CASE(STRING,    std::string);
        HANDLE_CASE(POINT,     ncvslideio::Point);
        HANDLE_CASE(POINT2F,   ncvslideio::Point2f);
        HANDLE_CASE(POINT3F,   ncvslideio::Point3f);
        HANDLE_CASE(SIZE,      ncvslideio::Size);
        HANDLE_CASE(RECT,      ncvslideio::Rect);
        HANDLE_CASE(SCALAR,    ncvslideio::Scalar);
        HANDLE_CASE(MAT,       ncvslideio::Mat);
        HANDLE_CASE(UNKNOWN,   ncvslideio::GArg);
        HANDLE_CASE(DRAW_PRIM, ncvslideio::gapi::wip::draw::Prim);
#undef HANDLE_CASE
#undef UNSUPPORTED
    }
    util::throw_error(std::logic_error("Unsupported type for GArrayT"));
}

static ncvslideio::GRunArg extract_run_arg(const ncvslideio::GTypeInfo& info, PyObject* item)
{
    switch (info.shape)
    {
        case ncvslideio::GShape::GMAT:
        {
            // NB: In case streaming it can be IStreamSource or ncvslideio::Mat
            if (PyObject_TypeCheck(item,
                        reinterpret_cast<PyTypeObject*>(pyopencv_gapi_wip_IStreamSource_TypePtr)))
            {
                ncvslideio::gapi::wip::IStreamSource::Ptr source =
                    reinterpret_cast<pyopencv_gapi_wip_IStreamSource_t*>(item)->v;
                return source;
            }
            ncvslideio::Mat obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain ncvslideio::Mat");
            return obj;
        }
        case ncvslideio::GShape::GSCALAR:
        {
            ncvslideio::Scalar obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain ncvslideio::Scalar");
            return obj;
        }
        case ncvslideio::GShape::GOPAQUE:
        {
            return extract_opaque_ref(item, info.kind);
        }
        case ncvslideio::GShape::GARRAY:
        {
            return extract_vector_ref(item, info.kind);
        }
        case ncvslideio::GShape::GFRAME:
        {
            // NB: Isn't supported yet.
            break;
        }
    }

    util::throw_error(std::logic_error("Unsupported output shape"));
}

static ncvslideio::GRunArgs extract_run_args(const ncvslideio::GTypesInfo& info, PyObject* py_args)
{
    GAPI_Assert(PyList_Check(py_args));

    ncvslideio::GRunArgs args;
    Py_ssize_t list_size = PyList_Size(py_args);
    args.reserve(list_size);

    for (int i = 0; i < list_size; ++i)
    {
        args.push_back(extract_run_arg(info[i], PyList_GetItem(py_args, i)));
    }

    return args;
}

static ncvslideio::GMetaArg extract_meta_arg(const ncvslideio::GTypeInfo& info, PyObject* item)
{
    switch (info.shape)
    {
        case ncvslideio::GShape::GMAT:
        {
            ncvslideio::Mat obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain ncvslideio::Mat");
            return ncvslideio::GMetaArg{ncvslideio::descr_of(obj)};
        }
        case ncvslideio::GShape::GSCALAR:
        {
            ncvslideio::Scalar obj;
            pyopencv_to_with_check(item, obj, "Failed to obtain ncvslideio::Scalar");
            return ncvslideio::GMetaArg{ncvslideio::descr_of(obj)};
        }
        case ncvslideio::GShape::GARRAY:
        {
            return ncvslideio::GMetaArg{ncvslideio::empty_array_desc()};
        }
        case ncvslideio::GShape::GOPAQUE:
        {
            return ncvslideio::GMetaArg{ncvslideio::empty_gopaque_desc()};
        }
        case ncvslideio::GShape::GFRAME:
        {
            // NB: Isn't supported yet.
            break;
        }
    }
    util::throw_error(std::logic_error("Unsupported output shape"));
}

static ncvslideio::GMetaArgs extract_meta_args(const ncvslideio::GTypesInfo& info, PyObject* py_args)
{
    GAPI_Assert(PyList_Check(py_args));

    ncvslideio::GMetaArgs metas;
    Py_ssize_t list_size = PyList_Size(py_args);
    metas.reserve(list_size);

    for (int i = 0; i < list_size; ++i)
    {
        metas.push_back(extract_meta_arg(info[i], PyList_GetItem(py_args, i)));
    }

    return metas;
}

static ncvslideio::GRunArgs run_py_kernel(ncvslideio::detail::PyObjectHolder kernel,
                                  const ncvslideio::gapi::python::GPythonContext &ctx)
{
    const auto& ins      = ctx.ins;
    const auto& in_metas = ctx.in_metas;
    const auto& out_info = ctx.out_info;

    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    ncvslideio::GRunArgs outs;
    try
    {
        // NB: Doesn't increase reference counter (false),
        // because PyObject already have ownership.
        // In case exception decrement reference counter.
        ncvslideio::detail::PyObjectHolder args(
                PyTuple_New(ctx.m_state.has_value() ? ins.size() + 1 : ins.size()), false);
        for (size_t i = 0; i < ins.size(); ++i)
        {
            // NB: If meta is monostate then object isn't associated with G-TYPE.
            if (ncvslideio::util::holds_alternative<ncvslideio::util::monostate>(in_metas[i]))
            {
                PyTuple_SetItem(args.get(), i, pyopencv_from(ins[i]));
                continue;
            }

            switch (in_metas[i].index())
            {
                case ncvslideio::GMetaArg::index_of<ncvslideio::GMatDesc>():
                    PyTuple_SetItem(args.get(), i, pyopencv_from(ins[i].get<ncvslideio::Mat>()));
                    break;
                case ncvslideio::GMetaArg::index_of<ncvslideio::GScalarDesc>():
                    PyTuple_SetItem(args.get(), i, pyopencv_from(ins[i].get<ncvslideio::Scalar>()));
                    break;
                case ncvslideio::GMetaArg::index_of<ncvslideio::GOpaqueDesc>():
                    PyTuple_SetItem(args.get(), i, pyopencv_from(ins[i].get<ncvslideio::detail::OpaqueRef>()));
                    break;
                case ncvslideio::GMetaArg::index_of<ncvslideio::GArrayDesc>():
                    PyTuple_SetItem(args.get(), i, pyopencv_from(ins[i].get<ncvslideio::detail::VectorRef>()));
                    break;
                case ncvslideio::GMetaArg::index_of<ncvslideio::GFrameDesc>():
                    util::throw_error(std::logic_error("GFrame isn't supported for custom operation"));
                    break;
            }
        }

        if (ctx.m_state.has_value())
        {
            PyTuple_SetItem(args.get(), ins.size(), pyopencv_from(ctx.m_state.value()));
        }

        // NB: Doesn't increase reference counter (false).
        // In case PyObject_CallObject return NULL, do nothing in destructor.
        ncvslideio::detail::PyObjectHolder result(
                PyObject_CallObject(kernel.get(), args.get()), false);

        if (PyErr_Occurred())
        {
            PyErr_PrintEx(0);
            PyErr_Clear();
            throw std::logic_error("Python kernel failed with error!");
        }
        // NB: In fact it's impossible situation, because errors were handled above.
        GAPI_Assert(result.get() && "Python kernel returned NULL!");

        if (out_info.size() == 1)
        {
            outs = ncvslideio::GRunArgs{extract_run_arg(out_info[0], result.get())};
        }
        else if (out_info.size() > 1)
        {
            GAPI_Assert(PyTuple_Check(result.get()));

            Py_ssize_t tuple_size = PyTuple_Size(result.get());
            outs.reserve(tuple_size);

            for (int i = 0; i < tuple_size; ++i)
            {
                outs.push_back(extract_run_arg(out_info[i], PyTuple_GetItem(result.get(), i)));
            }
        }
        else
        {
            // Seems to be impossible case.
            GAPI_Error("InternalError");
        }
    }
    catch (...)
    {
        PyGILState_Release(gstate);
        throw;
    }
    PyGILState_Release(gstate);

    return outs;
}

static void unpackMetasToTuple(const ncvslideio::GMetaArgs&        meta,
                               const ncvslideio::GArgs&            gargs,
                               ncvslideio::detail::PyObjectHolder& tuple)
{
    size_t idx = 0;
    for (auto&& m : meta)
    {
        switch (m.index())
        {
            case ncvslideio::GMetaArg::index_of<ncvslideio::GMatDesc>():
                PyTuple_SetItem(tuple.get(), idx, pyopencv_from(ncvslideio::util::get<ncvslideio::GMatDesc>(m)));
                break;
            case ncvslideio::GMetaArg::index_of<ncvslideio::GScalarDesc>():
                PyTuple_SetItem(tuple.get(), idx,
                        pyopencv_from(ncvslideio::util::get<ncvslideio::GScalarDesc>(m)));
                break;
            case ncvslideio::GMetaArg::index_of<ncvslideio::GArrayDesc>():
                PyTuple_SetItem(tuple.get(), idx,
                        pyopencv_from(ncvslideio::util::get<ncvslideio::GArrayDesc>(m)));
                break;
            case ncvslideio::GMetaArg::index_of<ncvslideio::GOpaqueDesc>():
                PyTuple_SetItem(tuple.get(), idx,
                        pyopencv_from(ncvslideio::util::get<ncvslideio::GOpaqueDesc>(m)));
                break;
            case ncvslideio::GMetaArg::index_of<ncvslideio::util::monostate>():
                PyTuple_SetItem(tuple.get(), idx, pyopencv_from(gargs[idx]));
                break;
            case ncvslideio::GMetaArg::index_of<ncvslideio::GFrameDesc>():
                util::throw_error(
                        std::logic_error("GFrame isn't supported for custom operation"));
                break;
        }
        ++idx;
    }
}

static ncvslideio::GArg run_py_setup(ncvslideio::detail::PyObjectHolder setup,
                             const ncvslideio::GMetaArgs        &meta,
                             const ncvslideio::GArgs            &gargs)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    ncvslideio::GArg state;
    try
    {
        // NB: Doesn't increase reference counter (false),
        // because PyObject already have ownership.
        // In case exception decrement reference counter.
        ncvslideio::detail::PyObjectHolder args(PyTuple_New(meta.size()), false);
        unpackMetasToTuple(meta, gargs, args);

        PyObject *py_kernel_state = PyObject_CallObject(setup.get(), args.get());
        if (PyErr_Occurred())
        {
            PyErr_PrintEx(0);
            PyErr_Clear();
            throw std::logic_error("Python kernel setup failed with error!");
        }
        // NB: In fact it's impossible situation, because errors were handled above.
        GAPI_Assert(py_kernel_state && "Python kernel setup returned NULL!");

        if (!pyopencv_to(py_kernel_state, state, ArgInfo("arg", false)))
        {
            util::throw_error(std::logic_error("Failed to convert python state"));
        }
    }
    catch (...)
    {
        PyGILState_Release(gstate);
        throw;
    }
    PyGILState_Release(gstate);
    return state;
}

static GMetaArg get_meta_arg(PyObject* obj)
{
    ncvslideio::GMetaArg arg;
    if (!pyopencv_to(obj, arg, ArgInfo("arg", false)))
    {
        util::throw_error(std::logic_error("Unsupported output meta type"));
    }
    return arg;
}

static ncvslideio::GMetaArgs get_meta_args(PyObject* tuple)
{
    size_t size = PyTuple_Size(tuple);

    ncvslideio::GMetaArgs metas;
    metas.reserve(size);
    for (size_t i = 0; i < size; ++i)
    {
        metas.push_back(get_meta_arg(PyTuple_GetItem(tuple, i)));
    }

    return metas;
}

static GMetaArgs run_py_meta(ncvslideio::detail::PyObjectHolder out_meta,
                             const ncvslideio::GMetaArgs        &meta,
                             const ncvslideio::GArgs            &gargs)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    ncvslideio::GMetaArgs out_metas;
    try
    {
        // NB: Doesn't increase reference counter (false),
        // because PyObject already have ownership.
        // In case exception decrement reference counter.
        ncvslideio::detail::PyObjectHolder args(PyTuple_New(meta.size()), false);
        unpackMetasToTuple(meta, gargs, args);
        // NB: Doesn't increase reference counter (false).
        // In case PyObject_CallObject return NULL, do nothing in destructor.
        ncvslideio::detail::PyObjectHolder result(
                PyObject_CallObject(out_meta.get(), args.get()), false);

        if (PyErr_Occurred())
        {
            PyErr_PrintEx(0);
            PyErr_Clear();
            throw std::logic_error("Python outMeta failed with error!");
        }
        // NB: In fact it's impossible situation, because errors were handled above.
        GAPI_Assert(result.get() && "Python outMeta returned NULL!");

        out_metas = PyTuple_Check(result.get()) ? get_meta_args(result.get())
                                                : ncvslideio::GMetaArgs{get_meta_arg(result.get())};
    }
    catch (...)
    {
        PyGILState_Release(gstate);
        throw;
    }
    PyGILState_Release(gstate);

    return out_metas;
}

static PyObject* pyopencv_cv_gapi_kernels(PyObject* , PyObject* py_args, PyObject*)
{
    using namespace ncvslideio;
    GKernelPackage pkg;
    Py_ssize_t size = PyTuple_Size(py_args);

    for (int i = 0; i < size; ++i)
    {
        PyObject* user_kernel = PyTuple_GetItem(py_args, i);

        PyObject* id_obj = PyObject_GetAttrString(user_kernel, "id");
        if (!id_obj)
        {
            PyErr_SetString(PyExc_TypeError,
                    "Python kernel should contain id, please use ncvslideio.gapi.kernel to define kernel");
            return NULL;
        }

        PyObject* out_meta = PyObject_GetAttrString(user_kernel, "outMeta");
        if (!out_meta)
        {
            PyErr_SetString(PyExc_TypeError,
                    "Python kernel should contain outMeta, please use ncvslideio.gapi.kernel to define kernel");
            return NULL;
        }

        PyObject* run  = PyObject_GetAttrString(user_kernel, "run");
        if (!run)
        {
            PyErr_SetString(PyExc_TypeError,
                    "Python kernel should contain run, please use ncvslideio.gapi.kernel to define kernel");
            return NULL;
        }
        PyObject* setup = nullptr;
        if (PyObject_HasAttrString(user_kernel, "setup")) {
            setup  = PyObject_GetAttrString(user_kernel, "setup");
        }

        std::string id;
        if (!pyopencv_to(id_obj, id, ArgInfo("id", false)))
        {
            PyErr_SetString(PyExc_TypeError, "Failed to obtain string");
            return NULL;
        }

        using namespace std::placeholders;

        if (setup)
        {
            gapi::python::GPythonFunctor f(
                id.c_str(), std::bind(run_py_meta, ncvslideio::detail::PyObjectHolder{out_meta}, _1, _2),
                std::bind(run_py_kernel, ncvslideio::detail::PyObjectHolder{run}, _1),
                std::bind(run_py_setup, ncvslideio::detail::PyObjectHolder{setup}, _1, _2));
            pkg.include(f);
        }
        else
        {
            gapi::python::GPythonFunctor f(
                id.c_str(), std::bind(run_py_meta, ncvslideio::detail::PyObjectHolder{out_meta}, _1, _2),
                std::bind(run_py_kernel, ncvslideio::detail::PyObjectHolder{run}, _1));
            pkg.include(f);
        }
    }
    return pyopencv_from(pkg);
}

static PyObject* pyopencv_cv_gapi_op(PyObject* , PyObject* py_args, PyObject*)
{
    using namespace ncvslideio;
    Py_ssize_t size = PyTuple_Size(py_args);
    std::string id;
    if (!pyopencv_to(PyTuple_GetItem(py_args, 0), id, ArgInfo("id", false)))
    {
        PyErr_SetString(PyExc_TypeError, "Failed to obtain: operation id must be a string");
        return NULL;
    }
    PyObject* outMeta = PyTuple_GetItem(py_args, 1);

    ncvslideio::GArgs args;
    for (int i = 2; i < size; i++)
    {
        PyObject* item = PyTuple_GetItem(py_args, i);
        if (PyObject_TypeCheck(item,
                    reinterpret_cast<PyTypeObject*>(pyopencv_GMat_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GMat_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item,
                           reinterpret_cast<PyTypeObject*>(pyopencv_GScalar_TypePtr)))
        {
            args.emplace_back(reinterpret_cast<pyopencv_GScalar_t*>(item)->v);
        }
        else if (PyObject_TypeCheck(item,
                           reinterpret_cast<PyTypeObject*>(pyopencv_GOpaqueT_TypePtr)))
        {
            auto&& arg = reinterpret_cast<pyopencv_GOpaqueT_t*>(item)->v.arg();
#define HC(T, K) case ncvslideio::GOpaqueT::Storage:: index_of<ncvslideio::GOpaque<T>>(): \
            args.emplace_back(ncvslideio::util::get<ncvslideio::GOpaque<T>>(arg));        \
            break;                                                        \

            SWITCH(arg.index(), GOPAQUE_TYPE_LIST_G, HC)
#undef HC
        }
        else if (PyObject_TypeCheck(item,
                           reinterpret_cast<PyTypeObject*>(pyopencv_GArrayT_TypePtr)))
        {
            auto&& arg = reinterpret_cast<pyopencv_GArrayT_t*>(item)->v.arg();
#define HC(T, K) case ncvslideio::GArrayT::Storage:: index_of<ncvslideio::GArray<T>>(): \
            args.emplace_back(ncvslideio::util::get<ncvslideio::GArray<T>>(arg));       \
            break;                                                      \

            SWITCH(arg.index(), GARRAY_TYPE_LIST_G, HC)
#undef HC
        }
        else
        {
            args.emplace_back(ncvslideio::GArg(ncvslideio::detail::PyObjectHolder{item}));
        }
    }

    ncvslideio::GKernel::M outMetaWrapper = std::bind(run_py_meta,
                                              ncvslideio::detail::PyObjectHolder{outMeta},
                                              std::placeholders::_1,
                                              std::placeholders::_2);
    return pyopencv_from(ncvslideio::gapi::wip::op(id, outMetaWrapper, std::move(args)));
}

template<>
bool pyopencv_to(PyObject* obj, ncvslideio::detail::ExtractArgsCallback& value, const ArgInfo&)
{
    ncvslideio::detail::PyObjectHolder holder{obj};
    value = ncvslideio::detail::ExtractArgsCallback{[=](const ncvslideio::GTypesInfo& info)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        ncvslideio::GRunArgs args;
        try
        {
            args = extract_run_args(info, holder.get());
        }
        catch (...)
        {
            PyGILState_Release(gstate);
            throw;
        }
        PyGILState_Release(gstate);
        return args;
    }};
    return true;
}

template<>
bool pyopencv_to(PyObject* obj, ncvslideio::detail::ExtractMetaCallback& value, const ArgInfo&)
{
    ncvslideio::detail::PyObjectHolder holder{obj};
    value = ncvslideio::detail::ExtractMetaCallback{[=](const ncvslideio::GTypesInfo& info)
    {
        PyGILState_STATE gstate;
        gstate = PyGILState_Ensure();

        ncvslideio::GMetaArgs args;
        try
        {
            args = extract_meta_args(info, holder.get());
        }
        catch (...)
        {
            PyGILState_Release(gstate);
            throw;
        }
        PyGILState_Release(gstate);
        return args;
    }};
    return true;
}

template<typename T>
struct PyOpenCV_Converter<ncvslideio::GArray<T>>
{
    static PyObject* from(const ncvslideio::GArray<T>& p)
    {
        return pyopencv_from(ncvslideio::GArrayT(p));
    }
    static bool to(PyObject *obj, ncvslideio::GArray<T>& value, const ArgInfo& info)
    {
        if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(pyopencv_GArrayT_TypePtr)))
        {
            auto& array = reinterpret_cast<pyopencv_GArrayT_t*>(obj)->v;
            try
            {
                value = ncvslideio::util::get<ncvslideio::GArray<T>>(array.arg());
            }
            catch (...)
            {
                return false;
            }
            return true;
        }
        return false;
    }
};

template<typename T>
struct PyOpenCV_Converter<ncvslideio::GOpaque<T>>
{
    static PyObject* from(const ncvslideio::GOpaque<T>& p)
    {
        return pyopencv_from(ncvslideio::GOpaqueT(p));
    }
    static bool to(PyObject *obj, ncvslideio::GOpaque<T>& value, const ArgInfo& info)
    {
        if (PyObject_TypeCheck(obj, reinterpret_cast<PyTypeObject*>(pyopencv_GOpaqueT_TypePtr)))
        {
            auto& opaque = reinterpret_cast<pyopencv_GOpaqueT_t*>(obj)->v;
            try
            {
                value = ncvslideio::util::get<ncvslideio::GOpaque<T>>(opaque.arg());
            }
            catch (...)
            {
                return false;
            }
            return true;
        }
        return false;
    }
};

template<>
bool pyopencv_to(PyObject* obj, ncvslideio::GProtoInputArgs& value, const ArgInfo& info)
{
    try
    {
        value = extract_proto_args<ncvslideio::GProtoInputArgs>(obj);
        return true;
    }
    catch (...)
    {
        failmsg("Can't parse ncvslideio::GProtoInputArgs");
        return false;
    }
}

template<>
bool pyopencv_to(PyObject* obj, ncvslideio::GProtoOutputArgs& value, const ArgInfo& info)
{
    try
    {
        value = extract_proto_args<ncvslideio::GProtoOutputArgs>(obj);
        return true;
    }
    catch (...)
    {
        failmsg("Can't parse ncvslideio::GProtoOutputArgs");
        return false;
    }
}

// extend ncvslideio.gapi methods
#define PYOPENCV_EXTRA_METHODS_GAPI \
  {"kernels", CV_PY_FN_WITH_KW(pyopencv_cv_gapi_kernels), "kernels(...) -> GKernelPackage"}, \
  {"__op", CV_PY_FN_WITH_KW(pyopencv_cv_gapi_op), "__op(...) -> retval\n"},


#endif  // HAVE_OPENCV_GAPI
#endif  // OPENCV_GAPI_PYOPENCV_GAPI_HPP
