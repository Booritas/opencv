#error This is a shadow header file, which is not intended for processing by any compiler. \
       Only bindings parser should handle this file.

namespace ncvslideio
{
struct GAPI_EXPORTS_W_SIMPLE GCompileArg
{
    GAPI_WRAP GCompileArg(GKernelPackage arg);
    GAPI_WRAP GCompileArg(gapi::GNetPackage arg);
    GAPI_WRAP GCompileArg(gapi::streaming::queue_capacity arg);
    GAPI_WRAP GCompileArg(gapi::ot::ObjectTrackerParams arg);
};

class GAPI_EXPORTS_W_SIMPLE GInferInputs
{
public:
    GAPI_WRAP GInferInputs();
    GAPI_WRAP GInferInputs& setInput(const std::string& name, const ncvslideio::GMat&   value);
    GAPI_WRAP GInferInputs& setInput(const std::string& name, const ncvslideio::GFrame& value);
};

class GAPI_EXPORTS_W_SIMPLE GInferListInputs
{
public:
    GAPI_WRAP GInferListInputs();
    GAPI_WRAP GInferListInputs setInput(const std::string& name, const ncvslideio::GArray<ncvslideio::GMat>& value);
    GAPI_WRAP GInferListInputs setInput(const std::string& name, const ncvslideio::GArray<ncvslideio::Rect>& value);
};

class GAPI_EXPORTS_W_SIMPLE GInferOutputs
{
public:
    GAPI_WRAP GInferOutputs();
    GAPI_WRAP ncvslideio::GMat at(const std::string& name);
};

class GAPI_EXPORTS_W_SIMPLE GInferListOutputs
{
public:
    GAPI_WRAP GInferListOutputs();
    GAPI_WRAP ncvslideio::GArray<ncvslideio::GMat> at(const std::string& name);
};

namespace gapi
{
namespace wip
{
class GAPI_EXPORTS_W IStreamSource { };
namespace draw
{
    // NB: These render primitives are partially wrapped in shadow file
    // because ncvslideio::Rect conflicts with ncvslideio::gapi::wip::draw::Rect in python generator
    // and ncvslideio::Rect2i breaks standalone mode.
    struct Rect
    {
        GAPI_WRAP Rect(const ncvslideio::Rect2i& rect_,
                       const ncvslideio::Scalar& color_,
                       int thick_ = 1,
                       int lt_ = 8,
                       int shift_ = 0);
    };

    struct Mosaic
    {
        GAPI_WRAP Mosaic(const ncvslideio::Rect2i& mos_, int cellSz_, int decim_);
    };
} // namespace draw
} // namespace wip
namespace streaming
{
    // FIXME: Extend to work with an arbitrary G-type.
    ncvslideio::GOpaque<int64_t> GAPI_EXPORTS_W timestamp(ncvslideio::GMat);
    ncvslideio::GOpaque<int64_t> GAPI_EXPORTS_W seqNo(ncvslideio::GMat);
    ncvslideio::GOpaque<int64_t> GAPI_EXPORTS_W seq_id(ncvslideio::GMat);

    GAPI_EXPORTS_W ncvslideio::GMat desync(const ncvslideio::GMat &g);
} // namespace streaming
} // namespace gapi

namespace detail
{
    gapi::GNetParam GAPI_EXPORTS_W strip(gapi::ie::PyParams params);
    gapi::GNetParam GAPI_EXPORTS_W strip(gapi::onnx::PyParams params);
    gapi::GNetParam GAPI_EXPORTS_W strip(gapi::ov::PyParams params);
} // namespace detail
} // namespace ncvslideio
