// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "precomp.hpp"

// needs to be included regardless if IE is present or not
// (ncvslideio::gapi::ov::backend() is still there and is defined always)
#include "backends/ov/govbackend.hpp"

#if defined HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000

#include "backends/ov/util.hpp"
#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!
#include "logger.hpp"

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/infer/ov.hpp>
#include <opencv2/core/utils/configuration.private.hpp> // getConfigurationParameterBool

#if defined(HAVE_TBB)
#  include <tbb/concurrent_queue.h> // FIXME: drop it from here!
template<typename T> using QueueClass = tbb::concurrent_bounded_queue<T>;
#else
#  include "executor/conc_queue.hpp"
template<typename T> using QueueClass = ncvslideio::gapi::own::concurrent_bounded_queue<T>;
#endif // TBB

#include "utils/itt.hpp"

#include <ade/util/zip_range.hpp>

#include <openvino/openvino.hpp>

#include <fstream>

using ParamDesc = ncvslideio::gapi::ov::detail::ParamDesc;

// NB: Some of OV plugins fail during ov::Core destroying in specific cases.
// Solution is allocate ov::Core in heap and doesn't destroy it, which cause
// leak, but fixes tests on CI. This behaviour is configurable by using
// OPENCV_GAPI_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND=0
static ov::Core create_OV_Core_pointer() {
    // NB: 'delete' is never called
    static ov::Core* core = new ov::Core();
    return *core;
}

static ov::Core create_OV_Core_instance() {
    static ov::Core core;
    return core;
}

ov::Core ncvslideio::gapi::ov::wrap::getCore() {
    // NB: to make happy memory leak tools use:
    // - OPENCV_GAPI_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND=0
    static bool param_GAPI_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND =
        utils::getConfigurationParameterBool(
                "OPENCV_GAPI_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND",
#if defined(_WIN32) || defined(__APPLE__)
                true
#else
                false
#endif
                );
    return param_GAPI_INFERENCE_ENGINE_CORE_LIFETIME_WORKAROUND
        ? create_OV_Core_pointer() : create_OV_Core_instance();
}

static ov::AnyMap toOV(const ParamDesc::PluginConfigT &config) {
    return {config.begin(), config.end()};
}

static std::map<std::string, ::ov::PartialShape>
toOV(const std::map<std::string, std::vector<size_t>> &shapes) {
    std::map<std::string, ::ov::PartialShape> ov_shapes;
    for (const auto &it : shapes) {
        ov_shapes.emplace(it.first, ::ov::Shape(it.second));
    }
    return ov_shapes;
}

static ov::element::Type toOV(int depth) {
    switch (depth) {
        case CV_8U:  return ov::element::u8;
        case CV_32S: return ov::element::i32;
        case CV_32F: return ov::element::f32;
        case CV_16F: return ov::element::f16;
        default: GAPI_Error("OV Backend: Unsupported data type");
    }
    return ov::element::undefined;
}

static ov::preprocess::ResizeAlgorithm toOVInterp(int interpolation) {
    namespace pp = ov::preprocess;
    switch (interpolation) {
        case ncvslideio::INTER_LINEAR:  return pp::ResizeAlgorithm::RESIZE_LINEAR;
        case ncvslideio::INTER_NEAREST: return pp::ResizeAlgorithm::RESIZE_NEAREST;
        case ncvslideio::INTER_CUBIC:   return pp::ResizeAlgorithm::RESIZE_CUBIC;
        default: GAPI_Error("OV Backend: Unsupported resize algorithm");
    }
    // Unreachable code
    GAPI_Assert(false);
}

static std::vector<int> toCV(const ov::Shape &shape) {
    std::vector<int> result;
    result.reserve(shape.size());
    for (auto dim : shape) {
        result.push_back(ade::util::checked_cast<int>(dim));
    }
    return result;
}

static int toCV(const ov::element::Type &type) {
    switch (type) {
        case ov::element::u8:  return CV_8U;
        case ov::element::f32: return CV_32F;
        case ov::element::i32: return CV_32S;
        case ov::element::i64: return CV_32S;
        case ov::element::f16: return CV_16F;
        default: GAPI_Error("OV Backend: Unsupported data type");
    }
    return -1;
}

static void copyFromOV(const ov::Tensor &tensor, ncvslideio::Mat &mat) {
    const auto total = mat.total() * mat.channels();
    if (toCV(tensor.get_element_type()) != mat.depth() ||
        tensor.get_size()               != total) {
        std::stringstream ss;
        ss << "Failed to copy data from ov::Tensor to ncvslideio::Mat."
           << " Data type or number of elements mismatch."
           << " ncvslideio::Mat: " << ncvslideio::descr_of(mat) << " and"
           << " ov::Tensor: " << tensor.get_element_type() << " "
           << tensor.get_shape();
        ncvslideio::util::throw_error(std::logic_error(ss.str()));
    }

    if (tensor.get_element_type() == ov::element::i64) {
        GAPI_LOG_WARNING(NULL, "INT64 isn't supported for ncvslideio::Mat. Conversion to INT32 is used.");
        ncvslideio::gimpl::convertInt64ToInt32(tensor.data<int64_t>(),
                                       mat.ptr<int>(),
                                       total);
    } else {
        std::copy_n(reinterpret_cast<uint8_t*>(tensor.data()),
                    tensor.get_byte_size(),
                    mat.ptr<uint8_t>());
    }
}

static ncvslideio::Mat wrapOV(const ncvslideio::MediaFrame::View& view,
               const ncvslideio::GFrameDesc& desc) {
    ncvslideio::Mat out;
    switch (desc.fmt) {
        case ncvslideio::MediaFormat::BGR: {
            out = ncvslideio::Mat(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            return out;
        }
        case ncvslideio::MediaFormat::NV12: {
            auto y_plane  = ncvslideio::Mat(desc.size, CV_8UC1, view.ptr[0], view.stride[0]);
            auto uv_plane = ncvslideio::Mat(desc.size / 2, CV_8UC2, view.ptr[1], view.stride[1]);
            cvtColorTwoPlane(y_plane, uv_plane, out, ncvslideio::COLOR_YUV2BGR_NV12);
            return out;
        }
        case ncvslideio::MediaFormat::GRAY: {
            out = ncvslideio::Mat(desc.size, CV_8UC1, view.ptr[0], view.stride[0]);
            return out;
        }
        default:
            GAPI_Error("OV Backend: Unsupported media format");
    }
    return out;
}

static void copyToOV(const ncvslideio::Mat &mat, ov::Tensor &tensor) {
    // TODO: Ideally there should be check that mat and tensor
    // dimensions are compatible.
    const auto total = mat.total() * mat.channels();
    if (toCV(tensor.get_element_type()) != mat.depth() ||
        tensor.get_size()               != total) {
        std::stringstream ss;
        ss << "Failed to copy data from ncvslideio::Mat to ov::Tensor."
           << " Data type or number of elements mismatch."
           << " ov::Tensor: " << tensor.get_element_type() << " "
           << tensor.get_shape() << " and"
           << " ncvslideio::Mat: " << ncvslideio::descr_of(mat);
        ncvslideio::util::throw_error(std::logic_error(ss.str()));
    }

    if (tensor.get_element_type() == ov::element::i64) {
        ncvslideio::gimpl::convertInt32ToInt64(mat.ptr<int>(),
                                       tensor.data<int64_t>(),
                                       total);
    } else {
        std::copy_n(mat.ptr<uint8_t>(),
                    tensor.get_byte_size(),
                    reinterpret_cast<uint8_t*>(tensor.data()));
    }
}

static void copyToOV(const ncvslideio::MediaFrame &frame, ov::Tensor &tensor) {
    const auto view = ncvslideio::MediaFrame::View(frame.access(ncvslideio::MediaFrame::Access::R));
    auto matFromFrame = wrapOV(view, frame.desc());
    copyToOV(matFromFrame, tensor);
}

std::vector<int> ncvslideio::gapi::ov::util::to_ocv(const ::ov::Shape &shape) {
    return toCV(shape);
}

int ncvslideio::gapi::ov::util::to_ocv(const ::ov::element::Type &type) {
    return toCV(type);
}

void ncvslideio::gapi::ov::util::to_ov(const ncvslideio::Mat &mat, ::ov::Tensor &tensor) {
    copyToOV(mat, tensor);
}

void ncvslideio::gapi::ov::util::to_ocv(const ::ov::Tensor &tensor, ncvslideio::Mat &mat) {
    copyFromOV(tensor, mat);
}

struct OVUnit {
    static const char *name() { return "OVUnit"; }

    explicit OVUnit(const ParamDesc &pd)
        : params(pd) {

        // FIXME: Can this logic be encapsulated to prevent checking every time?
        if (ncvslideio::util::holds_alternative<ParamDesc::Model>(params.kind)) {
            const auto desc = ncvslideio::util::get<ParamDesc::Model>(params.kind);
            model = ncvslideio::gapi::ov::wrap::getCore()
                .read_model(desc.model_path, desc.bin_path);
            GAPI_Assert(model);

            if (params.num_in == 1u && params.input_names.empty()) {
                params.input_names = { model->inputs().begin()->get_any_name() };
            }
            if (params.num_out == 1u && params.output_names.empty()) {
                params.output_names = { model->outputs().begin()->get_any_name() };
            }

        } else {
            GAPI_Assert(ncvslideio::util::holds_alternative<ParamDesc::CompiledModel>(params.kind));
            std::ifstream file(ncvslideio::util::get<ParamDesc::CompiledModel>(params.kind).blob_path,
                               std::ios_base::in | std::ios_base::binary);
            GAPI_Assert(file.is_open());
            compiled_model = ncvslideio::gapi::ov::wrap::getCore()
                .import_model(file, params.device, toOV(params.config));

            if (params.num_in == 1u && params.input_names.empty()) {
                params.input_names = { compiled_model.inputs().begin()->get_any_name() };
            }
            if (params.num_out == 1u && params.output_names.empty()) {
                params.output_names = { compiled_model.outputs().begin()->get_any_name() };
            }
        }
    };

    ncvslideio::gimpl::ov::OVCompiled compile() {
        if (ncvslideio::util::holds_alternative<ParamDesc::Model>(params.kind)) {
            compiled_model = ncvslideio::gapi::ov::wrap::getCore()
                .compile_model(model, params.device, toOV(params.config));
        }
        return {compiled_model};
    }

    ncvslideio::gapi::ov::detail::ParamDesc params;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
};

class OVCallContext
{
public:
    OVCallContext(const OVUnit                                      &  unit,
                  ncvslideio::gimpl::GIslandExecutable::IOutput             &  output,
                  const ncvslideio::GArgs                                   &  args,
                  const std::vector<ncvslideio::gimpl::RcDesc>              &  outs,
                  ncvslideio::GRunArg::Meta                                 && meta,
                  std::vector<ncvslideio::gimpl::GIslandExecutable::InObj>  && input_objs,
                  std::vector<ncvslideio::gimpl::GIslandExecutable::OutObj> && output_objs,
                  const ncvslideio::gimpl::ov::Options                      &  options);

    const ncvslideio::GArgs& inArgs() const;

    // Generic accessor API
    template<typename T>
    const T& inArg(std::size_t input) const {
        return m_args.at(input).get<T>();
    }

    template<typename T>
    std::vector<T>& outVecR(std::size_t output) {
        return outVecRef(output).wref<T>();
    }

    // Syntax sugar
          ncvslideio::GShape      inShape (std::size_t input) const;
    const ncvslideio::Mat&        inMat   (std::size_t input) const;
    const ncvslideio::MediaFrame& inFrame (std::size_t input) const;

    ncvslideio::GRunArgP output (std::size_t idx);
    ncvslideio::Mat&     outMatR(std::size_t idx);

    const OVUnit                          &uu;
    ncvslideio::gimpl::GIslandExecutable::IOutput &out;

    // To store exception appeared in callback.
    std::exception_ptr eptr;

    const ncvslideio::GRunArg::Meta& getMeta() { return m_meta; };

    const ncvslideio::gimpl::ov::Options& getOptions() const { return m_options; };

private:
    ncvslideio::detail::VectorRef& outVecRef(std::size_t idx);

    ncvslideio::GArg packArg(const ncvslideio::GArg &arg);

    // To propagate accumulated meta from all inputs to output.
    ncvslideio::GRunArg::Meta m_meta;

    // To store input/output data from frames
    std::vector<ncvslideio::gimpl::GIslandExecutable::InObj>  m_input_objs;
    std::vector<ncvslideio::gimpl::GIslandExecutable::OutObj> m_output_objs;

    // To simplify access to ncvslideio::Mat inside ncvslideio::RMat
    ncvslideio::gimpl::Mag m_res;

    std::unordered_map<std::size_t, ncvslideio::GRunArgP> m_results;

    // Input parameters passed to an inference operation.
    ncvslideio::GArgs m_args;
    ncvslideio::GShapes m_in_shapes;

    ncvslideio::gimpl::ov::Options m_options;
};

OVCallContext::OVCallContext(const OVUnit                                      &  unit,
                             ncvslideio::gimpl::GIslandExecutable::IOutput             &  output,
                             const ncvslideio::GArgs                                   &  args,
                             const std::vector<ncvslideio::gimpl::RcDesc>              &  outs,
                             ncvslideio::GRunArg::Meta                                 && meta,
                             std::vector<ncvslideio::gimpl::GIslandExecutable::InObj>  && input_objs,
                             std::vector<ncvslideio::gimpl::GIslandExecutable::OutObj> && output_objs,
                             const ncvslideio::gimpl::ov::Options                      &  options)
: uu(unit), out(output), m_meta(std::move(meta)),
  m_input_objs(std::move(input_objs)), m_output_objs(std::move(output_objs)),
  m_options(options)
{
    for (auto& it : m_input_objs)  ncvslideio::gimpl::magazine::bindInArg (m_res, it.first, it.second);
    for (auto& it : m_output_objs) ncvslideio::gimpl::magazine::bindOutArg(m_res, it.first, it.second);

    m_args.reserve(args.size());
    using namespace std::placeholders;
    ade::util::transform(args,
                         std::back_inserter(m_args),
                         std::bind(&OVCallContext::packArg, this, _1));

    ade::util::transform(args, std::back_inserter(m_in_shapes),
            [](const ncvslideio::GArg& arg) {
                return arg.get<ncvslideio::gimpl::RcDesc>().shape;
            });

    for (const auto out_it : ade::util::indexed(outs)) {
        // FIXME: Can the same GArg type resolution mechanism be reused here?
        const auto port  = ade::util::index(out_it);
        const auto desc  = ade::util::value(out_it);
        m_results[port] = ncvslideio::gimpl::magazine::getObjPtr(m_res, desc);
    }
}

const ncvslideio::GArgs& OVCallContext::inArgs() const {
    return m_args;
}

ncvslideio::GShape OVCallContext::inShape(std::size_t i) const {
    return m_in_shapes[i];
}

const ncvslideio::Mat& OVCallContext::inMat(std::size_t input) const {
    return inArg<ncvslideio::Mat>(input);
}

const ncvslideio::MediaFrame& OVCallContext::inFrame(std::size_t input) const {
    return inArg<ncvslideio::MediaFrame>(input);
}

ncvslideio::Mat& OVCallContext::outMatR(std::size_t idx) {
    return *ncvslideio::util::get<ncvslideio::Mat*>(m_results.at(idx));
}

ncvslideio::GRunArgP OVCallContext::output(std::size_t idx) {
    return m_output_objs[idx].second;
};

ncvslideio::detail::VectorRef& OVCallContext::outVecRef(std::size_t idx) {
    return ncvslideio::util::get<ncvslideio::detail::VectorRef>(m_results.at(idx));
}

ncvslideio::GArg OVCallContext::packArg(const ncvslideio::GArg &arg) {
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != ncvslideio::detail::ArgKind::GMAT
                && arg.kind != ncvslideio::detail::ArgKind::GSCALAR
                && arg.kind != ncvslideio::detail::ArgKind::GARRAY);

    if (arg.kind != ncvslideio::detail::ArgKind::GOBJREF) {
        ncvslideio::util::throw_error(std::logic_error("Inference supports G-types ONLY!"));
    }
    GAPI_Assert(arg.kind == ncvslideio::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const ncvslideio::gimpl::RcDesc &ref = arg.get<ncvslideio::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case ncvslideio::GShape::GMAT: return ncvslideio::GArg(m_res.slot<ncvslideio::Mat>()[ref.id]);

    // Note: .at() is intentional for GArray as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case ncvslideio::GShape::GARRAY:  return ncvslideio::GArg(m_res.slot<ncvslideio::detail::VectorRef>().at(ref.id));

    // Note: .at() is intentional for GOpaque as object MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case ncvslideio::GShape::GOPAQUE:  return ncvslideio::GArg(m_res.slot<ncvslideio::detail::OpaqueRef>().at(ref.id));

    case ncvslideio::GShape::GFRAME:  return ncvslideio::GArg(m_res.slot<ncvslideio::MediaFrame>()[ref.id]);

    default:
        ncvslideio::util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

struct OVCallable {
    static const char *name() { return "OVRequestCallable"; }
    using Run = std::function<void(std::shared_ptr<OVCallContext>,
                                   ncvslideio::gimpl::ov::RequestPool&)>;
    Run run;
};

struct KImpl {
    ncvslideio::gimpl::CustomMetaFunction::CM customMetaFunc;
    OVCallable::Run run;
};

using GOVModel = ade::TypedGraph
    < ncvslideio::gimpl::Protocol
    , ncvslideio::gimpl::Op
    , ncvslideio::gimpl::NetworkParams
    , ncvslideio::gimpl::CustomMetaFunction
    , OVUnit
    , OVCallable
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGOVModel = ade::ConstTypedGraph
    < ncvslideio::gimpl::Protocol
    , ncvslideio::gimpl::Op
    , ncvslideio::gimpl::NetworkParams
    , ncvslideio::gimpl::CustomMetaFunction
    , OVUnit
    , OVCallable
    >;

namespace {
class IInferExecutor {
public:
    using Ptr             = std::shared_ptr<IInferExecutor>;
    using NotifyCallbackF = std::function<void()>;
    using SetInputDataF   = std::function<void(::ov::InferRequest&)>;
    using ReadOutputDataF = std::function<void(::ov::InferRequest&, std::exception_ptr)>;

    // NB: The task is represented by:
    // SetInputDataF - function which set input data.
    // ReadOutputDataF - function which read output data.
    struct Task {
        SetInputDataF   set_input_data;
        ReadOutputDataF read_output_data;
    };

    IInferExecutor(::ov::InferRequest request, NotifyCallbackF notify)
        : m_request(std::move(request)),
          m_notify(std::move(notify)) {
    };

    virtual void execute(const Task& task) = 0;
    virtual ~IInferExecutor() = default;

protected:
    ::ov::InferRequest m_request;
    NotifyCallbackF    m_notify;
};

class SyncInferExecutor : public IInferExecutor {
    using IInferExecutor::IInferExecutor;
    virtual void execute(const IInferExecutor::Task &task) override;
};

void SyncInferExecutor::execute(const IInferExecutor::Task &task) {
    try {
        task.set_input_data(m_request);
        m_request.infer();
        task.read_output_data(m_request, nullptr);
    } catch (...) {
        m_notify();
        throw;
    }
    // NB: Notify pool that executor has finished.
    m_notify();
}

class AsyncInferExecutor : public IInferExecutor {
public:
    using IInferExecutor::IInferExecutor;
    virtual void execute(const IInferExecutor::Task& task) override;

private:
    void callback(Task task,
                  ::ov::InferRequest request,
                  std::exception_ptr eptr) noexcept;
};

void AsyncInferExecutor::execute(const IInferExecutor::Task& task) {
    using namespace std::placeholders;
    using callback_t = std::function<void(std::exception_ptr)>;
    m_request.set_callback(
            static_cast<callback_t>(
                std::bind(&AsyncInferExecutor::callback, this, task, m_request, _1)));
    try {
        task.set_input_data(m_request);
        m_request.start_async();
    } catch (...) {
        m_request.set_callback([](std::exception_ptr){});
        m_notify();
        throw;
    }
}

void AsyncInferExecutor::callback(IInferExecutor::Task task,
                                  ::ov::InferRequest   request,
                                  std::exception_ptr   eptr) noexcept {
    task.read_output_data(request, eptr);
    request.set_callback([](std::exception_ptr){});
    // NB: Notify pool that executor has finished.
    m_notify();
}

} // anonymous namespace

// TODO: Make it generic to reuse in IE and ONNX backends.
class ncvslideio::gimpl::ov::RequestPool {
public:
    explicit RequestPool(std::vector<::ov::InferRequest>&& requests);

    IInferExecutor::Ptr getIdleRequest();
    void waitAll();

private:
    void setup();
    void release(const size_t id);

    QueueClass<size_t>               m_idle_ids;
    std::vector<IInferExecutor::Ptr> m_requests;
};

void ncvslideio::gimpl::ov::RequestPool::release(const size_t id) {
    m_idle_ids.push(id);
}

ncvslideio::gimpl::ov::RequestPool::RequestPool(std::vector<::ov::InferRequest>&& requests) {
    GAPI_Assert(!requests.empty());
    if (requests.size() == 1u) {
        m_requests.push_back(
                std::make_shared<SyncInferExecutor>(
                    requests.front(), std::bind(&RequestPool::release, this, 0u)));
    } else {
        for (size_t i = 0; i < requests.size(); ++i) {
            m_requests.push_back(
                    std::make_shared<AsyncInferExecutor>(
                        requests[i], std::bind(&RequestPool::release, this, i)));
        }
    }
    setup();
}

void ncvslideio::gimpl::ov::RequestPool::setup() {
    for (size_t i = 0; i < m_requests.size(); ++i) {
        m_idle_ids.push(i);
    }
}

IInferExecutor::Ptr ncvslideio::gimpl::ov::RequestPool::getIdleRequest() {
    size_t id = 0u;
    m_idle_ids.pop(id);
    return m_requests[id];
}

// NB: Not thread-safe.
void ncvslideio::gimpl::ov::RequestPool::waitAll() {
    // NB: It will be blocked if at least one request is busy.
    for (size_t i = 0; i < m_requests.size(); ++i) {
        size_t id = 0u;
        m_idle_ids.pop(id);
    }
    setup();
}


// NB: This is a callback used by async infer
// to post outputs blobs (ncvslideio::GMat's).
static void PostOutputs(::ov::InferRequest             &infer_request,
                        std::exception_ptr             eptr,
                        std::shared_ptr<OVCallContext> ctx) {
    GAPI_ITT_STATIC_LOCAL_HANDLE(ov_cb_post_outputs_hndl, "OV_async_callback_PostOutputs");
    GAPI_ITT_AUTO_TRACE_GUARD(ov_cb_post_outputs_hndl);

    ctx->eptr = std::move(eptr);
    for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
        // NB: Copy data back only if execution finished successfully
        // and inference only mode is disabled.
        // Otherwise just post outputs to maintain streaming executor contract.
        if (!ctx->eptr && !ctx->getOptions().inference_only) {
            const auto& out_name = ctx->uu.params.output_names[i];
            copyFromOV(infer_request.get_tensor(out_name),
                       ctx->outMatR(i));
        }
        auto output = ctx->output(i);
        ctx->out.meta(output, ctx->getMeta());
        ctx->out.post(std::move(output), ctx->eptr);
    }
}

class PostOutputsList {
public:
    PostOutputsList(size_t size,
                    std::shared_ptr<OVCallContext> ctx);

    void operator()(::ov::InferRequest &infer_request,
                    std::exception_ptr eptr,
                    size_t             pos) const;

private:
    struct Priv {
        std::atomic<size_t> finished{0u};
        size_t size;
        std::shared_ptr<OVCallContext> ctx;
    };
    std::shared_ptr<Priv> m_priv;
};

PostOutputsList::PostOutputsList(size_t size,
                                 std::shared_ptr<OVCallContext> ctx)
    : m_priv(new Priv{}) {
    m_priv->size = size;
    m_priv->ctx = ctx;
}

void PostOutputsList::operator()(::ov::InferRequest &infer_request,
                                 std::exception_ptr eptr,
                                 size_t             pos) const {
    auto&& ctx         = m_priv->ctx;
    auto&& finished    = m_priv->finished;
    auto&& size        = m_priv->size;

    ctx->eptr = eptr;
    if (!ctx->eptr) {
        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            std::vector<ncvslideio::Mat> &out_vec = ctx->outVecR<ncvslideio::Mat>(i);

            const auto &out_name = ctx->uu.params.output_names[i];
            const auto &out_tensor = infer_request.get_tensor(out_name);

            out_vec[pos].create(toCV(out_tensor.get_shape()),
                                toCV(out_tensor.get_element_type()));
            copyFromOV(out_tensor, out_vec[pos]);
        }
    }
    ++finished;

    if (finished == size) {
        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            auto output = ctx->output(i);
            ctx->out.meta(output, ctx->getMeta());
            ctx->out.post(std::move(output), ctx->eptr);
        }
    }
}

static void copyToOV(std::shared_ptr<OVCallContext> ctx, uint32_t input_idx, ov::Tensor &tensor) {
    switch (ctx->inShape(input_idx)) {
        case ncvslideio::GShape::GMAT:
            copyToOV(ctx->inMat(input_idx), tensor);
            break;
        case ncvslideio::GShape::GFRAME:
            copyToOV(ctx->inFrame(input_idx), tensor);
            break;
        default:
            GAPI_Assert("Unsupported input shape for OV backend");
    }
}

namespace ncvslideio {
namespace gimpl {
namespace ov {

template <typename Attr>
using AttrMap = ncvslideio::gapi::ov::detail::AttrMap<Attr>;

template <typename Attr>
using LayerVariantAttr = ncvslideio::gapi::ov::detail::LayerVariantAttr<Attr>;

template <typename Attr> AttrMap<Attr>
broadcastLayerAttr(const LayerVariantAttr<Attr>   &layer_attr,
                   const std::vector<std::string> &layer_names) {
    AttrMap<Attr> map;
    if (ncvslideio::util::holds_alternative<AttrMap<Attr>>(layer_attr)) {
        map = ncvslideio::util::get<AttrMap<Attr>>(layer_attr);
        // NB: Validate map:
        std::unordered_set<std::string> existing_layers =
            {layer_names.begin(), layer_names.end()};

        for (const auto &p : map) {
            const auto it = existing_layers.find(p.first);
            if (it == existing_layers.end()) {
                ncvslideio::util::throw_error(
                        std::logic_error("OV Backend: Failed to"
                                         " find layer with name: " + p.first));
            }
        }
    } else if (ncvslideio::util::holds_alternative<Attr>(layer_attr)) {
        // NB: Broadcast value to all layers.
        auto elem = ncvslideio::util::get<Attr>(layer_attr);
        for (auto &&layer_name : layer_names) {
            map.emplace(layer_name, elem);
        }
    }
    return map;
}

template <typename K, typename V>
ncvslideio::optional<V> lookUp(const std::map<K, V> &map, const K& key) {
    const auto it = map.find(key);
    if (it == map.end()) {
        return {};
    }
    return ncvslideio::util::make_optional(std::move(it->second));
}

// NB: This function is used to preprocess input image
// for InferROI, InferList, InferList2 kernels.
static ncvslideio::Mat preprocess(const ncvslideio::Mat     &in_mat,
                          const ncvslideio::Rect    &roi,
                          const ::ov::Shape &model_shape) {
    ncvslideio::Mat out;
    // FIXME: Since there is no information about H and W positions
    // among tensor dimmensions assume that model layout is "NHWC".
    // (In fact "NHWC" is the only right layout for preprocessing because
    // it works only with images.
    GAPI_Assert(model_shape.size() == 4u);
    const auto H = model_shape[1];
    const auto W = model_shape[2];
    const auto C = model_shape[3];
    // NB: Soft check that at least number of channels matches.
    if (static_cast<int>(C) != in_mat.channels()) {
        std::stringstream ss;
        ss << "OV Backend: Failed to preprocess input data "
              " (Number of channels mismatch)."
              " Provided data: " << ncvslideio::descr_of(in_mat) <<
              " and Model shape: " << model_shape;
        util::throw_error(std::logic_error(ss.str()));
    }
    // NB: Crop roi and resize to model size.
    ncvslideio::resize(in_mat(roi), out, ncvslideio::Size(W, H));
    return out;
}

// NB: This function is used to preprocess input image
// for InferROI, InferList, InferList2 kernels.
static ncvslideio::Mat preprocess(MediaFrame::View&     view,
                          const ncvslideio::GFrameDesc& desc,
                          const ncvslideio::Rect&       roi,
                          const ::ov::Shape     &model_shape) {
    return preprocess(wrapOV(view, desc), roi, model_shape);
}

static void preprocess_and_copy(std::shared_ptr<OVCallContext> ctx,
                                uint32_t input_idx,
                                const ncvslideio::Rect &roi,
                                const ::ov::Shape &model_shape,
                                ::ov::Tensor& tensor) {
    switch (ctx->inShape(input_idx)) {
        case ncvslideio::GShape::GMAT: {
            auto roi_mat = preprocess(ctx->inMat(input_idx), roi, model_shape);
            copyToOV(roi_mat, tensor);
            break;
        }
        case ncvslideio::GShape::GFRAME: {
            auto currentFrame = ctx->inFrame(input_idx);
            auto view = ncvslideio::MediaFrame::View(currentFrame.access(ncvslideio::MediaFrame::Access::R));
            auto roi_mat = preprocess(view, currentFrame.desc(), roi, model_shape);
            copyToOV(roi_mat, tensor);
            break;
        }
        default:
            GAPI_Assert("Unsupported input shape for OV backend");
    }
}

static bool isImage(const ncvslideio::GMatDesc &desc,
                    const ::ov::Shape  &model_shape) {
    return (model_shape.size() == 4u)                      &&
           (!desc.isND())  /* dims == 2 */                 &&
           (desc.chan == 1 || desc.chan == 3)              &&
           (desc.size.height != 1 && desc.size.width != 1) &&
           (desc.depth == CV_8U);
}

static bool isImage(const ncvslideio::GMetaArg &meta,
                    const ::ov::Shape  &shape) {
    if (ncvslideio::util::holds_alternative<GFrameDesc>(meta)) {
        return true;
    }
    GAPI_Assert(ncvslideio::util::holds_alternative<GMatDesc>(meta));
    auto matdesc = ncvslideio::util::get<GMatDesc>(meta);
    return isImage(matdesc, shape);
}

class PrePostProcWrapper {
public:
    PrePostProcWrapper(std::shared_ptr<::ov::Model>   &model,
                       const ParamDesc::Model         &model_info,
                       const std::vector<std::string> &input_names,
                       const std::vector<std::string> &output_names)
        : m_ppp(model),
          m_model(model),
          m_model_info(model_info),
          m_input_names(input_names),
          m_output_names(output_names) {
        // NB: Do Reshape right away since it must be the first step of model modification
        // and applicable for all infer kernels.
        const auto new_shapes = broadcastLayerAttr(model_info.new_shapes, input_names);
        m_model->reshape(toOV(new_shapes));

        const auto &mi = m_model_info;
        m_input_tensor_layout = broadcastLayerAttr(mi.input_tensor_layout, m_input_names);
        m_input_model_layout  = broadcastLayerAttr(mi.input_model_layout, m_input_names);
        m_interpolation       = broadcastLayerAttr(mi.interpolation, m_input_names);
        m_mean_values         = broadcastLayerAttr(mi.mean_values, m_input_names);
        m_scale_values        = broadcastLayerAttr(mi.scale_values, m_input_names);
        m_interpolation       = broadcastLayerAttr(mi.interpolation, m_input_names);

        m_output_tensor_layout    = broadcastLayerAttr(mi.output_tensor_layout, m_output_names);
        m_output_model_layout     = broadcastLayerAttr(mi.output_model_layout, m_output_names);
        m_output_tensor_precision = broadcastLayerAttr(mi.output_tensor_precision, m_output_names);
     };

    void cfgLayouts(const std::string &input_name) {
        auto &input_info = m_ppp.input(input_name);
        const auto explicit_in_model_layout = lookUp(m_input_model_layout, input_name);
        if (explicit_in_model_layout) {
            input_info.model().set_layout(::ov::Layout(*explicit_in_model_layout));
        } else if (m_model->input(input_name).get_shape().size() == 4u) {
            const auto& input_layout = ::ov::layout::get_layout(m_model->input(input_name));
            if (!input_layout.empty()) {
                GAPI_LOG_INFO(NULL, "Model input layout " << input_name << " found: " << input_layout.to_string() << ".");
            } else {
                // NB: Back compatibility with IR's without any layout information.
                // Note that default is only applicable for 4D inputs in order to
                // support auto resize for image use cases.
                GAPI_LOG_WARNING(NULL, "Failed to find layout for input layer \""
                        << input_name << "\" - NCHW is set by default");
                const std::string default_layout = "NCHW";
                input_info.model().set_layout(::ov::Layout(default_layout));
                m_input_model_layout.emplace(input_name, default_layout);
            }
        }
        const auto explicit_in_tensor_layout = lookUp(m_input_tensor_layout, input_name);
        if (explicit_in_tensor_layout) {
            input_info.tensor().set_layout(::ov::Layout(*explicit_in_tensor_layout));
        }
    }

    void cfgScaleMean(const std::string &input_name,
                      const GMetaArg &input_meta) {
        auto &input_info = m_ppp.input(input_name);

        const auto mean_vec = lookUp(m_mean_values, input_name);
        const auto scale_vec = lookUp(m_scale_values, input_name);

        if (mean_vec || scale_vec) {
            GAPI_Assert(ncvslideio::util::holds_alternative<ncvslideio::GMatDesc>(input_meta));
            const auto depth = ncvslideio::util::get<ncvslideio::GMatDesc>(input_meta).depth;
            const bool depth_is_real = (depth == CV_32F) || (depth == CV_16F);
            if (!depth_is_real) {
                input_info.preprocess().convert_element_type(toOV(CV_32F));
            }
        }
        if (mean_vec) {
            input_info.preprocess().mean(*mean_vec);
        }
        if (scale_vec) {
            input_info.preprocess().scale(*scale_vec);
        }
    }

    // FIXME: Decompose this...
    void cfgPreProcessing(const std::string  &input_name,
                          const ncvslideio::GMetaArg &input_meta,
                          const bool         disable_img_resize = false) {
        GAPI_Assert(ncvslideio::util::holds_alternative<ncvslideio::GMatDesc>(input_meta) ||
                    ncvslideio::util::holds_alternative<ncvslideio::GFrameDesc>(input_meta));
        const auto explicit_in_tensor_layout = lookUp(m_input_tensor_layout, input_name);
        const auto explicit_in_model_layout  = lookUp(m_input_model_layout, input_name);
        const auto explicit_resize = lookUp(m_interpolation, input_name);

        if (disable_img_resize && explicit_resize.has_value()) {
            std::stringstream ss;
            util::throw_error(std::logic_error(
                "OV Backend: Resize for layer \"" + input_name + "\" will be performed"
                " on host via OpenCV so explicitly configured resize is prohibited."));
        }

        const auto &input_shape = m_model->input(input_name).get_shape();
        auto &input_info = m_ppp.input(input_name);

        auto isMat = ncvslideio::util::holds_alternative<ncvslideio::GMatDesc>(input_meta);
        auto prec  = isMat ? ncvslideio::util::get<ncvslideio::GMatDesc>(input_meta).depth : CV_8U;
        m_ppp.input(input_name).tensor().set_element_type(toOV(prec));

        const auto &matdesc   = isMat ? ncvslideio::util::get<ncvslideio::GMatDesc>(input_meta) : ncvslideio::GMatDesc();
        const auto &framedesc = !isMat ? ncvslideio::util::get<ncvslideio::GFrameDesc>(input_meta) : ncvslideio::GFrameDesc();
        if (isImage(input_meta, input_shape)) {
            // NB: Image case - all necessary preprocessng is configured automatically.
            GAPI_LOG_DEBUG(NULL, "OV Backend: Input: \"" << input_name << "\" is image.");
            if (explicit_in_tensor_layout && *explicit_in_tensor_layout != "NHWC") {
                std::stringstream desc_str;
                if (isMat) {
                    desc_str << matdesc;
                } else {
                    desc_str << framedesc;
                }
                std::stringstream ss;
                ss << "OV Backend: Provided tensor layout " << *explicit_in_tensor_layout
                << " is not compatible with input data " << desc_str.str() << " for layer \""
                << input_name << "\". Expecting NHWC";
                util::throw_error(std::logic_error(ss.str()));
            } else {
                input_info.tensor().set_layout(::ov::Layout("NHWC"));
            }

            if (!disable_img_resize) {
                const auto size = isMat ? ncvslideio::util::get<ncvslideio::GMatDesc>(input_meta).size : ncvslideio::util::get<ncvslideio::GFrameDesc>(input_meta).size;
                input_info.tensor().set_spatial_static_shape(size.height,
                                                             size.width);
                // NB: Even though resize is automatically configured
                // user have an opportunity to specify the interpolation algorithm.
                auto interp = explicit_resize
                    ? toOVInterp(*explicit_resize)
                    : ::ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR;
                input_info.preprocess().resize(interp);
            }
        } else {
            // NB: Tensor case - resize or layout conversions must be explicitly specified.
            GAPI_LOG_DEBUG(NULL, "OV Backend: Input: \"" << input_name << "\" is tensor.");

            if (explicit_resize) {
                if (matdesc.isND()) {
                    // NB: ND case - need to obtain "H" and "W" positions
                    // in order to configure resize.
                    const auto model_layout = explicit_in_model_layout
                        ? ::ov::Layout(*explicit_in_model_layout)
                        : ::ov::layout::get_layout(m_model->input(input_name));
                    if (!explicit_in_tensor_layout && model_layout.empty()) {
                        std::stringstream ss;
                        ss << "Resize for input layer: " << input_name
                        << "can't be configured."
                        << " Failed to extract H and W positions from layout.";
                        util::throw_error(std::logic_error(ss.str()));
                    } else {
                        const auto layout = explicit_in_tensor_layout
                            ? ::ov::Layout(*explicit_in_tensor_layout) : model_layout;
                        auto H_idx = ::ov::layout::height_idx(layout);
                        auto W_idx = ::ov::layout::width_idx(layout);
                        // NB: If layout is "...HW", H position is -2.
                        if (H_idx < 0) H_idx = matdesc.dims.size() + H_idx;
                        if (W_idx < 0) W_idx = matdesc.dims.size() + W_idx;
                        GAPI_Assert(H_idx >= 0 && H_idx < static_cast<int>(matdesc.dims.size()));
                        GAPI_Assert(W_idx >= 0 && W_idx < static_cast<int>(matdesc.dims.size()));
                        input_info.tensor().set_spatial_static_shape(matdesc.dims[H_idx],
                                                                     matdesc.dims[W_idx]);
                        input_info.preprocess().resize(toOVInterp(*explicit_resize));
                    }
                } else {
                    // NB: 2D case - We know exactly where H and W...
                    input_info.tensor().set_spatial_static_shape(matdesc.size.height,
                                                                 matdesc.size.width);
                    input_info.preprocess().resize(toOVInterp(*explicit_resize));
                }
            }
        }
    }

    void cfgPostProcessing() {
        for (const auto &output_name : m_output_names) {
            const auto explicit_out_tensor_layout =
                lookUp(m_output_tensor_layout, output_name);
            if (explicit_out_tensor_layout) {
                m_ppp.output(output_name).tensor()
                    .set_layout(::ov::Layout(*explicit_out_tensor_layout));
            }

            const auto explicit_out_model_layout =
                lookUp(m_output_model_layout, output_name);
            if (explicit_out_model_layout) {
                m_ppp.output(output_name).model()
                    .set_layout(::ov::Layout(*explicit_out_model_layout));
            }

            const auto explicit_out_tensor_prec =
                lookUp(m_output_tensor_precision, output_name);
            if (explicit_out_tensor_prec) {
                m_ppp.output(output_name).tensor()
                    .set_element_type(toOV(*explicit_out_tensor_prec));
            }
        }
    }

    void finalize() {
        GAPI_LOG_DEBUG(NULL, "OV Backend: PrePostProcessor: " << m_ppp);
        m_model = m_ppp.build();
    }

private:
    ::ov::preprocess::PrePostProcessor m_ppp;

    std::shared_ptr<::ov::Model>   &m_model;
    const ParamDesc::Model         &m_model_info;
    const std::vector<std::string> &m_input_names;
    const std::vector<std::string> &m_output_names;

    ncvslideio::gimpl::ov::AttrMap<std::string>        m_input_tensor_layout;
    ncvslideio::gimpl::ov::AttrMap<std::string>        m_input_model_layout;
    ncvslideio::gimpl::ov::AttrMap<int>                m_interpolation;
    ncvslideio::gimpl::ov::AttrMap<std::vector<float>> m_mean_values;
    ncvslideio::gimpl::ov::AttrMap<std::vector<float>> m_scale_values;
    ncvslideio::gimpl::ov::AttrMap<std::string>        m_output_tensor_layout;
    ncvslideio::gimpl::ov::AttrMap<std::string>        m_output_model_layout;
    ncvslideio::gimpl::ov::AttrMap<int>                m_output_tensor_precision;
};

struct Infer: public ncvslideio::detail::KernelTag {
    using API = ncvslideio::GInferBase;
    static ncvslideio::gapi::GBackend backend()  { return ncvslideio::gapi::ov::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static ncvslideio::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const ncvslideio::GMetaArgs   &in_metas,
                                 const ncvslideio::GArgs       &/*in_args*/) {
        ncvslideio::GMetaArgs result;

        GConstGOVModel gm(gr);
        const auto &uu = gm.metadata(nh).get<OVUnit>();
        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == in_metas.size()
                    && "Known input layers count doesn't match input meta count");

        // NB: Pre/Post processing configuration avaiable only for read models.
        if (ncvslideio::util::holds_alternative<ParamDesc::Model>(uu.params.kind)) {
            const auto &model_info = ncvslideio::util::get<ParamDesc::Model>(uu.params.kind);
            auto& model = const_cast<std::shared_ptr<::ov::Model>&>(uu.model);
            PrePostProcWrapper ppp {model, model_info,
                uu.params.input_names, uu.params.output_names};

            for (auto &&it : ade::util::zip(ade::util::toRange(uu.params.input_names),
                                            ade::util::toRange(in_metas))) {
                const auto &input_name = std::get<0>(it);
                const auto &mm = std::get<1>(it);
                ppp.cfgLayouts(input_name);
                ppp.cfgPreProcessing(input_name, mm);
                ppp.cfgScaleMean(input_name, mm);
            }
            ppp.cfgPostProcessing();
            ppp.finalize();
        }

        for (const auto &out_name : uu.params.output_names) {
            ncvslideio::GMatDesc outm;
            if (ncvslideio::util::holds_alternative<ParamDesc::Model>(uu.params.kind)) {
                const auto &out = uu.model->output(out_name);
                outm = ncvslideio::GMatDesc(toCV(out.get_element_type()),
                                    toCV(out.get_shape()));
            } else {
                GAPI_Assert(ncvslideio::util::holds_alternative<ParamDesc::CompiledModel>(uu.params.kind));
                const auto &out = uu.compiled_model.output(out_name);
                outm = ncvslideio::GMatDesc(toCV(out.get_element_type()),
                                    toCV(out.get_shape()));
            }
            result.emplace_back(std::move(outm));
        }

        return result;
    }

    static void run(std::shared_ptr<OVCallContext> ctx,
                    ncvslideio::gimpl::ov::RequestPool     &reqPool) {
        using namespace std::placeholders;
        reqPool.getIdleRequest()->execute(
                IInferExecutor::Task {
                    [ctx](::ov::InferRequest &infer_request) {
                        // NB: No need to populate model inputs with data
                        // if it's inference only mode.
                        if (ctx->getOptions().inference_only) {
                            return;
                        }
                        for (auto i : ade::util::iota(ctx->uu.params.num_in)) {
                            const auto& input_name = ctx->uu.params.input_names[i];
                            auto input_tensor = infer_request.get_tensor(input_name);
                            // TODO: In some cases wrapping existing data pointer
                            // might be faster than copy. Make it a strategy.
                            copyToOV(ctx, i, input_tensor);
                        }
                    },
                    std::bind(PostOutputs, _1, _2, ctx)
                }
        );
    }
};

struct InferROI: public ncvslideio::detail::KernelTag {
    using API = ncvslideio::GInferROIBase;
    static ncvslideio::gapi::GBackend backend()  { return ncvslideio::gapi::ov::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static ncvslideio::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const ncvslideio::GMetaArgs   &in_metas,
                                 const ncvslideio::GArgs       &/*in_args*/) {
        ncvslideio::GMetaArgs result;

        GConstGOVModel gm(gr);
        const auto &uu = gm.metadata(nh).get<OVUnit>();
        // Initialize input information
        // FIXME: So far it is pretty limited
        GAPI_Assert(1u == uu.params.input_names.size());
        GAPI_Assert(2u == in_metas.size());

        const auto &input_name = uu.params.input_names.at(0);
        const auto &mm = in_metas.at(1u);
        GAPI_Assert(ncvslideio::util::holds_alternative<ncvslideio::GMatDesc>(mm) ||
                    ncvslideio::util::holds_alternative<ncvslideio::GFrameDesc>(mm));
        const bool is_model = ncvslideio::util::holds_alternative<ParamDesc::Model>(uu.params.kind);
        const auto &input_shape = is_model ? uu.model->input(input_name).get_shape()
                                           : uu.compiled_model.input(input_name).get_shape();

        if (!isImage(mm, input_shape)) {
            util::throw_error(std::runtime_error(
                "OV Backend: InferROI supports only image as the 1th argument"));
        }

        if (is_model) {
            const auto &model_info = ncvslideio::util::get<ParamDesc::Model>(uu.params.kind);
            auto& model = const_cast<std::shared_ptr<::ov::Model>&>(uu.model);
            PrePostProcWrapper ppp {model, model_info,
                uu.params.input_names, uu.params.output_names};

            ppp.cfgLayouts(input_name);
            ppp.cfgPreProcessing(input_name, mm, true /*disable_img_resize*/);
            ppp.cfgScaleMean(input_name, mm);
            ppp.cfgPostProcessing();
            ppp.finalize();
        }

        for (const auto &out_name : uu.params.output_names) {
            ncvslideio::GMatDesc outm;
            if (ncvslideio::util::holds_alternative<ParamDesc::Model>(uu.params.kind)) {
                const auto &out = uu.model->output(out_name);
                outm = ncvslideio::GMatDesc(toCV(out.get_element_type()),
                                    toCV(out.get_shape()));
            } else {
                GAPI_Assert(ncvslideio::util::holds_alternative<ParamDesc::CompiledModel>(uu.params.kind));
                const auto &out = uu.compiled_model.output(out_name);
                outm = ncvslideio::GMatDesc(toCV(out.get_element_type()),
                                    toCV(out.get_shape()));
            }
            result.emplace_back(std::move(outm));
        }

        return result;
    }

    static void run(std::shared_ptr<OVCallContext> ctx,
                    ncvslideio::gimpl::ov::RequestPool     &reqPool) {
        using namespace std::placeholders;
        if (ctx->getOptions().inference_only) {
            ncvslideio::util::throw_error(
                    std::logic_error("OV Backend: Inference only mode is not supported for InferROI!"));
        }
        reqPool.getIdleRequest()->execute(
            IInferExecutor::Task {
                [ctx](::ov::InferRequest &infer_request) {
                    GAPI_Assert(ctx->uu.params.num_in == 1);
                    const auto &input_name = ctx->uu.params.input_names[0];
                    auto input_tensor = infer_request.get_tensor(input_name);
                    const auto &shape = input_tensor.get_shape();
                    const auto &roi = ctx->inArg<ncvslideio::detail::OpaqueRef>(0).rref<ncvslideio::Rect>();
                    preprocess_and_copy(ctx, 1, roi, shape, input_tensor);
                },
                std::bind(PostOutputs, _1, _2, ctx)
            }
        );
    }
};

struct InferList: public ncvslideio::detail::KernelTag {
    using API = ncvslideio::GInferListBase;
    static ncvslideio::gapi::GBackend backend()  { return ncvslideio::gapi::ov::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run};     }

    static ncvslideio::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const ncvslideio::GMetaArgs   &in_metas,
                                 const ncvslideio::GArgs       &/*in_args*/) {
        GConstGOVModel gm(gr);
        const auto &uu = gm.metadata(nh).get<OVUnit>();
        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == (in_metas.size() - 1u)
                    && "Known input layers count doesn't match input meta count");

        // NB: Pre/Post processing configuration avaiable only for read models.
        if (ncvslideio::util::holds_alternative<ParamDesc::Model>(uu.params.kind)) {
            const auto &model_info = ncvslideio::util::get<ParamDesc::Model>(uu.params.kind);
            auto& model = const_cast<std::shared_ptr<::ov::Model>&>(uu.model);
            PrePostProcWrapper ppp {model, model_info,
                uu.params.input_names, uu.params.output_names};

            size_t idx = 1u;
            for (auto &&input_name : uu.params.input_names) {
                const auto &mm = in_metas[idx++];
                GAPI_Assert(ncvslideio::util::holds_alternative<ncvslideio::GMatDesc>(mm) ||
                            ncvslideio::util::holds_alternative<ncvslideio::GFrameDesc>(mm));
                const auto &input_shape = uu.model->input(input_name).get_shape();

                if (!isImage(mm, input_shape)) {
                    util::throw_error(std::runtime_error(
                        "OV Backend: Only image is supported"
                        " as the " + std::to_string(idx) + "th argument for InferList"));
                }

                ppp.cfgLayouts(input_name);
                ppp.cfgPreProcessing(input_name, mm, true /*disable_img_resize*/);
                ppp.cfgScaleMean(input_name, mm);
            }
            ppp.cfgPostProcessing();
            ppp.finalize();
        }

        // roi-list version is much easier at the moment.
        // All our outputs are vectors which don't have
        // metadata at the moment - so just create a vector of
        // "empty" array metadatas of the required size.
        return ncvslideio::GMetaArgs(uu.params.output_names.size(),
                             ncvslideio::GMetaArg{ncvslideio::empty_array_desc()});
    }

    static void run(std::shared_ptr<OVCallContext> ctx,
                    ncvslideio::gimpl::ov::RequestPool     &reqPool) {
        if (ctx->getOptions().inference_only) {
            ncvslideio::util::throw_error(
                    std::logic_error("OV Backend: Inference only mode is not supported for InferList!"));
        }
        const auto& in_roi_vec = ctx->inArg<ncvslideio::detail::VectorRef>(0u).rref<ncvslideio::Rect>();
        // NB: In case there is no input data need to post output anyway
        if (in_roi_vec.empty()) {
            for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
                auto output = ctx->output(i);
                ctx->out.meta(output, ctx->getMeta());
                ctx->out.post(std::move(output));
            }
            return;
        }

        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            // FIXME: Isn't this should be done automatically
            // by some resetInternalData(), etc? (Probably at the GExecutor level)
            auto& out_vec = ctx->outVecR<ncvslideio::Mat>(i);
            out_vec.clear();
            out_vec.resize(in_roi_vec.size());
        }

        PostOutputsList callback(in_roi_vec.size(), ctx);
        for (auto&& it : ade::util::indexed(in_roi_vec)) {
            const auto pos = ade::util::index(it);
            const auto &rc = ade::util::value(it);
            reqPool.getIdleRequest()->execute(
                IInferExecutor::Task {
                    [ctx, rc](::ov::InferRequest &infer_request) {
                        const auto &input_name = ctx->uu.params.input_names[0];
                        auto input_tensor = infer_request.get_tensor(input_name);
                        const auto &shape = input_tensor.get_shape();
                        preprocess_and_copy(ctx, 1, rc, shape, input_tensor);
                    },
                    std::bind(callback, std::placeholders::_1, std::placeholders::_2, pos)
                }
            );
        }
    }
};

struct InferList2: public ncvslideio::detail::KernelTag {
    using API = ncvslideio::GInferList2Base;
    static ncvslideio::gapi::GBackend backend()  { return ncvslideio::gapi::ov::backend(); }
    static KImpl kernel()                { return KImpl{outMeta, run}; }

    static ncvslideio::GMetaArgs outMeta(const ade::Graph      &gr,
                                 const ade::NodeHandle &nh,
                                 const ncvslideio::GMetaArgs   &in_metas,
                                 const ncvslideio::GArgs       &/*in_args*/) {
        GConstGOVModel gm(gr);
        const auto &uu = gm.metadata(nh).get<OVUnit>();
        // Initialize input information
        // Note our input layers list order matches the API order and so
        // meta order.
        GAPI_Assert(uu.params.input_names.size() == (in_metas.size() - 1u)
                    && "Known input layers count doesn't match input meta count");

        const auto &op = gm.metadata(nh).get<Op>();

        // In contrast to InferList, the InferList2 has only one
        // "full-frame" image argument, and all the rest are arrays of
        // ether ROI or blobs. So here we set the 0th arg image format
        // to all inputs which are ROI-based (skipping the
        // "blob"-based ones)
        // FIXME: this is filtering not done, actually! GArrayDesc has
        // no hint for its underlying type!

        const auto &input_name_0 = uu.params.input_names.front();
        const auto &mm_0 = in_metas[0u];

        if (!(ncvslideio::util::holds_alternative<ncvslideio::GMatDesc>(mm_0) ||
              ncvslideio::util::holds_alternative<ncvslideio::GFrameDesc>(mm_0))) {
            util::throw_error(std::runtime_error(
                        "OV Backend: Unsupported input meta"
                        " for 0th argument in OV backend"));
        }

        const bool is_model = ncvslideio::util::holds_alternative<ParamDesc::Model>(uu.params.kind);
        const auto &input_shape = is_model ? uu.model->input(input_name_0).get_shape()
                                           : uu.compiled_model.input(input_name_0).get_shape();
        if (!isImage(mm_0, input_shape)) {
            util::throw_error(std::runtime_error(
                "OV Backend: InferList2 supports only image as the 0th argument"));
        }

        if (is_model) {
            const auto &model_info = ncvslideio::util::get<ParamDesc::Model>(uu.params.kind);
            auto& model = const_cast<std::shared_ptr<::ov::Model>&>(uu.model);
            PrePostProcWrapper ppp {model, model_info,
                uu.params.input_names, uu.params.output_names};

            size_t idx = 1u;
            for (auto &&input_name : uu.params.input_names) {
                GAPI_Assert(util::holds_alternative<ncvslideio::GArrayDesc>(in_metas[idx])
                            && "Non-array inputs are not supported");

                ppp.cfgLayouts(input_name);
                if (op.k.inKinds[idx] == ncvslideio::detail::OpaqueKind::CV_RECT) {
                    ppp.cfgPreProcessing(input_name, mm_0, true /*disable_img_resize*/);
                } else {
                    // This is a ncvslideio::GMat (equals to: ncvslideio::Mat)
                    // Just validate that it is really the type
                    // (other types are prohibited here)
                    GAPI_Assert(op.k.inKinds[idx] == ncvslideio::detail::OpaqueKind::CV_MAT);
                }

                ppp.cfgScaleMean(input_name, mm_0);
                idx++; // NB: Never forget to increment the counter
            }
            ppp.cfgPostProcessing();
            ppp.finalize();
        }

        // roi-list version is much easier at the moment.
        // All our outputs are vectors which don't have
        // metadata at the moment - so just create a vector of
        // "empty" array metadatas of the required size.
        return ncvslideio::GMetaArgs(uu.params.output_names.size(),
                             ncvslideio::GMetaArg{ncvslideio::empty_array_desc()});
    }

    static void run(std::shared_ptr<OVCallContext> ctx,
                    ncvslideio::gimpl::ov::RequestPool     &reqPool) {
        if (ctx->getOptions().inference_only) {
            ncvslideio::util::throw_error(
                    std::logic_error("OV Backend: Inference only mode is not supported for InferList2!"));
        }
        GAPI_Assert(ctx->inArgs().size() > 1u
                && "This operation must have at least two arguments");
        // NB: This blob will be used to make roi from its, so
        // it should be treated as image
        const auto list_size = ctx->inArg<ncvslideio::detail::VectorRef>(1u).size();
        if (list_size == 0u) {
            for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
                auto output = ctx->output(i);
                ctx->out.meta(output, ctx->getMeta());
                ctx->out.post(std::move(output));
            }
            return;
        }

        for (auto i : ade::util::iota(ctx->uu.params.num_out)) {
            // FIXME: Isn't this should be done automatically
            // by some resetInternalData(), etc? (Probably at the GExecutor level)
            auto& out_vec = ctx->outVecR<ncvslideio::Mat>(i);
            out_vec.clear();
            out_vec.resize(list_size);
        }

        PostOutputsList callback(list_size, ctx);
        for (const auto &list_idx : ade::util::iota(list_size)) {
            reqPool.getIdleRequest()->execute(
                IInferExecutor::Task {
                    [ctx, list_idx, list_size](::ov::InferRequest &infer_request) {
                        for (auto in_idx : ade::util::iota(ctx->uu.params.num_in)) {
                            const auto &this_vec = ctx->inArg<ncvslideio::detail::VectorRef>(in_idx+1u);
                            GAPI_Assert(this_vec.size() == list_size);
                            const auto &input_name = ctx->uu.params.input_names[in_idx];
                            auto input_tensor = infer_request.get_tensor(input_name);
                            const auto &shape = input_tensor.get_shape();
                            if (this_vec.getKind() == ncvslideio::detail::OpaqueKind::CV_RECT) {
                                const auto &vec = this_vec.rref<ncvslideio::Rect>();
                                const auto roi_mat = preprocess(ctx->inMat(0), vec[list_idx], shape);
                                copyToOV(roi_mat, input_tensor);
                            } else if (this_vec.getKind() == ncvslideio::detail::OpaqueKind::CV_MAT) {
                                const auto &vec = this_vec.rref<ncvslideio::Mat>();
                                const auto &mat = vec[list_idx];
                                copyToOV(mat, input_tensor);
                            } else {
                                GAPI_Assert(false &&
                                        "OV Backend: Only Rect and Mat types are supported for InferList2");
                            }
                        }
                    },
                    std::bind(callback, std::placeholders::_1, std::placeholders::_2, list_idx)
                } // task
            );
        } // for
    }
};

} // namespace ov
} // namespace gimpl
} // namespace ncvslideio

// IE backend implementation of GBackend::Priv ///////////////////////
namespace {
class GOVBackendImpl final: public ncvslideio::gapi::GBackend::Priv {
    virtual void unpackKernel(ade::Graph            &gr,
                              const ade::NodeHandle &nh,
                              const ncvslideio::GKernelImpl &ii) override {
        using namespace ncvslideio::gimpl;
        // FIXME: Introduce a DNNBackend interface which'd specify
        // the framework for this???
        GOVModel gm(gr);
        auto &np = gm.metadata(nh).get<NetworkParams>();
        auto &pp = ncvslideio::util::any_cast<ParamDesc>(np.opaque);
        const auto &ki = ncvslideio::util::any_cast<KImpl>(ii.opaque);

        GModel::Graph model(gr);
        auto& op = model.metadata(nh).get<Op>();

        // NB: In case generic infer, info about in/out names is stored in operation (op.params)
        if (pp.is_generic)
        {
            auto& info      = ncvslideio::util::any_cast<ncvslideio::detail::InOutInfo>(op.params);
            pp.input_names  = info.in_names;
            pp.output_names = info.out_names;
            pp.num_in       = info.in_names.size();
            pp.num_out      = info.out_names.size();
        }

        gm.metadata(nh).set(OVUnit{pp});
        gm.metadata(nh).set(OVCallable{ki.run});
        gm.metadata(nh).set(CustomMetaFunction{ki.customMetaFunc});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const ncvslideio::GCompileArgs &compileArgs,
                         const std::vector<ade::NodeHandle> &nodes) const override {
        return EPtr{new ncvslideio::gimpl::ov::GOVExecutable(graph, compileArgs, nodes)};
    }

    virtual ncvslideio::GKernelPackage auxiliaryKernels() const override {
        return ncvslideio::gapi::kernels< ncvslideio::gimpl::ov::Infer
                                , ncvslideio::gimpl::ov::InferROI
                                , ncvslideio::gimpl::ov::InferList
                                , ncvslideio::gimpl::ov::InferList2 >();
    }

    virtual bool controlsMerge() const override {
        return true;
    }

    virtual bool allowsMerge(const ncvslideio::gimpl::GIslandModel::Graph &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &) const override {
        return false;
    }
};

} // anonymous namespace

ncvslideio::gapi::GBackend ncvslideio::gapi::ov::backend() {
    static ncvslideio::gapi::GBackend this_backend(std::make_shared<GOVBackendImpl>());
    return this_backend;
}

static std::vector<::ov::InferRequest>
createInferRequests(::ov::CompiledModel &compiled_model,
                    size_t              num_infer_requests) {
    std::vector<::ov::InferRequest> infer_requests;
    for (size_t i = 0; i < num_infer_requests; ++i) {
        infer_requests.push_back(compiled_model.create_infer_request());
    }
    return infer_requests;
}

// GOVExecutable implementation //////////////////////////////////////////////
ncvslideio::gimpl::ov::GOVExecutable::GOVExecutable(const ade::Graph &g,
                                            const ncvslideio::GCompileArgs &compileArgs,
                                            const std::vector<ade::NodeHandle> &nodes)
    : m_g(g), m_gm(m_g) {

    m_options.inference_only =
        ncvslideio::gapi::getCompileArg<ncvslideio::gapi::wip::ov::benchmark_mode>(compileArgs).has_value();
    // FIXME: Currently this backend is capable to run a single inference node only.
    // Need to extend our island fusion with merge/not-to-merge decision making parametrization
    GConstGOVModel ovm(g);

    for (auto &nh : nodes) {
        switch (m_gm.metadata(nh).get<NodeType>().t) {
        case NodeType::OP:
            if (this_nh == nullptr) {
                this_nh = nh;
                const auto &unit = ovm.metadata(this_nh).get<OVUnit>();
                compiled = const_cast<OVUnit&>(unit).compile();
                m_reqPool.reset(new RequestPool(createInferRequests(
                                compiled.compiled_model, unit.params.nireq)));
            }
            else
                util::throw_error(std::logic_error("Multi-node inference is not supported!"));
            break;

        case NodeType::DATA: {
            m_dataNodes.push_back(nh);
            const auto &desc = m_gm.metadata(nh).get<Data>();
            if (desc.storage == Data::Storage::CONST_VAL) {
                util::throw_error(std::logic_error("No const data please!"));
            }
            if (desc.storage == Data::Storage::INTERNAL) {
                util::throw_error(std::logic_error("No internal data please!"));
            }
            break;
        }
        default: util::throw_error(std::logic_error("Unsupported NodeType type"));
        }
    }
}

void ncvslideio::gimpl::ov::GOVExecutable::run(ncvslideio::gimpl::GIslandExecutable::IInput  &in,
                                       ncvslideio::gimpl::GIslandExecutable::IOutput &out) {
    std::vector<InObj>  input_objs;
    std::vector<OutObj> output_objs;

    const auto &in_desc = in.desc();
          auto  in_msg  = in.get();

    if (ncvslideio::util::holds_alternative<ncvslideio::gimpl::EndOfStream>(in_msg))
    {
        m_reqPool->waitAll();
        out.post(ncvslideio::gimpl::EndOfStream{});
        return;
    }

    GAPI_Assert(ncvslideio::util::holds_alternative<ncvslideio::GRunArgs>(in_msg));
    const auto in_vector = ncvslideio::util::get<ncvslideio::GRunArgs>(in_msg);
    ncvslideio::GRunArg::Meta stub_meta;
    for (auto &&in_arg : in_vector)
    {
        stub_meta.insert(in_arg.meta.begin(), in_arg.meta.end());
    }

    input_objs.reserve(in_desc.size());
    for (auto &&it: ade::util::zip(ade::util::toRange(in_desc),
                    ade::util::toRange(in_vector)))
    {
        input_objs.emplace_back(std::get<0>(it), std::get<1>(it));
    }

    const auto &out_desc = out.desc();
    output_objs.reserve(out_desc.size());
    for (auto &&it: ade::util::indexed(ade::util::toRange(out_desc)))
    {
        output_objs.emplace_back(ade::util::value(it),
                out.get(ade::util::checked_cast<int>(ade::util::index(it))));
    }

    GConstGOVModel giem(m_g);
    const auto &uu = giem.metadata(this_nh).get<OVUnit>();
    const auto &op = m_gm.metadata(this_nh).get<Op>();

    auto ctx = std::make_shared<OVCallContext>(uu, out, op.args, op.outs,
            std::move(stub_meta), std::move(input_objs), std::move(output_objs), m_options);

    const auto &kk = giem.metadata(this_nh).get<OVCallable>();

    try {
        kk.run(ctx, *m_reqPool);
    } catch (...) {
        auto eptr = std::current_exception();
        for (auto i : ade::util::iota(ctx->uu.params.num_out))
        {
            auto output = ctx->output(i);
            ctx->out.meta(output, ctx->getMeta());
            ctx->out.post(std::move(output), eptr);
        }
        return;
    }

    if (!m_gm.metadata().contains<Streaming>()) {
        m_reqPool->waitAll();
    }
}

#else // HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000

ncvslideio::gapi::GBackend ncvslideio::gapi::ov::backend() {
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("G-API has been compiled without OpenVINO support"));
}

#endif // HAVE_INF_ENGINE && INF_ENGINE_RELEASE >= 2022010000
