// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <mutex>

#if !defined(GAPI_STANDALONE)
#include <opencv2/imgproc.hpp>
#endif // !defined(GAPI_STANDALONE)

#include <opencv2/gapi/util/throw.hpp> // throw_error
#include <opencv2/gapi/streaming/format.hpp> // kernels

#include "logger.hpp"
#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"

#include "gstreamingbackend.hpp"
#include "gstreamingkernel.hpp"

namespace {

struct StreamingCreateFunction
{
    static const char *name() { return "StreamingCreateFunction";  }
    ncvslideio::gapi::streaming::CreateActorFunction createActorFunction;
};

using StreamingGraph = ade::TypedGraph
    < ncvslideio::gimpl::Op
    , StreamingCreateFunction
    >;

using ConstStreamingGraph = ade::ConstTypedGraph
    < ncvslideio::gimpl::Op
    , StreamingCreateFunction
    >;

class GStreamingIntrinExecutable final: public ncvslideio::gimpl::GIslandExecutable
{
    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override {
        GAPI_Error("Not implemented");
    }

    virtual void run(GIslandExecutable::IInput &in,
                     GIslandExecutable::IOutput &out) override;

    virtual bool allocatesOutputs() const override { return true; }
    // Return an empty RMat since we will reuse the input.
    // There is no need to allocate and copy 4k image here.
    virtual ncvslideio::RMat allocate(const ncvslideio::GMatDesc&) const override { return {}; }

    virtual bool canReshape() const override { return true; }
    virtual void reshape(ade::Graph&, const ncvslideio::GCompileArgs&) override {
        // Do nothing here
    }

public:
    GStreamingIntrinExecutable(const ade::Graph                   &,
                               const ncvslideio::GCompileArgs             &,
                               const std::vector<ade::NodeHandle> &);

    const ade::Graph& m_g;
    ncvslideio::gimpl::GModel::ConstGraph m_gm;
    ncvslideio::gapi::streaming::IActor::Ptr m_actor;
};

void GStreamingIntrinExecutable::run(GIslandExecutable::IInput  &in,
                                     GIslandExecutable::IOutput &out)
{
    m_actor->run(in, out);
}

class GStreamingBackendImpl final: public ncvslideio::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &graph,
                              const ade::NodeHandle &op_node,
                              const ncvslideio::GKernelImpl &impl) override
    {
        StreamingGraph gm(graph);
        const auto &kimpl  = ncvslideio::util::any_cast<ncvslideio::gapi::streaming::GStreamingKernel>(impl.opaque);
        gm.metadata(op_node).set(StreamingCreateFunction{kimpl.createActorFunction});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const ncvslideio::GCompileArgs &args,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{new GStreamingIntrinExecutable(graph, args, nodes)};
    }

    virtual bool controlsMerge() const override
    {
        return true;
    }

    virtual bool allowsMerge(const ncvslideio::gimpl::GIslandModel::Graph &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &,
                             const ade::NodeHandle &) const override
    {
        return false;
    }
};

GStreamingIntrinExecutable::GStreamingIntrinExecutable(const ade::Graph& g,
                                                       const ncvslideio::GCompileArgs& args,
                                                       const std::vector<ade::NodeHandle>& nodes)
    : m_g(g), m_gm(m_g)
{
    using namespace ncvslideio::gimpl;
    const auto is_op = [this](const ade::NodeHandle &nh)
    {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    auto it = std::find_if(nodes.begin(), nodes.end(), is_op);
    GAPI_Assert(it != nodes.end() && "No operators found for this island?!");

    ConstStreamingGraph cag(m_g);
    m_actor = cag.metadata(*it).get<StreamingCreateFunction>().createActorFunction(args);

    // Ensure this the only op in the graph
    if (std::any_of(it+1, nodes.end(), is_op))
    {
        ncvslideio::util::throw_error
            (std::logic_error
             ("Internal error: Streaming subgraph has multiple operations"));
    }
}

} // anonymous namespace

ncvslideio::gapi::GBackend ncvslideio::gapi::streaming::backend()
{
    static ncvslideio::gapi::GBackend this_backend(std::make_shared<GStreamingBackendImpl>());
    return this_backend;
}

struct Copy: public ncvslideio::detail::KernelTag
{
    using API = ncvslideio::gimpl::streaming::GCopy;

    static ncvslideio::gapi::GBackend backend() { return ncvslideio::gapi::streaming::backend(); }

    class Actor final: public ncvslideio::gapi::streaming::IActor
    {
        public:
            explicit Actor(const ncvslideio::GCompileArgs&) {}
            virtual void run(ncvslideio::gimpl::GIslandExecutable::IInput  &in,
                             ncvslideio::gimpl::GIslandExecutable::IOutput &out) override;
    };

    static ncvslideio::gapi::streaming::IActor::Ptr create(const ncvslideio::GCompileArgs& args)
    {
        return ncvslideio::gapi::streaming::IActor::Ptr(new Actor(args));
    }

    static ncvslideio::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void Copy::Actor::run(ncvslideio::gimpl::GIslandExecutable::IInput  &in,
                      ncvslideio::gimpl::GIslandExecutable::IOutput &out)
{
    const auto in_msg = in.get();
    if (ncvslideio::util::holds_alternative<ncvslideio::gimpl::EndOfStream>(in_msg))
    {
        out.post(ncvslideio::gimpl::EndOfStream{});
        return;
    }

    GAPI_DbgAssert(ncvslideio::util::holds_alternative<ncvslideio::GRunArgs>(in_msg));
    const ncvslideio::GRunArgs &in_args = ncvslideio::util::get<ncvslideio::GRunArgs>(in_msg);
    GAPI_Assert(in_args.size() == 1u);

    const auto& in_arg = in_args[0];
    auto out_arg = out.get(0);
    using ncvslideio::util::get;
    switch (in_arg.index()) {
    case ncvslideio::GRunArg::index_of<ncvslideio::RMat>():
        *get<ncvslideio::RMat*>(out_arg) = get<ncvslideio::RMat>(in_arg);
        break;
    case ncvslideio::GRunArg::index_of<ncvslideio::MediaFrame>():
        *get<ncvslideio::MediaFrame*>(out_arg) = get<ncvslideio::MediaFrame>(in_arg);
        break;
    // FIXME: Add support for remaining types
    default:
        GAPI_Error("Copy: unsupported data type");
    }
    out.meta(out_arg, in_arg.meta);
    out.post(std::move(out_arg));
}

ncvslideio::GKernelPackage ncvslideio::gimpl::streaming::kernels()
{
    return ncvslideio::gapi::kernels<Copy>();
}

#if !defined(GAPI_STANDALONE)

class GAccessorActorBase : public ncvslideio::gapi::streaming::IActor {
public:
    explicit GAccessorActorBase(const ncvslideio::GCompileArgs&) {}
    virtual void run(ncvslideio::gimpl::GIslandExecutable::IInput  &in,
                     ncvslideio::gimpl::GIslandExecutable::IOutput &out) override {
        const auto in_msg = in.get();
        if (ncvslideio::util::holds_alternative<ncvslideio::gimpl::EndOfStream>(in_msg))
        {
            out.post(ncvslideio::gimpl::EndOfStream{});
            return;
        }

        GAPI_Assert(ncvslideio::util::holds_alternative<ncvslideio::GRunArgs>(in_msg));
        const ncvslideio::GRunArgs &in_args = ncvslideio::util::get<ncvslideio::GRunArgs>(in_msg);
        GAPI_Assert(in_args.size() == 1u);
        auto frame = ncvslideio::util::get<ncvslideio::MediaFrame>(in_args[0]);

        ncvslideio::GRunArgP out_arg = out.get(0);
        auto& rmat = *ncvslideio::util::get<ncvslideio::RMat*>(out_arg);

        extractRMat(frame, rmat);

        out.meta(out_arg, in_args[0].meta);
        out.post(std::move(out_arg));
    }

    virtual void extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat) = 0;

protected:
    std::once_flag m_warnFlag;
};

struct GOCVBGR: public ncvslideio::detail::KernelTag
{
    using API = ncvslideio::gapi::streaming::GBGR;
    static ncvslideio::gapi::GBackend backend() { return ncvslideio::gapi::streaming::backend(); }

    class Actor final: public GAccessorActorBase
    {
    public:
        using GAccessorActorBase::GAccessorActorBase;
        virtual void extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat) override;
    };

    static ncvslideio::gapi::streaming::IActor::Ptr create(const ncvslideio::GCompileArgs& args)
    {
        return ncvslideio::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static ncvslideio::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void GOCVBGR::Actor::extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat)
{
    const auto& desc = frame.desc();
    switch (desc.fmt)
    {
        case ncvslideio::MediaFormat::BGR:
        {
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatMediaFrameAdapter>(frame,
            [](const ncvslideio::GFrameDesc& d){ return ncvslideio::GMatDesc(CV_8U, 3, d.size); },
            [](const ncvslideio::GFrameDesc& d, const ncvslideio::MediaFrame::View& v){
                return ncvslideio::Mat(d.size, CV_8UC3, v.ptr[0], v.stride[0]);
            });
            break;
        }
        case ncvslideio::MediaFormat::NV12:
        {
            std::call_once(m_warnFlag,
                [](){
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from NV12 to BGR will happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve ncvslideio::Mat-s from NV12 ncvslideio::MediaFrame for free, you may use "
                        "ncvslideio::gapi::streaming::Y and ncvslideio::gapi::streaming::UV accessors.\n");
                });

            ncvslideio::Mat bgr;
            auto view = frame.access(ncvslideio::MediaFrame::Access::R);
            ncvslideio::Mat y_plane (desc.size,     CV_8UC1, view.ptr[0], view.stride[0]);
            ncvslideio::Mat uv_plane(desc.size / 2, CV_8UC2, view.ptr[1], view.stride[1]);
            ncvslideio::cvtColorTwoPlane(y_plane, uv_plane, bgr, ncvslideio::COLOR_YUV2BGR_NV12);
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatOnMat>(bgr);
            break;
        }
        case ncvslideio::MediaFormat::GRAY:
        {
            std::call_once(m_warnFlag,
                []() {
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from GRAY to BGR will happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve ncvslideio::Mat from GRAY ncvslideio::MediaFrame for free, you may use "
                        "ncvslideio::gapi::streaming::Y.\n");
                });
            ncvslideio::Mat bgr;
            auto view = frame.access(ncvslideio::MediaFrame::Access::R);
            ncvslideio::Mat gray(desc.size, CV_8UC1, view.ptr[0], view.stride[0]);
            ncvslideio::cvtColor(gray, bgr, ncvslideio::COLOR_GRAY2BGR);
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatOnMat>(bgr);
            break;
        }

        default:
            ncvslideio::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for ncvslideio::gapi::streaming::BGR"));
    }
}

struct GOCVY: public ncvslideio::detail::KernelTag
{
    using API = ncvslideio::gapi::streaming::GY;
    static ncvslideio::gapi::GBackend backend() { return ncvslideio::gapi::streaming::backend(); }

    class Actor final: public GAccessorActorBase
    {
    public:
        using GAccessorActorBase::GAccessorActorBase;
        virtual void extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat) override;
    };

    static ncvslideio::gapi::streaming::IActor::Ptr create(const ncvslideio::GCompileArgs& args)
    {
        return ncvslideio::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static ncvslideio::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void GOCVY::Actor::extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat)
{
    const auto& desc = frame.desc();
    switch (desc.fmt)
    {
        case ncvslideio::MediaFormat::BGR:
        {
            std::call_once(m_warnFlag,
                [](){
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from BGR to NV12 Y plane will "
                        "happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve ncvslideio::Mat from BGR ncvslideio::MediaFrame for free, you may use "
                        "ncvslideio::gapi::streaming::BGR accessor.\n");
                });

            auto view = frame.access(ncvslideio::MediaFrame::Access::R);
            ncvslideio::Mat tmp_bgr(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            ncvslideio::Mat yuv;
            cvtColor(tmp_bgr, yuv, ncvslideio::COLOR_BGR2YUV_I420);
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatOnMat>(yuv.rowRange(0, desc.size.height));
            break;
        }
        case ncvslideio::MediaFormat::NV12:
        {
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatMediaFrameAdapter>(frame,
            [](const ncvslideio::GFrameDesc& d){ return ncvslideio::GMatDesc(CV_8U, 1, d.size); },
            [](const ncvslideio::GFrameDesc& d, const ncvslideio::MediaFrame::View& v){
                return ncvslideio::Mat(d.size, CV_8UC1, v.ptr[0], v.stride[0]);
            });
            break;
        }
        case ncvslideio::MediaFormat::GRAY:
        {
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatMediaFrameAdapter>(frame,
            [](const ncvslideio::GFrameDesc& d) { return ncvslideio::GMatDesc(CV_8U, 1, d.size); },
            [](const ncvslideio::GFrameDesc& d, const ncvslideio::MediaFrame::View& v) {
                return ncvslideio::Mat(d.size, CV_8UC1, v.ptr[0], v.stride[0]);
            });
            break;
        }
        default:
            ncvslideio::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for ncvslideio::gapi::streaming::Y"));
    }
}

struct GOCVUV: public ncvslideio::detail::KernelTag
{
    using API = ncvslideio::gapi::streaming::GUV;
    static ncvslideio::gapi::GBackend backend() { return ncvslideio::gapi::streaming::backend(); }

    class Actor final: public GAccessorActorBase
    {
    public:
        using GAccessorActorBase::GAccessorActorBase;
        virtual void extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat) override;
    };

    static ncvslideio::gapi::streaming::IActor::Ptr create(const ncvslideio::GCompileArgs& args)
    {
        return ncvslideio::gapi::streaming::IActor::Ptr(new Actor(args));
    }
    static ncvslideio::gapi::streaming::GStreamingKernel kernel() { return {&create}; }
};

void GOCVUV::Actor::extractRMat(const ncvslideio::MediaFrame& frame, ncvslideio::RMat& rmat)
{
    const auto& desc = frame.desc();
    switch (desc.fmt)
    {
        case ncvslideio::MediaFormat::BGR:
        {
            std::call_once(m_warnFlag,
                [](){
                    GAPI_LOG_WARNING(NULL, "\nOn-the-fly conversion from BGR to NV12 UV plane will "
                        "happen.\n"
                        "Conversion may cost a lot for images with high resolution.\n"
                        "To retrieve ncvslideio::Mat from BGR ncvslideio::MediaFrame for free, you may use "
                        "ncvslideio::gapi::streaming::BGR accessor.\n");
                });

            auto view = frame.access(ncvslideio::MediaFrame::Access::R);

            ncvslideio::Mat tmp_bgr(desc.size, CV_8UC3, view.ptr[0], view.stride[0]);
            ncvslideio::Mat yuv;
            cvtColor(tmp_bgr, yuv, ncvslideio::COLOR_BGR2YUV_I420);

            ncvslideio::Mat uv;
            std::vector<int> dims = { desc.size.height / 2,
                                        desc.size.width / 2  };
            auto start = desc.size.height;
            auto range_h = desc.size.height / 4;
            std::vector<ncvslideio::Mat> uv_planes = {
                yuv.rowRange(start, start + range_h).reshape(0, dims),
                yuv.rowRange(start + range_h, start + range_h * 2).reshape(0, dims)
            };
            ncvslideio::merge(uv_planes, uv);
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatOnMat>(uv);
            break;
        }
        case ncvslideio::MediaFormat::NV12:
        {
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatMediaFrameAdapter>(frame,
            [](const ncvslideio::GFrameDesc& d){ return ncvslideio::GMatDesc(CV_8U, 2, d.size / 2); },
            [](const ncvslideio::GFrameDesc& d, const ncvslideio::MediaFrame::View& v){
                return ncvslideio::Mat(d.size / 2, CV_8UC2, v.ptr[1], v.stride[1]);
            });
            break;
        }
        case ncvslideio::MediaFormat::GRAY:
        {
            ncvslideio::Mat uv(desc.size / 2, CV_8UC2, ncvslideio::Scalar::all(127));
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatOnMat>(uv);
            break;
        }
        default:
            ncvslideio::util::throw_error(
                    std::logic_error("Unsupported MediaFormat for ncvslideio::gapi::streaming::UV"));
    }
}

ncvslideio::GKernelPackage ncvslideio::gapi::streaming::kernels()
{
    return ncvslideio::gapi::kernels<GOCVBGR, GOCVY, GOCVUV>();
}

#else

ncvslideio::GKernelPackage ncvslideio::gapi::streaming::kernels()
{
    // Still provide this symbol to avoid linking issues
    util::throw_error(std::runtime_error("ncvslideio::gapi::streaming::kernels() isn't supported in standalone"));
}

#endif // !defined(GAPI_STANDALONE)

ncvslideio::GMat ncvslideio::gapi::copy(const ncvslideio::GMat& in) {
    return ncvslideio::gimpl::streaming::GCopy::on<ncvslideio::GMat>(in);
}

ncvslideio::GFrame ncvslideio::gapi::copy(const ncvslideio::GFrame& in) {
    return ncvslideio::gimpl::streaming::GCopy::on<ncvslideio::GFrame>(in);
}
