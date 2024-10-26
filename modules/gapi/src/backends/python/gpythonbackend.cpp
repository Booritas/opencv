// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <ade/util/zip_range.hpp> // zip_range, indexed

#include "compiler/gmodel.hpp"
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/util/throw.hpp> // throw_error
#include <opencv2/gapi/python/python.hpp>

#include "api/gbackend_priv.hpp"
#include "backends/common/gbackend.hpp"

ncvslideio::gapi::python::GPythonKernel::GPythonKernel(ncvslideio::gapi::python::Impl  runf,
                                               ncvslideio::gapi::python::Setup setupf)
    : run(runf), setup(setupf), is_stateful(setup != nullptr)
{
}

ncvslideio::gapi::python::GPythonFunctor::GPythonFunctor(const char* id,
                                                 const ncvslideio::gapi::python::GPythonFunctor::Meta& meta,
                                                 const ncvslideio::gapi::python::Impl& impl,
                                                 const ncvslideio::gapi::python::Setup& setup)
    : gapi::GFunctor(id), impl_{GPythonKernel{impl, setup}, meta}
{
}

ncvslideio::GKernelImpl ncvslideio::gapi::python::GPythonFunctor::impl() const
{
    return impl_;
}

ncvslideio::gapi::GBackend ncvslideio::gapi::python::GPythonFunctor::backend() const
{
    return ncvslideio::gapi::python::backend();
}

namespace {

struct PythonUnit
{
    static const char *name() { return "PythonUnit"; }
    ncvslideio::gapi::python::GPythonKernel kernel;
};

using PythonModel = ade::TypedGraph
    < ncvslideio::gimpl::Op
    , PythonUnit
    >;

using ConstPythonModel = ade::ConstTypedGraph
    < ncvslideio::gimpl::Op
    , PythonUnit
    >;

class GPythonExecutable final: public ncvslideio::gimpl::GIslandExecutable
{
    virtual void run(std::vector<InObj>  &&,
                     std::vector<OutObj> &&) override;

    virtual bool allocatesOutputs() const override { return true; }
    // Return an empty RMat since we will reuse the input.
    // There is no need to allocate and copy 4k image here.
    virtual ncvslideio::RMat allocate(const ncvslideio::GMatDesc&) const override { return {}; }

    virtual bool canReshape() const override { return true; }
    virtual void handleNewStream() override;
    virtual void reshape(ade::Graph&, const ncvslideio::GCompileArgs&) override {
        // Do nothing here
    }

public:
    GPythonExecutable(const ade::Graph                   &,
                      const std::vector<ade::NodeHandle> &);

    const ade::Graph& m_g;
    ncvslideio::gimpl::GModel::ConstGraph m_gm;
    ncvslideio::gapi::python::GPythonKernel m_kernel;
    ade::NodeHandle m_op;
    ncvslideio::GArg m_node_state;

    ncvslideio::GTypesInfo m_out_info;
    ncvslideio::GMetaArgs  m_in_metas;
    ncvslideio::gimpl::Mag m_res;
};

static ncvslideio::GArg packArg(ncvslideio::gimpl::Mag& m_res, const ncvslideio::GArg &arg)
{
    // No API placeholders allowed at this point
    // FIXME: this check has to be done somewhere in compilation stage.
    GAPI_Assert(   arg.kind != ncvslideio::detail::ArgKind::GMAT
                && arg.kind != ncvslideio::detail::ArgKind::GSCALAR
                && arg.kind != ncvslideio::detail::ArgKind::GARRAY
                && arg.kind != ncvslideio::detail::ArgKind::GOPAQUE
                && arg.kind != ncvslideio::detail::ArgKind::GFRAME);

    if (arg.kind != ncvslideio::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == ncvslideio::detail::ArgKind::GOBJREF);

    // Wrap associated CPU object (either host or an internal one)
    // FIXME: object can be moved out!!! GExecutor faced that.
    const ncvslideio::gimpl::RcDesc &ref = arg.get<ncvslideio::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case ncvslideio::GShape::GMAT:    return ncvslideio::GArg(m_res.slot<ncvslideio::Mat>()   [ref.id]);
    case ncvslideio::GShape::GSCALAR: return ncvslideio::GArg(m_res.slot<ncvslideio::Scalar>()[ref.id]);
    // Note: .at() is intentional for GArray and GOpaque as objects MUST be already there
    //   (and constructed by either bindIn/Out or resetInternal)
    case ncvslideio::GShape::GARRAY:  return ncvslideio::GArg(m_res.slot<ncvslideio::detail::VectorRef>().at(ref.id));
    case ncvslideio::GShape::GOPAQUE: return ncvslideio::GArg(m_res.slot<ncvslideio::detail::OpaqueRef>().at(ref.id));
    case ncvslideio::GShape::GFRAME:  return ncvslideio::GArg(m_res.slot<ncvslideio::MediaFrame>().at(ref.id));
    default:
        ncvslideio::util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

static void writeBack(ncvslideio::GRunArg& arg, ncvslideio::GRunArgP& out)
{
    switch (arg.index())
    {
        case ncvslideio::GRunArg::index_of<ncvslideio::Mat>():
        {
            auto& rmat = *ncvslideio::util::get<ncvslideio::RMat*>(out);
            rmat = ncvslideio::make_rmat<ncvslideio::gimpl::RMatOnMat>(ncvslideio::util::get<ncvslideio::Mat>(arg));
            break;
        }
        case ncvslideio::GRunArg::index_of<ncvslideio::Scalar>():
        {
            *ncvslideio::util::get<ncvslideio::Scalar*>(out) = ncvslideio::util::get<ncvslideio::Scalar>(arg);
            break;
        }
        case ncvslideio::GRunArg::index_of<ncvslideio::detail::OpaqueRef>():
        {
            auto& oref = ncvslideio::util::get<ncvslideio::detail::OpaqueRef>(arg);
            ncvslideio::util::get<ncvslideio::detail::OpaqueRef>(out).mov(oref);
            break;
        }
        case ncvslideio::GRunArg::index_of<ncvslideio::detail::VectorRef>():
        {
            auto& vref = ncvslideio::util::get<ncvslideio::detail::VectorRef>(arg);
            ncvslideio::util::get<ncvslideio::detail::VectorRef>(out).mov(vref);
            break;
        }
        default:
            GAPI_Error("Unsupported output type");
    }
}

void GPythonExecutable::handleNewStream()
{
    if (!m_kernel.is_stateful)
        return;

    m_node_state = m_kernel.setup(ncvslideio::gimpl::GModel::collectInputMeta(m_gm, m_op),
                                  m_gm.metadata(m_op).get<ncvslideio::gimpl::Op>().args);
}

void GPythonExecutable::run(std::vector<InObj>  &&input_objs,
                            std::vector<OutObj> &&output_objs)
{
    const auto &op = m_gm.metadata(m_op).get<ncvslideio::gimpl::Op>();
    for (auto& it : input_objs) ncvslideio::gimpl::magazine::bindInArg(m_res, it.first, it.second);

    using namespace std::placeholders;
    ncvslideio::GArgs inputs;
    ade::util::transform(op.args,
                         std::back_inserter(inputs),
                         std::bind(&packArg, std::ref(m_res), _1));

    ncvslideio::gapi::python::GPythonContext ctx{inputs, m_in_metas, m_out_info, /*state*/{}};

    // NB: For stateful kernel add state to its execution context
    if (m_kernel.is_stateful)
    {
        ctx.m_state = ncvslideio::optional<ncvslideio::GArg>(m_node_state);
    }

    auto outs = m_kernel.run(ctx);

    for (auto&& it : ade::util::zip(outs, output_objs))
    {
        writeBack(std::get<0>(it), std::get<1>(it).second);
    }
}

class GPythonBackendImpl final: public ncvslideio::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &graph,
            const ade::NodeHandle &op_node,
            const ncvslideio::GKernelImpl &impl) override
    {
        PythonModel gm(graph);
        const auto &kernel  = ncvslideio::util::any_cast<ncvslideio::gapi::python::GPythonKernel>(impl.opaque);
        gm.metadata(op_node).set(PythonUnit{kernel});
    }

    virtual EPtr compile(const ade::Graph &graph,
                         const ncvslideio::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &nodes) const override
    {
        return EPtr{new GPythonExecutable(graph, nodes)};
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

GPythonExecutable::GPythonExecutable(const ade::Graph& g,
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

    ConstPythonModel cag(m_g);

    m_op = *it;
    m_kernel = cag.metadata(m_op).get<PythonUnit>().kernel;

    // If kernel is stateful then prepare storage for its state.
    if (m_kernel.is_stateful)
    {
        m_node_state = ncvslideio::GArg{ };
    }

    // Ensure this the only op in the graph
    if (std::any_of(it+1, nodes.end(), is_op))
    {
        ncvslideio::util::throw_error
            (std::logic_error
             ("Internal error: Python subgraph has multiple operations"));
    }

    m_out_info.reserve(m_op->outEdges().size());
    for (const auto &e : m_op->outEdges())
    {
        const auto& out_data = m_gm.metadata(e->dstNode()).get<ncvslideio::gimpl::Data>();
        m_out_info.push_back(ncvslideio::GTypeInfo{out_data.shape, out_data.kind, out_data.ctor});
    }

    const auto& op = m_gm.metadata(m_op).get<ncvslideio::gimpl::Op>();
    m_in_metas.resize(op.args.size());
    GAPI_Assert(m_op->inEdges().size() > 0);
    for (const auto &in_eh : m_op->inEdges())
    {
        const auto& input_port = m_gm.metadata(in_eh).get<Input>().port;
        const auto& input_nh   = in_eh->srcNode();
        const auto& input_meta = m_gm.metadata(input_nh).get<Data>().meta;
        m_in_metas.at(input_port) = input_meta;
    }
}

} // anonymous namespace

ncvslideio::gapi::GBackend ncvslideio::gapi::python::backend()
{
    static ncvslideio::gapi::GBackend this_backend(std::make_shared<GPythonBackendImpl>());
    return this_backend;
}
