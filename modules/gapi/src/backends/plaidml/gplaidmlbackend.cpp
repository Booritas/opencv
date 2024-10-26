// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2020 Intel Corporation


#ifdef HAVE_PLAIDML

#include "precomp.hpp"

#include <ade/util/algorithm.hpp>
#include <ade/util/range.hpp>
#include <ade/util/zip_range.hpp>
#include <ade/typed_graph.hpp>

#include <opencv2/gapi/gcommon.hpp>
#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/plaidml/plaidml.hpp>

#include "compiler/gobjref.hpp"
#include "compiler/gmodel.hpp"

#include "backends/plaidml/gplaidmlbackend.hpp"
#include "backends/plaidml/plaidml_util.hpp"

#include "api/gbackend_priv.hpp" // FIXME: Make it part of Backend SDK!

using GPlaidMLModel = ade::TypedGraph
    < ncvslideio::gimpl::PlaidMLUnit
    , ncvslideio::gimpl::Protocol
    >;

// FIXME: Same issue with Typed and ConstTyped
using GConstGPlaidMLModel = ade::ConstTypedGraph
    < ncvslideio::gimpl::PlaidMLUnit
    , ncvslideio::gimpl::Protocol
    >;

namespace
{
    class GPlaidMLBackendImpl final: public ncvslideio::gapi::GBackend::Priv
    {
        virtual void unpackKernel(ade::Graph            &graph,
                                  const ade::NodeHandle &op_node,
                                  const ncvslideio::GKernelImpl &impl) override
        {
            GPlaidMLModel gm(graph);
            auto plaidml_impl = ncvslideio::util::any_cast<ncvslideio::GPlaidMLKernel>(impl.opaque);
            gm.metadata(op_node).set(ncvslideio::gimpl::PlaidMLUnit{plaidml_impl});
        }

        virtual EPtr compile(const ade::Graph& graph,
                             const ncvslideio::GCompileArgs& args,
                             const std::vector<ade::NodeHandle>& nodes,
                             const std::vector<ncvslideio::gimpl::Data>& ins_data,
                             const std::vector<ncvslideio::gimpl::Data>& outs_data) const override
        {
            auto has_config = ncvslideio::gapi::getCompileArg<ncvslideio::gapi::plaidml::config>(args);

            if (!has_config)
            {
                ncvslideio::util::throw_error(std::runtime_error("Config not found!\n"
                                                         "You must pass ncvslideio::gapi::plaidml::config to the graph compile arguments"));
            }

            const auto& arg = has_config.value();
            return EPtr{new ncvslideio::gimpl::GPlaidMLExecutable(ncvslideio::gimpl::GPlaidMLExecutable::Config{arg.dev_id, arg.trg_id},
                                                          graph, nodes, ins_data, outs_data)};
        }
   };
}

ncvslideio::gapi::GBackend ncvslideio::gapi::plaidml::backend()
{
    static ncvslideio::gapi::GBackend this_backend(std::make_shared<GPlaidMLBackendImpl>());
    return this_backend;
}

void ncvslideio::gimpl::GPlaidMLExecutable::initBuffers(const std::vector<ncvslideio::gimpl::Data>& data,
                                                std::vector<plaidml::exec::Binding>& bindings)
{

    // NB: This is necessary because we keep a pointer to bindings elements to buffer_map
    // In order to them to remain valid it's required to prevant reallocation
    bindings.reserve(data.size());
    for (const auto& d : data)
    {
        GAPI_Assert(d.shape == GShape::GMAT &&
                    "Now PlaidML backend supports only ncvslideio::GMat's");

        const auto& desc = ncvslideio::util::get<ncvslideio::GMatDesc>(d.meta);

        auto placeholder = plaidml::edsl::Placeholder(
                           ncvslideio::util::plaidml::depth_from_ocv(desc.depth),
                           {desc.size.width, desc.size.height, desc.chan});

        const auto& shape = placeholder.shape();
        plaidml::TensorShape tshape(shape.dtype(), shape.int_dims());
        plaidml::Buffer buffer(m_cfg.dev_id, tshape);

        bindings.push_back(plaidml::exec::Binding{std::move(placeholder),
                                                  std::move(buffer)});

        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        // FIXME Avoid Copy here !!!
        tensor_map.emplace(d.rc, bindings.back().tensor);

        auto& buffer_map = m_res.slot<plaidml::Buffer*>();
        buffer_map.emplace(d.rc, &(bindings.back().buffer));
    }
}

void ncvslideio::gimpl::GPlaidMLExecutable::compile(const std::vector<ncvslideio::gimpl::Data>& ins_data,
                                            const std::vector<ncvslideio::gimpl::Data>& outs_data)
{
    initBuffers(ins_data,  input_bindings_);
    initBuffers(outs_data, output_bindings_);

    ade::util::transform(outs_data, std::back_inserter(output_ids_),
                         [](const ncvslideio::gimpl::Data& d) { return d.rc; });

    GConstGPlaidMLModel gcm(m_g);
    for (const auto& nh : m_all_ops)
    {
        const auto& k = gcm.metadata(nh).get<PlaidMLUnit>().k;
        GPlaidMLContext ctx;

        const auto &op = m_gm.metadata(nh).get<Op>();
        ctx.m_args.reserve(op.args.size());

        using namespace std::placeholders;
        ade::util::transform(op.args,
                std::back_inserter(ctx.m_args),
                std::bind(&GPlaidMLExecutable::packArg, this, _1));

        for (const auto &out_it : ade::util::indexed(op.outs))
        {
            const auto out_port  = ade::util::index(out_it);
            const auto out_desc  = ade::util::value(out_it);

            auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();

            // NB: Create tensor if need
            auto& tensor = tensor_map[out_desc.id];
            ctx.m_results[out_port] = GArg(&(tensor));
        }

        k.apply(ctx);
    }

    std::vector<plaidml::edsl::Tensor> output_tensors;
    for (const auto& out_id : output_ids_)
    {
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        // FIXME Avoid copy here !!!
        output_tensors.emplace_back(tensor_map[out_id]);
    }

    plaidml::edsl::Program program("Program", output_tensors);
    binder_.reset(new plaidml::exec::Binder(program));

    for (const auto& binding : input_bindings_)
    {
        binder_->set_input(binding.tensor, binding.buffer);
    }

    for (const auto& binding : output_bindings_)
    {
        binder_->set_output(binding.tensor, binding.buffer);
    }

    exec_ = binder_->compile();
}

ncvslideio::gimpl::GPlaidMLExecutable::GPlaidMLExecutable(ncvslideio::gimpl::GPlaidMLExecutable::Config cfg,
                                                  const ade::Graph& g,
                                                  const std::vector<ade::NodeHandle>& nodes,
                                                  const std::vector<ncvslideio::gimpl::Data>& ins_data,
                                                  const std::vector<ncvslideio::gimpl::Data>& outs_data)
    : m_cfg(std::move(cfg)), m_g(g), m_gm(m_g)
{
    auto is_op = [&](ade::NodeHandle nh) {
        return m_gm.metadata(nh).get<NodeType>().t == NodeType::OP;
    };

    std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(m_all_ops), is_op);

    compile(ins_data, outs_data);
}

void ncvslideio::gimpl::GPlaidMLExecutable::run(std::vector<InObj>  &&input_objs,
                                        std::vector<OutObj> &&output_objs)
{
    for (auto& it : input_objs) bindInArg (it.first, it.second);

    exec_->run();

    for (auto& it : output_objs) bindOutArg(it.first, it.second);
}

void ncvslideio::gimpl::GPlaidMLExecutable::bindInArg(const RcDesc &rc, const GRunArg  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        auto it = tensor_map.find(rc.id);
        GAPI_Assert(it != tensor_map.end());

        switch (arg.index())
        {
        case GRunArg::index_of<ncvslideio::RMat>():
        {
            auto& rmat = ncvslideio::util::get<ncvslideio::RMat>(arg);
            auto  view = rmat.access(ncvslideio::RMat::Access::R);
            auto  mat  = ncvslideio::gimpl::asMat(view);
            binder_->input(it->second).copy_from(mat.data);
        }
        break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
    }
    break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

void ncvslideio::gimpl::GPlaidMLExecutable::bindOutArg(const RcDesc &rc, const GRunArgP  &arg)
{
    switch (rc.shape)
    {
    case GShape::GMAT:
    {
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        auto it = tensor_map.find(rc.id);
        GAPI_Assert(it != tensor_map.end());

        switch (arg.index())
        {
        case GRunArgP::index_of<ncvslideio::RMat*>() :
        {
            auto& rmat = *ncvslideio::util::get<ncvslideio::RMat*>(arg);
            auto  view = rmat.access(ncvslideio::RMat::Access::W);
            auto  mat  = ncvslideio::gimpl::asMat(view);
            binder_->output(it->second).copy_into(mat.data);
        }
        break;
        default: util::throw_error(std::logic_error("content type of the runtime argument does not match to resource description ?"));
        }
    }
    break;

    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
    }
}

ncvslideio::GArg ncvslideio::gimpl::GPlaidMLExecutable::packArg(const GArg &arg)
{
    GAPI_Assert(   arg.kind != ncvslideio::detail::ArgKind::GMAT
              && arg.kind != ncvslideio::detail::ArgKind::GSCALAR
              && arg.kind != ncvslideio::detail::ArgKind::GARRAY
              && arg.kind != ncvslideio::detail::ArgKind::GOPAQUE);

    if (arg.kind != ncvslideio::detail::ArgKind::GOBJREF)
    {
        // All other cases - pass as-is, with no transformations to GArg contents.
        return arg;
    }
    GAPI_Assert(arg.kind == ncvslideio::detail::ArgKind::GOBJREF);

    const ncvslideio::gimpl::RcDesc &ref = arg.get<ncvslideio::gimpl::RcDesc>();
    switch (ref.shape)
    {
    case GShape::GMAT:
    {
        auto& tensor_map = m_res.slot<plaidml::edsl::Tensor>();
        auto it = tensor_map.find(ref.id);
        GAPI_Assert(it != tensor_map.end());
        return GArg(it->second);
    }
    break;
    default:
        util::throw_error(std::logic_error("Unsupported GShape type"));
        break;
    }
}

#endif // HAVE_PLAIDML
