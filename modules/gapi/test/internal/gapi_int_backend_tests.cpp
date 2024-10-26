// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "../gapi_mock_kernels.hpp"

#include "compiler/gmodel.hpp"
#include "compiler/gcompiler.hpp"

namespace opencv_test {

namespace {

struct MockMeta
{
    static const char* name() { return "MockMeta"; }
};

class GMockBackendImpl final: public ncvslideio::gapi::GBackend::Priv
{
    virtual void unpackKernel(ade::Graph            &,
                              const ade::NodeHandle &,
                              const ncvslideio::GKernelImpl &) override
    {
        // Do nothing here
    }

    virtual EPtr compile(const ade::Graph &,
                         const ncvslideio::GCompileArgs &,
                         const std::vector<ade::NodeHandle> &) const override
    {
        // Do nothing here as well
        return {};
    }

    virtual void addBackendPasses(ade::ExecutionEngineSetupContext &ectx) override
    {
        ectx.addPass("transform", "set_mock_meta", [](ade::passes::PassContext &ctx) {
                ade::TypedGraph<MockMeta> me(ctx.graph);
                for (const auto &nh : me.nodes())
                {
                    me.metadata(nh).set(MockMeta{});
                }
            });
    }
};

static ncvslideio::gapi::GBackend mock_backend(std::make_shared<GMockBackendImpl>());

GAPI_OCV_KERNEL(MockFoo, I::Foo)
{
    static void run(const ncvslideio::Mat &, ncvslideio::Mat &) { /*Do nothing*/ }
    static ncvslideio::gapi::GBackend backend() { return mock_backend; } // FIXME: Must be removed
};

} // anonymous namespace

TEST(GBackend, CustomPassesExecuted)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out = I::Foo::on(in);
    ncvslideio::GComputation c(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    const auto pkg     = ncvslideio::gapi::kernels<MockFoo>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(c, {in_meta}, ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the metadata written by Mock backend
    ade::TypedGraph<MockMeta> me(*graph);
    EXPECT_LT(0u, static_cast<std::size_t>(me.nodes().size()));
    for (const auto &nh : me.nodes())
    {
        EXPECT_TRUE(me.metadata(nh).contains<MockMeta>());
    }
}

} // namespace opencv_test
