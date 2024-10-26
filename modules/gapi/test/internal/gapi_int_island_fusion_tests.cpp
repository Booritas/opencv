// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include "compiler/transactions.hpp"

#include "../gapi_mock_kernels.hpp"

#include "compiler/gislandmodel.hpp"
#include "compiler/gcompiler.hpp"
#include "compiler/gmodel_priv.hpp"

namespace opencv_test
{

TEST(IslandFusion, TwoOps_OneIsland)
{
    namespace J = Jupiter; // see mock_kernels.cpp

    // Define a computation:
    //
    //    (in) -> J::Foo1 -> (tmp0) -> J::Foo2 -> (out)
    //          :                               :
    //          :          "island0"            :
    //          :<----------------------------->:

    ncvslideio::GMat in;
    ncvslideio::GMat tmp0 = I::Foo::on(in);
    ncvslideio::GMat out  = I::Foo::on(tmp0);
    ncvslideio::GComputation cc(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    const auto pkg     = ncvslideio::gapi::kernels<J::Foo>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, {in_meta}, ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);
    ncvslideio::gimpl::GModel::ConstLayoutGraph glm(*graph);

    auto in_nh  = ncvslideio::gimpl::GModel::dataNodeOf(glm, in);
    auto tmp_nh = ncvslideio::gimpl::GModel::dataNodeOf(glm, tmp0);
    auto out_nh = ncvslideio::gimpl::GModel::dataNodeOf(glm, out);

    // in/out mats shouldn't be assigned to any Island
    EXPECT_FALSE(gm.metadata(in_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<ncvslideio::gimpl::Island>());

    // Since tmp is surrounded by two J kernels, tmp should be assigned
    // to island J
    EXPECT_TRUE(gm.metadata(tmp_nh).contains<ncvslideio::gimpl::Island>());
}

TEST(IslandFusion, TwoOps_TwoIslands)
{
    namespace J = Jupiter; // see mock_kernels.cpp
    namespace S = Saturn;  // see mock_kernels.cpp

    // Define a computation:
    //
    //    (in) -> J::Foo --> (tmp0) -> S::Bar --> (out)
    //          :          :        ->          :
    //          :          :         :          :
    //          :<-------->:         :<-------->:

    ncvslideio::GMat in;
    ncvslideio::GMat tmp0 = I::Foo::on(in);
    ncvslideio::GMat out  = I::Bar::on(tmp0, tmp0);
    ncvslideio::GComputation cc(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    const auto pkg     = ncvslideio::gapi::kernels<J::Foo, S::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, {in_meta}, ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh  = ncvslideio::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto out_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out);

    // in/tmp/out mats shouldn't be assigned to any Island
    EXPECT_FALSE(gm.metadata(in_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp_nh).contains<ncvslideio::gimpl::Island>());

    auto isl_model = gm.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    // There should be two islands in the GIslandModel
    const auto is_island = [&](ade::NodeHandle nh) {
        return (ncvslideio::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<ncvslideio::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(2u, num_isl);

    auto isl_foo_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, tmp_nh);
    auto isl_bar_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, out_nh);
    ASSERT_NE(nullptr, isl_foo_nh);
    ASSERT_NE(nullptr, isl_bar_nh);

    // Islands should be different
    auto isl_foo_obj = gim.metadata(isl_foo_nh).get<ncvslideio::gimpl::FusedIsland>().object;
    auto isl_bar_obj = gim.metadata(isl_bar_nh).get<ncvslideio::gimpl::FusedIsland>().object;
    EXPECT_FALSE(isl_foo_obj == isl_bar_obj);
}

TEST(IslandFusion, ConsumerHasTwoInputs)
{
    namespace J = Jupiter; // see mock_kernels.cpp

    // Define a computation:     island
    //            ............................
    //    (in0) ->:J::Foo -> (tmp) -> S::Bar :--> (out)
    //            :....................^.....:
    //                                 |
    //    (in1) -----------------------`
    //

    // Check that island is build correctly, when consumer has two inputs

    GMat in[2];
    GMat tmp = I::Foo::on(in[0]);
    GMat out = I::Bar::on(tmp, in[1]);

    ncvslideio::GComputation cc(ncvslideio::GIn(in[0], in[1]), ncvslideio::GOut(out));

    // Prepare compilation parameters manually
    ncvslideio::GMetaArgs in_metas = {GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)}),
                              GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)})};
    const auto pkg = ncvslideio::gapi::kernels<J::Foo, J::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, std::move(in_metas), ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);

    auto in0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, in[0]);
    auto in1_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, in[1]);
    auto tmp_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp);
    auto out_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out);

    EXPECT_FALSE(gm.metadata(in0_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_TRUE(gm.metadata(tmp_nh).contains<ncvslideio::gimpl::Island>());

    auto isl_model = gm.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    const auto is_island = [&](ade::NodeHandle nh) {
        return (ncvslideio::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<ncvslideio::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(1u, num_isl);

    auto isl_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, out_nh);
    auto isl_obj = gim.metadata(isl_nh).get<ncvslideio::gimpl::FusedIsland>().object;

    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp_nh));

    EXPECT_EQ(2u, static_cast<std::size_t>(isl_nh->inNodes().size()));
    EXPECT_EQ(1u, static_cast<std::size_t>(isl_nh->outNodes().size()));
}

TEST(IslandFusion, DataNodeUsedDifferentBackend)
{
    // Define a computation:
    //
    //           internal isl            isl0
    //             ...........................
    //    (in1) -> :J::Foo--> (tmp) -> J::Foo: --> (out0)
    //             :............|............:
    //                          |     ........
    //                          `---->:S::Baz: --> (out1)
    //                                :......:

    // Check that the node was not dropped out of the island
    // because it is used by the kernel from another backend

    namespace J = Jupiter;
    namespace S = Saturn;

    ncvslideio::GMat in, tmp, out0;
    ncvslideio::GScalar out1;
    tmp  = I::Foo::on(in);
    out0 = I::Foo::on(tmp);
    out1 = I::Baz::on(tmp);

    ncvslideio::GComputation cc(ncvslideio::GIn(in), ncvslideio::GOut(out0, out1));

    // Prepare compilation parameters manually
    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    const auto pkg     = ncvslideio::gapi::kernels<J::Foo, S::Baz>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, {in_meta}, ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh   = ncvslideio::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp_nh  = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp);
    auto out0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out0);
    auto out1_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out1);

    EXPECT_TRUE(gm.metadata(tmp_nh).contains<ncvslideio::gimpl::Island>());

    auto isl_model = gm.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    auto isl_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, tmp_nh);
    auto isl_obj = gim.metadata(isl_nh).get<ncvslideio::gimpl::FusedIsland>().object;

    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp_nh));

    EXPECT_EQ(2u, static_cast<std::size_t>(isl_nh->outNodes().size()));
    EXPECT_EQ(7u, static_cast<std::size_t>(gm.nodes().size()));
    EXPECT_EQ(6u, static_cast<std::size_t>(gim.nodes().size()));
}

TEST(IslandFusion, LoopBetweenDifferentBackends)
{
    // Define a computation:
    //
    //
    //            .............................
    //    (in) -> :J::Baz -> (tmp0) -> J::Quux: -> (out0)
    //      |     :............|..........^....
    //      |     ........     |          |         ........
    //      `---->:S::Foo:     `----------|-------->:S::Qux:-> (out1)
    //            :....|.:                |         :....^.:
    //                 |                  |              |
    //                 `-------------- (tmp1) -----------`

    // Kernels S::Foo and S::Qux cannot merge, because there will be a cycle between islands

    namespace J = Jupiter;
    namespace S = Saturn;

    ncvslideio::GScalar tmp0;
    ncvslideio::GMat in, tmp1, out0, out1;

    tmp0 = I::Baz::on(in);
    tmp1 = I::Foo::on(in);
    out1 = I::Qux::on(tmp1, tmp0);
    out0 = I::Quux::on(tmp0, tmp1);

    ncvslideio::GComputation cc(ncvslideio::GIn(in), ncvslideio::GOut(out1, out0));

    // Prepare compilation parameters manually
    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    const auto pkg     = ncvslideio::gapi::kernels<J::Baz, J::Quux, S::Foo, S::Qux>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, {in_meta}, ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);
    auto isl_model = gm.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    auto in_nh   = ncvslideio::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto tmp1_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp1);
    auto out0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out0);
    auto out1_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out1);

    EXPECT_FALSE(gm.metadata(in_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out0_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out1_nh).contains<ncvslideio::gimpl::Island>());
    // The node does not belong to the island so as not to form a cycle
    EXPECT_FALSE(gm.metadata(tmp1_nh).contains<ncvslideio::gimpl::Island>());

    EXPECT_TRUE(gm.metadata(tmp0_nh).contains<ncvslideio::gimpl::Island>());

    // There should be three islands in the GIslandModel
    const auto is_island = [&](ade::NodeHandle nh) {
        return (ncvslideio::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<ncvslideio::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(3u, num_isl);
}

TEST(IslandsFusion, PartionOverlapUserIsland)
{
    // Define a computation:
    //
    //           internal isl            isl0
    //             ........            ........
    //    (in0) -> :J::Foo:--> (tmp) ->:S::Bar: --> (out)
    //             :......:            :......:
    //                                    ^
    //                                    |
    //    (in1) --------------------------`

    // Check that internal islands doesn't overlap user island

    namespace J = Jupiter;
    namespace S = Saturn;

    GMat in[2];
    GMat tmp = I::Foo::on(in[0]);
    GMat out = I::Bar::on(tmp, in[1]);

    ncvslideio::gapi::island("isl0", ncvslideio::GIn(tmp, in[1]), ncvslideio::GOut(out));
    ncvslideio::GComputation cc(ncvslideio::GIn(in[0], in[1]), ncvslideio::GOut(out));

    // Prepare compilation parameters manually
    ncvslideio::GMetaArgs in_metas = {GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)}),
                              GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)})};
    const auto pkg = ncvslideio::gapi::kernels<J::Foo, J::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, std::move(in_metas), ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);
    auto isl_model = gm.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    auto in0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, in[0]);
    auto in1_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, in[1]);
    auto tmp_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp);
    auto out_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, out);

    auto foo_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, tmp_nh);
    auto foo_obj = gim.metadata(foo_nh).get<ncvslideio::gimpl::FusedIsland>().object;

    auto bar_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, out_nh);
    auto bar_obj = gim.metadata(bar_nh).get<ncvslideio::gimpl::FusedIsland>().object;

    EXPECT_FALSE(gm.metadata(in0_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(in1_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(tmp_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(foo_obj->is_user_specified());
    EXPECT_TRUE(bar_obj->is_user_specified());
}

TEST(IslandsFusion, DISABLED_IslandContainsDifferentBackends)
{
    // Define a computation:
    //
    //                       isl0
    //             ............................
    //    (in0) -> :J::Foo:--> (tmp) -> S::Bar: --> (out)
    //             :..........................:
    //                                    ^
    //                                    |
    //    (in1) --------------------------`

    // Try create island contains different backends

    namespace J = Jupiter;
    namespace S = Saturn;

    GMat in[2];
    GMat tmp = I::Foo::on(in[0]);
    GMat out = I::Bar::on(tmp, in[1]);

    ncvslideio::gapi::island("isl0", ncvslideio::GIn(in[0], in[1]), ncvslideio::GOut(out));
    ncvslideio::GComputation cc(ncvslideio::GIn(in[0], in[1]), ncvslideio::GOut(out));

    // Prepare compilation parameters manually
    ncvslideio::GMetaArgs in_metas = {GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)}),
                              GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)})};
    const auto pkg = ncvslideio::gapi::kernels<J::Foo, S::Bar>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, std::move(in_metas), ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    EXPECT_ANY_THROW(compiler.runPasses(*graph));
}

TEST(IslandFusion, WithLoop)
{
    namespace J = Jupiter; // see mock_kernels.cpp

    // Define a computation:
    //
    //    (in) -> J::Foo --> (tmp0) -> J::Foo --> (tmp1) -> J::Qux -> (out)
    //                            :                        ^
    //                            '--> J::Baz --> (scl0) --'
    //
    // The whole thing should be merged to a single island
    // There's a cycle warning if Foo/Foo/Qux are merged first
    // Then this island both produces data for Baz and consumes data
    // from Baz. This is a cycle and it should be avoided by the merging code.
    //
    ncvslideio::GMat    in;
    ncvslideio::GMat    tmp0 = I::Foo::on(in);
    ncvslideio::GMat    tmp1 = I::Foo::on(tmp0);
    ncvslideio::GScalar scl0 = I::Baz::on(tmp0);
    ncvslideio::GMat    out  = I::Qux::on(tmp1, scl0);
    ncvslideio::GComputation cc(in, out);

    // Prepare compilation parameters manually
    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    const auto pkg     = ncvslideio::gapi::kernels<J::Foo, J::Baz, J::Qux>();

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(cc, {in_meta}, ncvslideio::compile_args(pkg));
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    // Inspect the graph and verify the islands configuration
    ncvslideio::gimpl::GModel::ConstGraph gm(*graph);

    auto in_nh   = ncvslideio::gimpl::GModel::dataNodeOf(gm, in);
    auto tmp0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp0);
    auto tmp1_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, tmp1);
    auto scl0_nh = ncvslideio::gimpl::GModel::dataNodeOf(gm, scl0);
    auto out_nh  = ncvslideio::gimpl::GModel::dataNodeOf(gm, out);

    // in/out mats shouldn't be assigned to any Island
    EXPECT_FALSE(gm.metadata(in_nh ).contains<ncvslideio::gimpl::Island>());
    EXPECT_FALSE(gm.metadata(out_nh).contains<ncvslideio::gimpl::Island>());

    // tmp0/tmp1/scl should be assigned to island
    EXPECT_TRUE(gm.metadata(tmp0_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_TRUE(gm.metadata(tmp1_nh).contains<ncvslideio::gimpl::Island>());
    EXPECT_TRUE(gm.metadata(scl0_nh).contains<ncvslideio::gimpl::Island>());

    // Check that there's a single island object and it contains all
    // that data object handles

    ncvslideio::gimpl::GModel::ConstGraph cg(*graph);
    auto isl_model = cg.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    const auto is_island = [&](ade::NodeHandle nh) {
        return (ncvslideio::gimpl::NodeKind::ISLAND
                == gim.metadata(nh).get<ncvslideio::gimpl::NodeKind>().k);
    };
    const std::size_t num_isl = std::count_if(gim.nodes().begin(),
                                              gim.nodes().end(),
                                              is_island);
    EXPECT_EQ(1u, num_isl);

    auto isl_nh  = ncvslideio::gimpl::GIslandModel::producerOf(gim, out_nh);
    auto isl_obj = gim.metadata(isl_nh).get<ncvslideio::gimpl::FusedIsland>().object;
    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp0_nh));
    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), tmp1_nh));
    EXPECT_TRUE(ade::util::contains(isl_obj->contents(), scl0_nh));
}

TEST(IslandFusion, Regression_ShouldFuseAll)
{
    // Initially the merge procedure didn't work as expected and
    // stopped fusion even if it could be continued (e.g. full
    // GModel graph could be fused into a single GIsland node).
    // Example of this is custom RGB 2 YUV pipeline as shown below:

    ncvslideio::GMat r, g, b;
    ncvslideio::GMat y = 0.299f*r + 0.587f*g + 0.114f*b;
    ncvslideio::GMat u = 0.492f*(b - y);
    ncvslideio::GMat v = 0.877f*(r - y);

    ncvslideio::GComputation customCvt({r, g, b}, {y, u, v});

    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});

    // Directly instantiate G-API graph compiler and run partial compilation
    ncvslideio::gimpl::GCompiler compiler(customCvt, {in_meta,in_meta,in_meta}, ncvslideio::compile_args());
    ncvslideio::gimpl::GCompiler::GPtr graph = compiler.generateGraph();
    compiler.runPasses(*graph);

    ncvslideio::gimpl::GModel::ConstGraph cg(*graph);
    auto isl_model = cg.metadata().get<ncvslideio::gimpl::IslandModel>().model;
    ncvslideio::gimpl::GIslandModel::ConstGraph gim(*isl_model);

    std::vector<ade::NodeHandle> data_nhs;
    std::vector<ade::NodeHandle> isl_nhs;
    for (auto &&nh : gim.nodes())
    {
        if (gim.metadata(nh).contains<ncvslideio::gimpl::FusedIsland>())
            isl_nhs.push_back(std::move(nh));
        else if (gim.metadata(nh).contains<ncvslideio::gimpl::DataSlot>())
            data_nhs.push_back(std::move(nh));
        else FAIL() << "GIslandModel node with unexpected metadata type";
    }

    EXPECT_EQ(6u, data_nhs.size()); // 3 input nodes + 3 output nodes
    EXPECT_EQ(1u, isl_nhs.size());  // 1 island
}

TEST(IslandFusion, Test_Desync_NoFuse)
{
    ncvslideio::GMat in;
    ncvslideio::GMat tmp1 = in*0.5f;
    ncvslideio::GMat tmp2 = tmp1 + in;

    ncvslideio::GMat tmp3 = ncvslideio::gapi::streaming::desync(tmp1);
    ncvslideio::GMat tmp4 = tmp3*0.1f;

    const auto in_meta = ncvslideio::GMetaArg(ncvslideio::GMatDesc{CV_8U,1,ncvslideio::Size(32,32)});
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(tmp2, tmp4));

    //////////////////////////////////////////////////////////////////
    // Compile the graph in "regular" mode, it should produce a single island
    // Note: with copy moved to a separate backend there is always 3 islands in this test
    {
        using namespace ncvslideio::gimpl;

        GCompiler compiler(comp, {in_meta}, ncvslideio::compile_args());
        GCompiler::GPtr graph = compiler.generateGraph();
        compiler.runPasses(*graph);

        auto isl_model = GModel::ConstGraph(*graph).metadata()
            .get<IslandModel>().model;
        GIslandModel::ConstGraph gim(*isl_model);

        const auto is_island = [&](ade::NodeHandle nh) {
            return (NodeKind::ISLAND == gim.metadata(nh).get<NodeKind>().k);
        };
        const auto num_isl = std::count_if(gim.nodes().begin(),
                                           gim.nodes().end(),
                                           is_island);
        EXPECT_EQ(3, num_isl);
    }
    //////////////////////////////////////////////////////////////////
    // Now compile the graph in the streaming mode.
    // It has to produce two islands
    // Note: with copy moved to a separate backend there is always 3 islands in this test
    {
        using namespace ncvslideio::gimpl;

        GCompiler compiler(comp, {in_meta}, ncvslideio::compile_args());
        GCompiler::GPtr graph = compiler.generateGraph();
        GModel::Graph(*graph).metadata().set(Streaming{});
        compiler.runPasses(*graph);

        auto isl_model = GModel::ConstGraph(*graph).metadata()
             .get<IslandModel>().model;
        GIslandModel::ConstGraph gim(*isl_model);

        const auto is_island = [&](ade::NodeHandle nh) {
            return (NodeKind::ISLAND == gim.metadata(nh).get<NodeKind>().k);
        };
        const auto num_isl = std::count_if(gim.nodes().begin(),
                                           gim.nodes().end(),
                                           is_island);
        EXPECT_EQ(3, num_isl);
    }
}

// Fixme: add more tests on mixed (hetero) graphs
// ADE-222, ADE-223

// FIXME: add test on combination of user-specified island
// which should be heterogeneous (based on kernel availability)
// but as we don't support this, compilation should fail

// FIXME: add tests on automatic inferred islands which are
// connected via 1) gmat 2) gscalar 3) garray,
// check the case with executor
// check the case when this 1/2/3 interim object is also gcomputation output

} // namespace opencv_test
