// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

#include "api/gcomputation_priv.hpp"

namespace opencv_test
{

TEST(GMetaArg, Traits_Is_Positive)
{
    using namespace ncvslideio::detail;

    static_assert(is_meta_descr<ncvslideio::GScalarDesc>::value,
                  "GScalarDesc is a meta description type");

    static_assert(is_meta_descr<ncvslideio::GMatDesc>::value,
                  "GMatDesc is a meta description type");
}

TEST(GMetaArg, Traits_Is_Negative)
{
    using namespace ncvslideio::detail;

    static_assert(!is_meta_descr<ncvslideio::GCompileArgs>::value,
                  "GCompileArgs is NOT a meta description type");

    static_assert(!is_meta_descr<int>::value,
                  "int is NOT a meta description type");

    static_assert(!is_meta_descr<std::string>::value,
                  "str::string is NOT a meta description type");
}

TEST(GMetaArg, Traits_Are_EntireList_Positive)
{
    using namespace ncvslideio::detail;

    static_assert(are_meta_descrs<ncvslideio::GScalarDesc>::value,
                  "GScalarDesc is a meta description type");

    static_assert(are_meta_descrs<ncvslideio::GMatDesc>::value,
                  "GMatDesc is a meta description type");

    static_assert(are_meta_descrs<ncvslideio::GMatDesc, ncvslideio::GScalarDesc>::value,
                  "Both GMatDesc and GScalarDesc are meta types");
}

TEST(GMetaArg, Traits_Are_EntireList_Negative)
{
    using namespace ncvslideio::detail;

    static_assert(!are_meta_descrs<ncvslideio::GCompileArgs>::value,
                  "GCompileArgs is NOT among meta types");

    static_assert(!are_meta_descrs<int, std::string>::value,
                  "Both int and std::string is NOT among meta types");

    static_assert(!are_meta_descrs<ncvslideio::GMatDesc, ncvslideio::GScalarDesc, int>::value,
                  "List of type is not valid for meta as there\'s int");

    static_assert(!are_meta_descrs<ncvslideio::GMatDesc, ncvslideio::GScalarDesc, ncvslideio::GCompileArgs>::value,
                  "List of type is not valid for meta as there\'s GCompileArgs");
}

TEST(GMetaArg, Traits_Are_ButLast_Positive)
{
    using namespace ncvslideio::detail;

    static_assert(are_meta_descrs_but_last<ncvslideio::GScalarDesc, int>::value,
                  "List is valid (int is omitted)");

    static_assert(are_meta_descrs_but_last<ncvslideio::GMatDesc, ncvslideio::GScalarDesc, ncvslideio::GCompileArgs>::value,
                  "List is valid (GCompileArgs are omitted)");
}

TEST(GMetaArg, Traits_Are_ButLast_Negative)
{
    using namespace ncvslideio::detail;

    static_assert(!are_meta_descrs_but_last<int, std::string>::value,
                  "Both int is NOT among meta types (std::string is omitted)");

    static_assert(!are_meta_descrs_but_last<ncvslideio::GMatDesc, ncvslideio::GScalarDesc, int, int>::value,
                  "List of type is not valid for meta as there\'s two ints");

    static_assert(!are_meta_descrs_but_last<ncvslideio::GMatDesc, ncvslideio::GScalarDesc, ncvslideio::GCompileArgs, float>::value,
                  "List of type is not valid for meta as there\'s GCompileArgs");
}

TEST(GMetaArg, Can_Get_Metas_From_Input_Run_Args)
{
    ncvslideio::Mat m(3, 3, CV_8UC3);
    ncvslideio::Scalar s;
    std::vector<int> v;

    GMatDesc m_desc;
    GMetaArgs meta_args = descr_of(ncvslideio::gin(m, s, v));

    EXPECT_EQ(3u, meta_args.size());
    EXPECT_NO_THROW(m_desc = util::get<ncvslideio::GMatDesc>(meta_args[0]));
    EXPECT_NO_THROW(util::get<ncvslideio::GScalarDesc>(meta_args[1]));
    EXPECT_NO_THROW(util::get<ncvslideio::GArrayDesc>(meta_args[2]));

    EXPECT_EQ(CV_8U, m_desc.depth);
    EXPECT_EQ(3, m_desc.chan);
    EXPECT_EQ(ncvslideio::gapi::own::Size(3, 3), m_desc.size);
}

TEST(GMetaArg, Can_Get_Metas_From_Output_Run_Args)
{
    ncvslideio::Mat m(3, 3, CV_8UC3);
    ncvslideio::Scalar s;
    std::vector<int> v;

    GMatDesc m_desc;
    GRunArgsP out_run_args = ncvslideio::gout(m, s, v);
    GMetaArg m_meta = descr_of(out_run_args[0]);
    GMetaArg s_meta = descr_of(out_run_args[1]);
    GMetaArg v_meta = descr_of(out_run_args[2]);

    EXPECT_NO_THROW(m_desc = util::get<ncvslideio::GMatDesc>(m_meta));
    EXPECT_NO_THROW(util::get<ncvslideio::GScalarDesc>(s_meta));
    EXPECT_NO_THROW(util::get<ncvslideio::GArrayDesc>(v_meta));

    EXPECT_EQ(CV_8U, m_desc.depth);
    EXPECT_EQ(3, m_desc.chan);
    EXPECT_EQ(ncvslideio::Size(3, 3), m_desc.size);
}

TEST(GMetaArg, Can_Describe_RunArg)
{
    ncvslideio::Mat m(3, 3, CV_8UC3);
    ncvslideio::UMat um(3, 3, CV_8UC3);
    ncvslideio::Scalar s;
    ncvslideio::Scalar os;
    std::vector<int> v;

    GMetaArgs metas = {GMetaArg(descr_of(m)),
                       GMetaArg(descr_of(um)),
                       GMetaArg(descr_of(s)),
                       GMetaArg(descr_of(os)),
                       GMetaArg(descr_of(v))};

    auto in_run_args = ncvslideio::gin(m, um, s, os, v);

    for (size_t i = 0; i < metas.size(); i++) {
        EXPECT_TRUE(can_describe(metas[i], in_run_args[i]));
    }
}

TEST(GMetaArg, Can_Describe_RunArgs)
{
    ncvslideio::Mat m(3, 3, CV_8UC3);
    ncvslideio::Scalar s;
    std::vector<int> v;

    GMetaArgs metas0 = {GMetaArg(descr_of(m)), GMetaArg(descr_of(s)), GMetaArg(descr_of(v))};
    auto in_run_args0  = ncvslideio::gin(m, s, v);

    EXPECT_TRUE(can_describe(metas0, in_run_args0));

    auto in_run_args01  = ncvslideio::gin(m, s);
    EXPECT_FALSE(can_describe(metas0, in_run_args01));
}

TEST(GMetaArg, Can_Describe_RunArgP)
{
    ncvslideio::Mat m(3, 3, CV_8UC3);
    ncvslideio::UMat um(3, 3, CV_8UC3);
    ncvslideio::Scalar s;
    ncvslideio::Scalar os;
    std::vector<int> v;

    GMetaArgs metas = {GMetaArg(descr_of(m)),
                       GMetaArg(descr_of(um)),
                       GMetaArg(descr_of(s)),
                       GMetaArg(descr_of(os)),
                       GMetaArg(descr_of(v))};

    auto out_run_args = ncvslideio::gout(m, um, s, os, v);

    for (size_t i = 0; i < metas.size(); i++) {
        EXPECT_TRUE(can_describe(metas[i], out_run_args[i]));
    }
}

} // namespace opencv_test
