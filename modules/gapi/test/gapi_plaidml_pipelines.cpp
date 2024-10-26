// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "test_precomp.hpp"

#include <stdexcept>
#include <ade/util/iota_range.hpp>
#include "logger.hpp"

#include <opencv2/gapi/plaidml/core.hpp>
#include <opencv2/gapi/plaidml/plaidml.hpp>

namespace opencv_test
{

#ifdef HAVE_PLAIDML

inline ncvslideio::gapi::plaidml::config getConfig()
{
    auto read_var_from_env = [](const char* env)
    {
        const char* raw = std::getenv(env);
        if (!raw)
        {
            ncvslideio::util::throw_error(std::runtime_error(std::string(env) + " is't set"));
        }

        return std::string(raw);
    };

    auto dev_id = read_var_from_env("PLAIDML_DEVICE");
    auto trg_id = read_var_from_env("PLAIDML_TARGET");

    return ncvslideio::gapi::plaidml::config{std::move(dev_id),
                                     std::move(trg_id)};
}

TEST(GAPI_PlaidML_Pipelines, SimpleArithmetic)
{
    ncvslideio::Size size(1920, 1080);
    int type = CV_8UC1;

    ncvslideio::Mat in_mat1(size, type);
    ncvslideio::Mat in_mat2(size, type);

    // NB: What about overflow ? PlaidML doesn't handle it
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(127));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(127));

    ncvslideio::Mat out_mat(size, type, ncvslideio::Scalar::all(0));
    ncvslideio::Mat ref_mat(size, type, ncvslideio::Scalar::all(0));

    ////////////////////////////// G-API //////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = in1 + in2;

    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    comp.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat),
               ncvslideio::compile_args(getConfig(),
                                ncvslideio::gapi::use_only{ncvslideio::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    ncvslideio::add(in_mat1, in_mat2, ref_mat, ncvslideio::noArray(), type);

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
}

// FIXME PlaidML cpu backend does't support bitwise operations
TEST(GAPI_PlaidML_Pipelines, DISABLED_ComplexArithmetic)
{
    ncvslideio::Size size(1920, 1080);
    int type = CV_8UC1;

    ncvslideio::Mat in_mat1(size, type);
    ncvslideio::Mat in_mat2(size, type);

    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    ncvslideio::Mat out_mat(size, type, ncvslideio::Scalar::all(0));
    ncvslideio::Mat ref_mat(size, type, ncvslideio::Scalar::all(0));

    ////////////////////////////// G-API //////////////////////////////////////
    ncvslideio::GMat in1, in2;
    auto out = in1 | (in2 ^ (in1 & (in2 + (in1 - in2))));

    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    comp.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat),
               ncvslideio::compile_args(getConfig(),
                                ncvslideio::gapi::use_only{ncvslideio::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    ncvslideio::subtract(in_mat1,  in_mat2, ref_mat, ncvslideio::noArray(), type);
    ncvslideio::add(in_mat2, ref_mat, ref_mat, ncvslideio::noArray(), type);
    ncvslideio::bitwise_and(in_mat1, ref_mat, ref_mat);
    ncvslideio::bitwise_xor(in_mat2, ref_mat, ref_mat);
    ncvslideio::bitwise_or(in_mat1, ref_mat, ref_mat);

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
}

TEST(GAPI_PlaidML_Pipelines, TwoInputOperations)
{
    ncvslideio::Size size(1920, 1080);
    int type = CV_8UC1;

    constexpr int kNumInputs = 4;
    std::vector<ncvslideio::Mat> in_mat(kNumInputs, ncvslideio::Mat(size, type));
    for (int i = 0; i < kNumInputs; ++i)
    {
        ncvslideio::randu(in_mat[i], ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(60));
    }

    ncvslideio::Mat out_mat(size, type, ncvslideio::Scalar::all(0));
    ncvslideio::Mat ref_mat(size, type, ncvslideio::Scalar::all(0));

    ////////////////////////////// G-API //////////////////////////////////////
    ncvslideio::GMat in[4];
    auto out = (in[3] - in[0]) + (in[2] - in[1]);

    ncvslideio::GComputation comp(ncvslideio::GIn(in[0], in[1], in[2], in[3]), ncvslideio::GOut(out));

    // FIXME Doesn't work just apply(in_mat, out_mat, ...)
    comp.apply(ncvslideio::gin(in_mat[0], in_mat[1], in_mat[2], in_mat[3]), ncvslideio::gout(out_mat),
               ncvslideio::compile_args(getConfig(),
                                ncvslideio::gapi::use_only{ncvslideio::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    ncvslideio::subtract(in_mat[3], in_mat[0],  ref_mat, ncvslideio::noArray(), type);
    ncvslideio::add(ref_mat, in_mat[2], ref_mat, ncvslideio::noArray(), type);
    ncvslideio::subtract(ref_mat, in_mat[1], ref_mat, ncvslideio::noArray(), type);

    EXPECT_EQ(0, ncvslideio::norm(out_mat, ref_mat));
}

TEST(GAPI_PlaidML_Pipelines, TwoOutputOperations)
{
    ncvslideio::Size size(1920, 1080);
    int type = CV_8UC1;

    constexpr int kNumInputs = 4;
    std::vector<ncvslideio::Mat> in_mat(kNumInputs, ncvslideio::Mat(size, type));
    for (int i = 0; i < kNumInputs; ++i)
    {
        ncvslideio::randu(in_mat[i], ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(60));
    }

    std::vector<ncvslideio::Mat> out_mat(kNumInputs, ncvslideio::Mat(size, type, ncvslideio::Scalar::all(0)));
    std::vector<ncvslideio::Mat> ref_mat(kNumInputs, ncvslideio::Mat(size, type, ncvslideio::Scalar::all(0)));

    ////////////////////////////// G-API //////////////////////////////////////
    ncvslideio::GMat in[4], out[2];
    out[0] = in[0] + in[3];
    out[1] = in[1] + in[2];

    ncvslideio::GComputation comp(ncvslideio::GIn(in[0], in[1], in[2], in[3]), ncvslideio::GOut(out[0], out[1]));

    // FIXME Doesn't work just apply(in_mat, out_mat, ...)
    comp.apply(ncvslideio::gin(in_mat[0], in_mat[1], in_mat[2], in_mat[3]),
               ncvslideio::gout(out_mat[0], out_mat[1]),
               ncvslideio::compile_args(getConfig(),
                                ncvslideio::gapi::use_only{ncvslideio::gapi::core::plaidml::kernels()}));

    ////////////////////////////// OpenCV /////////////////////////////////////
    ncvslideio::add(in_mat[0], in_mat[3], ref_mat[0], ncvslideio::noArray(), type);
    ncvslideio::add(in_mat[1], in_mat[2], ref_mat[1], ncvslideio::noArray(), type);

    EXPECT_EQ(0, ncvslideio::norm(out_mat[0], ref_mat[0]));
    EXPECT_EQ(0, ncvslideio::norm(out_mat[1], ref_mat[1]));
}

#else // HAVE_PLAIDML

TEST(GAPI_PlaidML_Pipelines, ThrowIfPlaidMLNotFound)
{
    ASSERT_ANY_THROW(ncvslideio::gapi::core::plaidml::kernels());
}

#endif // HAVE_PLAIDML

} // namespace opencv_test
