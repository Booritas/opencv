// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "common/gapi_tests_common.hpp"

namespace custom
{
G_TYPED_KERNEL(GKernelForGArrayGMatOut, <ncvslideio::GArray<ncvslideio::GMat>(ncvslideio::GMat)>,
               "custom.test.kernelForGArrayGMatOut")
{
    static ncvslideio::GArrayDesc outMeta(const ncvslideio::GMatDesc&)
    {
        return ncvslideio::empty_array_desc();
    }
};

GAPI_OCV_KERNEL(GCPUKernelForGArrayGMatOut, custom::GKernelForGArrayGMatOut)
{
    static void run(const ncvslideio::Mat &src, std::vector<ncvslideio::Mat> &out)
    {
        out[0] = src.clone();
    }
};

G_TYPED_KERNEL(GSizeOfVectorGMat, <ncvslideio::GOpaque<size_t>(ncvslideio::GArray<ncvslideio::GMat>)>,
               "custom.test.sizeOfVectorGMat")
{
    static ncvslideio::GOpaqueDesc outMeta(const ncvslideio::GArrayDesc&)
    {
        return ncvslideio::empty_gopaque_desc();
    }
};

GAPI_OCV_KERNEL(GCPUSizeOfVectorGMat, custom::GSizeOfVectorGMat)
{
    static void run(const std::vector<ncvslideio::Mat> &src, size_t &out)
    {
        out = src.size();
    }
};
}

namespace opencv_test
{

TEST(GAPI_Typed, UnaryOp)
{
    // Initialization //////////////////////////////////////////////////////////
    const ncvslideio::Size sz(32, 32);
    ncvslideio::Mat
        in_mat         (sz, CV_8UC3),
        out_mat_untyped(sz, CV_8UC3),
        out_mat_typed1 (sz, CV_8UC3),
        out_mat_typed2 (sz, CV_8UC3),
        out_mat_cv     (sz, CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    ncvslideio::GComputation cvtU([]()
    {
        ncvslideio::GMat in;
        ncvslideio::GMat out = ncvslideio::gapi::RGB2YUV(in);
        return ncvslideio::GComputation(in, out);
    });
    cvtU.apply(in_mat, out_mat_untyped);

    // Typed G-API /////////////////////////////////////////////////////////////
    ncvslideio::GComputationT<ncvslideio::GMat (ncvslideio::GMat)> cvtT(ncvslideio::gapi::RGB2YUV);
    auto cvtTComp = cvtT.compile(ncvslideio::descr_of(in_mat));

    cvtT.apply(in_mat, out_mat_typed1);
    cvtTComp(in_mat, out_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    ncvslideio::cvtColor(in_mat, out_mat_cv, ncvslideio::COLOR_RGB2YUV);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_untyped, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed2, NORM_INF));
}

TEST(GAPI_Typed, BinaryOp)
{
    // Initialization //////////////////////////////////////////////////////////
    const ncvslideio::Size sz(32, 32);
    ncvslideio::Mat
        in_mat1        (sz, CV_8UC1),
        in_mat2        (sz, CV_8UC1),
        out_mat_untyped(sz, CV_8UC1),
        out_mat_typed1 (sz, CV_8UC1),
        out_mat_typed2 (sz, CV_8UC1),
        out_mat_cv     (sz, CV_8UC1);
    ncvslideio::randu(in_mat1, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    ncvslideio::randu(in_mat2, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    ncvslideio::GComputation cvtU([]()
    {
        ncvslideio::GMat in1, in2;
        ncvslideio::GMat out = ncvslideio::gapi::add(in1, in2);
        return ncvslideio::GComputation({in1, in2}, {out});
    });
    std::vector<ncvslideio::Mat> u_ins  = {in_mat1, in_mat2};
    std::vector<ncvslideio::Mat> u_outs = {out_mat_untyped};
    cvtU.apply(u_ins, u_outs);

    // Typed G-API /////////////////////////////////////////////////////////////
    ncvslideio::GComputationT<ncvslideio::GMat (ncvslideio::GMat, ncvslideio::GMat)> cvtT([](ncvslideio::GMat m1, ncvslideio::GMat m2)
    {
        return m1+m2;
    });
    auto cvtTC =  cvtT.compile(ncvslideio::descr_of(in_mat1),
                               ncvslideio::descr_of(in_mat2));

    cvtT.apply(in_mat1, in_mat2, out_mat_typed1);
    cvtTC(in_mat1, in_mat2, out_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    ncvslideio::add(in_mat1, in_mat2, out_mat_cv);

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_untyped, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv, out_mat_typed2, NORM_INF));
}

TEST(GAPI_Typed, MultipleOuts)
{
    // Initialization //////////////////////////////////////////////////////////
    const ncvslideio::Size sz(32, 32);
    ncvslideio::Mat
        in_mat        (sz, CV_8UC1),
        out_mat_unt1  (sz, CV_8UC1),
        out_mat_unt2  (sz, CV_8UC1),
        out_mat_typed1(sz, CV_8UC1),
        out_mat_typed2(sz, CV_8UC1),
        out_mat_comp1 (sz, CV_8UC1),
        out_mat_comp2 (sz, CV_8UC1),
        out_mat_cv1   (sz, CV_8UC1),
        out_mat_cv2   (sz, CV_8UC1);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    // Untyped G-API ///////////////////////////////////////////////////////////
    ncvslideio::GComputation cvtU([]()
    {
        ncvslideio::GMat in;
        ncvslideio::GMat out1 = in * 2.f;
        ncvslideio::GMat out2 = in * 4.f;
        return ncvslideio::GComputation({in}, {out1, out2});
    });
    std::vector<ncvslideio::Mat> u_ins  = {in_mat};
    std::vector<ncvslideio::Mat> u_outs = {out_mat_unt1, out_mat_unt2};
    cvtU.apply(u_ins, u_outs);

    // Typed G-API /////////////////////////////////////////////////////////////
    ncvslideio::GComputationT<std::tuple<ncvslideio::GMat, ncvslideio::GMat> (ncvslideio::GMat)> cvtT([](ncvslideio::GMat in)
    {
        return std::make_tuple(in*2.f, in*4.f);
    });
    auto cvtTC =  cvtT.compile(ncvslideio::descr_of(in_mat));

    cvtT.apply(in_mat, out_mat_typed1, out_mat_typed2);
    cvtTC(in_mat, out_mat_comp1, out_mat_comp2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_mat_cv1 = in_mat * 2.f;
    out_mat_cv2 = in_mat * 4.f;

    // Comparison //////////////////////////////////////////////////////////////
    // FIXME: There must be OpenCV comparison test functions already available!
    EXPECT_EQ(0, cvtest::norm(out_mat_cv1, out_mat_unt1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv2, out_mat_unt2, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv1, out_mat_typed1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv2, out_mat_typed2, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv1, out_mat_comp1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_cv2, out_mat_comp2, NORM_INF));
}

TEST(GAPI_Typed, GArrayGMatOut)
{
    // Initialization //////////////////////////////////////////////////////////
    const ncvslideio::Size sz(32, 32);
    ncvslideio::Mat in_mat(sz, CV_8UC3);
    std::vector<ncvslideio::Mat> out_vec_mat_untyped(1),
                         out_vec_mat_typed1 (1),
                         out_vec_mat_typed2 (1),
                         out_vec_mat_cv     (1);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));

    auto customKernel = ncvslideio::gapi::kernels<custom::GCPUKernelForGArrayGMatOut>();
    auto absExactCompare = AbsExact().to_compare_f();

    // Untyped G-API ///////////////////////////////////////////////////////////
    ncvslideio::GComputation cptU([]()
    {
        ncvslideio::GMat in;
        ncvslideio::GArray<ncvslideio::GMat> out = custom::GKernelForGArrayGMatOut::on(in);
        return ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out));
    });
    cptU.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_vec_mat_untyped), ncvslideio::compile_args(customKernel));

    // Typed G-API /////////////////////////////////////////////////////////////
    ncvslideio::GComputationT<ncvslideio::GArray<ncvslideio::GMat> (ncvslideio::GMat)> cptT(custom::GKernelForGArrayGMatOut::on);
    auto cplT = cptT.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(customKernel));

    cptT.apply(in_mat, out_vec_mat_typed1, ncvslideio::compile_args(customKernel));
    cplT(in_mat, out_vec_mat_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_vec_mat_cv[0] = in_mat.clone();

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(absExactCompare(out_vec_mat_cv[0], out_vec_mat_untyped[0]));
    EXPECT_TRUE(absExactCompare(out_vec_mat_cv[0], out_vec_mat_typed1 [0]));
    EXPECT_TRUE(absExactCompare(out_vec_mat_cv[0], out_vec_mat_typed2 [0]));
}

TEST(GAPI_Typed, GArrayGMatIn)
{
    // Initialization //////////////////////////////////////////////////////////
    const ncvslideio::Size sz(32, 32);
    size_t vectorSize = 5;

    ncvslideio::Mat in_mat (sz, CV_8UC3);
    size_t out_size_t_untyped, out_size_t_typed1, out_size_t_typed2, out_size_t_cv;

    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(255));
    std::vector<ncvslideio::Mat> in_vec(vectorSize);
    for (size_t i = 0; i < vectorSize; i++)
        in_vec[i] = in_mat.clone();

    auto customKernel = ncvslideio::gapi::kernels<custom::GCPUSizeOfVectorGMat>();

    // Untyped G-API ///////////////////////////////////////////////////////////
    ncvslideio::GComputation cptU([]()
    {
        ncvslideio::GArray<ncvslideio::GMat> in;
        ncvslideio::GOpaque<size_t> out = custom::GSizeOfVectorGMat::on(in);
        return ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out));
    });
    cptU.apply(ncvslideio::gin(in_vec), ncvslideio::gout(out_size_t_untyped), ncvslideio::compile_args(customKernel));

    // Typed G-API /////////////////////////////////////////////////////////////
    ncvslideio::GComputationT<ncvslideio::GOpaque<size_t> (ncvslideio::GArray<ncvslideio::GMat>)> cptT(custom::GSizeOfVectorGMat::on);
    auto cplT = cptT.compile(ncvslideio::descr_of(in_vec), ncvslideio::compile_args(customKernel));

    cptT.apply(in_vec, out_size_t_typed1, ncvslideio::compile_args(customKernel));
    cplT(in_vec, out_size_t_typed2);

    // Plain OpenCV ////////////////////////////////////////////////////////////
    out_size_t_cv = in_vec.size();

    // Comparison //////////////////////////////////////////////////////////////
    EXPECT_TRUE(out_size_t_cv      == vectorSize);
    EXPECT_TRUE(out_size_t_untyped == vectorSize);
    EXPECT_TRUE(out_size_t_typed1  == vectorSize);
    EXPECT_TRUE(out_size_t_typed2  == vectorSize);
}
} // opencv_test
