// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


// FIXME: move out from Common

#include "../test_precomp.hpp"
#include <opencv2/gapi/cpu/core.hpp>

#include <ade/util/algorithm.hpp>

namespace opencv_test
{
namespace
{
    G_TYPED_KERNEL(GCompoundDoubleAddC, <GMat(GMat, GScalar)>, "org.opencv.test.compound_double_addC")
    {
        static GMatDesc outMeta(GMatDesc in, GScalarDesc) { return in; }
    };

    GAPI_COMPOUND_KERNEL(GCompoundDoubleAddCImpl, GCompoundDoubleAddC)
    {
        static GMat expand(ncvslideio::GMat in, ncvslideio::GScalar s)
        {
            return ncvslideio::gapi::addC(ncvslideio::gapi::addC(in, s), s);
        }
    };

    G_TYPED_KERNEL(GCompoundAddC, <GMat(GMat, GScalar)>, "org.opencv.test.compound_addC")
    {
        static GMatDesc outMeta(GMatDesc in, GScalarDesc) { return in; }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddCImpl, GCompoundAddC)
    {
        static GMat expand(ncvslideio::GMat in, ncvslideio::GScalar s)
        {
            return ncvslideio::gapi::addC(in, s);
        }
    };

    using GMat3 = std::tuple<GMat,GMat,GMat>;
    using GMat2 = std::tuple<GMat,GMat>;

    G_TYPED_KERNEL_M(GCompoundMergeWithSplit, <GMat3(GMat, GMat, GMat)>, "org.opencv.test.compound_merge_split")
    {
        static std::tuple<GMatDesc,GMatDesc,GMatDesc> outMeta(GMatDesc a, GMatDesc b, GMatDesc c)
        {
            return std::make_tuple(a, b, c);
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundMergeWithSplitImpl, GCompoundMergeWithSplit)
    {
        static GMat3 expand(ncvslideio::GMat a, ncvslideio::GMat b, ncvslideio::GMat c)
        {
            return ncvslideio::gapi::split3(ncvslideio::gapi::merge3(a, b, c));
        }
    };

    G_TYPED_KERNEL(GCompoundAddWithAddC, <GMat(GMat, GMat, GScalar)>, "org.opencv.test.compound_add_with_addc")
    {
        static GMatDesc outMeta(GMatDesc in, GMatDesc, GScalarDesc)
        {
            return in;
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddWithAddCImpl, GCompoundAddWithAddC)
    {
        static GMat expand(ncvslideio::GMat in1, ncvslideio::GMat in2, ncvslideio::GScalar s)
        {
            return ncvslideio::gapi::addC(ncvslideio::gapi::add(in1, in2), s);
        }
    };

    G_TYPED_KERNEL_M(GCompoundSplitWithAdd, <GMat2(GMat)>, "org.opencv.test.compound_split_with_add")
    {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc in)
        {
            const auto out_depth = in.depth;
            const auto out_desc  = in.withType(out_depth, 1);
            return std::make_tuple(out_desc, out_desc);
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundSplitWithAddImpl, GCompoundSplitWithAdd)
    {
        static GMat2 expand(ncvslideio::GMat in)
        {
            ncvslideio::GMat a, b, c;
            std::tie(a, b, c) = ncvslideio::gapi::split3(in);
            return std::make_tuple(ncvslideio::gapi::add(a, b), c);
        }
    };

    G_TYPED_KERNEL_M(GCompoundParallelAddC, <GMat2(GMat, GScalar)>, "org.opencv.test.compound_parallel_addc")
    {
        static std::tuple<GMatDesc, GMatDesc> outMeta(GMatDesc in, GScalarDesc)
        {
            return std::make_tuple(in, in);
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundParallelAddCImpl, GCompoundParallelAddC)
    {
        static GMat2 expand(ncvslideio::GMat in, ncvslideio::GScalar s)
        {
            return std::make_tuple(ncvslideio::gapi::addC(in, s), ncvslideio::gapi::addC(in, s));
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddImpl, ncvslideio::gapi::core::GAdd)
    {
        static GMat expand(ncvslideio::GMat in1, ncvslideio::GMat in2, int)
        {
            return ncvslideio::gapi::sub(ncvslideio::gapi::sub(in1, in2), in2);
        }
    };

    G_TYPED_KERNEL(GCompoundAddWithAddCWithDoubleAddC, <GMat(GMat, GMat, GScalar)>, "org.opencv.test.compound_add_with_addC_with_double_addC")
    {
        static GMatDesc outMeta(GMatDesc in, GMatDesc, GScalarDesc)
        {
            return in;
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundAddWithAddCWithDoubleAddCImpl, GCompoundAddWithAddCWithDoubleAddC)
    {
        static GMat expand(ncvslideio::GMat in1, ncvslideio::GMat in2, ncvslideio::GScalar s)
        {
            return GCompoundDoubleAddC::on(GCompoundAddWithAddC::on(in1, in2, s), s);
        }
    };

    using GDoubleArray = ncvslideio::GArray<double>;
    G_TYPED_KERNEL(GNegateArray, <GDoubleArray(GDoubleArray)>, "org.opencv.test.negate_array")
    {
        static GArrayDesc outMeta(const GArrayDesc&) { return empty_array_desc(); }
    };

    GAPI_OCV_KERNEL(GNegateArrayImpl, GNegateArray)
    {
        static void run(const std::vector<double>& in, std::vector<double>& out)
        {
            ade::util::transform(in, std::back_inserter(out), std::negate<double>());
        }
    };

    G_TYPED_KERNEL(GMaxInArray, <GScalar(GDoubleArray)>, "org.opencv.test.max_in_array")
    {
        static GScalarDesc outMeta(const GArrayDesc&) { return empty_scalar_desc(); }
    };

    GAPI_OCV_KERNEL(GMaxInArrayImpl, GMaxInArray)
    {
        static void run(const std::vector<double>& in, ncvslideio::Scalar& out)
        {
            out = *std::max_element(in.begin(), in.end());
        }
    };

    G_TYPED_KERNEL(GCompoundMaxInArray, <GScalar(GDoubleArray)>, "org.opencv.test.compound_max_in_array")
    {
        static GScalarDesc outMeta(const GArrayDesc&) { return empty_scalar_desc(); }
    };

    GAPI_COMPOUND_KERNEL(GCompoundMaxInArrayImpl, GCompoundMaxInArray)
    {
        static GScalar expand(GDoubleArray in)
        {
            return GMaxInArray::on(in);
        }
    };

    G_TYPED_KERNEL(GCompoundNegateArray, <GDoubleArray(GDoubleArray)>, "org.opencv.test.compound_negate_array")
    {
        static GArrayDesc outMeta(const GArrayDesc&) { return empty_array_desc(); }
    };

    GAPI_COMPOUND_KERNEL(GCompoundNegateArrayImpl, GCompoundNegateArray)
    {
        static GDoubleArray expand(GDoubleArray in)
        {
            return GNegateArray::on(in);
        }
    };

    G_TYPED_KERNEL(SetDiagKernel, <GMat(GMat, GDoubleArray)>, "org.opencv.test.empty_kernel")
    {
        static GMatDesc outMeta(GMatDesc in, GArrayDesc) { return in; }
    };

    void setDiag(ncvslideio::Mat& in, const std::vector<double>& diag)
    {
        GAPI_Assert(in.rows == static_cast<int>(diag.size()));
        GAPI_Assert(in.cols == static_cast<int>(diag.size()));
        for (int i = 0; i < in.rows; ++i)
        {
            in.at<uchar>(i, i) = static_cast<uchar>(diag[i]);
        }
    }

    GAPI_OCV_KERNEL(SetDiagKernelImpl, SetDiagKernel)
    {
        static void run(const ncvslideio::Mat& in, const std::vector<double>& v, ncvslideio::Mat& out)
        {
            in.copyTo(out);
            setDiag(out, v);
        }
    };

    G_TYPED_KERNEL(GCompoundGMatGArrayGMat, <GMat(GMat, GDoubleArray, GMat)>, "org.opencv.test.compound_gmat_garray_gmat")
    {
        static GMatDesc outMeta(GMatDesc in, GArrayDesc, GMatDesc) { return in; }
    };

    GAPI_COMPOUND_KERNEL(GCompoundGMatGArrayGMatImpl, GCompoundGMatGArrayGMat)
    {
        static GMat expand(GMat a, GDoubleArray b, GMat c)
        {
            return SetDiagKernel::on(ncvslideio::gapi::add(a, c), b);
        }
    };

    G_TYPED_KERNEL(GToInterleaved, <GMat(GMatP)>, "org.opencv.test.to_interleaved")
    {
        static GMatDesc outMeta(GMatDesc in)
        {
            GAPI_Assert(in.planar == true);
            GAPI_Assert(in.chan == 3);
            return in.asInterleaved();
        }
    };

    G_TYPED_KERNEL(GToPlanar, <GMatP(GMat)>, "org.opencv.test.to_planar")
    {
        static GMatDesc outMeta(GMatDesc in)
        {
            GAPI_Assert(in.planar == false);
            GAPI_Assert(in.chan == 3);
            return in.asPlanar();
        }
    };

    GAPI_OCV_KERNEL(GToInterleavedImpl, GToInterleaved)
    {
        static void run(const ncvslideio::Mat& in, ncvslideio::Mat& out)
        {
            constexpr int inPlanesCount = 3;
            int inPlaneHeight = in.rows / inPlanesCount;

            std::vector<ncvslideio::Mat> inPlanes(inPlanesCount);
            for (int i = 0; i < inPlanesCount; ++i)
            {
                int startRow = i * inPlaneHeight;
                int endRow = startRow + inPlaneHeight;
                inPlanes[i] = in.rowRange(startRow, endRow);
            }

            ncvslideio::merge(inPlanes, out);
        }
    };

    GAPI_OCV_KERNEL(GToPlanarImpl, GToPlanar)
    {
        static void run(const ncvslideio::Mat& in, ncvslideio::Mat& out)
        {
            std::vector<ncvslideio::Mat> inPlanes;
            ncvslideio::split(in, inPlanes);
            ncvslideio::vconcat(inPlanes, out);
        }
    };

    G_TYPED_KERNEL(GCompoundToInterleavedToPlanar, <GMatP(GMatP)>,
                   "org.opencv.test.compound_to_interleaved_to_planar")
    {
        static GMatDesc outMeta(GMatDesc in)
        {
            GAPI_Assert(in.planar == true);
            GAPI_Assert(in.chan == 3);
            return in;
        }
    };

    GAPI_COMPOUND_KERNEL(GCompoundToInterleavedToPlanarImpl, GCompoundToInterleavedToPlanar)
    {
        static GMatP expand(ncvslideio::GMatP in)
        {
            return GToPlanar::on(GToInterleaved::on(in));
        }
    };
} // namespace

// FIXME avoid ncvslideio::combine that use custom and default kernels together
TEST(GCompoundKernel, ReplaceDefaultKernel)
{
    ncvslideio::GMat in1, in2;
    auto out = ncvslideio::gapi::add(in1, in2);
    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundAddImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(ncvslideio::gapi::core::cpu::kernels(), custom_pkg);
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out));
    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
            out_mat(3, 3, CV_8UC1),
            ref_mat(3, 3, CV_8UC1);

    comp.apply(ncvslideio::gin(in_mat1, in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat1 - in_mat2 - in_mat2;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, DoubleAddC)
{
    ncvslideio::GMat in1, in2;
    ncvslideio::GScalar s;
    auto add_res   = ncvslideio::gapi::add(in1, in2);
    auto super     = GCompoundDoubleAddC::on(add_res, s);
    auto out       = ncvslideio::gapi::addC(super, s);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundDoubleAddCImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2, s), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    ncvslideio::Scalar scalar = 2;

    comp.apply(ncvslideio::gin(in_mat1, in_mat2, scalar), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + scalar + scalar;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, AddC)
{
    ncvslideio::GMat in1, in2;
    ncvslideio::GScalar s;
    auto add_res   = ncvslideio::gapi::add(in1, in2);
    auto super     = GCompoundAddC::on(add_res, s);
    auto out       = ncvslideio::gapi::addC(super, s);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundAddCImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2, s), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    ncvslideio::Scalar scalar = 2;

    comp.apply(ncvslideio::gin(in_mat1, in_mat2, scalar), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + scalar;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, MergeWithSplit)
{
    ncvslideio::GMat in, a1, b1, c1,
        a2, b2, c2;

    std::tie(a1, b1, c1) = ncvslideio::gapi::split3(in);
    std::tie(a2, b2, c2) = GCompoundMergeWithSplit::on(a1, b1, c1);
    auto out = ncvslideio::gapi::merge3(a2, b2, c2);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundMergeWithSplitImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC3), out_mat, ref_mat;
    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, AddWithAddC)
{
    ncvslideio::GMat in1, in2;
    ncvslideio::GScalar s;
    auto out = GCompoundAddWithAddC::on(in1, in2, s);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundAddWithAddCImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2, s), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    ncvslideio::Scalar scalar = 2;

    comp.apply(ncvslideio::gin(in_mat1, in_mat2, scalar), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, SplitWithAdd)
{
    ncvslideio::GMat in, out1, out2;
    std::tie(out1, out2) = GCompoundSplitWithAdd::on(in);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundSplitWithAddImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC3),
        out_mat1(3, 3, CV_8UC1),
        out_mat2(3, 3, CV_8UC1),
        ref_mat1(3, 3, CV_8UC1),
        ref_mat2(3, 3, CV_8UC1);

    comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat1, out_mat2), ncvslideio::compile_args(full_pkg));

    std::vector<ncvslideio::Mat> channels(3);
    ncvslideio::split(in_mat, channels);

    ref_mat1 = channels[0] + channels[1];
    ref_mat2 = channels[2];

    EXPECT_EQ(0, cvtest::norm(out_mat1, ref_mat1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat2, ref_mat2, NORM_INF));
}

TEST(GCompoundKernel, ParallelAddC)
{
    ncvslideio::GMat in1, out1, out2;
    ncvslideio::GScalar in2;
    std::tie(out1, out2) = GCompoundParallelAddC::on(in1, in2);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundParallelAddCImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2), ncvslideio::GOut(out1, out2));

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        out_mat1(3, 3, CV_8UC1),
        out_mat2(3, 3, CV_8UC1),
        ref_mat1(3, 3, CV_8UC1),
        ref_mat2(3, 3, CV_8UC1);

    ncvslideio::Scalar scalar = 2;

    comp.apply(ncvslideio::gin(in_mat, scalar), ncvslideio::gout(out_mat1, out_mat2), ncvslideio::compile_args(full_pkg));

    ref_mat1 = in_mat + scalar;
    ref_mat2 = in_mat + scalar;

    EXPECT_EQ(0, cvtest::norm(out_mat1, ref_mat1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat2, ref_mat2, NORM_INF));
}

TEST(GCompoundKernel, GCompundKernelAndDefaultUseOneData)
{
    ncvslideio::GMat in1, in2;
    ncvslideio::GScalar s;
    auto out = ncvslideio::gapi::add(GCompoundAddWithAddC::on(in1, in2, s), ncvslideio::gapi::addC(in2, s));

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundAddWithAddCImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2, s), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
        out_mat(3, 3, CV_8UC1),
        ref_mat(3, 3, CV_8UC1);

    ncvslideio::Scalar scalar = 2;

    comp.apply(ncvslideio::gin(in_mat1, in_mat2, scalar), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + in_mat2 + scalar;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, CompoundExpandedToCompound)
{
    ncvslideio::GMat in1, in2;
    ncvslideio::GScalar s;
    auto out = GCompoundAddWithAddCWithDoubleAddC::on(in1, in2, s);

    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundAddWithAddCWithDoubleAddCImpl,
                                              GCompoundAddWithAddCImpl,
                                              GCompoundDoubleAddCImpl>();

    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in1, in2, s), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
            out_mat(3, 3, CV_8UC1),
            ref_mat(3, 3, CV_8UC1);

    ncvslideio::Scalar scalar = 2;

    comp.apply(ncvslideio::gin(in_mat1, in_mat2, scalar), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));
    ref_mat = in_mat1 + in_mat2 + scalar + scalar + scalar;

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(GCompoundKernel, MaxInArray)
{
    GDoubleArray in;
    auto out = GCompoundMaxInArray::on(in);
    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundMaxInArrayImpl, GMaxInArrayImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));
    std::vector<double> v = { 1, 5, -2, 3, 10, 2};
    ncvslideio::Scalar out_scl;
    ncvslideio::Scalar ref_scl(*std::max_element(v.begin(), v.end()));

    comp.apply(ncvslideio::gin(v), ncvslideio::gout(out_scl), ncvslideio::compile_args(full_pkg));

    EXPECT_EQ(out_scl, ref_scl);
}

TEST(GCompoundKernel, NegateArray)
{
    GDoubleArray in;
    GDoubleArray out = GCompoundNegateArray::on(in);
    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundNegateArrayImpl, GNegateArrayImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));
    std::vector<double> in_v = {1, 5, -2, -10, 3};
    std::vector<double> out_v;
    std::vector<double> ref_v;
    ade::util::transform(in_v, std::back_inserter(ref_v), std::negate<double>());

    comp.apply(ncvslideio::gin(in_v), ncvslideio::gout(out_v), ncvslideio::compile_args(full_pkg));

    EXPECT_EQ(out_v, ref_v);
}

TEST(GCompoundKernel, RightGArrayHandle)
{
    ncvslideio::GMat in[2];
    GDoubleArray a;
    ncvslideio::GMat out = GCompoundGMatGArrayGMat::on(in[0], a, in[1]);
    const auto custom_pkg = ncvslideio::gapi::kernels<GCompoundGMatGArrayGMatImpl, SetDiagKernelImpl>();
    const auto full_pkg   = ncvslideio::gapi::combine(custom_pkg, ncvslideio::gapi::core::cpu::kernels());
    ncvslideio::GComputation comp(ncvslideio::GIn(in[0], a, in[1]), ncvslideio::GOut(out));
    std::vector<double> in_v(3, 1.0);
    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(ncvslideio::Size(3, 3), CV_8UC1),
            in_mat2 = ncvslideio::Mat::eye(ncvslideio::Size(3, 3), CV_8UC1),
            out_mat;
    ncvslideio::Mat ref_mat= in_mat1 + in_mat2;
    setDiag(ref_mat, in_v);

    comp.apply(ncvslideio::gin(in_mat1, in_v, in_mat2), ncvslideio::gout(out_mat), ncvslideio::compile_args(full_pkg));

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

}

TEST(GCompoundKernel, ToInterleavedToPlanar)
{
    ncvslideio::GMatP in;
    ncvslideio::GMatP out = GCompoundToInterleavedToPlanar::on(in);
    const auto pkg = ncvslideio::gapi::kernels<GCompoundToInterleavedToPlanarImpl,
                                       GToInterleavedImpl,
                                       GToPlanarImpl>();

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    constexpr int numPlanes = 3;
    ncvslideio::Mat in_mat(ncvslideio::Size(15, 15), CV_8UC1),
            out_mat,
            ref_mat;

    ncvslideio::randu(in_mat, 0, 255);
    ref_mat = in_mat;

    comp.compile(ncvslideio::descr_of(in_mat).asPlanar(numPlanes), ncvslideio::compile_args(pkg))
         (ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));

}
} // opencv_test
