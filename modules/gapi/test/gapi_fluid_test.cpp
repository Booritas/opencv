// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <opencv2/gapi/core.hpp>

#include <opencv2/gapi/fluid/gfluidbuffer.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

 // FIXME: move these tests with priv() to internal suite
#include "backends/fluid/gfluidbuffer_priv.hpp"

#include "gapi_fluid_test_kernels.hpp"
#include "logger.hpp"

namespace opencv_test
{

using namespace ncvslideio::gapi_test_kernels;

namespace
{
    void WriteFunction(uint8_t* row, int nr, int w) {
        for (int i = 0; i < w; i++)
            row[i] = static_cast<uint8_t>(nr+i);
    }
    void ReadFunction1x1(const uint8_t* row, int w) {
        for (int i = 0; i < w; i++)
            std::cout << std::setw(4) << static_cast<int>(row[i]) << " ";
        std::cout << "\n";
    }
    void ReadFunction3x3(const uint8_t* rows[3], int w) {
        for (int i = 0; i < 3; i++) {
            for (int j = -1; j < w+1; j++) {
                std::cout << std::setw(4) << static_cast<int>(rows[i][j]) << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

TEST(FluidBuffer, InputTest)
{
    const ncvslideio::Size buffer_size = {8,8};
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(buffer_size, CV_8U);

    ncvslideio::gapi::fluid::Buffer buffer(in_mat, true);
    ncvslideio::gapi::fluid::View  view = buffer.mkView(0, false);
    view.priv().allocate(1, {});
    view.priv().reset(1);
    int this_y = 0;

    while (this_y < buffer_size.height)
    {
        view.priv().prepareToRead();
        const uint8_t* rrow = view.InLine<uint8_t>(0);
        ReadFunction1x1(rrow, buffer_size.width);
        view.priv().readDone(1,1);

        ncvslideio::Mat from_buffer(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow));
        EXPECT_EQ(0, cvtest::norm(in_mat.row(this_y), from_buffer, NORM_INF));

        this_y++;
    }
}

TEST(FluidBuffer, CircularTest)
{
    const ncvslideio::Size buffer_size = {8,16};

    ncvslideio::gapi::fluid::Buffer buffer(ncvslideio::GMatDesc{CV_8U,1,buffer_size}, 3, 1, 0, 1,
        util::make_optional(ncvslideio::gapi::fluid::Border{ncvslideio::BORDER_CONSTANT, ncvslideio::Scalar(255)}));
    ncvslideio::gapi::fluid::View view = buffer.mkView(1, {});
    view.priv().reset(3);
    view.priv().allocate(3, {});
    buffer.debug(std::cout);

    const auto whole_line_is = [](const uint8_t *line, int len, int value)
    {
        return std::all_of(line, line+len, [&](const uint8_t v){return v == value;});
    };

    // Store all read/written data in separate Mats to compare with
    ncvslideio::Mat written_data(buffer_size, CV_8U);

    // Simulate write/read process
    int num_reads = 0, num_writes = 0;
    while (num_reads < buffer_size.height)
    {
        if (num_writes < buffer_size.height)
        {
            uint8_t* wrow = buffer.OutLine<uint8_t>();
            WriteFunction(wrow, num_writes, buffer_size.width);
            buffer.priv().writeDone();

            ncvslideio::Mat(1, buffer_size.width, CV_8U, wrow)
                .copyTo(written_data.row(num_writes));
            num_writes++;
        }
        buffer.debug(std::cout);

        if (view.ready())
        {
            view.priv().prepareToRead();
            const uint8_t* rrow[3] = {
                view.InLine<uint8_t>(-1),
                view.InLine<uint8_t>( 0),
                view.InLine<uint8_t>( 1),
            };
            ReadFunction3x3(rrow, buffer_size.width);
            view.priv().readDone(1,3);
            buffer.debug(std::cout);

            // Check borders right here
            EXPECT_EQ(255u, rrow[0][-1]);
            EXPECT_EQ(255u, rrow[0][buffer_size.width]);
            if (num_reads == 0)
            {
                EXPECT_TRUE(whole_line_is(rrow[0]-1, buffer_size.width+2, 255u));
            }
            if (num_reads == buffer_size.height-1)
            {
                EXPECT_TRUE(whole_line_is(rrow[2]-1, buffer_size.width+2, 255u));
            }

            // Check window (without borders)
            if (num_reads > 0 && num_reads < buffer_size.height-1)
            {
                // +1 everywhere since num_writes was just incremented above
                ncvslideio::Mat written_lastLine2 = written_data.row(num_writes - (2+1));
                ncvslideio::Mat written_lastLine1 = written_data.row(num_writes - (1+1));
                ncvslideio::Mat written_lastLine0 = written_data.row(num_writes - (0+1));

                ncvslideio::Mat read_prevLine(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow[0]));
                ncvslideio::Mat read_thisLine(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow[1]));
                ncvslideio::Mat read_nextLine(1, buffer_size.width, CV_8U, const_cast<uint8_t*>(rrow[2]));

                EXPECT_EQ(0, cvtest::norm(written_lastLine2, read_prevLine, NORM_INF));
                EXPECT_EQ(0, cvtest::norm(written_lastLine1, read_thisLine, NORM_INF));
                EXPECT_EQ(0, cvtest::norm(written_lastLine0, read_nextLine, NORM_INF));
            }
            num_reads++;
        }
    }
}

TEST(FluidBuffer, OutputTest)
{
    const ncvslideio::Size buffer_size = {8,16};
    ncvslideio::Mat out_mat = ncvslideio::Mat(buffer_size, CV_8U);

    ncvslideio::gapi::fluid::Buffer buffer(out_mat, false);
    int num_writes = 0;
    while (num_writes < buffer_size.height)
    {
        uint8_t* wrow = buffer.OutLine<uint8_t>();
        WriteFunction(wrow, num_writes, buffer_size.width);
        buffer.priv().writeDone();
        num_writes++;
    }

    GAPI_LOG_INFO(NULL, "\n" << out_mat);

    // Validity check
    for (int r = 0; r < buffer_size.height; r++)
    {
        for (int c = 0; c < buffer_size.width; c++)
        {
            EXPECT_EQ(r+c, out_mat.at<uint8_t>(r, c));
        }
    }
}

TEST(Fluid, AddC_WithScalar)
{
    ncvslideio::GMat in;
    ncvslideio::GScalar s;

    ncvslideio::GComputation c(ncvslideio::GIn(in, s), ncvslideio::GOut(TAddScalar::on(in, s)));
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC1), out_mat(3, 3, CV_8UC1), ref_mat;
    ncvslideio::Scalar in_s(100);

    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_s), ncvslideio::compile_args(fluidTestPackage));

    cc(ncvslideio::gin(in_mat, in_s), ncvslideio::gout(out_mat));
    ref_mat = in_mat + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Scalar_In_Middle_Graph)
{
    ncvslideio::GMat in;
    ncvslideio::GScalar s;

    ncvslideio::GComputation c(ncvslideio::GIn(in, s), ncvslideio::GOut(TAddScalar::on(TAddCSimple::on(in, 5), s)));
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC1), out_mat(3, 3, CV_8UC1), ref_mat;
    ncvslideio::Scalar in_s(100);

    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::descr_of(in_s), ncvslideio::compile_args(fluidTestPackage));

    cc(ncvslideio::gin(in_mat, in_s), ncvslideio::gout(out_mat));
    ref_mat = (in_mat + 5) + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Add_Scalar_To_Mat)
{
    ncvslideio::GMat in;
    ncvslideio::GScalar s;

    ncvslideio::GComputation c(ncvslideio::GIn(s, in), ncvslideio::GOut(TAddScalarToMat::on(s, in)));
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(3, 3, CV_8UC1), out_mat(3, 3, CV_8UC1), ref_mat;
    ncvslideio::Scalar in_s(100);

    auto cc = c.compile(ncvslideio::descr_of(in_s), ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));

    cc(ncvslideio::gin(in_s, in_mat), ncvslideio::gout(out_mat));
    ref_mat = in_mat + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Sum_2_Mats_And_Scalar)
{
    ncvslideio::GMat a, b;
    ncvslideio::GScalar s;

    ncvslideio::GComputation c(ncvslideio::GIn(a, s, b), ncvslideio::GOut(TSum2MatsAndScalar::on(a, s, b)));
    ncvslideio::Mat in_mat1 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = ncvslideio::Mat::eye(3, 3, CV_8UC1),
            out_mat(3, 3, CV_8UC1),
            ref_mat;
    ncvslideio::Scalar in_s(100);

    auto cc = c.compile(ncvslideio::descr_of(in_mat1), ncvslideio::descr_of(in_s), ncvslideio::descr_of(in_mat2), ncvslideio::compile_args(fluidTestPackage));

    cc(ncvslideio::gin(in_mat1, in_s, in_mat2), ncvslideio::gout(out_mat));
    ref_mat = in_mat1 + in_mat2 + in_s;
    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, EqualizeHist)
{
    ncvslideio::GMat in, out;
    ncvslideio::GComputation c(ncvslideio::GIn(in), ncvslideio::GOut(TEqualizeHist::on(in, TCalcHist::on(in))));

    ncvslideio::Mat in_mat(320, 480, CV_8UC1),
            out_mat(320, 480, CV_8UC1),
            ref_mat(320, 480, CV_8UC1);

    ncvslideio::randu(in_mat, 200, 240);

    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));

    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat));

    ncvslideio::equalizeHist(in_mat, ref_mat);

    EXPECT_EQ(0, cvtest::norm(out_mat, ref_mat, NORM_INF));
}

TEST(Fluid, Split3)
{
    ncvslideio::GMat bgr;
    ncvslideio::GMat r,g,b;
    std::tie(b,g,r) = ncvslideio::gapi::split3(bgr);
    auto rr = TAddSimple::on(r, TId::on(b));
    auto rrr = TAddSimple::on(TId::on(rr), g);
    ncvslideio::GComputation c(bgr, TId::on(rrr));

    ncvslideio::Size sz(5120, 5120);
    ncvslideio::Mat eye_1 = ncvslideio::Mat::eye(sz, CV_8UC1);
    std::vector<ncvslideio::Mat> eyes = {eye_1, eye_1, eye_1};
    ncvslideio::Mat in_mat;
    ncvslideio::merge(eyes, in_mat);
    ncvslideio::Mat out_mat(sz, CV_8UC1);

    // G-API
    auto cc = c.compile(ncvslideio::descr_of(in_mat),
                        ncvslideio::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);

    // OCV
    std::vector<ncvslideio::Mat> chans;
    ncvslideio::split(in_mat, chans);

    // Compare
    EXPECT_EQ(0, cvtest::norm(out_mat, Mat(chans[2]*3), NORM_INF));
}

TEST(Fluid, ScratchTest)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out = TPlusRow0::on(TPlusRow0::on(in));
    ncvslideio::GComputation c(in, out);

    ncvslideio::Size sz(8, 8);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat(sz, CV_8UC1);

    // OpenCV (reference)
    ncvslideio::Mat ref;
    {
        ncvslideio::Mat first_row = ncvslideio::Mat::zeros(1, sz.width, CV_8U);
        ncvslideio::Mat remaining = ncvslideio::repeat(in_mat.row(0), sz.height-1, 1);
        ncvslideio::Mat operand;
        ncvslideio::vconcat(first_row, 2*remaining, operand);
        ref = in_mat + operand;
    }
    GAPI_LOG_INFO(NULL, "\n" << ref);

    // G-API
    auto cc = c.compile(ncvslideio::descr_of(in_mat),
                        ncvslideio::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);
    GAPI_LOG_INFO(NULL, "\n" << out_mat);
    EXPECT_EQ(0, cvtest::norm(ref, out_mat, NORM_INF));

    cc(in_mat, out_mat);
    GAPI_LOG_INFO(NULL, "\n" << out_mat);
    EXPECT_EQ(0, cvtest::norm(ref, out_mat, NORM_INF));
}

TEST(Fluid, MultipleOutRowsTest)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out = TAddCSimple::on(TAddCSimple::on(in, 1), 2);
    ncvslideio::GComputation c(in, out);

    ncvslideio::Size sz(4, 4);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat(sz, CV_8UC1);

    auto cc = c.compile(ncvslideio::descr_of(in_mat),
                        ncvslideio::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);

    std::cout << out_mat << std::endl;

    ncvslideio::Mat ocv_ref = in_mat + 1 + 2;
    EXPECT_EQ(0, cvtest::norm(ocv_ref, out_mat, NORM_INF));
}


TEST(Fluid, LPIWindow)
{
    ncvslideio::GMat in;
    ncvslideio::GMat r,g,b;
    std::tie(r,g,b) = ncvslideio::gapi::split3(in);
    ncvslideio::GMat rr = TId7x7::on(r);
    ncvslideio::GMat tmp = TAddSimple::on(rr, g);
    ncvslideio::GMat out = TAddSimple::on(tmp, b);

    ncvslideio::GComputation c(in, out);

    ncvslideio::Size sz(8, 8);

    ncvslideio::Mat eye_1 = ncvslideio::Mat::eye(sz, CV_8UC1);
    std::vector<ncvslideio::Mat> eyes = {eye_1, eye_1, eye_1};
    ncvslideio::Mat in_mat;
    ncvslideio::merge(eyes, in_mat);

    ncvslideio::Mat out_mat(sz, CV_8U);
    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(in_mat, out_mat);

    //std::cout << out_mat << std::endl;

    // OpenCV reference
    ncvslideio::Mat ocv_ref = eyes[0]+eyes[1]+eyes[2];

    EXPECT_EQ(0, cvtest::norm(ocv_ref, out_mat, NORM_INF));
}

TEST(Fluid, MultipleReaders_SameLatency)
{
    //  in -> AddC -> a -> AddC -> b -> Add -> out
    //                '--> AddC -> c -'
    //
    // b and c have the same skew

    ncvslideio::GMat in;
    ncvslideio::GMat a = TAddCSimple::on(in, 1); // FIXME - align naming (G, non-G)
    ncvslideio::GMat b = TAddCSimple::on(a,  2);
    ncvslideio::GMat c = TAddCSimple::on(a,  3);
    ncvslideio::GMat out = TAddSimple::on(b, c);
    ncvslideio::GComputation comp(in, out);

    const auto sz = ncvslideio::Size(32, 32);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat_gapi(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv (sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(in_mat, out_mat_gapi);

    // Check with OpenCV
    ncvslideio::Mat tmp = in_mat + 1;
    out_mat_ocv = (tmp+2) + (tmp+3);
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

TEST(Fluid, MultipleReaders_DifferentLatency)
{
    //  in1 -> AddC -> a -> AddC -------------> b -> Add -> out
    //                 '--------------> Add --> c -'
    //                 '--> Id7x7-> d -'
    //
    // b and c have different skew (due to latency introduced by Id7x7)
    // a is ready by multiple views with different latency.

    ncvslideio::GMat in;
    ncvslideio::GMat a   = TAddCSimple::on(in, 1); // FIXME - align naming (G, non-G)
    ncvslideio::GMat b   = TAddCSimple::on(a,  2);
    ncvslideio::GMat d   = TId7x7::on(a);
    ncvslideio::GMat c   = TAddSimple::on(a, d);
    ncvslideio::GMat out = TAddSimple::on(b, c);
    ncvslideio::GComputation comp(in, out);

    const auto sz = ncvslideio::Size(32, 32);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat_gapi(sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(in_mat, out_mat_gapi);

    // Check with OpenCV
    ncvslideio::Mat ocv_a = in_mat + 1;
    ncvslideio::Mat ocv_b = ocv_a + 2;
    ncvslideio::Mat ocv_d = ocv_a;
    ncvslideio::Mat ocv_c = ocv_a + ocv_d;
    ncvslideio::Mat out_mat_ocv = ocv_b + ocv_c;
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

TEST(Fluid, MultipleOutputs)
{
    // in -> AddC -> a -> AddC ------------------> out1
    //               `--> Id7x7  --> b --> AddC -> out2

    ncvslideio::GMat in;
    ncvslideio::GMat a    = TAddCSimple::on(in, 1);
    ncvslideio::GMat b    = TId7x7::on(a);
    ncvslideio::GMat out1 = TAddCSimple::on(a, 2);
    ncvslideio::GMat out2 = TAddCSimple::on(b, 7);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out1, out2));

    const auto sz = ncvslideio::Size(32, 32);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat_gapi1(sz, CV_8UC1), out_mat_gapi2(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv1(sz, CV_8UC1), out_mat_ocv2(sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat_gapi1, out_mat_gapi2));

    // Check with OpenCV
    out_mat_ocv1 = in_mat + 1 + 2;
    out_mat_ocv2 = in_mat + 1 + 7;
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi1, out_mat_ocv1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi2, out_mat_ocv2, NORM_INF));
}

TEST(Fluid, EmptyOutputMatTest)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out = TAddCSimple::on(in, 2);
    ncvslideio::GComputation c(in, out);

    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(ncvslideio::Size(32, 24), CV_8UC1);
    ncvslideio::Mat out_mat;

    auto cc = c.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));

    cc(in_mat,    out_mat);
    EXPECT_EQ(CV_8UC1, out_mat.type());
    EXPECT_EQ(32, out_mat.cols);
    EXPECT_EQ(24, out_mat.rows);
    EXPECT_TRUE(out_mat.ptr() != nullptr);
}

struct LPISequenceTest : public TestWithParam<int>{};
TEST_P(LPISequenceTest, LPISequenceTest)
{
    // in -> AddC -> a -> Blur (2lpi) -> out

    int kernelSize = GetParam();
    ncvslideio::GMat in;
    ncvslideio::GMat a = TAddCSimple::on(in, 1);
    auto blur = kernelSize == 3 ? &TBlur3x3_2lpi::on : &TBlur5x5_2lpi::on;
    ncvslideio::GMat out = blur(a, ncvslideio::BORDER_CONSTANT, ncvslideio::Scalar(0));
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    const auto sz = ncvslideio::Size(8, 10);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat_gapi(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv(sz, CV_8UC1);

    // Run G-API
    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat_gapi));

    // Check with OpenCV
    ncvslideio::blur(in_mat + 1, out_mat_ocv, {kernelSize,kernelSize}, {-1,-1}, ncvslideio::BORDER_CONSTANT);
    EXPECT_EQ(0, cvtest::norm(out_mat_gapi, out_mat_ocv, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, LPISequenceTest,
                        Values(3, 5));

struct InputImageBorderTest : public TestWithParam <std::tuple<int, int>> {};
TEST_P(InputImageBorderTest, InputImageBorderTest)
{
    ncvslideio::Size sz_in = { 320, 240 };

    int ks         = 0;
    int borderType = 0;
    std::tie(ks, borderType) = GetParam();
    ncvslideio::Mat in_mat1(sz_in, CV_8UC1);
    ncvslideio::Scalar mean   = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);

    ncvslideio::Size kernelSize = {ks, ks};
    ncvslideio::Point anchor = {-1, -1};
    ncvslideio::Scalar borderValue(0);

    auto gblur = ks == 3 ? &TBlur3x3::on : &TBlur5x5::on;

    GMat in;
    auto out = gblur(in, borderType, borderValue);

    Mat out_mat_gapi = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out));
    auto cc = c.compile(descr_of(in_mat1), ncvslideio::compile_args(fluidTestPackage));
    cc(gin(in_mat1), gout(out_mat_gapi));

    ncvslideio::Mat out_mat_ocv = Mat::zeros(sz_in, CV_8UC1);
    ncvslideio::blur(in_mat1, out_mat_ocv, kernelSize, anchor, borderType);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, InputImageBorderTest,
                        Combine(Values(3, 5),
                                Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101)));

struct SequenceOfBlursTest : public TestWithParam <std::tuple<int>> {};
TEST_P(SequenceOfBlursTest, Test)
{
    ncvslideio::Size sz_in = { 320, 240 };

    int borderType = 0;;
    std::tie(borderType) = GetParam();
    ncvslideio::Mat in_mat(sz_in, CV_8UC1);
    ncvslideio::Scalar mean   = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);

    ncvslideio::randn(in_mat, mean, stddev);

    ncvslideio::Point anchor = {-1, -1};
    ncvslideio::Scalar borderValue(0);

    GMat in;
    auto mid = TBlur3x3::on(in,  borderType, borderValue);
    auto out = TBlur5x5::on(mid, borderType, borderValue);

    Mat out_mat_gapi = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out));
    auto cc = c.compile(descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(gin(in_mat), gout(out_mat_gapi));

    ncvslideio::Mat mid_mat_ocv = Mat::zeros(sz_in, CV_8UC1);
    ncvslideio::Mat out_mat_ocv = Mat::zeros(sz_in, CV_8UC1);
    ncvslideio::blur(in_mat, mid_mat_ocv, {3,3}, anchor, borderType);
    ncvslideio::blur(mid_mat_ocv, out_mat_ocv, {5,5}, anchor, borderType);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv, out_mat_gapi, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, SequenceOfBlursTest,
                               Values(BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT_101));

struct TwoBlursTest : public TestWithParam <std::tuple<int, int, int, int, int, int, bool>> {};
TEST_P(TwoBlursTest, Test)
{
    ncvslideio::Size sz_in = { 320, 240 };

    int kernelSize1 = 0, kernelSize2 = 0;
    int borderType1 = -1, borderType2 = -1;
    ncvslideio::Scalar borderValue1{}, borderValue2{};
    bool readFromInput = false;
    std::tie(kernelSize1, borderType1, borderValue1, kernelSize2, borderType2, borderValue2, readFromInput) = GetParam();
    ncvslideio::Mat in_mat(sz_in, CV_8UC1);
    ncvslideio::Scalar mean   = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);

    ncvslideio::randn(in_mat, mean, stddev);

    ncvslideio::Point anchor = {-1, -1};

    auto blur1 = kernelSize1 == 3 ? &TBlur3x3::on : TBlur5x5::on;
    auto blur2 = kernelSize2 == 3 ? &TBlur3x3::on : TBlur5x5::on;

    GMat in, out1, out2;
    if (readFromInput)
    {
        out1 = blur1(in, borderType1, borderValue1);
        out2 = blur2(in, borderType2, borderValue2);
    }
    else
    {
        auto mid = TAddCSimple::on(in, 0);
        out1 = blur1(mid, borderType1, borderValue1);
        out2 = blur2(mid, borderType2, borderValue2);
    }

    Mat out_mat_gapi1 = Mat::zeros(sz_in, CV_8UC1);
    Mat out_mat_gapi2 = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out1, out2));
    auto cc = c.compile(descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(gin(in_mat), gout(out_mat_gapi1, out_mat_gapi2));

    ncvslideio::Mat out_mat_ocv1 = Mat::zeros(sz_in, CV_8UC1);
    ncvslideio::Mat out_mat_ocv2 = Mat::zeros(sz_in, CV_8UC1);
    ncvslideio::blur(in_mat, out_mat_ocv1, {kernelSize1, kernelSize1}, anchor, borderType1);
    ncvslideio::blur(in_mat, out_mat_ocv2, {kernelSize2, kernelSize2}, anchor, borderType2);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv1, out_mat_gapi1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat_gapi2, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, TwoBlursTest,
                               Combine(Values(3, 5),
                                       Values(ncvslideio::BORDER_CONSTANT, ncvslideio::BORDER_REPLICATE, ncvslideio::BORDER_REFLECT_101),
                                       Values(0),
                                       Values(3, 5),
                                       Values(ncvslideio::BORDER_CONSTANT, ncvslideio::BORDER_REPLICATE, ncvslideio::BORDER_REFLECT_101),
                                       Values(0),
                                       testing::Bool())); // Read from input directly or place a copy node at start

struct TwoReadersTest : public TestWithParam <std::tuple<int, int, int, bool>> {};
TEST_P(TwoReadersTest, Test)
{
    ncvslideio::Size sz_in = { 320, 240 };

    int kernelSize = 0;
    int borderType = -1;
    ncvslideio::Scalar borderValue;
    bool readFromInput = false;
    std::tie(kernelSize, borderType, borderValue, readFromInput) = GetParam();
    ncvslideio::Mat in_mat(sz_in, CV_8UC1);
    ncvslideio::Scalar mean   = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);

    ncvslideio::randn(in_mat, mean, stddev);

    ncvslideio::Point anchor = {-1, -1};

    auto blur = kernelSize == 3 ? &TBlur3x3::on : TBlur5x5::on;

    GMat in, out1, out2;
    if (readFromInput)
    {
        out1 = TAddCSimple::on(in, 0);
        out2 = blur(in, borderType, borderValue);
    }
    else
    {
        auto mid = TAddCSimple::on(in, 0);
        out1 = TAddCSimple::on(mid, 0);
        out2 = blur(mid, borderType, borderValue);
    }

    Mat out_mat_gapi1 = Mat::zeros(sz_in, CV_8UC1);
    Mat out_mat_gapi2 = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in), GOut(out1, out2));
    auto cc = c.compile(descr_of(in_mat), ncvslideio::compile_args(fluidTestPackage));
    cc(gin(in_mat), gout(out_mat_gapi1, out_mat_gapi2));

    ncvslideio::Mat out_mat_ocv1 = Mat::zeros(sz_in, CV_8UC1);
    ncvslideio::Mat out_mat_ocv2 = Mat::zeros(sz_in, CV_8UC1);
    out_mat_ocv1 = in_mat;
    ncvslideio::blur(in_mat, out_mat_ocv2, {kernelSize, kernelSize}, anchor, borderType);

    EXPECT_EQ(0, cvtest::norm(out_mat_ocv1, out_mat_gapi1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(out_mat_ocv2, out_mat_gapi2, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, TwoReadersTest,
                               Combine(Values(3, 5),
                                       Values(ncvslideio::BORDER_CONSTANT, ncvslideio::BORDER_REPLICATE, ncvslideio::BORDER_REFLECT_101),
                                       Values(0),
                                       testing::Bool())); // Read from input directly or place a copy node at start

TEST(FluidTwoIslands, SanityTest)
{
    ncvslideio::Size sz_in{8,8};

    GMat in1, in2;
    auto out1 = TAddScalar::on(in1, {0});
    auto out2 = TAddScalar::on(in2, {0});

    ncvslideio::Mat in_mat1(sz_in, CV_8UC1);
    ncvslideio::Mat in_mat2(sz_in, CV_8UC1);
    ncvslideio::Scalar mean   = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);

    ncvslideio::randn(in_mat1, mean, stddev);
    ncvslideio::randn(in_mat2, mean, stddev);

    Mat out_mat1 = Mat::zeros(sz_in, CV_8UC1);
    Mat out_mat2 = Mat::zeros(sz_in, CV_8UC1);

    GComputation c(GIn(in1, in2), GOut(out1, out2));
    EXPECT_NO_THROW(c.apply(gin(in_mat1, in_mat2), gout(out_mat1, out_mat2), ncvslideio::compile_args(fluidTestPackage)));
    EXPECT_EQ(0, cvtest::norm(in_mat1, out_mat1, NORM_INF));
    EXPECT_EQ(0, cvtest::norm(in_mat2, out_mat2, NORM_INF));
}

struct NV12RoiTest : public TestWithParam <std::pair<ncvslideio::Size, ncvslideio::Rect>> {};
TEST_P(NV12RoiTest, Test)
{
    ncvslideio::Size y_sz;
    ncvslideio::Rect roi;
    std::tie(y_sz, roi) = GetParam();

    ncvslideio::Size uv_sz(y_sz.width / 2, y_sz.height / 2);
    ncvslideio::Size in_sz(y_sz.width, y_sz.height*3/2);

    ncvslideio::Mat in_mat = ncvslideio::Mat(in_sz, CV_8UC1);

    ncvslideio::Scalar mean   = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);
    ncvslideio::randn(in_mat, mean, stddev);

    ncvslideio::Mat y_mat  = ncvslideio::Mat(y_sz, CV_8UC1, in_mat.data);
    ncvslideio::Mat uv_mat = ncvslideio::Mat(uv_sz, CV_8UC2, in_mat.data + in_mat.step1() * y_sz.height);
    ncvslideio::Mat out_mat, out_mat_ocv;

    ncvslideio::GMat y, uv;
    auto rgb = ncvslideio::gapi::NV12toRGB(y, uv);
    ncvslideio::GComputation c(ncvslideio::GIn(y, uv), ncvslideio::GOut(rgb));

    c.apply(ncvslideio::gin(y_mat, uv_mat), ncvslideio::gout(out_mat), ncvslideio::compile_args(fluidTestPackage, ncvslideio::GFluidOutputRois{{roi}}));

    ncvslideio::cvtColor(in_mat, out_mat_ocv, ncvslideio::COLOR_YUV2RGB_NV12);

    EXPECT_EQ(0, cvtest::norm(out_mat(roi), out_mat_ocv(roi), NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Fluid, NV12RoiTest,
                        Values(std::make_pair(ncvslideio::Size{8, 8}, ncvslideio::Rect{0, 0, 8, 2})
                              ,std::make_pair(ncvslideio::Size{8, 8}, ncvslideio::Rect{0, 2, 8, 2})
                              ,std::make_pair(ncvslideio::Size{8, 8}, ncvslideio::Rect{0, 4, 8, 2})
                              ,std::make_pair(ncvslideio::Size{8, 8}, ncvslideio::Rect{0, 6, 8, 2})
                              ,std::make_pair(ncvslideio::Size{1920, 1080}, ncvslideio::Rect{0,   0, 1920, 270})
                              ,std::make_pair(ncvslideio::Size{1920, 1080}, ncvslideio::Rect{0, 270, 1920, 270})
                              ,std::make_pair(ncvslideio::Size{1920, 1080}, ncvslideio::Rect{0, 540, 1920, 270})
                              ,std::make_pair(ncvslideio::Size{1920, 1080}, ncvslideio::Rect{0, 710, 1920, 270})
                              ));

TEST(Fluid, UnusedNodeOutputCompileTest)
{
    ncvslideio::GMat in;
    ncvslideio::GMat a, b, c, d;
    std::tie(a, b, c, d) = ncvslideio::gapi::split4(in);
    ncvslideio::GMat out = ncvslideio::gapi::merge3(a, b, c);

    ncvslideio::Mat in_mat(ncvslideio::Size(8, 8), CV_8UC4);
    ncvslideio::Mat out_mat(ncvslideio::Size(8, 8), CV_8UC3);

    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    ASSERT_NO_THROW(comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat),
        ncvslideio::compile_args(ncvslideio::gapi::core::fluid::kernels())));
}

TEST(Fluid, UnusedNodeOutputReshapeTest)
{
    const auto test_size = ncvslideio::Size(8, 8);

    const auto get_compile_args = [] () {
        return ncvslideio::compile_args(
            ncvslideio::gapi::combine(
                ncvslideio::gapi::core::fluid::kernels(),
                ncvslideio::gapi::imgproc::fluid::kernels()
            )
        );
    };

    ncvslideio::GMat in;
    ncvslideio::GMat a, b, c, d;
    std::tie(a, b, c, d) = ncvslideio::gapi::split4(in);
    ncvslideio::GMat out = ncvslideio::gapi::resize(ncvslideio::gapi::merge3(a, b, c), test_size, 0.0, 0.0,
        ncvslideio::INTER_LINEAR);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    ncvslideio::Mat in_mat(test_size, CV_8UC4);
    ncvslideio::Mat out_mat(test_size, CV_8UC3);

    ncvslideio::GCompiled compiled;
    ASSERT_NO_THROW(compiled = comp.compile(descr_of(in_mat), get_compile_args()));

    in_mat = ncvslideio::Mat(test_size * 2, CV_8UC4);
    ASSERT_TRUE(compiled.canReshape());
    ASSERT_NO_THROW(compiled.reshape(descr_of(gin(in_mat)), get_compile_args()));
    ASSERT_NO_THROW(compiled(in_mat, out_mat));
}

TEST(Fluid, InvalidROIs)
{
    ncvslideio::GMat in;
    ncvslideio::GMat out = ncvslideio::gapi::add(in, in);

    ncvslideio::Mat in_mat(ncvslideio::Size(8, 8), CV_8UC3);
    ncvslideio::Mat out_mat = in_mat.clone();
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(100));

    std::vector<ncvslideio::Rect> invalid_rois =
    {
        ncvslideio::Rect(1, 0, 0, 0),
        ncvslideio::Rect(0, 1, 0, 0),
        ncvslideio::Rect(0, 0, 1, 0),
        ncvslideio::Rect(0, 0, 0, 1),
        ncvslideio::Rect(0, 0, out_mat.cols, 0),
        ncvslideio::Rect(0, 0, 0, out_mat.rows),
        ncvslideio::Rect(0, out_mat.rows, out_mat.cols, out_mat.rows),
        ncvslideio::Rect(out_mat.cols, 0, out_mat.cols, out_mat.rows),
    };

    const auto compile_args = [] (ncvslideio::Rect roi) {
        return ncvslideio::compile_args(ncvslideio::gapi::core::fluid::kernels(), GFluidOutputRois{{roi}});
    };

    for (const auto& roi : invalid_rois)
    {
        ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));
        EXPECT_THROW(comp.apply(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat), compile_args(roi)),
            std::exception);
    }
}


namespace
{
#if defined(__linux__)
uint64_t currMemoryConsumption()
{
    // check self-state via /proc information
    constexpr const char stat_file_path[] = "/proc/self/statm";
    std::ifstream proc_stat(stat_file_path);
    if (!proc_stat.is_open() || !proc_stat.good())
    {
        CV_LOG_WARNING(NULL, "Failed to open stat file: " << stat_file_path);
        return static_cast<uint64_t>(0);
    }
    std::string stat_line;
    std::getline(proc_stat, stat_line);
    uint64_t unused, data_and_stack;
    std::istringstream(stat_line) >> unused >> unused >> unused >> unused >> unused
                                  >> data_and_stack;
    CV_Assert(data_and_stack != 0);
    return data_and_stack;
}
#else
// FIXME: implement this part (at least for Windows?), right now it's enough to check Linux only
uint64_t currMemoryConsumption() { return static_cast<uint64_t>(0); }
#endif
}  // anonymous namespace

TEST(Fluid, MemoryConsumptionDoesNotGrowOnReshape)
{
    ncvslideio::GMat in;
    ncvslideio::GMat a, b, c;
    std::tie(a, b, c) = ncvslideio::gapi::split3(in);
    ncvslideio::GMat merged = ncvslideio::gapi::merge4(a, b, c, a);
    ncvslideio::GMat d, e, f, g;
    std::tie(d, e, f, g) = ncvslideio::gapi::split4(merged);
    ncvslideio::GMat out = ncvslideio::gapi::merge3(d, e, f);

    ncvslideio::Mat in_mat(ncvslideio::Size(8, 8), CV_8UC3);
    ncvslideio::randu(in_mat, ncvslideio::Scalar::all(0), ncvslideio::Scalar::all(100));
    ncvslideio::Mat out_mat;

    const auto compile_args = [] () {
        return ncvslideio::compile_args(ncvslideio::gapi::core::fluid::kernels());
    };

    ncvslideio::GCompiled compiled = ncvslideio::GComputation(ncvslideio::GIn(in), ncvslideio::GOut(out)).compile(
        ncvslideio::descr_of(in_mat), compile_args());
    ASSERT_TRUE(compiled.canReshape());

    const auto mem_before = currMemoryConsumption();
    for (int _ = 0; _ < 1000; ++_) compiled.reshape(ncvslideio::descr_of(ncvslideio::gin(in_mat)), compile_args());
    const auto mem_after = currMemoryConsumption();

    ASSERT_GE(mem_before, mem_after);
}

} // namespace opencv_test
