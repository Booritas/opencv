// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP
#define OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"

namespace opencv_test
{
enum operation
{
    ADD,  SUB,  MUL,  DIV,
    ADDR, SUBR, MULR, DIVR,
    GT,  LT,  GE,  LE,  EQ,  NE,
    GTR, LTR, GER, LER, EQR, NER,
    AND,  OR,  XOR,
    ANDR, ORR, XORR
};

// Note: namespace must match the namespace of the type of the printed object
inline std::ostream& operator<<(std::ostream& os, operation op)
{
#define CASE(v) case operation::v: os << #v; break
    switch (op)
    {
        CASE(ADD);  CASE(SUB);  CASE(MUL);  CASE(DIV);
        CASE(ADDR); CASE(SUBR); CASE(MULR); CASE(DIVR);
        CASE(GT);  CASE(LT);  CASE(GE);  CASE(LE);  CASE(EQ);  CASE(NE);
        CASE(GTR); CASE(LTR); CASE(GER); CASE(LER); CASE(EQR); CASE(NER);
        CASE(AND);  CASE(OR);  CASE(XOR);
        CASE(ANDR); CASE(ORR); CASE(XORR);
        default: GAPI_Error("unknown operation value");
    }
#undef CASE
    return os;
}

namespace
{
// declare test cases for matrix and scalar operators
auto opADD_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in + c;};
auto opADD_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::add(in, c, out);};

auto opADDR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c + in;};
auto opADDR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::add(c, in, out);};

auto opSUB_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in - c;};
auto opSUB_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::subtract(in, c, out);};

auto opSUBR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c - in;};
auto opSUBR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::subtract(c, in, out);};

auto opMUL_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in * c;};
auto opMUL_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::multiply(in, c, out);};

auto opMULR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c * in;};
auto opMULR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::multiply(c, in, out);};

auto opDIV_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in / c;};
auto opDIV_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::divide(in, c, out);};

auto opDIVR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c / in;};
auto opDIVR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::divide(c, in, out);};


auto opGT_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in > c;};
auto opGT_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(in, c, out,ncvslideio::CMP_GT);};

auto opGTR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c > in;};
auto opGTR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(c, in, out,ncvslideio::CMP_GT);};

auto opLT_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in < c;};
auto opLT_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(in, c, out,ncvslideio::CMP_LT);};

auto opLTR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c < in;};
auto opLTR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(c, in, out,ncvslideio::CMP_LT);};

auto opGE_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in >= c;};
auto opGE_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(in, c, out,ncvslideio::CMP_GE);};

auto opGER_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c >= in;};
auto opGER_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(c, in, out,ncvslideio::CMP_GE);};

auto opLE_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in <= c;};
auto opLE_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(in, c, out,ncvslideio::CMP_LE);};

auto opLER_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c <= in;};
auto opLER_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(c, in, out,ncvslideio::CMP_LE);};

auto opEQ_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in == c;};
auto opEQ_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(in, c, out,ncvslideio::CMP_EQ);};

auto opEQR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c == in;};
auto opEQR_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(c, in, out,ncvslideio::CMP_EQ);};

auto opNE_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in != c;};
auto opNE_ocv   = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(in, c, out,ncvslideio::CMP_NE);};

auto opNER_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c != in;};
auto opNER_ocv  = [](const ncvslideio::Mat& in, ncvslideio::Scalar c, ncvslideio::Mat& out){ncvslideio::compare(c, in, out,ncvslideio::CMP_NE);};


auto opAND_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in & c;};
auto opAND_ocv   = [](const ncvslideio::Mat& in, const ncvslideio::Scalar& c, ncvslideio::Mat& out){ncvslideio::bitwise_and(in, c, out);};

auto opOR_gapi   = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in | c;};
auto opOR_ocv    = [](const ncvslideio::Mat& in, const ncvslideio::Scalar& c, ncvslideio::Mat& out){ncvslideio::bitwise_or(in, c, out);};

auto opXOR_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return in ^ c;};
auto opXOR_ocv   = [](const ncvslideio::Mat& in, const ncvslideio::Scalar& c, ncvslideio::Mat& out){ncvslideio::bitwise_xor(in, c, out);};

auto opANDR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c & in;};
auto opANDR_ocv  = [](const ncvslideio::Mat& in, const ncvslideio::Scalar& c, ncvslideio::Mat& out){ncvslideio::bitwise_and(c, in, out);};

auto opORR_gapi  = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c | in;};
auto opORR_ocv   = [](const ncvslideio::Mat& in, const ncvslideio::Scalar& c, ncvslideio::Mat& out){ncvslideio::bitwise_or(c, in, out);};

auto opXORR_gapi = [](ncvslideio::GMat in,ncvslideio::GScalar c){return c ^ in;};
auto opXORR_ocv  = [](const ncvslideio::Mat& in, const ncvslideio::Scalar& c, ncvslideio::Mat& out){ncvslideio::bitwise_xor(c, in, out);};

// declare test cases for matrix and matrix operators
auto opADDM_gapi = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 + in2;};
auto opADDM_ocv  = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::add(in1, in2, out);};

auto opSUBM_gapi = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 - in2;};
auto opSUBM_ocv  = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::subtract(in1, in2, out);};

auto opDIVM_gapi = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 / in2;};
auto opDIVM_ocv  = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::divide(in1, in2, out);};


auto opGTM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 > in2;};
auto opGTM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::compare(in1, in2, out, ncvslideio::CMP_GT);};

auto opGEM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 >= in2;};
auto opGEM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::compare(in1, in2, out, ncvslideio::CMP_GE);};

auto opLTM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 < in2;};
auto opLTM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::compare(in1, in2, out, ncvslideio::CMP_LT);};

auto opLEM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 <= in2;};
auto opLEM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::compare(in1, in2, out, ncvslideio::CMP_LE);};

auto opEQM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 == in2;};
auto opEQM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::compare(in1, in2, out, ncvslideio::CMP_EQ);};

auto opNEM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 != in2;};
auto opNEM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::compare(in1, in2, out, ncvslideio::CMP_NE);};


auto opANDM_gapi = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 & in2;};
auto opANDM_ocv  = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::bitwise_and(in1, in2, out);};

auto opORM_gapi  = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 | in2;};
auto opORM_ocv   = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::bitwise_or(in1, in2, out);};

auto opXORM_gapi = [](ncvslideio::GMat in1,ncvslideio::GMat in2){return in1 ^ in2;};
auto opXORM_ocv  = [](const ncvslideio::Mat& in1, const ncvslideio::Mat& in2, ncvslideio::Mat& out){ncvslideio::bitwise_xor(in1, in2, out);};
} // anonymous namespace

struct g_api_ocv_pair_mat_scalar {
    using g_api_function_t = std::function<ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GScalar)>;
    using ocv_function_t   = std::function<void(ncvslideio::Mat const&, ncvslideio::Scalar, ncvslideio::Mat&)>;

    g_api_function_t g_api_function;
    ocv_function_t   ocv_function;

    g_api_ocv_pair_mat_scalar() = default;

#define CASE(v) case operation::v: \
    g_api_function = op##v##_gapi; \
    ocv_function   = op##v##_ocv;  \
    break

    g_api_ocv_pair_mat_scalar(operation op)
    {
        switch (op)
        {
            CASE(ADD);  CASE(SUB);  CASE(MUL);  CASE(DIV);
            CASE(ADDR); CASE(SUBR); CASE(MULR); CASE(DIVR);
            CASE(GT);  CASE(LT);  CASE(GE);  CASE(LE);  CASE(EQ);  CASE(NE);
            CASE(GTR); CASE(LTR); CASE(GER); CASE(LER); CASE(EQR); CASE(NER);
            CASE(AND);  CASE(OR);  CASE(XOR);
            CASE(ANDR); CASE(ORR); CASE(XORR);
            default: GAPI_Error("unknown operation value");
        }
    }
#undef CASE
};

struct g_api_ocv_pair_mat_mat {
    using g_api_function_t = std::function<ncvslideio::GMat(ncvslideio::GMat,ncvslideio::GMat)>;
    using ocv_function_t   = std::function<void(ncvslideio::Mat const&, ncvslideio::Mat const&, ncvslideio::Mat&)>;

    g_api_function_t g_api_function;
    ocv_function_t   ocv_function;

    g_api_ocv_pair_mat_mat() = default;

#define CASE(v) case operation::v:  \
    g_api_function = op##v##M_gapi; \
    ocv_function   = op##v##M_ocv;  \
    break

    g_api_ocv_pair_mat_mat(operation op)
    {
        switch (op)
        {
            CASE(ADD);  CASE(SUB);  CASE(DIV);
            CASE(GT); CASE(LT); CASE(GE); CASE(LE); CASE(EQ); CASE(NE);
            CASE(AND); CASE(OR); CASE(XOR);
            default: GAPI_Error("unknown operation value");
        }
    }
#undef CASE
};

// Create new value-parameterized test fixture:
// MathOperatorMatScalarTest - fixture name
// initMatsRandU - function that is used to initialize input/output data
// FIXTURE_API(CompareMats, g_api_ocv_pair_mat_scalar) - test-specific parameters (types)
// 2 - number of test-specific parameters
// cmpF, op - test-spcific parameters (names)
//
// We get:
// 1. Default parameters: int type, ncvslideio::Size sz, int dtype, getCompileArgs() function
//      - available in test body
// 2. Input/output matrices will be initialized by initMatsRandU (in this fixture)
// 3. Specific parameters: cmpF, op of corresponding types
//      - created (and initialized) automatically
//      - available in test body
// Note: all parameter _values_ (e.g. type CV_8UC3) are set via INSTANTIATE_TEST_CASE_P macro
GAPI_TEST_FIXTURE(MathOperatorMatScalarTest, initMatsRandU,
    FIXTURE_API(CompareMats, operation), 2, cmpF, op)
GAPI_TEST_FIXTURE(MathOperatorMatMatTest, initMatsRandU,
    FIXTURE_API(CompareMats, operation), 2, cmpF, op)
GAPI_TEST_FIXTURE(NotOperatorTest, initMatrixRandU, <>, 0)
} // opencv_test

#endif // OPENCV_GAPI_OPERATOR_TESTS_COMMON_HPP
