// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"


#include "logger.hpp"
#include "common/gapi_tests_common.hpp"
#include <opencv2/gapi/gpu/ggpukernel.hpp>
#include "opencl_kernels_test_gapi.hpp"


namespace ncvslideio
{

#ifdef HAVE_OPENCL

    static void reference_symm7x7_CPU(const ncvslideio::Mat& in, const ncvslideio::Mat& kernel_coeff, int shift, ncvslideio::Mat &out)
    {
        ncvslideio::Point anchor = { -1, -1 };
        double delta = 0;

        const int* ci = kernel_coeff.ptr<int>();

        float c_float[10];
        float divisor = (float)(1 << shift);
        for (int i = 0; i < 10; i++)
        {
            c_float[i] = ci[i] / divisor;
        }
        // J & I & H & G & H & I & J
        // I & F & E & D & E & F & I
        // H & E & C & B & C & E & H
        // G & D & B & A & B & D & G
        // H & E & C & B & C & E & H
        // I & F & E & D & E & F & I
        // J & I & H & G & H & I & J

        // A & B & C & D & E & F & G & H & I & J

        // 9 & 8 & 7 & 6 & 7 & 8 & 9
        // 8 & 5 & 4 & 3 & 4 & 5 & 8
        // 7 & 4 & 2 & 1 & 2 & 4 & 7
        // 6 & 3 & 1 & 0 & 1 & 3 & 6
        // 7 & 4 & 2 & 1 & 2 & 4 & 7
        // 8 & 5 & 4 & 3 & 4 & 5 & 8
        // 9 & 8 & 7 & 6 & 7 & 8 & 9

        float coefficients[49] =
        {
            c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9],
            c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
            c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
            c_float[6], c_float[3], c_float[1], c_float[0], c_float[1], c_float[3], c_float[6],
            c_float[7], c_float[4], c_float[2], c_float[1], c_float[2], c_float[4], c_float[7],
            c_float[8], c_float[5], c_float[4], c_float[3], c_float[4], c_float[5], c_float[8],
            c_float[9], c_float[8], c_float[7], c_float[6], c_float[7], c_float[8], c_float[9]
        };

        ncvslideio::Mat kernel = ncvslideio::Mat(7, 7, CV_32FC1);
        float* cf = kernel.ptr<float>();
        for (int i = 0; i < 49; i++)
        {
            cf[i] = coefficients[i];
        }

        ncvslideio::filter2D(in, out, CV_8UC1, kernel, anchor, delta, ncvslideio::BORDER_REPLICATE);
    }

    namespace gapi_test_kernels
    {
        G_TYPED_KERNEL(TSymm7x7_test, <GMat(GMat, Mat, int)>, "org.opencv.imgproc.symm7x7_test") {
            static GMatDesc outMeta(GMatDesc in, Mat, int) {
                return in.withType(CV_8U, 1);
            }
        };


        GAPI_GPU_KERNEL(GGPUSymm7x7_test, TSymm7x7_test)
        {
            static void run(const ncvslideio::UMat& in, const ncvslideio::Mat& kernel_coeff, int shift, ncvslideio::UMat &out)
            {
                if (ncvslideio::ocl::isOpenCLActivated())
                {
                    ncvslideio::Size size = in.size();
                    size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };

                    const ncvslideio::String moduleName = "gapi";
                    ncvslideio::ocl::ProgramSource source(moduleName, "symm7x7", opencl_symm7x7_src, "");

                    static const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE", "BORDER_UNDEFINED" };
                    std::string build_options = " -D BORDER_CONSTANT_VALUE=" + std::to_string(0) +
                        " -D " + borderMap[1] +
                        " -D SCALE=1.f/" + std::to_string(1 << shift) + ".f";

                    ncvslideio::String errmsg;
                    ncvslideio::ocl::Program program(source, build_options, errmsg);
                    if (program.ptr() == NULL)
                    {
                        CV_Error_(ncvslideio::Error::OpenCLInitError, ("symm_7x7_test Can't compile OpenCL program: = %s with build_options = %s\n", errmsg.c_str(), build_options.c_str()));
                    }
                    if (!errmsg.empty())
                    {
                        std::cout << "OpenCL program build log:" << std::endl << errmsg << std::endl;
                    }

                    ncvslideio::ocl::Kernel kernel("symm_7x7_test", program);
                    if (kernel.empty())
                    {
                        CV_Error(ncvslideio::Error::OpenCLInitError, "symm_7x7_test Can't get OpenCL kernel\n");
                    }

                    ncvslideio::UMat gKer;
                    kernel_coeff.copyTo(gKer);

                    int tile_y = 0;

                    int idxArg = kernel.set(0, ncvslideio::ocl::KernelArg::PtrReadOnly(in));
                    idxArg = kernel.set(idxArg, (int)in.step);
                    idxArg = kernel.set(idxArg, (int)size.width);
                    idxArg = kernel.set(idxArg, (int)size.height);
                    idxArg = kernel.set(idxArg, ncvslideio::ocl::KernelArg::PtrWriteOnly(out));
                    idxArg = kernel.set(idxArg, (int)out.step);
                    idxArg = kernel.set(idxArg, (int)size.height);
                    idxArg = kernel.set(idxArg, (int)size.width);
                    idxArg = kernel.set(idxArg, (int)tile_y);
                    idxArg = kernel.set(idxArg, ncvslideio::ocl::KernelArg::PtrReadOnly(gKer));

                    if (!kernel.run(2, globalsize, NULL, false))
                    {
                        CV_Error(ncvslideio::Error::OpenCLApiCallError, "symm_7x7_test OpenCL kernel run failed\n");
                    }
                }
                else
                {
                    //CPU fallback
                    ncvslideio::Mat in_Mat, out_Mat;
                    in_Mat = in.getMat(ACCESS_READ);
                    out_Mat = out.getMat(ACCESS_WRITE);
                    reference_symm7x7_CPU(in_Mat, kernel_coeff, shift, out_Mat);
                }
            }
        };

        ncvslideio::GKernelPackage gpuTestPackage = ncvslideio::gapi::kernels
            <GGPUSymm7x7_test
            >();

    } // namespace gapi_test_kernels
#endif //HAVE_OPENCL

} // namespace ncvslideio


namespace opencv_test
{

#ifdef HAVE_OPENCL

using namespace ncvslideio::gapi_test_kernels;

TEST(GPU, Symm7x7_test)
{
    const auto sz = ncvslideio::Size(1280, 720);
    ncvslideio::Mat in_mat = ncvslideio::Mat::eye(sz, CV_8UC1);
    ncvslideio::Mat out_mat_gapi(sz, CV_8UC1);
    ncvslideio::Mat out_mat_ocv(sz, CV_8UC1);
    ncvslideio::Scalar mean = ncvslideio::Scalar(127.0f);
    ncvslideio::Scalar stddev = ncvslideio::Scalar(40.f);
    ncvslideio::randn(in_mat, mean, stddev);

    //Symm7x7 coefficients and shift
    int coefficients_symm7x7[10] = { 1140, -118, 526, 290, -236, 64, -128, -5, -87, -7 };
    int shift = 10;
    ncvslideio::Mat kernel_coeff(10, 1, CV_32S);
    int* ci = kernel_coeff.ptr<int>();
    for (int i = 0; i < 10; i++)
    {
        ci[i] = coefficients_symm7x7[i];
    }

    // Run G-API
    ncvslideio::GMat in;
    auto out = TSymm7x7_test::on(in, kernel_coeff, shift);
    ncvslideio::GComputation comp(ncvslideio::GIn(in), ncvslideio::GOut(out));

    auto cc = comp.compile(ncvslideio::descr_of(in_mat), ncvslideio::compile_args(gpuTestPackage));
    cc(ncvslideio::gin(in_mat), ncvslideio::gout(out_mat_gapi));

    // Run OpenCV
    reference_symm7x7_CPU(in_mat, kernel_coeff, shift, out_mat_ocv);

    compare_f cmpF = AbsSimilarPoints(1, 0.05).to_compare_f();

    // Comparison //////////////////////////////////////////////////////////////
    {
        EXPECT_TRUE(cmpF(out_mat_gapi, out_mat_ocv));
        EXPECT_EQ(out_mat_gapi.size(), sz);
    }
}
#endif

} // namespace opencv_test
